import os
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoProcessor, ASTModel, VivitImageProcessor, VivitModel
from modules.foundation_model import Qwen2ForVQA
from modules.token_transmitter import text_transmitter, img_transmitter, audio_transmitter
from modules.channel import ChannelSimulator
from modules.token_receiver import TokenReceiver
from utils.build_avqa_dataset import AVQADataset, AVQACollactor
import matplotlib.pyplot as plt
from openpyxl import Workbook

DO_TRAIN = True
DO_VAL = True
DO_TEST = True
USE_CKPT = True

SEED = 1111
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

# initialize
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model_path = "/mnt/Qwen/Qwen2.5-1.5B"
model = Qwen2ForVQA(model_path).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_path)
img_encoder = VivitModel.from_pretrained("/datadisk/google/vivit-b-16x2").to(device)
img_processor = VivitImageProcessor.from_pretrained("/datadisk/google/vivit-b-16x2")
audio_encoder = ASTModel.from_pretrained("/datadisk/MIT/ast-finetuned-audioset-10-10-0.4593").to(device)
audio_processor = AutoProcessor.from_pretrained("/datadisk/MIT/ast-finetuned-audioset-10-10-0.4593")

hidden_size = model.model.config.hidden_size
audio_token_len = 128
img_token_len = 128
snr = 12.0


text_tx = text_transmitter(model=model, text_tokenizer=tokenizer, if_train_encoder=False).to(device)
img_tx = img_transmitter(img_encoder=img_encoder, img_processor=img_processor, img_token_len=img_token_len, output_dim=hidden_size, if_train_encoder=False).to(device)
audio_tx = audio_transmitter(audio_encoder=audio_encoder, audio_processor=audio_processor, audio_token_len=audio_token_len, output_dim=hidden_size, if_train_encoder=False).to(device)
text_receiver = TokenReceiver(hidden_size, hidden_size).to(device)
img_receiver = TokenReceiver(hidden_size, hidden_size).to(device)
audio_receiver = TokenReceiver(hidden_size, hidden_size).to(device)
channel = ChannelSimulator(snr_db=snr)

if USE_CKPT:
    print("-----------------CKPT LOADING--------------")
    checkpoint = torch.load('/mnt/checkpoints/007-multimodal_transceivers.pth', map_location='cpu')
    img_tx.load_state_dict(checkpoint['img_tx'])
    audio_tx.load_state_dict(checkpoint['audio_tx'])
    text_receiver.load_state_dict(checkpoint['text_receiver'])
    img_receiver.load_state_dict(checkpoint['img_receiver'])
    audio_receiver.load_state_dict(checkpoint['audio_receiver'])   

# optimizer
optimizer = torch.optim.AdamW(
    list(model.parameters()) + list(text_tx.parameters()) + 
    list(img_tx.parameters()) + list(audio_tx.parameters()) + 
    list(text_receiver.parameters()) + list(audio_receiver.parameters()) +
    list(img_receiver.parameters()) , lr=1e-4
)

# lists to store results
train_losses = []
val_losses = []
test_accuracies = []

# ---------- train_step, val_step, test_step ----------
def train_step(batch):
    model.train()
    text_tx.train()
    img_tx.train()
    audio_tx.train()
    text_receiver.train()
    img_receiver.train()
    audio_receiver.train()

    text_embeds = text_receiver(channel(batch["text_embeds"].to(device)))
    img_embeds = img_receiver(channel(batch["img_embeds"].to(device)))
    audio_embeds = audio_receiver(channel(batch["audio_embeds"].to(device)))
    labels = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    outputs = model(audio_embeds, img_embeds, text_embeds, attention_mask, labels)
    loss = outputs["loss"]

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def val_step(batch):
    model.eval()
    text_tx.eval()
    img_tx.eval()
    audio_tx.eval()
    text_receiver.eval()
    img_receiver.eval()
    audio_receiver.eval()

    text_embeds = text_receiver(channel(batch["text_embeds"].to(device)))
    img_embeds = img_receiver(channel(batch["img_embeds"].to(device)))
    audio_embeds = audio_receiver(channel(batch["audio_embeds"].to(device)))
    labels = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    outputs = model(audio_embeds, img_embeds, text_embeds, attention_mask, labels)
    return outputs["loss"].item()


def test_step(batch):
    model.eval()
    text_tx.eval()
    img_tx.eval()
    audio_tx.eval()
    text_receiver.eval()
    img_receiver.eval()
    audio_receiver.eval()

    text_embeds = text_receiver(channel(batch["text_embeds"].to(device)))
    img_embeds = img_receiver(channel(batch["img_embeds"].to(device)))
    audio_embeds = audio_receiver(channel(batch["audio_embeds"].to(device)))
    labels = batch["labels"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    outputs = model(audio_embeds, img_embeds, text_embeds, attention_mask, labels)
    preds = torch.argmax(outputs["logits"], dim=-1)
    
    labels = labels.squeeze(1)
    correct = (preds == labels).sum().item()
    total = labels.size(0)
    return correct, total

# ---------- save_results_to_excel_and_plot ----------
def save_results_to_excel_and_plot(train_losses, val_losses, test_accuracies, save_dir="avqa_results"):
    os.makedirs(save_dir, exist_ok=True)

    wb = Workbook()
    ws = wb.active
    ws.title = "Metrics"
    ws.append(["Iter", "Train Loss", "Val Loss", "Test Accuracy"])

    max_len = max(len(train_losses), len(val_losses), len(test_accuracies))
    for i in range(max_len):
        iter_idx = i + 1
        train_loss = train_losses[i] if i < len(train_losses) else ""
        val_loss = val_losses[i] if i < len(val_losses) else ""
        test_acc = test_accuracies[i] if i < len(test_accuracies) else ""
        ws.append([iter_idx, train_loss, val_loss, test_acc])

    excel_path = os.path.join(save_dir, "avqa_results.xlsx")
    wb.save(excel_path)
    print(f"[Saved] Excel saved to {excel_path}")

    plt.figure(figsize=(10, 6))
    if train_losses:
        plt.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss")
    if val_losses:
        plt.plot(range(1, len(val_losses)+1), val_losses, label="Val Loss")
    if test_accuracies:
        plt.plot(range(1, len(test_accuracies)+1), test_accuracies, label="Test Accuracy")

    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.title("Train / Val Loss & Test Accuracy per Iteration")
    plt.legend()
    plt.grid(True)
    fig_path = os.path.join(save_dir, "avqa_plot.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"[Saved] Plot saved to {fig_path}")

# ------main-------
if __name__ == '__main__':
    train_dataset = AVQADataset("/datadisk/datasets/MUSIC-AVQA", dataset_name="avqa-train")
    val_dataset = AVQADataset("/datadisk/datasets/MUSIC-AVQA", dataset_name="avqa-val")
    test_dataset = AVQADataset("/datadisk/datasets/MUSIC-AVQA", dataset_name="avqa-test")

    collator = AVQACollactor(tokenizer, text_tx, img_tx, audio_tx)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True, collate_fn=collator)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True, collate_fn=collator)

    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    test_iter = iter(test_loader)

    max_iters = 300
    for iter_idx in range(max_iters):
        # print(f"\n---------- Iteration {iter_idx+1}/{max_iters} ----------")

        # --- train ---
            if DO_TRAIN:
                train_batch = next(train_iter)
                loss = train_step(train_batch)
                train_losses.append(loss)
                print(f"[Train] Iter {iter_idx+1} Loss: {loss:.4f}")

            # --- val ---
            if DO_VAL:
                val_batch = next(val_iter)
                val_loss = val_step(val_batch)
                val_losses.append(val_loss)
                print(f"[Val] Iter {iter_idx+1} Loss: {val_loss:.4f}")

            torch.cuda.empty_cache()
    
    if DO_TEST:
        total_correct = 0
        total_samples = 0
        for test_batch in test_loader:
            correct, total = test_step(test_batch)
            total_correct += correct
            total_samples += total
            print(f"[Test Progress] Processed {total_samples}/{len(test_loader.dataset)} samples")
            print(f"[Test] Correct: {total_correct}, Total: {total_samples}")
            accuracy = total_correct / total_samples
            print(f"[Test] Accuracy: {accuracy:.4f}")
            print("------------------------------------------------------------")

        overall_acc = total_correct / total_samples if total_samples > 0 else 0.0
        test_accuracies.append(overall_acc)
        print(f"[Test] Overall Accuracy: {overall_acc:.4f}")
    

    save_results_to_excel_and_plot(train_losses, val_losses, test_accuracies)
