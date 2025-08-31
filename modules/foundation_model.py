import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2ForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'

class Qwen2ForVQA(nn.Module):
    def __init__(self, model_path, lora_r=4, lora_alpha=16, lora_dropout=0.1):
        super().__init__()
        model = Qwen2ForCausalLM.from_pretrained(model_path).to(device)


        # LoRA configuration
        lora_config = LoraConfig(
            r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj"]
        )

        self.model = get_peft_model(model, lora_config).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    def forward(self, audio_embeds, img_embeds, text_embeds, attention_mask, labels=None):
        input_embeds = torch.cat([img_embeds, audio_embeds, text_embeds], dim=1).to(device)

        answer_token_ids = labels  # shape: [batch_size, 1]
        answer_embeds = self.model.get_input_embeddings()(answer_token_ids)


        full_input_embeds = torch.cat([input_embeds, answer_embeds], dim=1) 


        extra_column = torch.ones(attention_mask.shape[0], 1, device=attention_mask.device)
        full_attention_mask =torch.cat([attention_mask, extra_column], dim=1) 

        # build labels: length=total_seq_len (the last token is the answer)
        # using -100 to ignore the loss of tokens except the last one
        batch_size, seq_len = full_input_embeds.shape[:2]
        labels_id = torch.full((batch_size, seq_len), -100, device=full_input_embeds.device)
        labels_id[:, -1:] = answer_token_ids


        # or the forward process will still use the raw input_embeds [batch_size, 4383, dim] rather than full_input_embeds [batch_size, seq_len, dim].
        input_embeds = full_input_embeds
        attention_mask = full_attention_mask
        labels = labels_id # or labels will still be [batch_size, 1] at the forward process.
        
        # forward
        outputs = self.model(
            inputs_embeds=input_embeds, # [batch_size, seq_len, dim]
            attention_mask=attention_mask, # [batch_size, seq_len]
            labels=labels, # torch.Size([batch_size, seq_len)
            logits_to_keep=0
        )

        loss = outputs.loss
        logits = outputs.logits  # [batch_size, seq_len, vocab_size]
        pred_token_logits = logits[:, -2, :]  # [batch_size, vocab_size] Note: It's "-2" here, not "-1".

        # # get predicted token_ids
        # pred_token_ids = torch.argmax(pred_token_logits, dim=-1)  # [batch_size]

        # # decode predicted token_ids
        # decoded_predictions = [self.tokenizer.decode(token_id, skip_special_tokens=True) for token_id in pred_token_ids]
        # decoded_targets = [self.tokenizer.decode(label_id, skip_special_tokens=True) for label_id in labels[:, -1]]
        # for i, (pred, target) in enumerate(zip(decoded_predictions, decoded_targets)):
        #     print(f"[{i}] Pred: {pred}  |  Target: {target}")



        return {"loss": loss, "logits": pred_token_logits}
