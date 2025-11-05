# Task-Oriented Multimodal Token Transmission in Resource-Constrained Multiuser Networks

This repository is the official implementation of the paper: Task-Oriented Multimodal Token Transmission in Resource-Constrained Multiuser Networks.
[IEEE WCL](https://ieeexplore.ieee.org/document/11224844) [arXiv](https://arxiv.org/abs/2505.07841).

---

## Dataset Preparation

The training consists of two stages: **cross-modal alignment** and **task-oriented fine-tuning**.

- **Cross-modal alignment**: [VALOR-32K](https://casia-iva-group.github.io/projects/VALOR/data.html) dataset.  
- **Task-oriented fine-tuning**: [MUSIC-AVQA](https://gewu-lab.github.io/MUSIC-AVQA/) dataset.  

After downloading the raw data, run `utils/raw_video_preprocess.py` to split videos into image frames and extract audio. Using VALOR as an example, the processed folder structure should be:

```
VALOR/
├── annotations/
│   ├── desc_train
│   ├── desc_val
│   └── desc_test
└── extraction/
    ├── ast_audio/
    │   ├── xxxxx0.wav
    │   ├── xxxxx1.wav
    └── vivit_frames/
        ├── xxxxx0/
        │   ├── frame_0001.png
        │   ├── frame_0002.png
        └── xxxxx1/
            ├── frame_0001.png
            ├── frame_0002.png
```

---

## Model Preparation

- **Foundation model / Text**: [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B)  
- **Audio**: [AST (Audio Spectrogram Transformer)](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593)  
- **Visual**: [ViViT (Video Vision Transformer)](https://huggingface.co/google/vivit-b-16x2)  

Models can be downloaded from HuggingFace or using `utils/hf-download.py`.

---

## Training

### Cross-Modal Alignment  
```bash
python task_align.py
```
- Checkpoints will be saved in `checkpoint/xxx.pth` (create the folder beforehand).

We also provide **t-SNE plots** corresponding to different contrastive temperatures.

#### t-SNE Plots with Different Contrastive Temperatures τ
<p align="center">
  <img src="imgs/000.png" width="200"/>
  <img src="imgs/003.png" width="200"/>
  <img src="imgs/007.png" width="200"/>
  <img src="imgs/013.png" width="200"/>
</p>
From left to right: without alignment, τ = 0.03, τ = 0.07, τ = 0.13.

### Task-Oriented Fine-Tuning  
```bash
python task_avqa.py
```

It’s noteworthy that we evaluate the autoregressive generation capability of the foundation model rather than using a classification head as the origin implementation does [(MUSIC-AVQA)](https://github.com/GeWu-Lab/MUSIC-AVQA). Specifically, the output space of our model spans the entire vocabulary instead of a limited set of label indices.  See `modules/foundation_model.py` for details.

---

For beginners, it is recommended to set breakpoints to check the shape of tensors at each step of the process :)

---

## Citation

If you find our work helpful, please consider citing:

```bibtex
@ARTICLE{junhe2025wcl,
  author={Zhang, Junhe and Ni, Wanli and Wang, Pengwei and Wang, Dongyu},
  journal={IEEE Wireless Communications Letters}, 
  title={Task-Oriented Multimodal Token Transmission in Resource-Constrained Multiuser Networks}, 
  year={2025},
  doi={10.1109/LWC.2025.3628928},
  note={early access}
}
```
