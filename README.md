[![License](https://img.shields.io/badge/license-CC--BY%204.0-blue)](https://creativecommons.org/licenses/by/4.0/)
[![arXiv](https://img.shields.io/badge/cs.CV-%09arXiv%3A2205.00823-red)](https://arxiv.org/abs/2410.20508)

## Referring Human Pose and Mask Estimation In the Wild

This is the official implementation of our NeurIPS 2024 paper 
"[Referring Human Pose and Mask Estimation In the Wild](https://arxiv.org/abs/2410.20508)".

## Introduction

* We propose **Referring Human Pose and Mask Estimation (R-HPM)** in the wild, 
a new task that requires a unified model to predict both body keypoints and mask for a specified individual 
using text or positional prompts. 
This enables comprehensive and identity-aware human representations to enhance human-AI interaction.
* We introduce **RefHuman**, a benchmark dataset with comprehensive human annotations, including
<u>pose, mask, text and positional prompts</u> in unconstrained environments.
* We propose **UniPHD**, an end-to-end promptable model that supports various prompt types for R-HPM and achieves top-tier performance.

## ⭐ RefHuman Dataset

Our RefHuman dataset is available for download from [GoogleDrive](https://drive.google.com/drive/folders/128R4SMIC1BlO3bFNuHYO6jeYClZtGnA3?usp=drive_link),
with parsing code provided in ```./datasets/refhuman.py```.
```
path/to/refhuman/
├── images/  
└── RefHuman_train.json   # annotations for train split
└── RefHuman_val.json     # annotations for val split
```


## ⭐ Code

The code is being prepared and will be released before the conference.


## Citation
☀️ If you find this work useful, please kindly cite our paper! ☀️

```
@InProceedings{Miao_2024_NeurIPS,
    author    = {Miao, Bo and Feng, Mingtao and Wu, Zijie and Bennamoun, Mohammed and Gao, Yongsheng and Mian, Ajmal},
    title     = {Referring Human Pose and Mask Estimation In the Wild},
    booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
    year      = {2024},
}
```
