[![License](https://img.shields.io/badge/license-CC--BY%204.0-blue)](https://creativecommons.org/licenses/by/4.0/)
[![arXiv](https://img.shields.io/badge/cs.CV-%09arXiv%3A2205.00823-red)](https://arxiv.org/pdf/2410.20508)

## Referring Human Pose and Mask Estimation In the Wild


This is the official pytorch implementation of our NeurIPS 2024 paper 
"[Referring Human Pose and Mask Estimation In the Wild](https://arxiv.org/pdf/2410.20508)".


## Introduction

* We propose **Referring Human Pose and Mask Estimation (R-HPM)** in the wild, 
a new task that requires a unified model to predict both body keypoints and mask for a specified individual 
using text or positional prompts. 
This enables comprehensive and identity-aware human representations to enhance human-AI interaction.
* We introduce **RefHuman**, a benchmark dataset with comprehensive human annotations, including
<u>pose, mask, text and positional prompts</u> in unconstrained environments.
* We propose **UniPHD**, an end-to-end promptable model that supports various prompt types for R-HPM and achieves top-tier performance.

 
## ⭐ UniPHD
![method](figures/uniphd.png "model arch")

## ⭐ RefHuman Dataset
The images and validation split annotations for the RefHuman dataset are available for download from [GoogleDrive](https://drive.google.com/drive/folders/128R4SMIC1BlO3bFNuHYO6jeYClZtGnA3?usp=drive_link).

To request access to the training annotations, please fill out [this form](https://docs.google.com/document/d/1A-SOAu6DX2gIYID68vi-59f2wsZILs52/edit?usp=sharing&ouid=108610218897155132169&rtpof=true&sd=true) and send it to bomiaobbb@gmail.com using your .edu or company email address. We will respond as soon as possible.

The parsing code is provided in ```./datasets/refhuman.py```.

```
path/to/refhuman/
├── images/  
└── RefHuman_train.json   # annotations for train split
└── RefHuman_val.json     # annotations for val split
```

## Setup

The code is developed with ```python=3.9.0,pytorch=1.11.0,cuda=11.3```.

First, clone the repository locally and install related packages.
```
git clone https://github.com/bo-miao/RefHuman
```

```
pip install pycocotools timm termcolor opencv-python addict yapf scipy
```

Then, compile the CUDA operators.
```shell
cd models/uniphd/ops
python setup.py build install
python test.py  # check if correctly installed
```


## Evaluation on RefHuman

Our UniPHD checkpoint trained on **RefHuman** is available at [GoogleDrive](https://drive.google.com/file/d/1w4boI-xAMZaYn56bXWAzg-jUgndgEHiJ/view?usp=sharing).

The evaluation script is located in `./scripts/eval_coco_swint.sh`. (Please first download the pretrained [Swin-T checkpoint](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth))

You can also run the following command (use --eval_trigger to control the prompt type) :

```
GPU_NUM=4
BATCH=24
python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port 22222 \
    main.py -c config/uniphd.py \
    --backbone swin_T_224_1k \
    --options batch_size=${BATCH} \
    --resume ${CKPT_PATH} \
    --eval \
    --eval_trigger 'text'\
```

## Acknowledgements

This project is built on the open source repositories 
[SgMg](https://github.com/bo-miao/SgMg), [GroupPose](https://github.com/Michel-liu/GroupPose), and [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR).
Thanks them for their well-organized codes!


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
