# CGDC
The code of paper: 《Edge Matters: Center-Edge-Based Gaussianized Distribution Calibration for Few-Shot Images Classification》
# Requirement
This repo was tested with Ubuntu 18.04.5 LTS, Python 3.10, Pytorch 2.0.0, and CUDA 11.8. You will need at least 64GB RAM and 24GB VRAM(i.e. Nvidia RTX-3090) for running full experiments in this repo.
# Datasets
## _mini_Imagenet
Please follow [mini-imagenet-tools](https://github.com/yaoyao-liu/mini-imagenet-tools) to obtain the miniImageNet dataset and put it in ./filelists/miniImagenet/.
## CIFAR-FS
Please follow [download_cifar_fs.sh](https://github.com/mrkshllr/FewTURE/blob/main/datasets/download_cifar_fs.sh) to obtain the CIFAR-FS dataset and put it in ./filelists/cifar/.
## CUB
Please follow [CUB](https://www.kaggle.com/datasets/cyizhuo/cub-200-2011-by-classes-folder) to obtain the CUB_200_2011 dataset and put it in ./filelists/CUB/.
# Reference
[Charting the Right Manifold: Manifold Mixup for Few-shot Learning](https://arxiv.org/pdf/1907.12087v3.pdf)<br>
[https://github.com/nupurkmr9/S2M2_fewshot](https://github.com/nupurkmr9/S2M2_fewshot)<br>
[Free Lunch for Few-Shot Learning: Distribution Calibration](https://openreview.net/forum?id=JWOiYxMG92s)<br>
[https://github.com/ShuoYang-1998/Few_Shot_Distribution_Calibration](https://github.com/ShuoYang-1998/Few_Shot_Distribution_Calibration)<br>
[L-GM_loss_pytorch](https://github.com/ChaofWang/L-GM_loss_pytorch)<br>
[CPEA](https://github.com/FushengHao/CPEA)
