# Center-Edge-Based Gaussianized Distribution Calibration for Few-Shot Images Classification
PyTorch implementation of the paper Center-Edge-Based Gaussianized Distribution Calibration for Few-Shot Images Classification.

# Requirement
This repo was tested with Ubuntu 18.04.5 LTS, Python 3.10, Pytorch 2.0.0, and CUDA 11.8. You will need at least 64GB RAM and 24GB VRAM(i.e. Nvidia RTX-3090) for running full experiments in this repo.

# Datasets
* **_mini_ Imagenet:**
Please follow [miniImageNet](https://github.com/yaoyao-liu/mini-imagenet-tools) to obtain the _mini_ ImageNet dataset and put it in ./filelists/miniImagenet/.<br>
* **CIFAR-FS:**
Please follow [CIFAR-FS](https://github.com/mrkshllr/FewTURE/blob/main/datasets/download_cifar_fs.sh) to obtain the CIFAR-FS dataset and put it in ./filelists/cifar/.<br>
* **CUB:**
Please follow [CUB](https://github.com/cyizhuo/CUB-200-2011-dataset) to obtain the CUB_200_2011 dataset and put it in ./filelists/CUB/.<br>
***
After download all datasets, please run the make_josn.py in each dataset folder to generate json files which include filepath-label pairs of each image.
# Pretraining
You can directly download the extracted features from the link:<br>
https://drive.google.com/file/d/1kvdfKsyS6Atr9-T8wOIYexSPrdM9-nuy/view?usp=drive_link
***
After downloading the extracted features, put them in ./features/WideResNet28_10_S2M2_R/[dataset]/ respectively.
# Evaluate our CGDC
To evaluate our method, run: <br>
```
python CGDC.py
```
The code will conduct 5way1shot setting experiments on _mini_ ImageNet by default. You can change the experimental settings in configs/5ways and specify run 5way1shot experiments on CIFAR-FS like this:<br>
```
configid=[
  '5ways/CIFAR-FS_1s5w.json',
  '5ways/CIFAR-FS_5s5w.json',
  '5ways/miniImagenet_1s5w.json',
  '5ways/miniImagenet_5s5w.json',
  '5ways/CUB_1s5w.json',
  '5ways/CUB_5s5w.json'
]
python CGDC.py --configid 5ways/CIFAR-FS_1s5w.json
```
# Reference
[Charting the Right Manifold: Manifold Mixup for Few-shot Learning](https://arxiv.org/pdf/1907.12087v3.pdf)<br>
[https://github.com/nupurkmr9/S2M2_fewshot](https://github.com/nupurkmr9/S2M2_fewshot)<br>
[Free Lunch for Few-Shot Learning: Distribution Calibration](https://openreview.net/forum?id=JWOiYxMG92s)<br>
[https://github.com/ShuoYang-1998/Few_Shot_Distribution_Calibration](https://github.com/ShuoYang-1998/Few_Shot_Distribution_Calibration)<br>
[L-GM_loss_pytorch](https://github.com/ChaofWang/L-GM_loss_pytorch)<br>
[CPEA](https://github.com/FushengHao/CPEA)
