# Task 2 of Final project of "Neural Network and Deep Learning"

-----

Table of Contents
=================

* [This is the task 2 of final project of "Neural Network and Deep Learning"](#this-is-the-final-project-of-neural-network-and-deep-learning)
   * [Introduction](#introduction)
   * [Dependencies](#dependencies)
   * [Structure](#structure)
   * [Key Features](#key-features)
   * [Training](#training)
   * [Trained Models](#trained-models)
   * [Testing](#testing)
   * [Results](#results)

## Introduction

The project is the final project of **DATA620004**. 
In this project, we use the **ResNet-18** and **ViT** to run image classification on *CIFAR-100* dataset, and use three different data augmentation methods ([Mixup](https://arxiv.org/pdf/1710.09412v2.pdf), [Cutmix](https://arxiv.org/pdf/1905.04899.pdf), [Cutout](https://arxiv.org/pdf/1708.04552.pdf)).

## Dependencies

- `PyTorch`
- `numpy`
- `matplotlib`
- `einops`

## Structure
```angular2html
.
├── README.md
├── augmentation
│   ├── __init__.py
│   ├── cutmix.py
│   ├── cutout.py
│   └── mixup.py
├── conf
│   ├── __init__.py
│   └── global_settings.py
├── data
│   ├── __init__.py
│   └── dataloader.py
├── functions.py
├── models
│   ├── attention.py
│   ├── densenet.py
│   ├── googlenet.py
│   ├── inceptionv3.py
│   ├── inceptionv4.py
│   ├── mobilenet.py
│   ├── mobilenetv2.py
│   ├── nasnet.py
│   ├── preactresnet.py
│   ├── resnet.py
│   ├── resnext.py
│   ├── rir.py
│   ├── senet.py
│   ├── shufflenet.py
│   ├── shufflenetv2.py
│   ├── squeezenet.py
│   ├── stochasticdepth.py
│   ├── vgg.py
│   ├── wideresidual.py
│   ├── vit.py
│   └── xception.py
├── plot.py
├── res
│   └── *.csv
├── resnet18_grid_search.sh
├── run.sh
├── test.py
├── train.py
├── utils.py
└── vit_grid_search.sh
```

- `models`: There are many models such as **ResNet**, **ViT**, **VGG**, **DenseNet** and you can also put your models in the file. (**Note: The implementation of these models is a reference to [pytorch-cifar100](https://github.com/weiaicunzai/pytorch-cifar100)**)
- `augmentation`: This file contains three different data augmentation methods: Mixup, Cutmix and Cutout
- `conf`: The settings of training models can be found in this file, such as the number of epochs, the path for saving models
- `data`: The `DataLoader` for *CIFAR10* and *CIFAR100* is in this module
- `res`: The csv files of all results of accuracy at each epoch
- `utils.py`: Some useful functions
- `functions.py`: The functions for training evaluating models
- `plot.py`: Image visualization of the *CIFAR100* dataset
- `train.py`: Training the model
- `test.py`: Testing the model
- `run.sh`: a simple bash script to run the **ResNet-18** and **ViT** models with best parameters
- `resnet18_grid_search.sh`: a bash script to run hyper-parameter grid search for **ResNet-18**
- `vit_grid_search.sh`: a bash script to run hyper-parameter grid search for **ViT**

## Key Features

Data augmentation:
- [Mixup](https://arxiv.org/pdf/1710.09412v2.pdf). The code is in `./augmentation/mixup.py`, and is used in lines 43~45 of `./functions.py`
- [Cutmix](https://arxiv.org/pdf/1905.04899.pdf). The code is in `./augmentation/cutmix.py`, and is used in lines 47~49 of `./functions.py`
- [Cutout](https://arxiv.org/pdf/1708.04552.pdf). The code is in `./augmentation/cutout.py`, and is used in line 38 of `./data/dataloader.py`

Hyper-parameter Tuning:
- We use grid search to tune the hyper-parameters of both **ResNet-18** and **ViT** models by using bash scripts `resnet18_grid_search.sh` and `vit_grid_search.sh` 


## Training

Run command below to reproduce results on *CIFAR100* (dataset auto-downloads on first use).

```bash
$ python main.py
    --resume False                  # resume training, default False
    --device cuda:0                 # the device, default cpu
    --lr 0.1                        # the initial learning rate, default 0.1
    --optimizer sgd                 # the optimizer, default sgd and you can also use adam for Adam optimizer
    --criterion CrossEntropyLoss    # the loss criterion, default CrossEntropyLoss
    --batch-size 128                # the batch size, default 128
    --net resnet18                  # the neural network, default resnet18
    --data cifar100                 # the dataset, default cifar100
    --warmup-num 0                  # the epoch number of warmup, default 0, means no warmup
```

## Trained Models

You can download our trained model from [This Link (Password: qqpb)](https://pan.baidu.com/s/10g1PU_lTJH7ghU21uZkGzw). 
It has a folder named `checkpoint`, which should be put it in current folder.

## Testing

Run command below to reproduce results on *CIFAR100* (dataset auto-downloads on first use), which automatically uses the latest checkpoint for the specified model.

```bash
$ python test.py
    --device cuda:0                 # the device, default cpu
    --net resnet18                  # the neural network, default resnet18
    --data cifar100                 # the dataset, default cifar100
    --batch-size 128                # the batch size, default 128
```

## Results

| **Dataset** | **Model** | **Data Augmentation** | **Parameter** | **Accuracy** | **Uncertainty** |
|:----------------:|:--------------:|:---------------------------:|:------------------:|:---------------------------:|:-----------------------------:|
|     CIFAR-10     |      VGG16     |             None            |               \              |            93.80            |                   0.28                  |
|     CIFAR-10     |      VGG16     |            Mixup            |              0.2             |            94.21            |                   0.22                  |
|     CIFAR-10     |      VGG16     |            Mixup            |              0.4             |            94.36            |                   0.18                  |
|     CIFAR-10     |      VGG16     |            Mixup            |              0.6             |            94.31            |                   0.09                  |
|     CIFAR-10     |      VGG16     |            Mixup            |              0.8             |            94.24            |                   0.25                  |
|     CIFAR-10     |      VGG16     |            Mixup            |               1              |            94.17            |                   0.22                  |
|     CIFAR-10     |      VGG16     |            Cutout           |               4              |            93.89            |                   0.16                  |
|     CIFAR-10     |      VGG16     |            Cutout           |               8              |            94.20            |                   0.26                  |
|     CIFAR-10     |      VGG16     |            Cutout           |              12              |            94.40            |                   0.15                  |
|     CIFAR-10     |      VGG16     |            Cutout           |              16              |            94.72            |                   0.08                  |
|     CIFAR-10     |      VGG16     |            Cutout           |              20              |            94.65            |                   0.08                  |
|     CIFAR-10     |      VGG16     |            Cutmix           |              0.2             |            94.86            |                   0.08                  |
|     CIFAR-10     |      VGG16     |            Cutmix           |              0.4             |            94.89            |                   0.14                  |
|     CIFAR-10     |      VGG16     |            Cutmix           |              0.6             |            94.87            |                   0.12                  |
|     CIFAR-10     |      VGG16     |            Cutmix           |              0.8             |            94.82            |                   0.14                  |
|     CIFAR-10     |      VGG16     |            Cutmix           |               1              |            94.83            |                   0.15                  |
|     CIFAR-10     |    ResNet-18   |             None            |               \              |            95.06            |                   0.16                  |
|     CIFAR-10     |    ResNet-18   |            Mixup            |              0.2             |            95.41            |                   0.14                  |
|     CIFAR-10     |    ResNet-18   |            Cutout           |              16              |            95.64            |                   0.19                  |
|     CIFAR-10     |    ResNet-18   |            Cutmix           |              0.2             |            96.12            |                   0.13                  |
|     CIFAR-100    |      VGG16     |             None            |          \         |            72.44            |              0.55             |
|     CIFAR-100    |      VGG16     |            Mixup            |         0.2        |            73.66            |              0.21             |
|     CIFAR-100    |      VGG16     |            Mixup            |         0.4        |            73.75            |              0.18             |
|     CIFAR-100    |      VGG16     |            Mixup            |         0.6        |            73.43            |              0.33             |
|     CIFAR-100    |      VGG16     |            Mixup            |         0.8        |            73.67            |              0.32             |
|     CIFAR-100    |      VGG16     |            Mixup            |          1         |            73.51            |              0.46             |
|     CIFAR-100    |      VGG16     |            Cutout           |          4         |            72.35            |              0.30             |
|     CIFAR-100    |      VGG16     |            Cutout           |          8         |            72.18            |              0.27             |
|     CIFAR-100    |      VGG16     |            Cutout           |         12         |            72.53            |              0.22             |
|     CIFAR-100    |      VGG16     |            Cutout           |         16         |            72.56            |              0.16             |
|     CIFAR-100    |      VGG16     |            Cutout           |         20         |            72.26            |              0.14             |
|     CIFAR-100    |      VGG16     |            Cutmix           |         0.2        |            74.74            |              0.23             |
|     CIFAR-100    |      VGG16     |            Cutmix           |         0.4        |            74.59            |              0.23             |
|     CIFAR-100    |      VGG16     |            Cutmix           |         0.6        |            74.37            |              0.42             |
|     CIFAR-100    |      VGG16     |            Cutmix           |         0.8        |            74.35            |              0.37             |
|     CIFAR-100    |      VGG16     |            Cutmix           |          1         |            74.20            |              0.15             |
|     CIFAR-100    |    ResNet-18   |             None            |          \         |            76.18            |              0.18             |
|     CIFAR-100    |    ResNet-18   |            Mixup            |         0.2        |            77.39            |              0.19             |
|     CIFAR-100    |    ResNet-18   |            Cutout           |         16         |            75.41            |              0.28             |
|     CIFAR-100    |    ResNet-18   |            Cutmix           |         0.2        |            79.12            |              0.35             |