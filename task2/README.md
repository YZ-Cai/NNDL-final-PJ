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

In this project, we use the **ResNet-18** and **ViT** to run image classification on *CIFAR-100* dataset, and use three different data augmentation methods ([Mixup](https://arxiv.org/pdf/1710.09412v2.pdf), [Cutmix](https://arxiv.org/pdf/1905.04899.pdf), [Cutout](https://arxiv.org/pdf/1708.04552.pdf)).
Moreover, we also try larger models such as ResNet-50, ResNet-152, ViT-base, and ViT-large, while the latter two models use different patch sizes in experiments.

## Dependencies

- `PyTorch`
- `numpy`
- `matplotlib`
- `einops`
- `tensorboard`
- `tabulate`
- `pandas`

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
│   ├── ViT.py
│   └── xception.py
├── res
│   └── *.csv
├── ResNet-18_grid_search.sh
├── run.sh
├── plot.py
├── test.py
├── train.py
├── utils.py
└── ViT_grid_search.sh
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
- `run.sh`: The simple bash script to run the **ResNet-18** and **ViT** models with best parameters
- `ResNet-18_grid_search.sh`: The bash script to run hyper-parameter grid search for **ResNet-18**
- `ViT_grid_search.sh`: The bash script to run hyper-parameter grid search for **ViT**

## Key Features

By modifying model hyper-parameters, both models have similar size:
- ResNet-18: 11,220,132 parameters
- ViT: 11,220,132 parameters

Data augmentation:
- [Mixup](https://arxiv.org/pdf/1710.09412v2.pdf). The code is in `./augmentation/mixup.py`, and is used in lines 43~45 of `./functions.py`
- [Cutmix](https://arxiv.org/pdf/1905.04899.pdf). The code is in `./augmentation/cutmix.py`, and is used in lines 47~49 of `./functions.py`
- [Cutout](https://arxiv.org/pdf/1708.04552.pdf). The code is in `./augmentation/cutout.py`, and is used in line 38 of `./data/dataloader.py`

Hyper-parameter Tuning:
- We use grid search to tune the hyper-parameters of both **ResNet-18** and **ViT** models by using bash scripts `ResNet-18_grid_search.sh` and `ViT_grid_search.sh` 


## Training

Run command below to reproduce results on *CIFAR100* (dataset auto-downloads on first use).

```bash
$ python train.py
    --resume False                  # resume training, default False
    --device cuda:0                 # the device, default cpu
    --lr 0.1                        # the initial learning rate, default 0.1
    --optimizer sgd                 # the optimizer, default SGD and you can also use adam for Adam optimizer
    --criterion CrossEntropyLoss    # the loss criterion, default CrossEntropyLoss
    --batch-size 128                # the batch size, default 128
    --net resnet18                  # the neural network, default resnet18
    --patch-size                    # the patch size for ViT, default 4
    --data cifar100                 # the dataset, default cifar100
    --warmup-num 0                  # the epoch number of warmup, default 0, means no warmup
```

## Trained Models

You can download our trained model from [this link (password: NNDL)](https://pan.baidu.com/s/1l_gyF6x9SNJnw8hQthxrZw). 
It has a folder named `task2/checkpoint`, which should be put it in current folder.

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

| Model     | Optimizer   |   Learning rate |   Batch size | ViT patch size   |   Test accuracy |
|:----------|:------------|----------------:|-------------:|:-----------------|----------------:|
| ResNet-50  | SGD         |           0.1   |           64 | -                |          0.7946 |
| ResNet-152 | SGD         |           0.1   |           64 | -                |          0.7911 |
| ResNet-18  | SGD         |           0.1   |           64 | -                |          0.7826 |
| ResNet-18  | SGD         |           0.01  |           32 | -                |          0.7814 |
| ResNet-18  | SGD         |           0.1   |          128 | -                |          0.7757 |
| ResNet-18  | SGD         |           0.1   |          256 | -                |          0.7687 |
| ResNet-18  | SGD         |           0.1   |           32 | -                |          0.7683 |
| ResNet-18  | SGD         |           0.01  |           64 | -                |          0.7672 |
| ResNet-18  | SGD         |           0.001 |           32 | -                |          0.7567 |
| ResNet-18  | SGD         |           0.01  |          128 | -                |          0.7561 |
| ResNet-18  | Adam        |           0.001 |          256 | -                |          0.7468 |
| ResNet-18  | SGD         |           0.01  |          256 | -                |          0.7463 |
| ResNet-18  | Adam        |           0.001 |          128 | -                |          0.7319 |
| ResNet-18  | SGD         |           0.001 |           64 | -                |          0.729  |
| ResNet-18  | SGD         |           0.001 |          128 | -                |          0.6956 |
| ResNet-18  | Adam        |           0.001 |           64 | -                |          0.6788 |
| ResNet-18  | Adam        |           0.001 |           32 | -                |          0.6383 |
| ResNet-18  | SGD         |           0.001 |          256 | -                |          0.6377 |
| ViT       | SGD         |           0.01  |           64 | 4                |          0.5362 |
| ViT-base  | SGD         |           0.01  |           64 | 4                |          0.5341 |
| ViT-large | SGD         |           0.01  |           64 | 4                |          0.5337 |
| ViT       | SGD         |           0.01  |           32 | 4                |          0.5319 |
| ViT       | SGD         |           0.01  |          128 | 4                |          0.5261 |
| ViT       | SGD         |           0.001 |           32 | 4                |          0.5089 |
| ViT       | SGD         |           0.01  |          256 | 4                |          0.5077 |
| ViT       | SGD         |           0.1   |          256 | 4                |          0.4909 |
| ViT       | SGD         |           0.001 |           64 | 4                |          0.4741 |
| ResNet-18  | Adam        |           0.01  |          256 | -                |          0.4565 |
| ViT       | Adam        |           0.001 |          256 | 4                |          0.4492 |
| ViT       | SGD         |           0.1   |          128 | 4                |          0.4407 |
| ViT-large | SGD         |           0.01  |           64 | 8                |          0.4384 |
| ViT       | Adam        |           0.001 |          128 | 4                |          0.4297 |
| ResNet-18  | Adam        |           0.01  |          128 | -                |          0.4273 |
| ViT       | SGD         |           0.001 |          128 | 4                |          0.4129 |
| ViT-base  | SGD         |           0.01  |           64 | 8                |          0.4058 |
| ViT       | Adam        |           0.001 |           64 | 4                |          0.4001 |
| ResNet-18  | Adam        |           0.01  |           64 | -                |          0.3931 |
| ViT       | SGD         |           0.1   |           64 | 4                |          0.3895 |
| ViT       | Adam        |           0.001 |           32 | 4                |          0.3802 |
| ResNet-18  | Adam        |           0.01  |           32 | -                |          0.3418 |
| ViT       | SGD         |           0.001 |          256 | 4                |          0.332  |
| ResNet-18  | Adam        |           0.1   |          128 | -                |          0.3258 |
| ResNet-18  | Adam        |           0.1   |           64 | -                |          0.2939 |
| ViT-large | SGD         |           0.01  |           64 | 16               |          0.2864 |
| ViT       | Adam        |           0.01  |          256 | 4                |          0.2844 |
| ViT       | SGD         |           0.1   |           32 | 4                |          0.2782 |
| ResNet-18  | Adam        |           0.1   |          256 | -                |          0.2763 |
| ResNet-18  | Adam        |           0.1   |           32 | -                |          0.2733 |
| ViT-base  | SGD         |           0.01  |           64 | 16               |          0.2703 |
| ViT       | Adam        |           0.01  |          128 | 4                |          0.2453 |
| ViT       | Adam        |           0.01  |           64 | 4                |          0.2315 |
| ViT       | Adam        |           0.01  |           32 | 4                |          0.2276 |
| ViT       | Adam        |           0.1   |           64 | 4                |          0.2164 |
| ViT       | Adam        |           0.1   |          128 | 4                |          0.205  |
| ViT       | Adam        |           0.1   |          256 | 4                |          0.1804 |
| ViT       | Adam        |           0.1   |           32 | 4                |          0.1586 |