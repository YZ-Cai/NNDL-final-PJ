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
- `tensorboard`

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

You can download our trained model from [This Link (Password: TODO)](TODO). 
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

TODO