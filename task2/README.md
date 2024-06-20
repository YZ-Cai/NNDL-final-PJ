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
- `run.sh`: The simple bash script to run the **ResNet-18** and **ViT** models with best parameters
- `resnet18_grid_search.sh`: The bash script to run hyper-parameter grid search for **ResNet-18**
- `vit_grid_search.sh`: The bash script to run hyper-parameter grid search for **ViT**

## Key Features

By modifying model hyper-parameters, both models have similar size:
- ResNet-18: 11,220,132 parameters
- ViT: 11,220,132 parameters

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

| model    | optimizer   |   learning_rate |   batch_size |   best_test_accuracy |
|:---------|:------------|----------------:|-------------:|---------------------:|
| resnet18 | sgd         |           0.1   |           64 |               0.7826 |
| resnet18 | sgd         |           0.01  |           32 |               0.7814 |
| resnet18 | sgd         |           0.1   |          128 |               0.7757 |
| resnet18 | sgd         |           0.1   |          256 |               0.7687 |
| resnet18 | sgd         |           0.1   |           32 |               0.7683 |
| resnet18 | sgd         |           0.01  |           64 |               0.7672 |
| resnet18 | sgd         |           0.001 |           32 |               0.7567 |
| resnet18 | sgd         |           0.01  |          128 |               0.7561 |
| resnet18 | adam        |           0.001 |          256 |               0.7468 |
| resnet18 | sgd         |           0.01  |          256 |               0.7463 |
| resnet18 | adam        |           0.001 |          128 |               0.7319 |
| resnet18 | sgd         |           0.001 |           64 |               0.729  |
| resnet18 | sgd         |           0.001 |          128 |               0.6956 |
| resnet18 | adam        |           0.001 |           64 |               0.6788 |
| resnet18 | adam        |           0.001 |           32 |               0.6383 |
| resnet18 | sgd         |           0.001 |          256 |               0.6377 |
| vit      | sgd         |           0.01  |           64 |               0.5378 |
| vit      | sgd         |           0.01  |           32 |               0.5319 |
| vit      | sgd         |           0.01  |          128 |               0.5261 |
| vit      | sgd         |           0.001 |           32 |               0.5089 |
| vit      | sgd         |           0.01  |          256 |               0.5077 |
| vit      | sgd         |           0.1   |          256 |               0.4909 |
| vit      | sgd         |           0.001 |           64 |               0.4741 |
| resnet18 | adam        |           0.01  |          256 |               0.4565 |
| vit      | adam        |           0.001 |          256 |               0.4492 |
| vit      | sgd         |           0.1   |          128 |               0.4407 |
| vit      | adam        |           0.001 |          128 |               0.4297 |
| resnet18 | adam        |           0.01  |          128 |               0.4273 |
| vit      | sgd         |           0.001 |          128 |               0.4129 |
| vit      | adam        |           0.001 |           64 |               0.4001 |
| resnet18 | adam        |           0.01  |           64 |               0.3931 |
| vit      | sgd         |           0.1   |           64 |               0.3895 |
| vit      | adam        |           0.001 |           32 |               0.3802 |
| resnet18 | adam        |           0.01  |           32 |               0.3418 |
| vit      | sgd         |           0.001 |          256 |               0.332  |
| resnet18 | adam        |           0.1   |          128 |               0.3258 |
| resnet18 | adam        |           0.1   |           64 |               0.2939 |
| vit      | adam        |           0.01  |          256 |               0.2844 |
| vit      | sgd         |           0.1   |           32 |               0.2782 |
| resnet18 | adam        |           0.1   |          256 |               0.2763 |
| resnet18 | adam        |           0.1   |           32 |               0.2733 |
| vit      | adam        |           0.01  |          128 |               0.2453 |
| vit      | adam        |           0.01  |           64 |               0.2315 |
| vit      | adam        |           0.01  |           32 |               0.2276 |
| vit      | adam        |           0.1   |           64 |               0.2164 |
| vit      | adam        |           0.1   |          128 |               0.205  |
| vit      | adam        |           0.1   |          256 |               0.1804 |
| vit      | adam        |           0.1   |           32 |               0.1586 |