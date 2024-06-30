# Task 1 of Final project of "Neural Network and Deep Learning"

-----

Table of Contents
=================

* [This is the task 1 of final project of "Neural Network and Deep Learning"](#this-is-the-final-project-of-neural-network-and-deep-learning)
   * [Introduction](#introduction)
   * [Dependencies](#dependencies)
   * [Structure](#structure)
   * [Key Features](#key-features)
   * [Self-suprevised Learning](#self-suprevised-learning)
   * [Transfer Learning and Training from Scratch](#transfer-learning-and-training-from-scratch)
   * [Suprevised Pre-trained Model](#suprevised-pre-trained-model)
   * [Trained Models](#trained-models)
   * [Testing](#testing)
   * [Results](#results)

## Introduction

Based on ResNet-18 model, we compare the following five different methods:

- **Self-supervised pretrain + Finetune**: We use [SimCLR](https://proceedings.mlr.press/v119/chen20j/chen20j.pdf) built upon ResNet-18 for self-supervised pretraining on [ImageNet](https://www.image-net.org), and the source codes come from a [pytorch implementation](https://github.com/sthalles/SimCLR). Then, we finetune all its parameters on CIFAR-100.
- **Self-supervised pretrain + Linear classifier**: After pre-training the ResNet-18 on ImageNet with self-supervised learning framework SimCLR, based on the features from the last linear layer of the pretrained model, we train a linear classifier on CIFAR-100.
- **Supervised pretrain + Finetune**: The supervised pretrained ResNet-18 model on [ImageNet](https://www.image-net.org) comes from [pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch). Then, we finetune all its parameters on CIFAR-100.
- **Supervised pretrain + Linear classifier**: After pre-training the ResNet-18 on ImageNet with supervised learning, bsed on the features from the last linear layer of the pretrained model, we train a linear classifier on CIFAR-100.
- **Supervised train from scratch**: We directly train the ResNet-18 model on CIFAR-100.

# Dependencies

- `PyTorch`
- `numpy`
- `matplotlib`
- `pyyaml`
- `tensorboard`

## Structure
```angular2html
.
├── README.md
├── conf
│   ├── __init__.py
│   └── global_settings.py
├── data
│   ├── __init__.py
│   └── dataloader.py
├── self_supervised_pretrained
│   └── files editted based on https://github.com/sthalles/SimCLR
├── supervised
│   └── resnet.py
├── supervised_pretrained
│   └── files editted based on https://github.com/Cadene/pretrained-models.pytorch
├── res
│   └── *.csv
├── functions.py
├── pretrain.py
├── run.sh
├── self_supervised_pretrained_finetune_grid_search.sh
├── self_supervised_pretrained_grid_search.sh
├── supervised_grid_search.sh
├── supervised_pretrained_finetune_grid_search.sh
├── supervised_pretrained_grid_search.sh
├── test.py
├── train.py
└── utils.py
```

- `conf`: The settings of training models can be found in this file, such as the number of epochs, the path for saving models
- `data`: The `DataLoader` for *CIFAR10* and *CIFAR100* is in this module
- `self_supervised_pretrained`: The **ResNet-18** model pre-trained with self-supervised learning framework SimCLR on ImageNet
- `supervised_pretrained`: The **ResNet-18** model pre-trained on ImageNet
- `supervised`: The **ResNet-18** model that will be trained from scratch
- `res`: The csv files of all results of accuracy at each epoch
- `utils.py`: Some useful functions
- `functions.py`: The functions for training evaluating models
- `train.py`: Training the model
- `test.py`: Testing the model
- `run.sh`: The simple bash script to run all the five methods with best parameters
- `self_supervised_pretrained_finetune_grid_search.sh`: The bash script to run hyper-parameter grid search for the finetuning step of **Self-supervised pretrain + finetune** method
- `self_supervised_pretrained_grid_search.sh`: The bash script to run hyper-parameter grid search for the linear classifier of **Self-supervised pretrain + linear classifier** method
- `supervised_pretrained_finetune_grid_search.sh`: The bash script to run hyper-parameter grid search for the finetuning step of **Supervised pretrain + finetune** method
- `supervised_pretrained_grid_search.sh`: The bash script to run hyper-parameter grid search for the linear classifier of **Supervised pretrain + linear classifier** method
- `supervised_grid_search.sh`: The bash script to run hyper-parameter grid search for **Supervised training from scratch**


# Key Features

How the scale of data impacts self-supervised learning:
- We run self-supervised learning on the ImageNet dataset with different sampling ratios of 0.1, 0.3, 0.5 and 1, which helps to investigate how the scale of the dataset affects the learning quality.

Hyper-parameter Tuning:
- We use grid search to tune the hyper-parameters of the training process on CIFAR-100, which includes the finetune, linear classification, and train from scratch. It is achieved by using bash scripts `self_supervised_pretrained_finetune_grid_search.sh`, `self_supervised_pretrained_grid_search.sh`, `supervised_grid_search.sh`, `supervised_pretrained_finetune_grid_search.sh`  and `supervised_pretrained_grid_search.sh`.

Moreover, we also try different sampling ratio of ImageNet dataset for self-supervised learning, and 

To run the codes, please follow the following instructions.

# Self-suprevised Learning

Download necessary files of [ImageNet](https://www.image-net.org/challenges/LSVRC/2012/2012-downloads.php) dataset into `./data`:
```bash
mkdir -p ./data/imagenet/train
cd ./data/imagenet
wget https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz
wget https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
```

Unzip the training files
```bash
tar -xvf ILSVRC2012_img_train.tar -C ./train
cd ./train
for f in *.tar; do
    dir=$(basename $f .tar)
    mkdir -p $dir
    tar -xvf $f -C $dir
done
```

Run command below to pre-train the ResNet-18 with self-supervised framework SimCLR.
```bash
$ python pretrain.py
    --device cuda:0                             # the device, default cpu
    --lr 0.0001                                 # the initial learning rate, default 0.0001
    --optimizer adam                            # the optimizer, default adam and you can also use sgd for SGD optimizer
    --batch-size 128                            # the batch size, default 128
    --model_type SelfSupervisedPretraining      # the pretraining model type, default SelfSupervisedPretraining
    --data imagenet                             # the dataset, default imagenet
    --sample-ratio                              # the sampling ratio of the dataset for pretraining, default 0.1
    --epochs                                    # number of total epochs to run, default 200
    --arch                                      # the underlying model architecture, default resnet18
    --out_dim                                   # feature dimension of self-supervised pretraining, default 128
```

# Suprevised Pre-trained Model

To load the supervised pre-trained model, specify the path before training linear classifier or finetuning:
```bash
export TORCH_HOME="checkpoint/SupervisedPretraining"
```

The, setup for running the pretrained models:
```bash
cd supervised_pretrained
python setup.py install
```

# Transfer Learning and Training from Scratch

Run command below to reproduce results on *CIFAR100* (dataset auto-downloads on first use).

```bash
$ python train.py
    --resume False                              # resume training, default False
    --device cuda:0                             # the device, default cpu
    --lr 0.1                                    # the initial learning rate, default 0.1
    --optimizer sgd                             # the optimizer, default SGD and you can also use adam for Adam optimizer
    --criterion CrossEntropyLoss                # the loss criterion, default CrossEntropyLoss
    --batch-size 128                            # the batch size, default 128
    --model-type SelfSupervisedPretrained       # the model type: SelfSupervisedPretrained or SelfSupervisedPretrainedFinetune or SupervisedPretrained or SSupervisedPretrainedFinetune or Supervised
    --data cifar100                             # the dataset, default cifar100
    --pretrained-model-path                     # the path to load the pretrained model
```

## Trained Models

You can download our trained model from [this link (password: NNDL)](https://pan.baidu.com/s/1l_gyF6x9SNJnw8hQthxrZw). 
It has a folder named `task1/checkpoint`, which should be put it in current folder.

## Testing

Run command below to reproduce results on *CIFAR100* (dataset auto-downloads on first use), which automatically uses the latest checkpoint for the specified model.

```bash
$ python test.py
    --device cuda:0                 # the device, default cpu
    --model-type SelfSupervisedPretrained       # the model type: SelfSupervisedPretrained or SelfSupervisedPretrainedFinetune or SupervisedPretrained or SSupervisedPretrainedFinetune or Supervised
    --data cifar100                 # the dataset, default cifar100
    --batch-size 128                # the batch size, default 128
```

## Results

| Settting                                      | Optimizer   |   Learning_ ate |   Batch size | Pretraining Sample ratio   |   Test accuracy |
|:----------------------------------------------|:------------|----------------:|-------------:|:---------------------------|----------------:|
| Supervised Pretrained (Finetune)              | sgd         |           0.01  |          128 | -                          |          0.8045 |
| Supervised Pretrained (Finetune)              | sgd         |           0.01  |          256 | -                          |          0.8016 |
| Supervised Pretrained (Finetune)              | sgd         |           0.001 |           32 | -                          |          0.7996 |
| Supervised Pretrained (Finetune)              | sgd         |           0.01  |           64 | -                          |          0.7988 |
| Supervised Pretrained (Finetune)              | sgd         |           0.001 |           64 | -                          |          0.764  |
| Supervised Pretrained (Finetune)              | sgd         |           0.01  |           32 | -                          |          0.752  |
| Supervised Pretrained (Finetune)              | sgd         |           0.001 |          128 | -                          |          0.7393 |
| Supervsed (from scratch)                      | sgd         |           0.001 |           32 | -                          |          0.7175 |
| Supervised Pretrained (Finetune)              | sgd         |           0.001 |          256 | -                          |          0.7159 |
| Supervised Pretrained (Linear)                | sgd         |           0.001 |          128 | -                          |          0.7043 |
| Supervised Pretrained (Linear)                | sgd         |           0.01  |          128 | -                          |          0.7036 |
| Supervsed (from scratch)                      | sgd         |           0.001 |           64 | -                          |          0.7029 |
| Supervised Pretrained (Linear)                | adam        |           0.001 |          128 | -                          |          0.701  |
| Supervised Pretrained (Finetune)              | sgd         |           0.1   |          256 | -                          |          0.6972 |
| Supervsed (from scratch)                      | sgd         |           0.01  |          256 | -                          |          0.6965 |
| Self-supervised Pretrained (Finetune)         | sgd         |           0.01  |          128 | 0.3                        |          0.6965 |
| Self-supervised Pretrained (Finetune)         | sgd         |           0.01  |          128 | 1.0                        |          0.6962 |
| Self-supervised Pretrained (Finetune)         | sgd         |           0.01  |          128 | 0.5                        |          0.695  |
| Supervsed (from scratch)                      | sgd         |           0.01  |          128 | -                          |          0.6902 |
| Supervsed (from scratch)                      | sgd         |           0.001 |          128 | -                          |          0.6874 |
| Supervsed (from scratch)                      | sgd         |           0.01  |           64 | -                          |          0.6857 |
| Supervsed (from scratch)                      | sgd         |           0.01  |           32 | -                          |          0.6852 |
| Self-supervised Pretrained (Finetune)         | sgd         |           0.01  |          128 | 0.1                        |          0.6788 |
| Supervsed (from scratch)                      | adam        |           0.001 |          256 | -                          |          0.6787 |
| Supervsed (from scratch)                      | adam        |           0.001 |          128 | -                          |          0.6706 |
| Supervsed (from scratch)                      | sgd         |           0.001 |          256 | -                          |          0.6572 |
| Supervsed (from scratch)                      | adam        |           0.001 |           64 | -                          |          0.6514 |
| Supervised Pretrained (Finetune)              | adam        |           0.001 |          256 | -                          |          0.6468 |
| Supervised Pretrained (Linear)                | sgd         |           0.1   |          128 | -                          |          0.6468 |
| Supervised Pretrained (Linear)                | adam        |           0.01  |          128 | -                          |          0.6444 |
| Supervsed (from scratch)                      | sgd         |           0.1   |          256 | -                          |          0.6371 |
| Supervised Pretrained (Finetune)              | adam        |           0.001 |          128 | -                          |          0.6343 |
| Supervised Pretrained (Finetune)              | adam        |           0.001 |           64 | -                          |          0.6277 |
| Self-supervised Pretrained (Finetune)         | sgd         |           0.001 |          128 | 1.0                        |          0.6145 |
| Supervised Pretrained (Finetune)              | adam        |           0.001 |           32 | -                          |          0.6109 |
| Self-supervised Pretrained (Finetune)         | sgd         |           0.001 |          128 | 0.3                        |          0.6069 |
| Supervsed (from scratch)                      | sgd         |           0.1   |          128 | -                          |          0.6055 |
| Supervsed (from scratch)                      | adam        |           0.001 |           32 | -                          |          0.6049 |
| Self-supervised Pretrained (Finetune)         | sgd         |           0.001 |          128 | 0.1                        |          0.6032 |
| Self-supervised Pretrained (Finetune)         | sgd         |           0.001 |          128 | 0.5                        |          0.5788 |
| Self-supervised Pretrained (Finetune)         | adam        |           0.001 |          128 | 0.3                        |          0.5688 |
| Self-supervised Pretrained (Finetune)         | adam        |           0.001 |          128 | 0.5                        |          0.5664 |
| Self-supervised Pretrained (Finetune)         | adam        |           0.001 |          128 | 0.1                        |          0.5663 |
| Self-supervised Pretrained (Finetune)         | adam        |           0.001 |          128 | 1.0                        |          0.5537 |
| Supervsed (from scratch)                      | sgd         |           0.1   |           64 | -                          |          0.5482 |
| Self-supervised Pretrained (Finetune)         | sgd         |           0.1   |          128 | 0.1                        |          0.5464 |
| Self-supervised Pretrained (Finetune)         | sgd         |           0.1   |          128 | 0.3                        |          0.5463 |
| Supervised Pretrained (Finetune)              | sgd         |           0.1   |          128 | -                          |          0.5437 |
| Supervised Pretrained (Finetune)              | sgd         |           0.1   |           64 | -                          |          0.5435 |
| Self-supervised Pretrained (Finetune)         | sgd         |           0.1   |          128 | 0.5                        |          0.5389 |
| Self-supervised Pretrained (Finetune)         | sgd         |           0.1   |          128 | 1.0                        |          0.5325 |
| Supervised Pretrained (Linear)                | adam        |           0.1   |          128 | -                          |          0.5229 |
| Self-supervised Pretrained (Linear)           | sgd         |           0.01  |          128 | 0.1                        |          0.4946 |
| Self-supervised Pretrained (Linear)           | adam        |           0.001 |          128 | 0.1                        |          0.4889 |
| Supervised Pretrained (Finetune)              | sgd         |           0.1   |           32 | -                          |          0.4745 |
| Self-supervised Pretrained (Linear)           | sgd         |           0.01  |          128 | 0.3                        |          0.4662 |
| Self-supervised Pretrained (Linear)           | sgd         |           0.001 |          128 | 0.1                        |          0.4661 |
| Supervsed (from scratch)                      | sgd         |           0.1   |           32 | -                          |          0.4646 |
| Self-supervised Pretrained (Linear)           | adam        |           0.001 |          128 | 0.3                        |          0.463  |
| Self-supervised Pretrained (Linear)           | sgd         |           0.1   |          128 | 0.3                        |          0.4525 |
| Self-supervised Pretrained (Linear)           | sgd         |           0.1   |          128 | 0.1                        |          0.4456 |
| Self-supervised Pretrained (Linear)           | sgd         |           0.01  |          128 | 1.0                        |          0.4312 |
| Self-supervised Pretrained (Linear)           | adam        |           0.001 |          128 | 1.0                        |          0.4305 |
| Self-supervised Pretrained (Linear)           | adam        |           0.01  |          128 | 0.1                        |          0.4268 |
| Self-supervised Pretrained (Linear)           | sgd         |           0.001 |          128 | 0.3                        |          0.4235 |
| Self-supervised Pretrained (Linear)           | adam        |           0.01  |          128 | 0.3                        |          0.4218 |
| Self-supervised Pretrained (Linear)           | sgd         |           0.1   |          128 | 1.0                        |          0.4217 |
| Self-supervised Pretrained (Linear)           | sgd         |           0.01  |          128 | 0.5                        |          0.4146 |
| Self-supervised Pretrained (Linear)           | sgd         |           0.1   |          128 | 0.5                        |          0.4117 |
| Self-supervised Pretrained (Linear)           | adam        |           0.001 |          128 | 0.5                        |          0.4116 |
| Supervised Pretrained (Finetune)              | adam        |           0.01  |          256 | -                          |          0.4047 |
| Self-supervised Pretrained (Linear)           | adam        |           0.01  |          128 | 1.0                        |          0.3888 |
| Self-supervised Pretrained (Linear, STL-10)   | adam        |           0.001 |          128 | -                          |          0.3868 |
| Self-supervised Pretrained (Linear)           | adam        |           0.01  |          128 | 0.5                        |          0.3837 |
| Self-supervised Pretrained (Linear)           | sgd         |           0.001 |          128 | 1.0                        |          0.3833 |
| Self-supervised Pretrained (Finetune)         | adam        |           0.01  |          128 | 0.3                        |          0.3814 |
| Supervsed (from scratch)                      | adam        |           0.01  |          256 | -                          |          0.3738 |
| Self-supervised Pretrained (Finetune)         | adam        |           0.01  |          128 | 1.0                        |          0.3667 |
| Self-supervised Pretrained (Finetune)         | adam        |           0.01  |          128 | 0.5                        |          0.364  |
| Supervised Pretrained (Finetune)              | adam        |           0.01  |          128 | -                          |          0.3602 |
| Self-supervised Pretrained (Finetune)         | adam        |           0.01  |          128 | 0.1                        |          0.3587 |
| Self-supervised Pretrained (Linear)           | sgd         |           0.001 |          128 | 0.5                        |          0.353  |
| Supervsed (from scratch)                      | adam        |           0.01  |          128 | -                          |          0.3336 |
| Supervised Pretrained (Finetune)              | adam        |           0.01  |           64 | -                          |          0.2881 |
| Self-supervised Pretrained (Linear)           | adam        |           0.1   |          128 | 0.3                        |          0.2839 |
| Supervsed (from scratch)                      | adam        |           0.01  |           64 | -                          |          0.2789 |
| Self-supervised Pretrained (Linear)           | adam        |           0.1   |          128 | 0.1                        |          0.278  |
| Self-supervised Pretrained (Linear, CIFAR-10) | adam        |           0.001 |          128 | -                          |          0.2707 |
| Self-supervised Pretrained (Linear)           | adam        |           0.1   |          128 | 0.5                        |          0.2698 |
| Self-supervised Pretrained (Linear)           | adam        |           0.1   |          128 | 1.0                        |          0.2539 |
| Supervised Pretrained (Finetune)              | adam        |           0.01  |           32 | -                          |          0.2396 |
| Supervsed (from scratch)                      | adam        |           0.01  |           32 | -                          |          0.1902 |
| Supervised Pretrained (Finetune)              | adam        |           0.1   |          256 | -                          |          0.0862 |
| Supervised Pretrained (Finetune)              | adam        |           0.1   |          128 | -                          |          0.0835 |
| Supervsed (from scratch)                      | adam        |           0.1   |          256 | -                          |          0.0634 |
| Self-supervised Pretrained (Finetune)         | adam        |           0.1   |          128 | 1.0                        |          0.0591 |
| Self-supervised Pretrained (Finetune)         | adam        |           0.1   |          128 | 0.1                        |          0.0568 |
| Supervised Pretrained (Finetune)              | adam        |           0.1   |           64 | -                          |          0.0448 |
| Self-supervised Pretrained (Finetune)         | adam        |           0.1   |          128 | 0.3                        |          0.0399 |
| Self-supervised Pretrained (Finetune)         | adam        |           0.1   |          128 | 0.5                        |          0.0361 |
| Supervised Pretrained (Finetune)              | adam        |           0.1   |           32 | -                          |          0.0337 |
| Supervsed (from scratch)                      | adam        |           0.1   |           64 | -                          |          0.0311 |
| Supervsed (from scratch)                      | adam        |           0.1   |          128 | -                          |          0.0299 |
| Supervsed (from scratch)                      | adam        |           0.1   |           32 | -                          |          0.0231 |
