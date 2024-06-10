#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：NNDL_final 
@File    ：dataloader.py
@Author  ：Iker Zhe, Yuzheng Cai
@Date    ：2024/6/10 21:30 
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from argumentation.cutout import Cutout


def get_training_dataloader(mean, std, batch_size=16, num_workers=4, shuffle=True, data="cifar100",
                            cutout_length=16):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
        data: cifar100 or cifar10
        cutout_length: the length of patch in cutout
    Returns: train_data_loader:torch dataloader object
    """
    # construct transform
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # use cutout
    transform_train.transforms.append(Cutout(n_holes=1, length=cutout_length))

    # load data
    if data == 'cifar10':
        train_dataset = datasets.CIFAR10(root='data/',
                                         train=True,
                                         transform=transform_train,
                                         download=True)
    elif data == 'cifar100':
        train_dataset = datasets.CIFAR100(root='data/',
                                          train=True,
                                          transform=transform_train,
                                          download=True)
    else:
        raise ValueError('Invalid data, the data should be cifar100 or cifar10, but {} is given'.format(data))

    # construct dataloader
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              pin_memory=True,
                              num_workers=num_workers)

    return train_loader


def get_test_dataloader(mean, std, batch_size=16, num_workers=4, shuffle=True, data="cifar10"):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
        data: cifar100 or cifar10
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # load data
    if data == 'cifar10':
        test_dataset = datasets.CIFAR10(root='data/',
                                        train=False,
                                        transform=transform_test,
                                        download=True)
    elif data == 'cifar100':
        test_dataset = datasets.CIFAR100(root='data/',
                                         train=False,
                                         transform=transform_test,
                                         download=True)
    else:
        raise ValueError('Invalid data, the data should be cifar100 or cifar10, but {} is given'.format(data))

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             pin_memory=True,
                             num_workers=num_workers)

    return test_loader
