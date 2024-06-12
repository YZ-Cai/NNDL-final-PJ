#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：NNDL_final 
@File    ：plot.py
@Author  ：Iker Zhe
@Date    ：2021/6/8 16:28 
"""
from augmentation.mixup import mixup_data
from augmentation.cutmix import cutmix_data
from conf import settings
from utils import plot_images
from data.dataloader import get_training_dataloader


if __name__ == '__main__':
    train_loader = get_training_dataloader(
        mean=settings.CIFAR100_TRAIN_MEAN,
        std=settings.CIFAR100_TRAIN_STD,
        batch_size=64,
        data="cifar100",
        augmentation="cutout"
    )

    for batch_index, (images, labels) in enumerate(train_loader):
        if batch_index == 0:
            plot_images(images)

            # mixup
            mixup_images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=1)
            plot_images(mixup_images)

            # cutmix
            cutmix_images, _, _, _ = cutmix_data(images, labels, alpha=1)
            plot_images(cutmix_images)

        else:
            break
