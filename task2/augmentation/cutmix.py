#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：NNDL_final 
@File    ：cutmix.py
@Author  ：Iker Zhe
@Date    ：2021/6/5 16:59 
"""
import numpy as np
import torch


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_data(x, y, alpha=1.):
    """
    :param x: a 4-dimensional tensor (batch_size, 32, 32, 3)
    :param y: a 1-dimensional tensor, aligned to x
    :param alpha: the parameter of beta distribution
    :return: a batch of cutmix data
    """
    device = x.device
    if alpha > 0.:

        lam = np.random.beta(alpha, alpha)

    else:

        lam = 1.

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    size = x.size()

    bbx1, bby1, bbx2, bby2 = rand_bbox(size, lam)

    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

    y_a, y_b = y, y[index]

    return x, y_a, y_b, lam


def cutmix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
