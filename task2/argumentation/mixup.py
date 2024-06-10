#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：NNDL_final 
@File    ：mixup.py
@Author  ：Iker Zhe
@Date    ：2021/6/5 15:26 
"""
import numpy as np
import torch


def mixup_data(x, y, alpha=1.0):
    """
    :param x: a 4-dimensional tensor (batch_size, 32, 32, 3)
    :param y: a 1-dimensional tensor, aligned to x
    :param alpha: the parameter of beta distribution
    :return: a batch of mixup data
    """
    device = x.device
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]

    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)
