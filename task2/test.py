#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：NNDL_final 
@File    ：test.py
@Author  ：Yuzheng Cai
@Date    ：2024/6/12 16:10 
"""
import os
import argparse
import torch
from data.dataloader import get_test_dataloader
from conf import settings
from utils import get_network, most_recent_folder, best_acc_weights, get_num_parameters
from functions import eval_testing


if __name__ == '__main__':
    # Settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', help='the device: cpu or gpu')
    parser.add_argument('--net', type=str, default="resnet18", help='the models name')
    parser.add_argument('--data', type=str, default='cifar100', help="the dataset cifar100 or cifar10")
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    args = parser.parse_args()

    # load model
    net = get_network(args)
    print(f"the number of parameters of the model is {get_num_parameters(net)}")

    # load data
    if args.data == "cifar100":
        mean = settings.CIFAR100_TRAIN_MEAN
        std = settings.CIFAR100_TRAIN_STD
    elif args.data == "cifar10":
        mean = settings.CIFAR10_TRAIN_MEAN
        std = settings.CIFAR10_TRAIN_STD
    else:
        raise ValueError("the data should be cifar10 or cifar100, but {} is given.".format(args.data))

    test_loader = get_test_dataloader(
        mean=mean,
        std=std,
        batch_size=args.batch_size,
        data=args.data,
    )

    # load from checkpoint
    recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net),
                                       fmt=settings.DATE_FORMAT)
    if not recent_folder:
        raise Exception('no recent folder were found')
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, '{net}-{epoch}-{type}.pth')
    
    # test accuracy
    best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
    if best_weights:
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
        print('found best acc weights file:{}'.format(weights_path))
        print('load best checkpoint to test acc...')
        net.load_state_dict(torch.load(weights_path))
        best_acc = eval_testing(
            model=net,
            data_loader=test_loader,
            device=args.device
            )
