#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：NNDL_final 
@File    ：pretrain.py
@Author  ：Yuzheng Cai
@Date    ：2024/6/14 19:06
"""

import os
import torch
import argparse
import torch.nn as nn
from torch import optim
from conf import settings
from self_supervised_pretrained import ContrastiveLearningDataset, ResNetSimCLR, SimCLR



if __name__ == '__main__':
    # Settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', help='the device: cpu or gpu')
    parser.add_argument('--lr', type=float, default=0.005, help='initial learning rate')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--model-type', type=str, default="SelfSupervisedPretraining", help='the model type: SelfSupervisedPretraining')
    parser.add_argument('--data', type=str, default='imagenet', help="the dataset stl10 or cifar10 or imagenet")
    parser.add_argument('--sample-ratio', type=float, default=0.1, help='the sample ratio of the dataset')
    parser.add_argument('--n-views', default=2, type=int, help='Number of views for contrastive learning training.')
    parser.add_argument('--temperature', default=0.07, type=float, help='softmax temperature (default: 0.07)')
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--fp16-precision', action='store_true', help='Whether or not to use 16-bit precision GPU training.')
    parser.add_argument('--log-every-n-steps', default=20, type=int, help='Log every n steps')
    parser.add_argument('--arch', type=str, default='resnet18', help='model architecture: resnet18')
    parser.add_argument('--out_dim', default=128, type=int, help='feature dimension (default: 128)')
    args = parser.parse_args()
    
    # data
    dataset = ContrastiveLearningDataset('./data/')
    if args.data == "imagenet":
        mean = settings.IMAGENET_TRAIN_MEAN
        std = settings.IMAGENET_TRAIN_STD
    elif args.data == "cifar10":
        mean = settings.CIFAR10_TRAIN_MEAN
        std = settings.CIFAR10_TRAIN_STD
    elif args.data == "stl10":
        mean = settings.STL10_TRAIN_MEAN
        std = settings.STL10_TRAIN_STD
    else:
        raise ValueError("the data should be cifar10 or stl10 or imagenet, but {} is given.".format(args.data))
    train_dataset = dataset.get_dataset(args.data, args.n_views, mean, std)
    
    # sample some data for training
    num_train = int(len(train_dataset) * args.sample_ratio) // args.batch_size * args.batch_size
    indices = torch.randperm(len(train_dataset))[:num_train]
    train_dataset = torch.utils.data.Subset(train_dataset, indices)    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=32, pin_memory=True, drop_last=True)
    print(f"the number of images for training is {num_train}")

    # model
    if args.model_type == "SelfSupervisedPretraining":
        model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)
    else:
        raise ValueError("the model type should be SelfSupervisedPretraining, but {} is given".format(args.model_type))
    
    # optimizer
    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    else:
        raise ValueError("the optimizer should be 'adam' or 'sgd', but got '%s'" % args.optimizer)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)
    
    # specify the directory
    args.log_dir = os.path.join(settings.LOG_DIR, args.model_type,
                                settings.TIME_NOW + "_" + args.arch + "_" + args.data) + \
                                "_" + args.optimizer + "_lr" + str(args.lr) + "_bs" + str(args.batch_size) + \
                                "_sr" + str(args.sample_ratio)
    args.checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.model_type,
                                        settings.TIME_NOW + "_" + args.model_type + "_" + args.data + 
                                        "_" + args.optimizer + "_lr" + str(args.lr) + "_bs" + str(args.batch_size))+ \
                                        "_sr" + str(args.sample_ratio)
    if not os.path.exists(args.checkpoint_path):
        os.makedirs(args.checkpoint_path)
    
    # train the self-supervised pretraining model          
    simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
    simclr.train(train_loader)
        
    