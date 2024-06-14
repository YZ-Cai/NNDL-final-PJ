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
from unsupervised_pretrained import ContrastiveLearningDataset, ResNetSimCLR, SimCLR



if __name__ == '__main__':
    # Settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', help='the device: cpu or gpu')
    parser.add_argument('--lr', type=float, default=0.0005, help='initial learning rate')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
    parser.add_argument('--criterion', type=str, default='CrossEntropyLoss', help='criterion')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--model_type', type=str, default="unsupervised", help='the model type: unsupervised')
    parser.add_argument('--data', type=str, default='imagenet', help="the dataset stl10 or cifar10 or imagenet")
    parser.add_argument('--n-views', default=2, type=int, help='Number of views for contrastive learning training.')
    parser.add_argument('--temperature', default=0.07, type=float, help='softmax temperature (default: 0.07)')
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')
    parser.add_argument('--fp16-precision', action='store_true', help='Whether or not to use 16-bit precision GPU training.')
    parser.add_argument('--log-every-n-steps', default=100, type=int, help='Log every n steps')
    parser.add_argument('--arch', type=str, default='resnet18', help='model architecture: resnet18')
    args = parser.parse_args()
    
    # data
    dataset = ContrastiveLearningDataset('./data')
    train_dataset = dataset.get_dataset(args.data, args.n_views)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True, drop_last=True)
    
    # print how many images in the data
    print(f"the number of images in the dataset is {len(train_dataset)}")

    # model
    if args.model_type == "unsupervised":
        model = ResNetSimCLR(base_model=args.arch, out_dim=args.out_dim)
    else:
        raise ValueError("the model type should be UnsupervisedPretrained SupervisedPretrained or Supervised, \
                          but {} is given".format(args.model_type))
        
    # loss function
    if args.criterion == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError("the loss function should be cross entropy loss, but {} is given".format(args.criterion))
    
    # optimizer
    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    else:
        raise ValueError("the optimizer should be 'adam' or 'sgd', but got '%s'" % args.optimizer)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                           last_epoch=-1)
    
    # specify the tensor board log directory
    args.log_dir = os.path.join(settings.LOG_DIR, 'unsupervised_pretrained',
                                settings.TIME_NOW + "_" + args.arch + "_" + args.data) + \
                                "_" + args.optimizer + "_lr" + str(args.lr) + "_bs" + str(args.batch_size)
    
    # train the unsupervised pretraining model          
    simclr = SimCLR(model=model, optimizer=optimizer, scheduler=scheduler, args=args)
    simclr.train(train_loader)
        
    