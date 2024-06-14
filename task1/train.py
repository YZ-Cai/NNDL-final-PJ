#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：NNDL_final 
@File    ：train.py
@Author  ：Yuzheng Cai
@Date    ：2024/6/13 20:30 
"""

import os
import argparse
import torch.nn as nn
from torch import optim
from conf import settings
from supervised_pretrained import PretrainedModel
from data.dataloader import get_training_dataloader, get_test_dataloader



if __name__ == '__main__':
    # Settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', help='the device: cpu or gpu')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
    parser.add_argument('--criterion', type=str, default='CrossEntropyLoss', help='criterion')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--model_type', type=str, default="SupervisedPretrained", 
                        help='the model type: SupervisedPretrained or Supervised or UnsupervisedPretrained')
    parser.add_argument('--data', type=str, default='cifar100', help="the dataset cifar100 or cifar10")
    args = parser.parse_args()
        
    # load data
    if args.data == "cifar100":
        mean = settings.CIFAR100_TRAIN_MEAN
        std = settings.CIFAR100_TRAIN_STD
    elif args.data == "cifar10":
        mean = settings.CIFAR10_TRAIN_MEAN
        std = settings.CIFAR10_TRAIN_STD
    else:
        raise ValueError("the data should be cifar10 or cifar100, but {} is given.".format(args.data))
    
    train_loader = get_training_dataloader(
        mean=mean,
        std=std,
        batch_size=args.batch_size,
        data=args.data,
        height=224 if "Pretrained" in args.model_type else 32,
        width=224 if "Pretrained" in args.model_type else 32
    )
    
    test_loader = get_test_dataloader(
        mean=mean,
        std=std,
        batch_size=args.batch_size,
        data=args.data,
    )

    # model, TODO: function
    device = args.device
    if args.model_type == "SupervisedPretrained":
        pretrained_model = PretrainedModel('resnet18', device)
        feature_dim = pretrained_model.get_feature_dim()
        model = nn.Linear(feature_dim, 100, device=device)
        
    # loss function
    if args.criterion == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError("the loss function should be cross entropy loss, but {} is given".format(args.criterion))
    
    # optimizer
    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    elif args.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
    else:
        raise ValueError("the optimizer should be 'adam' or 'sgd', but got '%s'" % args.optimizer)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2)
    
    for epoch in range(1, settings.EPOCH + 1):
        
        # train, TODO: function
        data_loader = train_loader
        batch_size = args.batch_size
        import time
        start = time.time()
        model.train()
        for batch_index, (images, labels) in enumerate(data_loader):
            if device != "cpu":
                labels = labels.to(device)
                images = images.to(device)
            optimizer.zero_grad()
            
            output_features = pretrained_model.get_features(images)
            outputs = model(output_features)
            
            # train loss and accuracy
            loss = criterion(outputs, labels)
            
            # step
            loss.backward()
            optimizer.step()
            
            print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
                loss.item(),
                optimizer.param_groups[0]['lr'],
                epoch=epoch,
                trained_samples=batch_index * batch_size + len(images),
                total_samples=len(data_loader.dataset),
            ))
            
        finish = time.time()
        print('Epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))
            
        # test, TODO: function
        
        print(f"batch {batch_index} loss: {loss.item()}, acc: {acc.item()}")
    