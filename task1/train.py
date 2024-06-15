#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：NNDL_final 
@File    ：train.py
@Author  ：Yuzheng Cai
@Date    ：2024/6/13 20:30 
"""

import os
import torch
import argparse
import torch.nn as nn
from torch import optim
from conf import settings
from utils import get_models
from functions import train, eval_training
from torch.utils.tensorboard import SummaryWriter
from data.dataloader import get_training_dataloader, get_test_dataloader



if __name__ == '__main__':
    # Settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', help='the device: cpu or gpu')
    parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer')
    parser.add_argument('--criterion', type=str, default='CrossEntropyLoss', help='criterion')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--model-type', type=str, default="SupervisedPretrained", 
                        help='the model type: SupervisedPretrained or Supervised or UnsupervisedPretrained')
    parser.add_argument('--data', type=str, default='cifar100', help="the dataset cifar100 or cifar10")
    args = parser.parse_args()
    
    # load model
    pretrained_model, model = get_models(args)
        
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
        data=args.data
    )
    
    test_loader = get_test_dataloader(
        mean=mean,
        std=std,
        batch_size=args.batch_size,
        data=args.data,
    )
        
    # loss function
    if args.criterion == "CrossEntropyLoss":
        loss_function = nn.CrossEntropyLoss()
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
    
    # use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)
        
    # since tensorboard can't overwrite old values, the only way is to create a new tensorboard log
    writer = SummaryWriter(log_dir=os.path.join(
                           settings.LOG_DIR, args.model_type,
                           settings.TIME_NOW + "_" + args.model_type + "_" + args.data) + 
                           "_" + args.optimizer + "_lr" + str(args.lr) + "_bs" + str(args.batch_size))
    
    # create checkpoint folder to save models
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.model_type,
                                   settings.TIME_NOW + "_" + args.model_type + "_" + args.data + 
                                   "_" + args.optimizer + "_lr" + str(args.lr) + "_bs" + str(args.batch_size))
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{model_type}-{epoch}-{type}.pth')
    
    # create result folder to save accuracy
    res_path = os.path.join(settings.RES_DIR, args.model_type)
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    res_path = os.path.join(res_path,
                            args.model_type + "_" + args.data + "_" + args.optimizer + "_lr" + str(args.lr) + 
                            "_bs" + str(args.batch_size) + "_accuracy.csv")
    with open(res_path, 'w') as f:
        f.write("epoch,test_acc\n")
    
    # train
    best_acc = 0.0
    old_path = None
    for epoch in range(1, settings.EPOCH + 1):
        
        # train the model
        train(
            pretrained_model=pretrained_model,
            model=model,
            data_loader=train_loader,
            device=args.device,
            optimizer=optimizer,
            loss_function=loss_function,
            epoch=epoch,
            batch_size=args.batch_size,
            writer=writer
        )
        
        # test
        acc = eval_training(
            pretrained_model=pretrained_model,
            model=model,
            data_loader=test_loader,
            device=args.device,
            loss_function=loss_function,
            epoch=epoch,
            writer=writer
        )
        with open(res_path, 'a') as f:
            f.write(f"{epoch},{acc}\n")
        
        # start to save best performance models after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            weights_path = checkpoint_path.format(model_type=args.model_type, epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(model.state_dict(), weights_path)
            # delete old best model
            if old_path is not None:
                os.remove(old_path)
            # update
            old_path = weights_path
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(model_type=args.model_type, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(model.state_dict(), weights_path)

    writer.close()
    