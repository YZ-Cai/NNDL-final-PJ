#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：NNDL_final 
@File    ：functions.py
@Author  ：Iker Zhe, Yuzheng Cai
@Date    ：2024/6/15 09:56 
"""
import os
import time
import torch


def train(pretrained_model, model, data_loader, device, optimizer, loss_function, epoch, batch_size, writer=None):
    """
    :param pretrained_model: pretrained model
    :param model: the final classifier model
    :param data_loader: the data loader
    :param device: cpu or gpu
    :param optimizer: the optimizer
    :param loss_function: the loss function
    :param epoch: the number of iteration
    :param batch_size: the batch sizes
    :param writer: whether to writer logs or not
    """
    start = time.time()
    model.train()
    for batch_index, (images, labels) in enumerate(data_loader):
        if device != "cpu":
            labels = labels.to(device)
            images = images.to(device)
        optimizer.zero_grad()
        
        # model output
        if pretrained_model == None:
            outputs = model(images)
        else:
            output_features = pretrained_model.get_features(images)
            outputs = model(output_features)
            
        # train loss
        loss = loss_function(outputs, labels)
        
        # step
        loss.backward()
        optimizer.step()
        
        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * batch_size + len(images),
            total_samples=len(data_loader.dataset)
        ))

        # update training loss for each iteration
        n_iter = (epoch - 1) * len(data_loader) + batch_index + 1
        if writer is not None:
            writer.add_scalar('Train/loss', loss.item(), n_iter)

    if writer is not None:
        for name, param in model.named_parameters():
            layer, attr = os.path.splitext(name)
            attr = attr[1:]
            writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('Epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))



@torch.no_grad()
def eval_training(pretrained_model, model, data_loader, device, loss_function, epoch=0, writer=None):
    start = time.time()
    model.eval()

    test_loss = 0.0 
    correct = 0.0

    for (images, labels) in data_loader:
        if device != "cpu":
            labels = labels.to(device)
            images = images.to(device)

        # model output
        if pretrained_model == None:
            outputs = model(images)
        else:
            output_features = pretrained_model.get_features(images)
            outputs = model(output_features)
        
        # loss and accuracy
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
        
    finish = time.time()
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(data_loader.dataset),
        correct.float() / len(data_loader.dataset),
        finish - start
    ))
    print()

    # add information to tensorboard
    if writer is not None:
        writer.add_scalar('Test/Average loss', test_loss / len(data_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(data_loader.dataset), epoch)

    return correct.float() / len(data_loader.dataset)



@torch.no_grad()
def eval_testing(pretrained_model, model, data_loader, device):
    start = time.time()
    model.eval()

    correct = 0.0
    for (images, labels) in data_loader:
        if device != "cpu":
            labels = labels.to(device)
            images = images.to(device)

        # model output
        if pretrained_model == None:
            outputs = model(images)
        else:
            output_features = pretrained_model.get_features(images)
            outputs = model(output_features)
            
        # accuracy
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    print('Test set Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        correct.float() / len(data_loader.dataset),
        finish - start
    ))
    print()

