#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：NNDL_final 
@File    ：functions.py
@Author  ：Iker Zhe, Yuzheng Cai
@Date    ：2024/6/10 21:30 
"""
import os
import time
import random
import torch
from augmentation.cutmix import cutmix_data, cutmix_criterion
from augmentation.mixup import mixup_data, mixup_criterion


def train(model, data_loader, device, optimizer, loss_function, epoch, batch_size, warmup_dict=None, writer=None,
          alpha=0.2):
    """
    :param model: cnn model
    :param data_loader: the data loader
    :param device: cpu or gpu
    :param optimizer: the optimizer
    :param loss_function: the loss function
    :param epoch: the number of iteration
    :param batch_size: the batch size
    :param warmup_dict: whether to warmup or not
    :param writer: whether to writer logs or not
    :param alpha: the parameter of beta(\alpha, \alpha), default=0.2
    :return:
    """
    start = time.time()
    model.train()
    for batch_index, (images, labels) in enumerate(data_loader):
        if device != "cpu":
            labels = labels.to(device)
            images = images.to(device)
        optimizer.zero_grad()
        
        # cutout has been processed in data loader
        # randomly use mixup or cutmix
        if random.random() < 0.5:
            images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=alpha)
            outputs = model(images)
            loss = mixup_criterion(loss_function, outputs, labels_a, labels_b, lam)
        else:
            images, labels_a, labels_b, lam = cutmix_data(images, labels, alpha=alpha)
            outputs = model(images)
            loss = cutmix_criterion(loss_function, outputs, labels_a, labels_b, lam)
        
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(data_loader) + batch_index + 1

        last_layer = list(model.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * batch_size + len(images),
            total_samples=len(data_loader.dataset)
        ))

        # update training loss for each iteration
        if writer is not None:
            writer.add_scalar('Train/loss', loss.item(), n_iter)

        if warmup_dict is not None:
            warmup_scheduler = warmup_dict["warmup_scheduler"]
            if epoch <= warmup_dict["warmup_num"]:
                warmup_scheduler.step()

    if writer is not None:
        for name, param in model.named_parameters():
            layer, attr = os.path.splitext(name)
            attr = attr[1:]
            writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('Epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))


@torch.no_grad()
def eval_training(model, data_loader, device, loss_function, epoch=0, writer=None):
    start = time.time()
    model.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0

    for (images, labels) in data_loader:

        if device != "cpu":
            labels = labels.to(device)
            images = images.to(device)

        outputs = model(images)
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
def eval_testing(model, data_loader, device):
    start = time.time()
    model.eval()

    correct = 0.0

    for (images, labels) in data_loader:

        if device != "cpu":
            labels = labels.to(device)
            images = images.to(device)

        outputs = model(images)
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    print('Test set Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        correct.float() / len(data_loader.dataset),
        finish - start
    ))
    print()

