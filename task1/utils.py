"""
useful functions
@Project ：NNDL_final
@File    ：global_settings.py
@Author  ：Iker Zhe
@Date    ：2021/6/6 14:21
"""

import os
import re
import numpy
import math
import torch
import datetime
import torchvision
import torch.nn as nn
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from supervised.resnet import resnet18
from supervised_pretrained import PretrainedModel
from self_supervised_pretrained import ResNetSimCLR


def get_models(args):
    """ 
    return given network
    """
    device = args.device
    if args.model_type == "SelfSupervisedPretrained":
        pretrained_model = None
        model = ResNetSimCLR('resnet18', 100, mode='test').to(device)
        model.load(args.pretrained_model_path, device, freeze_backbone=True)
    elif args.model_type == "SelfSupervisedPretrainedFinetune":
        pretrained_model = None
        model = ResNetSimCLR('resnet18', 100, mode='test').to(device)
        model.load(args.pretrained_model_path, device, freeze_backbone=False)
    elif args.model_type == "SupervisedPretrained":
        pretrained_model = PretrainedModel('resnet18', device)
        model = nn.Linear(pretrained_model.feature_dim(), 100, device=device)
    elif args.model_type == "SupervisedPretrainedFinetune":
        pretrained_model = None
        model = PretrainedModel('resnet18', device, finetune=True, num_classes=100).model
    elif args.model_type == "Supervised":
        pretrained_model = None
        model = resnet18().to(device)
    else:
        raise ValueError("Do not support {}!".format(args.model_type))
    return pretrained_model, model


def get_training_dataloader(mean, std, batch_size=16, num_workers=4, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 training dataset
        std: std of cifar100 training dataset
        path: path to cifar100 training python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: train_data_loader:torch dataloader object
    """

    transform_train = transforms.Compose([
        # transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # cifar100_training = CIFAR100Train(path, transform=transform_train)
    cifar100_training = torchvision.datasets.CIFAR100(root='./data', train=True, download=True,
                                                      transform=transform_train)
    cifar100_training_loader = DataLoader(
        cifar100_training, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=True)

    return cifar100_training_loader


def get_test_dataloader(mean, std, batch_size=16, num_workers=4, shuffle=True):
    """ return training dataloader
    Args:
        mean: mean of cifar100 test dataset
        std: std of cifar100 test dataset
        path: path to cifar100 test python dataset
        batch_size: dataloader batchsize
        num_workers: dataloader num_works
        shuffle: whether to shuffle
    Returns: cifar100_test_loader:torch dataloader object
    """

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    # cifar100_test = CIFAR100Test(path, transform=transform_test)
    cifar100_test = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
    cifar100_test_loader = DataLoader(
        cifar100_test, shuffle=shuffle, num_workers=num_workers, batch_size=batch_size, pin_memory=True)

    return cifar100_test_loader


def compute_mean_std(cifar100_dataset):
    """compute the mean and std of cifar100 dataset
    Args:
        cifar100_training_dataset or cifar100_test_dataset
        witch derived from class torch.utils.data

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([cifar100_dataset[i][1][:, :, 0] for i in range(len(cifar100_dataset))])
    data_g = numpy.dstack([cifar100_dataset[i][1][:, :, 1] for i in range(len(cifar100_dataset))])
    data_b = numpy.dstack([cifar100_dataset[i][1][:, :, 2] for i in range(len(cifar100_dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


def most_recent_folder(net_weights, fmt):
    """
        return most recent created folder under net_weights
        if no none-empty folder were found, return empty folder
    """
    # get subfolders in net_weights
    folders = os.listdir(net_weights)

    # filter out empty folders
    folders = [(f, '_'.join(f.split('_' )[:len(fmt.split('_'))]))
               for f in folders if len(os.listdir(os.path.join(net_weights, f)))]
    if len(folders) == 0:
        return ''

    # sort folders by folder created time
    folders = sorted(folders, key=lambda f: datetime.datetime.strptime(f[1], fmt))
    return folders[-1][0]


def most_recent_weights(weights_folder):
    """
        return most recent created weights file
        if folder is empty return empty string
    """
    weight_files = os.listdir(weights_folder)
    if len(weights_folder) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'

    # sort files by epoch
    weight_files = sorted(weight_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))

    return weight_files[-1]


def last_epoch(weights_folder):
    weight_file = most_recent_weights(weights_folder)
    if not weight_file:
        raise Exception('no recent weights were found')
    resume_epoch = int(weight_file.split('-')[1])

    return resume_epoch


def best_acc_weights(weights_folder):
    """
        return the best acc .pth file in given folder, if no
        best acc weights file were found, return empty string
    """
    files = os.listdir(weights_folder)
    if len(files) == 0:
        return ''

    regex_str = r'([A-Za-z0-9]+)-([0-9]+)-(regular|best)'
    best_files = [w for w in files if re.search(regex_str, w).groups()[2] == 'best']
    if len(best_files) == 0:
        return ''

    best_files = sorted(best_files, key=lambda w: int(re.search(regex_str, w).groups()[1]))
    return best_files[-1]


# plot
def plot_images(images, img_per_row=8):
    # Plot image grid with labels

    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()

    bs, _, h, w = images.shape  # batch size, _, height, width

    plt.figure()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    gs1 = gridspec.GridSpec(math.ceil(bs / img_per_row), img_per_row)
    gs1.update(wspace=0.025, hspace=0.05)
    for i in range(bs):
        plt.subplot(gs1[i])
        plt.imshow(images[i].transpose(1, 2, 0))
        plt.xticks([])
        plt.yticks([])
    plt.show()


# get number of parameters in the model
def get_num_parameters(model):
    return sum(p.numel() for p in model.parameters())