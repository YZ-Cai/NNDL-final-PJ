#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：NNDL_final 
@File    ：__init__.py.py
@Author  ：Yuzheng Cai
@Date    ：2024/6/13 20:25 
"""

import torch
import pretrainedmodels
import pretrainedmodels.utils as utils


class PretrainedModel:
    def __init__(self, model_name, device='cpu'):
        self.model = pretrainedmodels.__dict__[model_name](num_classes=1000, pretrained='imagenet')
        if device != "cpu":
            self.model = self.model.to(device)
        self.model.eval()

    def get_features(self, input):
        input = torch.autograd.Variable(input, requires_grad=False)
        return self.model.features(input).view(input.shape[0], -1)
    
    def get_feature_dim(self):
        return 25088



if __name__ == '__main__':
    model_name = 'resnet18'
    pretrained_model = PretrainedModel(model_name)

    # data
    path_img = 'data/cat.jpg'
    load_img = utils.LoadImage()
    tf_img = utils.TransformImage(pretrained_model.model)        
    
    # test simple input data
    input_img = load_img(path_img)
    input_tensor = tf_img(input_img)     
    input_tensor = input_tensor.unsqueeze(0)   
    print(input_tensor.shape)
    
    # obtain features
    output_features = pretrained_model.get_features(input_tensor)
    print(output_features.shape)