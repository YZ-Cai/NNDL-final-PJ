#!/bin/bash

#######################################
# Self Supervised Pretrained Classifier
#######################################

# run self-supervised pretraining
nohup python pretrain.py --device cuda:1 > run_self_supervised_pretraining.txt 2>&1 &

# run self-supervised pretrained model
nohup python train.py --model-type SelfSupervisedPretrained \
                      --pretrained-model-path ./checkpoint/SelfSupervisedPretraining/15_June_2024_15h_09m_25s_SelfSupervisedPretraining_imagenet_sgd_lr0.0005_bs128/checkpoint_2.pth.tar \
                      --device cuda:1 > run_self_supervised_pretrained.txt 2>&1 &

#######################################
# Supervised Pretrained Classifier
#######################################

# specify the path to the checkpoint directory
export TORCH_HOME="checkpoint/SupervisedPretrained"

# run supervised pretrained model
nohup python train.py --model-type SupervisedPretrained --device cuda:1 > run_supervised_pretrained.txt 2>&1 &

# run grid search for supervised pretrained model
nohup bash supervised_pretrained_grid_search.sh > supervised_pretrained_grid_search.txt 2>&1 &


#######################################
# Supervised ResNet Classifier
#######################################

# run supervised model
nohup python train.py --model-type Supervised --device cuda:1 > run_supervised.txt 2>&1 &

# run grid search for supervised model
nohup bash supervised_grid_search.sh > supervised_grid_search.txt 2>&1 &