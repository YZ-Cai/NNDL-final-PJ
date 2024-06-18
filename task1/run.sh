#!/bin/bash

#######################################
# Self Supervised Pretrained Classifier
#######################################

# run self-supervised pretraining
nohup python pretrain.py --device cuda:1 > run_self_supervised_pretraining.txt 2>&1 &

nohup python pretrain.py --device cuda:2 --lr 0.1 > run_self_supervised_pretraining.txt 2>&1 &

nohup python pretrain.py --device cuda:2 --data cifar10 --lr 0.0003 --sample-ratio 1 > run_self_supervised_pretraining_cifar10.txt 2>&1 &

nohup python pretrain.py --device cuda:3 --sample-ratio 1 > run_self_supervised_pretraining_1.txt 2>&1 &

# run self-supervised pretrained model
nohup python train.py --model-type SelfSupervisedPretrained --lr 0.0003 \
                      --pretrained-model-path ./checkpoint/SelfSupervisedPretraining/17_June_2024_13h_21m_44s_SelfSupervisedPretraining_imagenet_sgd_lr0.005_bs128_sr0.1/checkpoint_50.pth.tar \
                      --device cuda:1 > run_self_supervised_pretrained.txt 2>&1 &

nohup python train.py --model-type SelfSupervisedPretrained --lr 0.0003 --data cifar10 \
                      --pretrained-model-path ./checkpoint/SelfSupervisedPretraining/18_June_2024_09h_28m_50s_SelfSupervisedPretraining_cifar10_adam_lr0.0003_bs128_sr1.0/checkpoint_200.pth.tar \
                      --device cuda:1 > run_self_supervised_pretrained_cifar10.txt 2>&1 &

# run grid search for self-supervised pretraining
nohup bash self_supervised_pretraining_grid_search.sh > self_supervised_pretraining_grid_search.txt 2>&1 &

# run grid search for self-supervised pretrained model
nohup bash self_supervised_pretrained_grid_search.sh > self_supervised_pretrained_grid_search.txt 2>&1 &


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