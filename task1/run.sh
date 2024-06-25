#!/bin/bash

#######################################
# Self Supervised Pretraining
#######################################

# run self-supervised pretraining with different sample ratios
nohup python pretrain.py --device cuda:0 --sample-ratio 0.1 > run_self_supervised_pretraining_sr0.1.txt 2>&1 &
nohup python pretrain.py --device cuda:1 --sample-ratio 0.3 > run_self_supervised_pretraining_sr0.3.txt 2>&1 &
nohup python pretrain.py --device cuda:2 --sample-ratio 0.5 > run_self_supervised_pretraining_sr0.5.txt 2>&1 &
nohup python pretrain.py --device cuda:3 --sample-ratio 1.0 > run_self_supervised_pretraining_sr1.0.txt 2>&1 &


#######################################
# Self Supervised Pretrained Classifier
#######################################

# run self-supervised pretrained model
nohup python train.py --model-type SelfSupervisedPretrained --optimizer sgd --lr 0.01 \
                      --pretrained-model-path ./checkpoint/SelfSupervisedPretraining/19_June_2024_03h_17m_03s_SelfSupervisedPretraining_imagenet_adam_lr0.0001_bs128_sr1.0/checkpoint_50.pth.tar \
                      --device cuda:1 > run_self_supervised_pretrained_sr1.txt 2>&1 &

# run grid search for self-supervised pretrained model
nohup bash self_supervised_pretrained_grid_search.sh > self_supervised_pretrained_grid_search.txt 2>&1 &


#######################################
# Self Supervised Pretrained Finetune
#######################################

# run self-supervised pretrained model that finetunes all parameters
nohup python train.py --model-type SelfSupervisedPretrainedFinetune \
                      --pretrained-model-path ./checkpoint/SelfSupervisedPretraining/19_June_2024_03h_17m_03s_SelfSupervisedPretraining_imagenet_adam_lr0.0001_bs128_sr1.0/checkpoint_50.pth.tar \
                      --device cuda:3 > run_self_supervised_pretrained_finetune_sr1.txt 2>&1 &

# run grid search for self-supervised pretrained model that finetunes all parameters
nohup bash self_supervised_pretrained_finetune_grid_search.sh > self_supervised_pretrained_finetune_grid_search.txt 2>&1 &


#######################################
# Supervised Pretrained Classifier
#######################################

# specify the path to the checkpoint directory
export TORCH_HOME="checkpoint/SupervisedPretraining"

# run supervised pretrained model
nohup python train.py --model-type SupervisedPretrained --device cuda:1 --optimizer sgd --lr 0.001 --batch-size 32 > run_supervised_pretrained.txt 2>&1 &
python test.py --model-type SupervisedPretrained --device cuda:1

# run grid search for supervised pretrained model
nohup bash supervised_pretrained_grid_search.sh > supervised_pretrained_grid_search.txt 2>&1 &


#######################################
# Supervised Pretrained Finetune
#######################################

# specify the path to the checkpoint directory
export TORCH_HOME="checkpoint/SupervisedPretraining"

# run supervised pretrained model that finetunes all parameters
nohup python train.py --model-type SupervisedPretrainedFinetune --device cuda:3 > run_supervised_pretrained_finetune.txt 2>&1 &
python test.py --model-type SupervisedPretrainedFinetune --device cuda:3

# run grid search for supervised pretrained model
nohup bash supervised_pretrained_finetune_grid_search.sh > supervised_pretrained_finetune_grid_search.txt 2>&1 &


#######################################
# Supervised ResNet Classifier
#######################################

# run supervised model
nohup python train.py --model-type Supervised --device cuda:1 > run_supervised.txt 2>&1 &
python test.py --model-type Supervised --device cuda:1

# run grid search for supervised model
nohup bash supervised_grid_search.sh > supervised_grid_search.txt 2>&1 &