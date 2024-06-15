#!/bin/bash

# specify the path to the checkpoint directory
export TORCH_HOME="checkpoint/SupervisedPretrained"

# test supervised pretrained model
nohup python train.py --model_type SupervisedPretrained --device cuda:1 > test_supervised_pretrained.txt 2>&1 &

# test supervised model
nohup python train.py --model_type Supervised --device cuda:1 > test_supervised.txt 2>&1 &