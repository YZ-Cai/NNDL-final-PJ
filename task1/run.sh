#!/bin/bash

# test supervised pretrained model
nohup python train.py --model_type SupervisedPretrained --device cuda:1 --lr 0.001 > test_supervised_pretrained.txt 2>&1 &

# test supervised model
nohup python train.py --model_type Supervised --device cuda:1 > test_supervised.txt 2>&1 &