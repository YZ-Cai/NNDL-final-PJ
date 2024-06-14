#!/bin/bash

# test supervised pretrained model
nohup python train.py --model_type SupervisedPretrained --device cuda:1 > test_supervised_pretrained.txt 2>&1 &
