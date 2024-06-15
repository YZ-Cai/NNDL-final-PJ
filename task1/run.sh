#!/bin/bash

# specify the path to the checkpoint directory
export TORCH_HOME="checkpoint/SupervisedPretrained"

# run supervised pretrained model
nohup python train.py --model-type SupervisedPretrained --device cuda:1 > run_supervised_pretrained.txt 2>&1 &

# run supervised model
nohup python train.py --model-type Supervised --device cuda:1 > run_supervised.txt 2>&1 &

# run grid search for supervised pretrained model
nohup bash supervised_pretrained_grid_search.sh > supervised_pretrained_grid_search.txt 2>&1 &

# run grid search for supervised model
nohup bash supervised_grid_search.sh > supervised_grid_search.txt 2>&1 &