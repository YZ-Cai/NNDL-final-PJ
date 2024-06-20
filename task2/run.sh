#!/bin/bash

# run resnet18
nohup python train.py --net resnet18 --device cuda:1 --optimizer sgd --lr 0.1 --batch-size 64 > run_resnet18.txt 2>&1 &
python test.py --net resnet18 --device cuda:1

# run vit
nohup python train.py --net vit --device cuda:1 --optimizer sgd --lr 0.01 --batch-size 64 > run_vit.txt 2>&1 &
python test.py --net vit --device cuda:1

# run grid search for resnet18
nohup bash resnet18_grid_search.sh > resnet18_grid_search.txt 2>&1 &

# run grid search for vit
nohup bash vit_grid_search.sh > vit_grid_search.txt 2>&1 &