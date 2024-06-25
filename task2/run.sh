#!/bin/bash

# run resnet18: 11220132 parameters
nohup python train.py --net resnet18 --device cuda:1 --optimizer sgd --lr 0.1 --batch-size 64 > run_resnet18.txt 2>&1 &
python test.py --net resnet18 --device cuda:1

# run resnet50: 23705252 parameters
nohup python train.py --net resnet50 --device cuda:0 --optimizer sgd --lr 0.1 --batch-size 64 > run_resnet50.txt 2>&1 &
python test.py --net resnet50 --device cuda:0

# run resnet152: 58341028 parameters
nohup python train.py --net resnet152 --device cuda:0 --optimizer sgd --lr 0.1 --batch-size 64 > run_resnet152.txt 2>&1 &
python test.py --net resnet152 --device cuda:0

# run vit: 11146564 parameters
nohup python train.py --net vit --device cuda:1 --optimizer sgd --lr 0.01 --batch-size 64 > run_vit.txt 2>&1 &
python test.py --net vit --device cuda:1

# run vit-base: 85703524 parameters
nohup python train.py --net vit-base --device cuda:0 --optimizer sgd --lr 0.01 --batch-size 64 > run_vit_base.txt 2>&1 &
python test.py --net vit-base --device cuda:0

# run vit-large: 303137380 parameters
nohup python train.py --net vit-large --device cuda:0 --optimizer sgd --lr 0.01 --batch-size 64 > run_vit_large.txt 2>&1 &
python test.py --net vit-large --device cuda:0

# run grid search for resnet18
nohup bash resnet18_grid_search.sh > resnet18_grid_search.txt 2>&1 &

# run grid search for vit
nohup bash vit_grid_search.sh > vit_grid_search.txt 2>&1 &