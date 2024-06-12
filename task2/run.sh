#!/bin/bash

# run resnet18
nohup python main.py --net resnet18 --device cuda:3 > run_resnet18.txt 2>&1 &

# run vit
nohup python main.py --net vit --device cuda:2 > run_vit.txt 2>&1 &

# run grid search for resnet18
nohup bash resnet18_grid_search.sh > resnet18_grid_search.txt 2>&1 &

# run grid search for vit
nohup bash vit_grid_search.sh > vit_grid_search.txt 2>&1 &