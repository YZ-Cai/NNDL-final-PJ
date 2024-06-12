#!/bin/bash

# parameters for grid search
lrs=(0.001 0.01 0.1)
optimizers=("adam" "sgd")
batch_sizes=(32 64 128 256)

# run grid search
for lr in "${lrs[@]}"; do
    for optimizer in "${optimizers[@]}"; do
        for batch_size in "${batch_sizes[@]}"; do
            echo "-------------------------------------------------------"
            echo "lr: $lr, optimizer: $optimizer, batch_size: $batch_size"
            python main.py \
                --net "vit" \
                --lr "$lr" \
                --optimizer "$optimizer" \
                --batch-size "$batch_size" \
                --data "cifar100" \
                --device "cuda:2"
        done
    done
done