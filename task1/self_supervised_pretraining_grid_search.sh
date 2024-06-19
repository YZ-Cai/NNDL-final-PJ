#!/bin/bash

# parameters for grid search
lrs=(0.005 0.0005 0.00005)
optimizers=("adam" "sgd")
batch_sizes=(128)

# run grid search
for lr in "${lrs[@]}"; do
    for optimizer in "${optimizers[@]}"; do
        for batch_size in "${batch_sizes[@]}"; do
            echo "-------------------------------------------------------"
            echo "lr: $lr, optimizer: $optimizer, batch_size: $batch_size"
            python pretrain.py \
                --model-type "SelfSupervisedPretraining" \
                --lr "$lr" \
                --optimizer "$optimizer" \
                --batch-size "$batch_size" \
                --data "imagenet" \
                --device "cuda:2"
        done
    done
done