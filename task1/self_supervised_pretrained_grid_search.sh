#!/bin/bash

# parameters for grid search
lrs=(0.001 0.01 0.1)
optimizers=("adam" "sgd")
batch_sizes=(128)

# run grid search
for lr in "${lrs[@]}"; do
    for optimizer in "${optimizers[@]}"; do
        for batch_size in "${batch_sizes[@]}"; do
            echo "-------------------------------------------------------"
            echo "lr: $lr, optimizer: $optimizer, batch_size: $batch_size"
            python train.py \
                --model-type "SelfSupervisedPretrained" \
                --lr "$lr" \
                --optimizer "$optimizer" \
                --batch-size "$batch_size" \
                --data "cifar100" \
                --device "cuda:3" \
                --pretrained-model-path "./checkpoint/SelfSupervisedPretraining/15_June_2024_23h_29m_25s_SelfSupervisedPretraining_imagenet_sgd_lr0.0005_bs128_sr0.01/checkpoint_200.pth.tar"
        done
    done
done