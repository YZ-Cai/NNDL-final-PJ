#!/bin/bash

# parameters for grid search
lrs=(0.001 0.01 0.1)
optimizers=("adam" "sgd")
batch_sizes=(128)

# run grid search
# for lr in "${lrs[@]}"; do
#     for optimizer in "${optimizers[@]}"; do
#         for batch_size in "${batch_sizes[@]}"; do
#             echo "-------------------------------------------------------"
#             echo "lr: $lr, optimizer: $optimizer, batch_size: $batch_size"
#             python train.py \
#                 --model-type "SelfSupervisedPretrained" \
#                 --lr "$lr" \
#                 --optimizer "$optimizer" \
#                 --batch-size "$batch_size" \
#                 --data "cifar100" \
#                 --device "cuda:3" \
#                 --pretrained-model-path "./checkpoint/SelfSupervisedPretraining/19_June_2024_03h_17m_03s_SelfSupervisedPretraining_imagenet_adam_lr0.0001_bs128_sr1.0/checkpoint_50.pth.tar"
#         done
#     done
# done

# for lr in "${lrs[@]}"; do
#     for optimizer in "${optimizers[@]}"; do
#         for batch_size in "${batch_sizes[@]}"; do
#             echo "-------------------------------------------------------"
#             echo "lr: $lr, optimizer: $optimizer, batch_size: $batch_size"
#             python train.py \
#                 --model-type "SelfSupervisedPretrained" \
#                 --lr "$lr" \
#                 --optimizer "$optimizer" \
#                 --batch-size "$batch_size" \
#                 --data "cifar100" \
#                 --device "cuda:3" \
#                 --pretrained-model-path "./checkpoint/SelfSupervisedPretraining/20_June_2024_13h_58m_37s_SelfSupervisedPretraining_imagenet_adam_lr0.0001_bs128_sr0.5/checkpoint_50.pth.tar"
#         done
#     done
# done

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
                --pretrained-model-path "./checkpoint/SelfSupervisedPretraining/20_June_2024_08h_21m_41s_SelfSupervisedPretraining_imagenet_adam_lr0.0001_bs128_sr0.3/checkpoint_50.pth.tar"
        done
    done
done

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
                --pretrained-model-path "./checkpoint/SelfSupervisedPretraining/20_June_2024_08h_36m_27s_SelfSupervisedPretraining_imagenet_adam_lr0.0001_bs128_sr0.1/checkpoint_50.pth.tar"
        done
    done
done