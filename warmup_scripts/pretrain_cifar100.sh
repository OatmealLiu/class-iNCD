#!/usr/bin/env bash

python -W ignore supervised_learning_wo_ssl.py \
        --dataset_name cifar100 \
        --epochs 200 \
        --batch_size 128 \
        --num_unlabeled_classes 20 \
        --num_labeled_classes 80 \
        --dataset_root ./data/datasets/CIFAR/ \
        --model_name warmup_cifar100_resnet_wo_ssl \
        --wandb_mode online \
        --wandb_entity oatmealliu