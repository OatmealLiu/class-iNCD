#!/usr/bin/env bash

python -W ignore supervised_learning_wo_ssl.py \
        --dataset_name cifar10 \
        --epochs 200 \
        --batch_size 128 \
        --num_unlabeled_classes 5 \
        --num_labeled_classes 5 \
        --dataset_root ./data/datasets/CIFAR/ \
        --model_name warmup_resnet_wo_ssl \
        --wandb_mode online \
        --wandb_entity oatmealliu