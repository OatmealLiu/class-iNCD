#!/usr/bin/env bash

python -W ignore supervised_learning_wo_ssl.py \
        --dataset_name tinyimagenet \
        --epochs 200 \
        --batch_size 128 \
        --num_unlabeled_classes 20 \
        --num_labeled_classes 180 \
        --dataset_root ./data/datasets/tiny-imagenet-200/ \
        --model_name warmup_TinyImageNet_resnet_wo_ssl \
        --wandb_mode online \
        --wandb_entity oatmealliu