#!/usr/bin/env bash

python -W ignore incd_ablation_expt.py \
        --dataset_root ./data/datasets/CIFAR/ \
        --exp_root ./data/experiments/ \
        --warmup_model_dir ./data/experiments/supervised_learning_wo_ssl/warmup_cifar100_resnet_wo_ssl.pth \
        --lr 0.1 \
        --gamma 0.1 \
        --weight_decay 1e-4 \
        --step_size 340 \
        --batch_size 256 \
        --epochs 400 \
        --rampup_length 300 \
        --rampup_coefficient 25 \
        --num_unlabeled_classes 20 \
        --num_labeled_classes 80 \
        --dataset_name cifar100 \
        --seed 10 \
        --model_name incd_OGwoProto_kd10_cifar100 \
        --increment_coefficient 0.01 \
        --IL_version OGwoProto \
        --labeled_center -1 \
        --w_kd 10 \
        --mode train \
        --lambda_proto 1 \
        --wandb_mode online \
        --wandb_entity oatmealliu