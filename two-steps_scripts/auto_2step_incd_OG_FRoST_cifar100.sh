#!/usr/bin/env bash

python -W ignore incd_2step_cifar100.py \
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
        --num_unlabeled_classes1 10 \
        --num_unlabeled_classes2 10 \
        --num_labeled_classes 80 \
        --dataset_name cifar100 \
        --seed 10 \
        --model_name ProINCD_1st_OG_kd10_p1_cifar100 \
        --increment_coefficient 0.01 \
        --IL_version OG \
        --labeled_center 10 \
        --w_kd 10 \
        --mode train \
        --lambda_proto 1 \
        --wandb_mode online \
        --wandb_entity oatmealliu \
        --step first

python -W ignore incd_2step_cifar100.py \
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
        --num_unlabeled_classes1 10 \
        --num_unlabeled_classes2 10 \
        --num_labeled_classes 80 \
        --dataset_name cifar100 \
        --seed 10 \
        --model_name ProINCD_2nd_OG_kd10_p1_cifar100 \
        --increment_coefficient 0.01 \
        --IL_version OG \
        --labeled_center 10 \
        --w_kd 10 \
        --mode train \
        --lambda_proto 1 \
        --wandb_mode online \
        --wandb_entity oatmealliu \
        --step second \
        --first_step_dir ./data/experiments/incd_2step_cifar100_cifar100/first_ProINCD_1st_OG_kd10_p1_cifar100.pth

