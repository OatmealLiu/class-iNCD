#!/usr/bin/env bash

python -W ignore incd_ablation_expt.py \
        --dataset_root ./data/datasets/CIFAR/ \
        --exp_root ./data/experiments/ \
        --warmup_model_dir ./data/experiments/supervised_learning_wo_ssl/warmup_cifar100_resnet_wo_ssl.pth \
        --lr 0.1 \
        --gamma 0.1 \
        --weight_decay 1e-4 \
        --step_size 170 \
        --batch_size 128 \
        --epochs 200 \
        --rampup_length 150 \
        --rampup_coefficient 50 \
        --num_unlabeled_classes 20 \
        --num_labeled_classes 80 \
        --dataset_name cifar100 \
        --seed 10 \
        --model_name incd_OG_kd10_p1_cifar100 \
        --increment_coefficient 0.01 \
        --IL_version OG \
        --labeled_center 10 \
        --w_kd 10 \
        --mode eval \
        --lambda_proto 1