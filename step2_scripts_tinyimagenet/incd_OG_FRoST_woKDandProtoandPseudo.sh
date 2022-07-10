#!/usr/bin/env bash

python -W ignore incd_ablation_expt.py \
        --dataset_root ./data/datasets/tiny-imagenet-200/ \
        --exp_root ./data/experiments/ \
        --warmup_model_dir ./data/experiments/supervised_learning_wo_ssl/warmup_TinyImageNet_resnet_wo_ssl.pth \
        --lr 0.1 \
        --gamma 0.1 \
        --weight_decay 1e-4 \
        --step_size 170 \
        --batch_size 64 \
        --epochs 300 \
        --rampup_length 150 \
        --rampup_coefficient 50 \
        --num_unlabeled_classes 20 \
        --num_labeled_classes 180 \
        --dataset_name tinyimagenet \
        --seed 10 \
        --model_name incd_OGwoKDwoProtowoPseudo_kd10_p1_tinyimagenet \
        --increment_coefficient 0.01 \
        --IL_version OGwoKDwoProtowoPseudo \
        --labeled_center -1 \
        --w_kd -1 \
        --mode train \
        --lambda_proto 0 \
        --wandb_mode online \
        --wandb_entity oatmealliu