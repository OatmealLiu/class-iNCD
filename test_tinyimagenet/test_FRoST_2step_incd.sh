#!/usr/bin/env bash

python -W ignore incd_2step_tinyimagenet.py \
        --dataset_root ./data/datasets/tiny-imagenet-200/ \
        --exp_root ./data/experiments/ \
        --warmup_model_dir ./data/experiments/supervised_learning_wo_ssl/warmup_TinyImageNet_resnet_wo_ssl.pth \
        --lr 0.1 \
        --gamma 0.1 \
        --weight_decay 1e-4 \
        --step_size 170 \
        --batch_size 128 \
        --epochs 200 \
        --rampup_length 150 \
        --rampup_coefficient 50 \
        --num_unlabeled_classes1 10 \
        --num_unlabeled_classes2 10 \
        --num_labeled_classes 180 \
        --dataset_name tinyimagenet \
        --seed 10 \
        --model_name FRoST_1st_OG_kd10_p1_tinyimagenet \
        --increment_coefficient 0.01 \
        --IL_version OG \
        --labeled_center 1 \
        --w_kd 10 \
        --mode eval \
        --lambda_proto 1 \
        --step first

python -W ignore incd_2step_tinyimagenet.py \
        --dataset_root ./data/datasets/tiny-imagenet-200/ \
        --exp_root ./data/experiments/ \
        --warmup_model_dir ./data/experiments/supervised_learning_wo_ssl/warmup_TinyImageNet_resnet_wo_ssl.pth \
        --lr 0.1 \
        --gamma 0.1 \
        --weight_decay 1e-4 \
        --step_size 170 \
        --batch_size 128 \
        --epochs 200 \
        --rampup_length 150 \
        --rampup_coefficient 50 \
        --num_unlabeled_classes1 10 \
        --num_unlabeled_classes2 10 \
        --num_labeled_classes 180 \
        --dataset_name tinyimagenet \
        --seed 10 \
        --model_name FRoST_2nd_OG_kd10_p1_tinyimagenet \
        --increment_coefficient 0.01 \
        --IL_version OG \
        --labeled_center 1 \
        --w_kd 10 \
        --mode eval \
        --lambda_proto 1 \
        --step second \
        --first_step_dir ./data/experiments/incd_2step_tinyimagenet_tinyimagenet/first_FRoST_1st_OG_kd10_p1_tinyimagenet.pth
