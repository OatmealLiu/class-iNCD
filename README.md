# Class-incremental Novel Class Discovery (ECCV2022)
Class-incremental Novel Class Discovery (class-iNCD)

![](figures/framework.pdf)

## Introduction

## Environment

## Dataset
By default, we put the datasets in `./data/datasets/` and save trained models in `./data/experiments/` (soft link is suggested). You may also use any other directories you like by setting the `--dataset_root` argument to `/your/data/path/`, and the `--exp_root` argument to `/your/experiment/path/` when running all experiments below.

- For CIFAR-10, CIFAR-100, and SVHN, simply download the datasets and put into `./data/datasets/`.

- For TinyImagenet, to download and generate image folders, please follow https://github.com/tjmoon0104/pytorch-tiny-imagenet

- For ImageNet, we provide the exact split files used in the experiments following existing work. To download the split files, run the command:
``
sh scripts/download_imagenet_splits.sh
``
. The ImageNet dataset folder is organized in the following way:

    ```
    ImageNet/imagenet_rand118 #downloaded by the above command
    ImageNet/images/train #standard ImageNet training split
    ImageNet/images/val #standard ImageNet validation split
    ```
## Step 1: Supervised learning with labelled data
```shell
# For CIFAR10
CUDA_VISIBLE_DEVICES=0 python supervised_learning_wo_ssl.py --dataset_name cifar10 --model_name resnet_cifar10

# For CIFAR100
CUDA_VISIBLE_DEVICES=0 python supervised_learning_wo_ssl.py --dataset_name cifar100 --model_name resnet_cifar100 --num_labeled_classes 80 --num_unlabeled_classes 20

# For TinyImagenet
CUDA_VISIBLE_DEVICES=0 python supervised_learning_wo_ssl.py --dataset_name tinyimagenet --model_name resnet_tinyimagenet --num_labeled_classes 150 --num_unlabeled_classes 50
```
## Step 2: Class-incremental Novel Class Discovery (class-iNCD) with unlabeled data
```shell
# Train on CIFAR10
CUDA_VISIBLE_DEVICES=0 sh scripts/incd_center_cifar10.sh

# Train on CIFAR100
CUDA_VISIBLE_DEVICES=0 sh scripts/incd_center_cifar100.sh

# Train on TinyImagenet
CUDA_VISIBLE_DEVICES=0 sh scripts/incd_center_tinyimagenet.sh
```
## Two-steps class-iNCD
```shell
# Train on CIFAR10

# Train on CIFAR100

# Train on TinyImagenet
```
