# Class-incremental Novel Class Discovery (ECCV2022)
This Github repository presents the PyTorch implementation for the paper [Class-incremental Novel Class Discovery](), accepted with a poster presentation at European Conference on Computer Vision (ECCV) held at Tel Aviv International Convention Center on October 23-27, 2022.

![](figures/framework.png)


## Preparation
### Environment
```shell
Python >= 3.8.8
PyTorch >= 1.10.0 
```

`environment.yaml` includes all the dependencies for conda installation. To install (Please pre-install [Anaconda](https://www.anaconda.com/)):
```shell
conda env create -f environment.yaml
```
To activate the installed environment:
```shell
conda activate iNCD
```

### Dataset
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

## Training and Testing
![](figures/setting.png)

### Step 1: Supervised learning with labelled data
```shell
# For CIFAR10
CUDA_VISIBLE_DEVICES=0 sh step1_scripts/pretrain_cifar10.sh

# For CIFAR100
CUDA_VISIBLE_DEVICES=0 sh step1_scripts/pretrain_cifar100.sh

# For TinyImagenet
CUDA_VISIBLE_DEVICES=0 sh step1_scripts/pretrain_tinyimagenet.sh
```

### Step 2: Class-incremental Novel Class Discovery (class-iNCD) with unlabeled data
```shell
# Train on CIFAR10
CUDA_VISIBLE_DEVICES=0 sh step2_scripts_cifar10/incd_OG_FRoST.sh

# Train on CIFAR100
CUDA_VISIBLE_DEVICES=0 sh step2_scripts_cifar100/incd_OG_FRoST.sh

# Train on TinyImagenet
CUDA_VISIBLE_DEVICES=0 sh step2_scripts_tinyimagenet/incd_OG_FRoST.sh
```

### Two-steps class-iNCD
```shell
# Train on CIFAR100
CUDA_VISIBLE_DEVICES=0 sh two-steps_scripts/auto_2step_incd_OG_FRoST_cifar100.sh

# Train on TinyImagenet
CUDA_VISIBLE_DEVICES=0 sh two-steps_scripts/auto_2step_incd_OG_FRoST_tinyimagenet.sh
```

## Evaluation Protocol
![](figures/evalutation.png)

## Evaluation results
Table 1: Comparison with state-of-the-art methods in class-iNCD

![](figures/results_SOTA-HM.png)

Table 2: Comparison with the state-of-the-art methods in the two-step class-iNCD setting where new classes arrive in two episodes, instead of one. New-1-J: new classes performance from joint head at first step, New-1-N: new classes performance from novel head at first step, etc

![](figures/results_2step-iNCD.png)

Table 3: Ablation study on the proposed feature distillation (FD), feature replay (FR) and self-training (ST) that form our FRoST

![](figures/results_ablation.png)

Table 4: Ablation study comparing FRoST with LwF (logits-KD)

![](figures/results_LwF.png)

Table 5: Ablation study on having a single and separated heads for old and new classes. Joint: class-agnostic head; Novel: new classes classifier head

![](figures/results_heads.png)





