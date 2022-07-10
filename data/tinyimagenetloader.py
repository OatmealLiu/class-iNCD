from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch.backends.cudnn as cudnn
import random
import torch.utils.data as data
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataloader import default_collate, DataLoader
from .utils import TransformTwice, TransformKtimes, RandomTranslateWithReflect, TwoStreamBatchSampler
from .concat import ConcatDataset


def find_classes_from_folder(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def find_classes_from_file(file_path):
    with open(file_path) as f:
        classes = f.readlines()
    classes = [x.strip() for x in classes]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, classes, class_to_idx):
    samples = []
    for target in classes:
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                item = (path, class_to_idx[target])
                if 'JPEG' in path or 'jpg' in path:
                    samples.append(item)

    return samples


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def pil_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, transform=None, target_transform=None, samples=None, loader=pil_loader):

        if len(samples) == 0:
            raise (RuntimeError("Found 0 images in subfolders \n"
                                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        path = self.samples[index][0]
        target = self.samples[index][1]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target, index

    def __len__(self):
        return len(self.samples)


def TinyImageNet200(aug=None, subfolder='train', class_list=range(150), path='./data/datasets/tiny-imagenet-200/'):
    # img_split = 'images/'+subfolder
    img_split = subfolder
    classes_200, class_to_idx_200 = find_classes_from_file(os.path.join(path, 'tinyimagenet_200.txt'))

    classes_sel = [classes_200[i] for i in class_list]

    samples = make_dataset(path + img_split, classes_sel, class_to_idx_200)

    if aug == None:
        transform = transforms.Compose([
            transforms.Resize(64),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif aug == 'once':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif aug == 'twice':
        transform = TransformTwice(transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]))
    elif aug == 'ktimes':
        transform = TransformKtimes(transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]), k=10)
    dataset = ImageFolder(transform=transform, samples=samples)
    return dataset


def TinyImageNetLoader(batch_size, num_workers=4, path='./data/datasets/tiny-imagenet-200/', aug=None, shuffle=False,
                       class_list=range(150), subfolder='train'):
    dataset = TinyImageNet200(aug=aug, subfolder=subfolder, class_list=class_list, path=path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    return dataloader


