import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from utils.util import cluster_acc, Identity, AverageMeter, CentroidTracker
from models.resnet import ResNet, BasicBlock
from data.cifarloader import CIFAR10Loader, CIFAR100Loader
from data.svhnloader import SVHNLoader
from tqdm import tqdm
import numpy as np
import os
import wandb
from data.tinyimagenetloader import TinyImageNetLoader


def train(model, train_loader, labeled_eval_loader, args, cntr_tracker=None, track_interval=10):
    """
    Stage-I: supervised-learning
    :param model: give a model
    :param train_loader: dataloader for the training dataset of labeled data
    :param labeled_eval_loader: dataloader for the validation dataset of labeled data
    :param args: contains the hyperparameters for training
    :return: N/A
    """
    # create DNN components
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                    weight_decay=args.weight_decay)  # create optimizer
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)  # create LR shceduler
    criterion1 = nn.CrossEntropyLoss()  # create CE loss

    # start training epoch-by-epoch
    for epoch in range(args.epochs):
        # LOOK: we should calculate and temp-save the feature here
        #     : we should save the extracted feature for each class epoch-wise during Stage-I training
        #     : in order to monitor the movement of the Gaussion
        if cntr_tracker:
            if track_interval != 1:
                if epoch % track_interval - 1 == 0:
                    cntr_tracker.generate(epoch)
            else:
                cntr_tracker.generate(epoch)

        # create loss statistics recoder
        loss_record = AverageMeter()
        # turn on the training mode of the model
        model.train()
        # update LR scheduler
        exp_lr_scheduler.step()

        # start the training for the current epoch batch-by-batch
        for batch_idx, (x, label, idx) in enumerate(tqdm(train_loader)):
            # normalize the prototypes
            if args.l2_classifier:
                model.l2_classifier = True
                with torch.no_grad():
                    w_head = model.head1.weight.data.clone()
                    w_head = F.normalize(w_head, dim=1, p=2)
                    model.head1.weight.copy_(w_head)
            else:
                model.l2_classifier = False

            x, label = x.to(device), label.to(device)  # sent the variables to CUDA
            output1, _, _ = model(x)  # forward-prop: head-1 output, head-2 output, extracted feature
            loss = criterion1(output1, label)  # compute the CE loss
            loss_record.update(loss.item(), x.size(0))  # record the loss statistics
            optimizer.zero_grad()  # zero the gradients of the model parameters
            loss.backward()  # compute the gradients w.r.t. the model parameters
            optimizer.step()  # back-prop: update the model parameters

        # Complete the current epoch, print the statistics and validate the current learned model
        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))
        print('test on labeled classes')
        args.head = 'head1'
        _, acc_head1_lb_warmup = test(model, labeled_eval_loader, args)
        wandb.log({"val_acc/head1_lb_warm": acc_head1_lb_warmup}, step=epoch)
        # # LOOK: we should calculate and temp-save the feature here
        # #     : we should save the extracted feature for each class epoch-wise during Stage-I training
        # #     : in order to monitor the movement of the Gaussion
        # if cntr_tracker:
        #     if epoch % track_interval-1 == 0:
        #         cntr_tracker.generate(epoch)


def test(model, test_loader, args):
    model.eval()
    preds = np.array([])
    targets = np.array([])
    for batch_idx, (x, label, _) in enumerate(tqdm(test_loader)):
        x, label = x.to(device), label.to(device)
        output1, output2, _ = model(x)
        if args.head == 'head1':
            output = output1
        else:
            output = output2
        _, pred = output.max(1)
        targets = np.append(targets, label.cpu().numpy())
        preds = np.append(preds, pred.cpu().numpy())
    # DEBUG
    # acc, nmi, ari = cluster_acc(targets.astype(int), preds.astype(int)), nmi_score(targets, preds), ari_score(targets, preds)
    acc = cluster_acc(targets.astype(int), preds.astype(int))
    nmi = nmi_score(targets, preds)
    ari = ari_score(targets, preds)
    print('Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))
    return preds, acc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='cluster',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--step_size', default=30, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_unlabeled_classes', default=5, type=int)
    parser.add_argument('--num_labeled_classes', default=5, type=int)
    parser.add_argument('--dataset_root', type=str, default='./data/datasets/CIFAR/')
    parser.add_argument('--l2_classifier', action='store_true', default=False, help='L2 normalize classifier')
    parser.add_argument('--exp_root', type=str, default='./data/experiments/')
    parser.add_argument('--rotnet_dir', type=str,
                        default='./data/experiments/selfsupervised_learning/rotnet_cifar10.pth')
    parser.add_argument('--model_name', type=str, default='resnet_wo_ssl')
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='options: cifar10, cifar100, svhn, d')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--wandb_entity', type=str, default='unitn-mhug')
    parser.add_argument('--track_centroid', action='store_true', default=False, help='track the centroid epoch-wise')
    parser.add_argument('--track_interval', default=10, type=int, help="the frequency to save the feature statistics")

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    runner_name = os.path.basename(__file__).split(".")[0]
    model_dir = os.path.join(args.exp_root, runner_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    args.model_dir = model_dir + '/' + '{}.pth'.format(args.model_name)

    # WandB setting
    # use wandb logging
    wandb_run_name = args.model_name + args.dataset_name
    wandb.init(project='incd_dev_miu',
               entity=args.wandb_entity,
               name=wandb_run_name,
               mode=args.wandb_mode)

    model = ResNet(BasicBlock, [2, 2, 2, 2], args.num_labeled_classes, args.num_unlabeled_classes).to(device)

    num_classes = args.num_labeled_classes + args.num_unlabeled_classes

    # state_dict = torch.load(args.rotnet_dir)
    # del state_dict['linear.weight']
    # del state_dict['linear.bias']
    # model.load_state_dict(state_dict, strict=False)
    # for name, param in model.named_parameters():
    #     if 'head' not in name and 'layer4' not in name:
    #         param.requires_grad = False

    if args.dataset_name == 'cifar10':
        print("Create CIFAR-10 dataloader")
        labeled_train_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='train',
                                             aug='once', shuffle=True, target_list=range(args.num_labeled_classes))
        labeled_eval_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None,
                                            shuffle=False, target_list=range(args.num_labeled_classes))
    elif args.dataset_name == 'cifar100':
        print("Create CIFAR-100 dataloader")
        labeled_train_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train',
                                              aug='once', shuffle=True, target_list=range(args.num_labeled_classes))
        labeled_eval_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None,
                                             shuffle=False, target_list=range(args.num_labeled_classes))
    elif args.dataset_name == 'svhn':
        labeled_train_loader = SVHNLoader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='once',
                                          shuffle=True, target_list=range(args.num_labeled_classes))
        labeled_eval_loader = SVHNLoader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None,
                                         shuffle=False, target_list=range(args.num_labeled_classes))
    elif args.dataset_name == 'tinyimagenet':
        print("Create TinyImageNet dataloader")
        labeled_train_loader = TinyImageNetLoader(batch_size=args.batch_size, num_workers=8, path=args.dataset_root,
                                                  aug='once', shuffle=True, class_list=range(args.num_labeled_classes),
                                                  subfolder='train')
        labeled_eval_loader = TinyImageNetLoader(batch_size=args.batch_size, num_workers=8, path=args.dataset_root,
                                                 aug=None, shuffle=False, class_list=range(args.num_labeled_classes),
                                                 subfolder='val')

    # LOOK:NEW
    # create the centroid tracker if tracking mode is on
    if args.track_centroid:
        cntr_tracker = CentroidTracker(model, labeled_train_loader, args.num_labeled_classes, device,
                                       args.dataset_name, 'Supervised', save_root=model_dir)
    else:
        cntr_tracker = None

    if args.mode == 'train':
        # train the model
        train(model, labeled_train_loader, labeled_eval_loader, args,
              cntr_tracker=cntr_tracker, track_interval=args.track_interval)
        # save the warmed-up model
        torch.save(model.state_dict(), args.model_dir)
        print("model saved to {}.".format(args.model_dir))
    elif args.mode == 'test':
        print("model loaded from {}.".format(args.model_dir))
        model.load_state_dict(torch.load(args.model_dir))
    print('test on labeled classes')
    args.head = 'head1'
    test(model, labeled_eval_loader, args)
