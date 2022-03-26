#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
For experiment 1 #CIFAR10, run python main.py --exp='CIFAR10' --resolution=3072
For experiment 2 #MNIST, run python main.py --exp='MNIST' 
'''


import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
from model import SparseAutoEncoder, AutoEncoder
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--exp', dest='experiment', help='MNIST or CIFAR10', type=str, default='MNIST')
parser.add_argument('--resolution', dest='res', type=int, default=28*28)
parser.add_argument('--hidden', dest='hidden', type=int, default=1000)
parser.add_argument('--sparsity', dest='sparsity', type=int, default=50)
parser.add_argument('--lr', dest='learning_rate', type=float, default=0.001)
parser.add_argument('--num_epoch', dest='num_epoch', type=int, default=30)
parser.add_argument('--batch_size', dest='batch_size', type=int, default=512)
parser.add_argument('--test_batch_size', dest='test_batch_size', type=int, default=10)
args = parser.parse_args()


if args.experiment == 'CIFAR10':
    transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.ToTensor()])
    trainset = datasets.CIFAR10("../data/", train=True, transform=transform, download=True)
    testset = datasets.CIFAR10("../data/", train=False, transform=transforms.ToTensor(), download=True)
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    testloader = data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False)
    
    net = SparseAutoEncoder(args.res, args.hidden, args.res, sparsity=args.sparsity).cuda()
    train(args, trainloader, net)
    autoencoder = AutoEncoder(args.res, args.hidden, args.res).cuda()
    train(args, trainloader, autoencoder)
    CIFAR_visualize(args, testloader)
    
elif args.experiment == 'MNIST':
    # In experiment 2, only train SAE and visualize filters, so no need for testset.
    trainset = datasets.MNIST("../data/", train=True, transform=transforms.ToTensor())
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    
    for sparsity in [70, 40, 25, 10]:
        net = SparseAutoEncoder(args.res, args.hidden, args.res, sparsity=sparsity).cuda()
        train(args, trainloader, net)
    MNIST_visualize(args)

