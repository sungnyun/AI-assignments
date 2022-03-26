#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import VOCSegmentation
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models import *
from utility import *
import time
import numpy as np
import argparse


'''
run python train.py with --mode='train' --model='FCN32s' first.
Then, run with other models: FCN16s -> FCN8s.
Finally, python train.py --mode='valid' --model=whatever you want.
'''

parser = argparse.ArgumentParser()
parser.add_argument('--mode', dest='mode', help='train or valid or visualize', type=str, default='train')
parser.add_argument('--model', dest='model', help='FCN32s/16s/8s', type=str, default='FCN32s')
parser.add_argument('--num_epoch', dest='num_epoch', help='number of epochs', type=int, default=300)  # at least 175
parser.add_argument('--n_class', dest='n_class', help='number of classes', type=int, default=21)  # 20 classes + background
parser.add_argument('--learning_rate', dest='learning_rate', help='learning rate', type=float, default=1e-4) 
parser.add_argument('--momentum', dest='momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', dest='weight_decay', type=float, default= 5**(-4))
parser.add_argument('--batch_size', dest='batch_size', help='batch size', type=int, default=20)
parser.add_argument('--test_batch_size', dest='test_batch_size', help='test batch size', type=int, default=1)
args = parser.parse_args()
gpu_ids = [1, 0]
void_idx = 255   # set in the voc dataset
    

def train(args):
    
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(256),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    target_transform = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(256),
                                           MaskToTensor()]) 
    trainset = VOCSegmentation(root='../data', year='2011', image_set='train', download=True, transform=transform, target_transform=target_transform)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=5)
    
    if args.model == 'FCN32s':
        vgg16 = VGG16()
        net = nn.DataParallel(FCN32s(pretrained_net=vgg16, n_class=args.n_class).cuda(device=gpu_ids[0]), device_ids=gpu_ids)
    elif args.model == 'FCN16s':
        vgg16 = VGG16()
        net = nn.DataParallel(FCN16s(pretrained_net=vgg16, n_class=args.n_class).cuda(device=gpu_ids[0]), device_ids=gpu_ids)
        net.load_state_dict(torch.load('./save/FCN32s.pth'))
    elif args.model == 'FCN8s':
        vgg16 = VGG16()
        net = nn.DataParallel(FCN8s(pretrained_net=vgg16, n_class=args.n_class).cuda(device=gpu_ids[0]), device_ids=gpu_ids)
        net.load_state_dict(torch.load('./save/FCN32s.pth'))
    
    # cross-entropy loss supports 2d image 
    criterion = nn.CrossEntropyLoss(ignore_index=void_idx)
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    
    net.train()
    
    for epoch in range(args.num_epoch):
        running_loss = 0
        t = time.time()
        for data, target in trainloader:
            data = data.cuda(device=gpu_ids[0])
            target = target.cuda(device=gpu_ids[0])
            
            optimizer.zero_grad()
            output = net(data)
            assert output.size()[2:] == target.size()[1:]
            assert output.size()[1] == args.n_class
            
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        t = time.time() - t
        print("Epoch [{}/{}], loss: {:.4f}, training time: {:.1f} s".format(epoch+1, args.num_epoch, running_loss/len(trainloader), t))
        
    torch.save(net.state_dict(), "./save/%s.pth" % args.model)
        
        



# In[ ]:


def valid(args):

    if args.model == 'FCN32s':
        vgg16 = VGG16()
        net = nn.DataParallel(FCN32s(pretrained_net=vgg16, n_class=args.n_class).cuda(device=gpu_ids[0]), device_ids=gpu_ids)
        net.load_state_dict(torch.load('./save/FCN32s.pth'))
        transform = transforms.Compose([Rescale(32),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        target_transform = transforms.Compose([Rescale(32), MaskToTensor()])
    elif args.model == 'FCN16s':
        vgg16 = VGG16()
        net = nn.DataParallel(FCN16s(pretrained_net=vgg16, n_class=args.n_class).cuda(device=gpu_ids[0]), device_ids=gpu_ids)
        net.load_state_dict(torch.load('./save/FCN16s.pth'))
        transform = transforms.Compose([Rescale(16),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        target_transform = transforms.Compose([Rescale(16), MaskToTensor()])
    elif args.model == 'FCN8s':
        vgg16 = VGG16()
        net = nn.DataParallel(FCN8s(pretrained_net=vgg16, n_class=args.n_class).cuda(device=gpu_ids[0]), device_ids=gpu_ids)
        net.load_state_dict(torch.load('./save/FCN8s.pth'))
        transform = transforms.Compose([Rescale(16),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        target_transform = transforms.Compose([Rescale(16), MaskToTensor()])
        
    validset = VOCSegmentation(root='../data', year='2011', image_set='val', download=True, transform=transform, target_transform=target_transform)
    validloader = DataLoader(validset, batch_size=args.test_batch_size, shuffle=False)
        
    total_pixel_acc = 0
    total_mean_acc = 0
    total_mean_IU = 0 
    total_fw_IU = 0
    
    net.eval()
    with torch.no_grad():
        for data, target in validloader:
            data = data.cuda(device=gpu_ids[0])
            target = target.cuda(device=gpu_ids[0])
            output = net(data)

            assert output.size()[2:] == target.size()[1:]
            assert output.size()[1] == args.n_class

            pixel_acc, mean_acc, mean_IU, fw_IU = evaluate(output, target, args.n_class)
            total_pixel_acc += pixel_acc
            total_mean_acc += mean_acc
            total_mean_IU += mean_IU
            total_fw_IU += fw_IU
            #debug
            print("pixel acc: {:.2f}%, mean acc: {:.2f}%, mean IU: {:.2f}%, f.w. IU: {:.2f}%".format(100*pixel_acc, 100*mean_acc, 100*mean_IU, 100*fw_IU))

        total_pixel_acc = 100 * total_pixel_acc / len(validloader)
        total_mean_acc = 100 * total_mean_acc / len(validloader)
        total_mean_IU = 100 * total_mean_IU / len(validloader)
        total_fw_IU = 100 * total_fw_IU / len(validloader)
        print("===== %s =====" % args.model)
        print("pixel acc: {:.2f}%, mean acc: {:.2f}%, mean IU: {:.2f}%, f.w. IU: {:.2f}%".format(total_pixel_acc, total_mean_acc, total_mean_IU, total_fw_IU))
    
        
# In[ ]:



if args.mode == 'train':
    train(args)
    
elif args.mode == 'valid':
    valid(args)

elif args.mode =='visualize':
    import visualize
    visualize.visualize(args)