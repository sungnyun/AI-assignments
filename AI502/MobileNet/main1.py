#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import time
from scipy import io
import torch
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import import_ipynb
from model import *


# In[ ]:


transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(32, padding=4),
                                transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='../data/',
                                             train=True, 
                                             transform=transform,
                                             download=True)
testset = torchvision.datasets.CIFAR10(root='../data/',
                                            train=False, 
                                            transform=transforms.ToTensor(),
                                            download=True)

train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                           batch_size=128,
                                           shuffle=True) 

test_loader = torch.utils.data.DataLoader(dataset=testset,
                                          batch_size=128, 
                                          shuffle=False)


# In[ ]:


device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
gpu_ids = [3, 0, 1, 2]

net1 = torch.nn.DataParallel(PlainNet().cuda(device=gpu_ids[0]), device_ids=gpu_ids)
net2 = torch.nn.DataParallel(MobileNet32(alpha=1).cuda(device=gpu_ids[0]), device_ids=gpu_ids)
net3 = torch.nn.DataParallel(MobileNet32(alpha=0.75).cuda(device=gpu_ids[0]), device_ids=gpu_ids)    


# In[ ]:


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(net, num_epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
    total_time = 0
    
    for epoch in range(num_epochs):
        t = time.time()
        for image, label in train_loader:
            image = image.cuda(device=gpu_ids[0])
            label = label.cuda(device=gpu_ids[0])
            output = net(image)
            loss = criterion(output, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        t = time.time() - t
        total_time += t
        
        if epoch == 80:
            learning_rate /= 10
            update_lr(optimizer, learning_rate) 
    
    print("Training Complete. total time: {:.1f}min".format(total_time/60))
    
    # test
    net.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (image, label) in enumerate(test_loader):
            image = image.cuda(device=gpu_ids[0])
            label = label.cuda(device=gpu_ids[0])
            output = net(image)
            _, predict = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predict==label).sum().item()

        print('Test Accuracy: {} %'.format(100*(correct/total)))    


# In[ ]:


# hyperparameters
num_epochs = 150

print("====== Baseline ======")
train(net1, num_epochs, learning_rate=0.01)
print(" ")
print("====== MobileNet ======")
train(net2, num_epochs, learning_rate=0.1)
print(" ")
print("====== 0.75 MobileNet ======")
train(net3, num_epochs, learning_rate=0.1)


# In[ ]:




