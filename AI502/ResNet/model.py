#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.sampler import SubsetRandomSampler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(3)


# In[2]:


def conv3x3(nin, nout, stride=1, padding=1):
    return nn.Conv2d(nin, nout, kernel_size=3, stride=stride, padding=padding, bias=False)


class residual_Layer(nn.Module):
    def __init__(self, nin, nout, stride=1, padding=1, subsample=False):
        super(residual_Layer, self).__init__()
        self.subsample = subsample
        self.nin = nin
        self.nout = nout
        if self.subsample is False:
            self.conv1 = conv3x3(self.nin, self.nout)
        else:
            self.conv1 = conv3x3(self.nin, self.nout, stride=2)
            self.subsample_layer = conv3x3(self.nin, self.nout, stride=2)
        self.bn1 = nn.BatchNorm2d(self.nout)
        self.bn2 = nn.BatchNorm2d(self.nout)
        self.conv2 = conv3x3(self.nout, self.nout)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.subsample is not False:
            residual = self.subsample_layer(residual)
        out += residual
        out = self.relu(out)
        
        return out


class ResNet(nn.Module):
    def __init__(self, nlayer):
        super(ResNet, self).__init__()
        self.nin = 16
        self.nlayer = nlayer
        self.block1 = self.block(16, subsample=False)
        self.block2 = self.block(32, subsample=True)
        self.block3 = self.block(64, subsample=True)
        self.conv1 = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.avg = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, 10)
        
    def block(self, nout, subsample):
        layers = []
        layers.append(residual_Layer(self.nin, nout, subsample=subsample))
        for i in range(self.nlayer - 1):
            layers.append(residual_Layer(nout, nout))
        self.nin = nout
        
        return nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.avg(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
    
    
def ResNet20():
    return ResNet(nlayer=3)

def ResNet32():
    return ResNet(nlayer=5)

def ResNet44():
    return ResNet(nlayer=7)

def ResNet56():
    return ResNet(nlayer=9)


# In[3]:


# For the seoncd experiment, made plain network without residual
class plain_Layer(nn.Module):
    def __init__(self, nin, nout, stride=1, padding=1, subsample=False):
        super(plain_Layer, self).__init__()
        self.subsample = subsample
        self.nin = nin
        self.nout = nout
        if self.subsample is False:
            self.conv1 = conv3x3(self.nin, self.nout)
        else:
            self.conv1 = conv3x3(self.nin, self.nout, stride=2)
        self.bn1 = nn.BatchNorm2d(self.nout)
        self.bn2 = nn.BatchNorm2d(self.nout)
        self.conv2 = conv3x3(self.nout, self.nout)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        return out

    
class PlainNet(nn.Module):
    def __init__(self, nlayer):
        super(PlainNet, self).__init__()
        self.nlayer = nlayer
        self.nin = 16
        self.block1 = self.block(16, subsample=False)
        self.block2 = self.block(32, subsample=True)
        self.block3 = self.block(64, subsample=True)
        self.conv1 = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.avg = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, 10)
        
    def block(self, nout, subsample):
        layers = []
        layers.append(plain_Layer(self.nin, nout, subsample=subsample))
        for i in range(self.nlayer - 1):
            layers.append(plain_Layer(nout, nout))
        self.nin = nout
        
        return nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.avg(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

    
def Plain20():
    return PlainNet(nlayer=3)

def Plain32():
    return PlainNet(nlayer=5)

def Plain44():
    return PlainNet(nlayer=7)

def Plain56():
    return PlainNet(nlayer=9)


# In[ ]:




