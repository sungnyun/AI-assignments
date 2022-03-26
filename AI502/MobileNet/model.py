#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np
import torch
import torch.nn as nn


# In[36]:


def conv3x3(nin, nout, stride=1, padding=1, groups=1):
    return nn.Conv2d(nin, nout, kernel_size=3, stride=stride, padding=padding, bias=False, groups=groups)
  
def conv1x1(nin, nout, stride=1, padding=1):
    return nn.Conv2d(nin, nout, kernel_size=1, stride=stride, padding=padding, bias=False)
  

class dwBlock(nn.Module):
    def __init__(self, nin, nout, alpha, downsample=False):
        super(dwBlock, self).__init__()
        self.alpha = alpha
        self.nin = int(nin * self.alpha)
        self.nout = int(nout * self.alpha)
        self.downsample = downsample
        if self.downsample is False:
            self.conv1 = conv3x3(self.nin, self.nin, stride=1, padding=1, groups=self.nin)    # pointwise conv filter
        else:
            self.conv1 = conv3x3(self.nin, self.nin, stride=2, padding=1, groups=self.nin)
        self.conv2 = conv1x1(self.nin, self.nout, stride=1, padding=0)                        # depthwise conv filetr
        self.bn1 = nn.BatchNorm2d(self.nin)
        self.bn2 = nn.BatchNorm2d(self.nout)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        return out
    
    
class MobileNet(nn.Module):
    def __init__(self, alpha=1, resolution=224):
        super(MobileNet, self).__init__()
        self.alpha = alpha
        self.resolution = resolution
        
        # input size: 3x224x224
        self.conv = conv3x3(3, int(32 * self.alpha), stride=2, padding=1, groups=1)
        self.layers = nn.Sequential(dwBlock(32, 64, self.alpha, downsample=False),
                                    dwBlock(64, 128, self.alpha, downsample=True),
                                    dwBlock(128, 128, self.alpha, downsample=False),
                                    dwBlock(128, 256, self.alpha, downsample=True),
                                    dwBlock(256, 256, self.alpha, downsample=False),
                                    dwBlock(256, 512, self.alpha, downsample=True),
                                    dwBlock(512, 512, self.alpha, downsample=False),
                                    dwBlock(512, 512, self.alpha, downsample=False),
                                    dwBlock(512, 512, self.alpha, downsample=False),
                                    dwBlock(512, 512, self.alpha, downsample=False),
                                    dwBlock(512, 512, self.alpha, downsample=False),
                                    dwBlock(512, 1024, self.alpha, downsample=True),
                                    dwBlock(1024, 1024, self.alpha, downsample=False),
                                    )
        
        # image resolution multiplier (rho)
        if self.resolution == 160:
            self.avgpool = nn.AvgPool2d(5)
        elif self.resolution == 192:
            self.avgpool = nn.AvgPool2d(6)
        elif self.resolution == 224:
            self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(int(1024 * self.alpha), 1000)
        #self.softmax = nn.Softmax2d()
    
    def forward(self, x):
        out = self.conv(x)
        out = self.layers(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        #out = self.softmax(out)
        
        return out


# In[37]:


# MobileNet for cifar-10 dataset (resolution 32)
class MobileNet32(nn.Module):
    
    def __init__(self, alpha=1):
        super(MobileNet32, self).__init__()
        self.alpha = alpha
        # input size: 3x32x32
        self.conv = conv3x3(3, int(32 * self.alpha), stride=1, padding=1, groups=1)
        self.layers = nn.Sequential(dwBlock(32, 64, self.alpha, downsample=False),
                                    dwBlock(64, 128, self.alpha, downsample=False),
                                    dwBlock(128, 128, self.alpha, downsample=False),
                                    dwBlock(128, 256, self.alpha, downsample=True),
                                    dwBlock(256, 256, self.alpha, downsample=False),
                                    dwBlock(256, 512, self.alpha, downsample=False),
                                    dwBlock(512, 512, self.alpha, downsample=True),
                                    dwBlock(512, 512, self.alpha, downsample=False),
                                    )
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(int(512 * self.alpha), 10)

    def forward(self, x):
        out = self.conv(x)
        out = self.layers(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out
        
        
# baseline model for cifar-10
class PlainNet(nn.Module):
    def __init__(self):
        super(PlainNet, self).__init__()
        # input size: 3x32x32
        self.layers = nn.Sequential(conv3x3(3, 32),
                                    conv3x3(32, 64),
                                    conv3x3(64, 128),
                                    conv3x3(128, 128),
                                    conv3x3(128, 256, stride=2),
                                    conv3x3(256, 256),
                                    conv3x3(256, 512),
                                    conv3x3(512, 512, stride=2),
                                    conv3x3(512, 512),
                                    nn.AvgPool2d(8),
                                    )
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        out = self.layers(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out


# In[40]:


if __name__ == '__main__':
    from thop import profile
    import torchsummary
    
    net1 = PlainNet()
    net2 = MobileNet32(alpha=1)
    net3 = MobileNet32(alpha=0.75)
    
    for net in [net1, net2, net3]:
        inputs = torch.Tensor(torch.randn(1, 3, 32, 32))
        flops, params = profile(net, inputs=(inputs, ), verbose=False)
        print('%0.2f M FLOPs/image'%(flops/1000000))
        
        torchsummary.summary(net.cuda(), (3, 32, 32))
        
        print(" ")


# In[ ]:




