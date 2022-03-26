#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class VGG16(nn.Module):
    def __init__(self, remove_fc=True):
        super(VGG16, self).__init__()
        self.model = models.vgg16(pretrained=True, progress=True)        
        if remove_fc == True:
            del self.model.classifier
    
    def forward(self, x):
        out = {}
        i = 0
        for layer in self.model.features:
            x = layer(x)
            if isinstance(layer, nn.MaxPool2d):
                out["x%d"%(i+1)] = x
                i += 1
        
        return out
    

class FCN32s(nn.Module):
    def __init__(self, pretrained_net, n_class=21):
        super(FCN32s, self).__init__()
        self.pretrained_net = pretrained_net
        self.n_class = n_class
        self.deconv1 = nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, self.n_class, 1)
        
    def forward(self, x):
        out = self.pretrained_net(x)
        x5 = out['x5']
        
        output = self.bn1(F.relu(self.deconv1(x5), inplace=True))
        output = self.bn2(F.relu(self.deconv2(output), inplace=True))
        output = self.bn3(F.relu(self.deconv3(output), inplace=True))
        output = self.bn4(F.relu(self.deconv4(output), inplace=True))
        output = self.bn5(F.relu(self.deconv5(output), inplace=True))
        output = self.classifier(output)
        
        return output

    
class FCN16s(nn.Module):
    def __init__(self, pretrained_net, n_class=21):
        super(FCN16s, self).__init__()
        self.pretrained_net = pretrained_net
        self.n_class = n_class
        self.deconv1 = nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, self.n_class, 1)
        
    def forward(self, x):
        out = self.pretrained_net(x)
        x5 = out['x5']
        x4 = out['x4']

        output = F.relu(self.deconv1(x5), inplace=True)
        output = self.bn1(output + x4)
        output = self.bn2(F.relu(self.deconv2(output), inplace=True))
        output = self.bn3(F.relu(self.deconv3(output), inplace=True))
        output = self.bn4(F.relu(self.deconv4(output), inplace=True))
        output = self.bn5(F.relu(self.deconv5(output), inplace=True))
        output = self.classifier(output)
        
        return output
    
    
class FCN8s(nn.Module):
    def __init__(self, pretrained_net, n_class=21):
        super(FCN8s, self).__init__()
        self.pretrained_net = pretrained_net
        self.n_class = n_class
        self.deconv1 = nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, self.n_class, 1)
        
    def forward(self, x):
        out = self.pretrained_net(x)
        x5 = out['x5']
        x4 = out['x4']
        x3 = out['x3']
        
        output = F.relu(self.deconv1(x5), inplace=True)
        output = self.bn1(output + x4)
        output = F.relu(self.deconv2(output), inplace=True)
        output = self.bn2(output + x3)
        output = self.bn3(F.relu(self.deconv3(output), inplace=True))
        output = self.bn4(F.relu(self.deconv4(output), inplace=True))
        output = self.bn5(F.relu(self.deconv5(output), inplace=True))
        output = self.classifier(output)
        
        return output