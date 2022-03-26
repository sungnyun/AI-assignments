#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import torchvision.utils as utils

num_epoch = 50
batch_size = 128
learning_rate = 0.0002
destruction_rate = 0.5

trainset = MNIST("../data/", train=True, transform=transforms.ToTensor(), download=True)
testset = MNIST("../data/", train=False, transform=transforms.ToTensor(), download=True)
trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = data.DataLoader(testset, batch_size=10, shuffle=False)


# DAE model structure from 'https://github.com/GunhoChoi/Kind-PyTorch-Tutorial/tree/master'
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.layer1 = nn.Sequential(
                        nn.Conv2d(1,32,3,padding=1),   # batch x 32 x 28 x 28
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        nn.Conv2d(32,32,3,padding=1),   # batch x 32 x 28 x 28
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        nn.Conv2d(32,64,3,padding=1),  # batch x 64 x 28 x 28
                        nn.ReLU(),
                        nn.BatchNorm2d(64),
                        nn.Conv2d(64,64,3,padding=1),  # batch x 64 x 28 x 28
                        nn.ReLU(),
                        nn.BatchNorm2d(64),
                        nn.MaxPool2d(2,2)   # batch x 64 x 14 x 14
        )
        self.layer2 = nn.Sequential(
                        nn.Conv2d(64,128,3,padding=1),  # batch x 128 x 14 x 14
                        nn.ReLU(),
                        nn.BatchNorm2d(128),
                        nn.Conv2d(128,128,3,padding=1),  # batch x 128 x 14 x 14
                        nn.ReLU(),
                        nn.BatchNorm2d(128),
                        nn.MaxPool2d(2,2),
                        nn.Conv2d(128,256,3,padding=1),  # batch x 256 x 7 x 7
                        nn.ReLU()
        )
        
                
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(x.size(0), -1)
        return out
    
encoder = Encoder().cuda()


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.layer1 = nn.Sequential(
                        nn.ConvTranspose2d(256,128,3,2,1,1), # batch x 128 x 14 x 14
                        nn.ReLU(),
                        nn.BatchNorm2d(128),
                        nn.ConvTranspose2d(128,128,3,1,1),   # batch x 128 x 14 x 14
                        nn.ReLU(),
                        nn.BatchNorm2d(128),
                        nn.ConvTranspose2d(128,64,3,1,1),    # batch x 64 x 14 x 14
                        nn.ReLU(),
                        nn.BatchNorm2d(64),
                        nn.ConvTranspose2d(64,64,3,1,1),     # batch x 64 x 14 x 14
                        nn.ReLU(),
                        nn.BatchNorm2d(64)
        )
        self.layer2 = nn.Sequential(
                        nn.ConvTranspose2d(64,32,3,1,1),     # batch x 32 x 14 x 14
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        nn.ConvTranspose2d(32,32,3,1,1),     # batch x 32 x 14 x 14
                        nn.ReLU(),
                        nn.BatchNorm2d(32),
                        nn.ConvTranspose2d(32,1,3,2,1,1),    # batch x 1 x 28 x 28
                        nn.ReLU()
        )
        
    def forward(self,x):
        out = x.view(x.size(0), 256, 7, 7)
        out = self.layer1(out)
        out = self.layer2(out)
        return out

decoder = Decoder().cuda()


# train
parameters = list(encoder.parameters()) + list(decoder.parameters())
criterion = nn.MSELoss()
optimizer = optim.Adam(parameters, lr=learning_rate)

print("Training...")
for epoch in range(num_epoch):
    for image, label in trainloader:
        noised_image = image + destruction_rate * torch.randn(image.size(0), 1, 28, 28)
        noised_image = torch.clamp(noised_image, 0, 1)
        image = image.cuda()
        noised_image = noised_image.cuda()
        output = encoder(noised_image)
        output = decoder(output)
        loss = criterion(output, image)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print("Epoch [{}/{}] training loss: {:.4f}".format(epoch+1, num_epoch, loss.item()))
        
# visualize
encoder.eval()
decoder.eval()
with torch.no_grad():
    for image, label in testloader:
        noised_image = image + destruction_rate * torch.randn(image.size(0), 1, 28, 28)
        noised_image = torch.clamp(noised_image, 0, 1)
        noised_image = noised_image.cuda()
        output = encoder(noised_image)
        output = decoder(output)
        noised_image = noised_image.cpu()
        output = output.cpu()
        
        break
    
    grid = torch.cat((image, noised_image, output), dim=0)
    grid = utils.make_grid(grid, nrow=10, padding=5)
    utils.save_image(grid, './denoised.png', nrow=10, padding=5)


# In[ ]:




