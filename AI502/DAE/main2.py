#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
import random


class Autoencoder(nn.Module):
    def __init__(self, nin, nout, hidden):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Linear(nin, hidden), nn.Sigmoid())
        self.decoder = nn.Sequential(nn.Linear(hidden, nout), nn.Sigmoid())
        #self.relu = nn.ReLU(inplace=True)
        #self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.encoder(x)
        out = self.decoder(out)
        
        return out

class Classifier(nn.Module):
    def __init__(self, nin, nout):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(nin, nout)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.classifier(x)
        x = self.sigmoid(x)
        return x
    
    
num_epoch = 10
cl_epoch = 20
batch_size = 128
learning_rate = 0.001

# They conversed training set and test set    
trainset = MNIST("../data/", train=False, transform=transforms.ToTensor(), download=True)
testset = MNIST("../data/", train=True, transform=transforms.ToTensor(), download=True)
trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)
testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=4)

def train(destruction_rate=0.1):
    criterion = nn.MSELoss()
    AE1 = Autoencoder(nin=28*28, nout=28*28, hidden=2000).cuda()
    optimizer = optim.Adam(AE1.parameters(), lr=learning_rate)
    print("1st Pretraining...")
    for epoch in range(num_epoch):
        for image, label in trainloader:
            image = image.view(image.size(0), -1)
            noised_image = image + destruction_rate*torch.randn(image.size(0), image.size(1))
            
            image = image.cuda()
            noised_image = noised_image.cuda()
            output = AE1(noised_image)
            loss = criterion(output, image)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Epoch [{}/{}] training loss: {:.4f}".format(epoch+1, num_epoch, loss.item()))
    for param in AE1.parameters():     # freezing autoencoder 1
        param.requires_grad = False
    
    
    AE2 = Autoencoder(nin=2000, nout=2000, hidden=2000).cuda() 
    optimizer = optim.Adam(AE2.parameters(), lr=learning_rate)
    print("2nd Pretraining...")
    for epoch in range(10):
        for image, label in trainloader:
            image = image.view(image.size(0), -1).cuda()
            image = AE1.encoder(image)
            noised_image = image + destruction_rate*torch.randn(image.size(0), image.size(1)).cuda()
            
            noised_image = noised_image.cuda()
            output = AE2(noised_image)
            loss = criterion(output, image)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Epoch [{}/{}] training loss: {:.4f}".format(epoch+1, num_epoch, loss.item()))
    for param in AE2.parameters():      # freezeing autoencoder 2
        param.requires_grad = False
        
        
    AE3 = Autoencoder(nin=2000, nout=2000, hidden=2000).cuda() 
    optimizer = optim.Adam(AE3.parameters(), lr=learning_rate)
    print("3rd Pretraining...")
    for epoch in range(num_epoch):
        for image, label in trainloader:
            image = image.view(image.size(0), -1).cuda()
            image = AE1.encoder(image)
            image = AE2.encoder(image)
            noised_image = image + destruction_rate*torch.randn(image.size(0), image.size(1)).cuda()
            
            noised_image = noised_image.cuda()
            output = AE3(noised_image)
            loss = criterion(output, image)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("Epoch [{}/{}] training loss: {:.4f}".format(epoch+1, num_epoch, loss.item()))
    for param in AE1.parameters():      # unfreezeing
        param.requires_grad = True
    for param in AE2.parameters():
        param.requires_grad = True
    
    # classifying with stacked auto-associators
    classifier = Classifier(nin=2000, nout=10).cuda()
    parameters = list(AE1.parameters())+list(AE2.parameters())+list(AE3.parameters())+list(classifier.parameters())
    optimizer = optim.Adam(parameters, lr=learning_rate)
    print("Training...")
    criterion = nn.CrossEntropyLoss()
    for epoch in range(cl_epoch):
        correct = 0
        total = 0
        for image, label in trainloader:
            image = image.view(image.size(0), -1).cuda()
            image = AE1.encoder(image)
            image = AE2.encoder(image)
            image = AE3.encoder(image)
            output = classifier(image)
            label = label.cuda()
            loss = criterion(output, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            _, idx = torch.max(output, dim=1)
            correct += (idx == label).sum().item()
            total += label.size(0)
            
        print("Epoch [{}/{}] training loss: {:.4f}, accuracy: {:.2f}%".format(epoch+1, cl_epoch, loss.item(), (correct/total)*100))
    
    torch.save(AE1.state_dict(), './save/Autoencoder1_%d%%' %(destruction_rate*100))
    torch.save(AE2.state_dict(), './save/Autoencoder2_%d%%' %(destruction_rate*100))
    torch.save(AE3.state_dict(), './save/Autoencoder3_%d%%' %(destruction_rate*100))
    torch.save(classifier.state_dict(), './save/Classifier_%d%%' %(destruction_rate*100))

def test(destruction_rate=0.1, bg_rand=False):
    AE1 = Autoencoder(nin=28*28, nout=28*28, hidden=2000).cuda()
    AE2 = Autoencoder(nin=2000, nout=2000, hidden=2000).cuda()
    AE3 = Autoencoder(nin=2000, nout=2000, hidden=2000).cuda()
    classifier = Classifier(nin=2000, nout=10).cuda()
    
    AE1.load_state_dict(torch.load('./save/Autoencoder1_%d%%' %(destruction_rate*100)))
    AE2.load_state_dict(torch.load('./save/Autoencoder2_%d%%' %(destruction_rate*100)))
    AE3.load_state_dict(torch.load('./save/Autoencoder3_%d%%' %(destruction_rate*100)))
    classifier.load_state_dict(torch.load('./save/Classifier_%d%%' %(destruction_rate*100)))
        
    with torch.no_grad():
        correct = 0
        total = 0
        for image, label in testloader:
            label = label.cuda()    
            if bg_rand == True:
                image = image + 0.4*torch.randn(image.size(0), 1, 28, 28)
            image = image.view(image.size(0), -1).cuda()
            output = AE1.encoder(image)
            output = AE2.encoder(output)
            output = AE3.encoder(output)
            output = classifier(output)
            
            _, idx = torch.max(output, dim=1)
            correct += (idx == label).sum().item()
            total += label.size(0)
        print("===Test Result===")
        if destruction_rate > 0:
            print("Test Accuracy for SdA-3: {:.2f}%".format((correct/total)*100))
        else:
            print("Test Accuracy for SAA-3: {:.2f}%".format((correct/total)*100))

# In[2]:
train(destruction_rate=0.1)
train(destruction_rate=0.0)
train(destruction_rate=0.4)

test(destruction_rate=0.1)
test(destruction_rate=0.0)
test(destruction_rate=0.4, bg_rand=True)
test(destruction_rate=0.0, bg_rand=True)