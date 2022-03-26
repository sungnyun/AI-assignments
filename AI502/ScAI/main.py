#!/usr/bin/env python
# coding: utf-8

# In[7]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import time
import import_ipynb
from model import *
from dataloader import OrderDataset
from torch.utils.data.sampler import SubsetRandomSampler
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

random_seed = 10
seq_length = 50
batch_size = 512
hidden_size = 64
num_epoch = 10
learning_rate = 0.0001
validation_ratio = 0.1

trainset = OrderDataset(seq_length=seq_length, train=True)
validset = OrderDataset(seq_length=seq_length, train=True)

num_features = trainset.get_features()
num_data = len(trainset)

splitidx = int((1-validation_ratio) * num_data)
indices = list(range(num_data - seq_length))
#np.random.seed(random_seed)
#np.random.shuffle(indices)
train_idx = indices[:(splitidx-seq_length)]
valid_idx = indices[splitidx:]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

trainloader = data.DataLoader(dataset=trainset, batch_size=batch_size, sampler=train_sampler, num_workers=5)
validloader = data.DataLoader(dataset=validset, batch_size=batch_size, sampler=valid_sampler, num_workers=5)

net = SimpleRNN(num_features, name='LSTM', seq_length=seq_length, hidden_size=hidden_size).cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

def lr_decay(optimizer, lr):
    for param in optimizer.param_groups:
        param['lr'] = lr

def train(net, learning_rate):
    training_loss = []
    current_lr = learning_rate
    for epoch in range(num_epoch):
        total = 0
        correct = 0
        t = time.time()
        for data, target in trainloader:
            data = data.cuda()
            target = target.cuda()
            output = net(data)
            loss = criterion(output, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total += data.size(0)
            correct += (torch.round(output) == target).sum().item()
            
        t = time.time() - t
        if (epoch+1) % 1==0:
            print('Epoch [{}/{}], training loss: {:.4f}, time: {:.1f} s, acc: {:.2f}%'.format(epoch+1, num_epoch, loss.item(), t, (correct/total)*100))
            training_loss.append(loss.item())
    
        if epoch == 100:
            current_lr /= 10
            lr_decay(optimizer, lr=current_lr)
        
    torch.save(net.state_dict(), "./save/simpleLSTM.pth")
    
train(net, learning_rate)


# In[2]:


import csv

f = open('./result.csv', 'w', encoding='utf-8', newline='')
writer = csv.writer(f)
writer.writerow(['prediction', 'date', 'difference', 'anomaly_detection'])

net = SimpleRNN(num_features, name='LSTM', seq_length=seq_length, hidden_size=hidden_size).cuda(device=1)
net.load_state_dict(torch.load('./save/simpleLSTM.pth'))

net.eval()
with torch.no_grad():
    for data, target in validloader:
        data = data.cuda(device=1)
        target = target.cuda(device=1)
        output = net(data)

        output = output[:, -1]
        target = target[:, -1]
        for i in range(target.size(0)):
            prediction = round(output[i].item())
            date = target[i].item()
            difference = abs(prediction - date)
            
            if date > 5:
                if prediction > 5:
                    anomaly_detection = 1
                else:
                    anomaly_detection = 0
            else:
                anomaly_detection = -1
            
            writer.writerow([prediction, date, difference, anomaly_detection])
                
                #if difference <= 50:
                    #writer.writerow([prediction, date, difference])
                    
                
    f.close()


# In[12]:


import csv

f = open('./result.csv', 'w', encoding='utf-8', newline='')
writer = csv.writer(f)
writer.writerow(['prediction', 'date'])

net = SimpleRNN(num_features, name='LSTM', seq_length=seq_length, hidden_size=hidden_size).cuda(device=1)
net.load_state_dict(torch.load('./save/simpleLSTM.pth'))

net.eval()
with torch.no_grad():
    for data, target in validloader:
        data = data.cuda(device=1)
        target = target.cuda(device=1)
        output = net(data)

        output = output[:, -1]
        target = target[:, -1]
        for i in range(target.size(0)):
            prediction = output[i].item()
            
            if prediction > 0.65:
                prediction = 1
            else:
                prediction = 0
            
            date = target[i].item()
            writer.writerow([prediction, date])
                
                #if difference <= 50:
                    #writer.writerow([prediction, date, difference])
                    
                
    f.close()


# In[ ]:




