#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
from os.path import join
import time
from scipy import io
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import import_ipynb
from model import *
import warnings
warnings.filterwarnings("ignore")


# In[2]:


root = '/home/ftpuser/hdd/ImageNet_pytorch'
traindir = os.path.join(root, 'train')
valdir = os.path.join(root, 'val')

train_transform = transforms.Compose([transforms.RandomResizedCrop(224, ratio=(1,1.3)), 
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])])
valid_transform = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224), 
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                           std=[0.229, 0.224, 0.225])])

trainset = datasets.ImageFolder(root=traindir, transform=train_transform)    # 1280k images
validset = datasets.ImageFolder(root=valdir, transform=valid_transform)      # 50k images

train_loader = data.DataLoader(dataset=trainset, batch_size=256, shuffle=True, num_workers=5)
valid_loader = data.DataLoader(dataset=validset, batch_size=256, shuffle=False, num_workers=5)


# In[3]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpu_ids=[3, 0, 1, 2]


# In[4]:


net = torch.nn.DataParallel(MobileNet(alpha=1, resolution=224).cuda(device=gpu_ids[0]), device_ids=gpu_ids)


# In[5]:


# hyperparameters - no info. in the paper
learning_rate = 0.1
num_epoch = 50
criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(net.parameters(), lr=learning_rate)

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

train_loss = []        
valid_acc = []

# train
for epoch in range(num_epoch):
    t = time.time()
    net.train()
    for image, label in train_loader:
        image = image.cuda(device=gpu_ids[0])
        label = label.cuda(device=gpu_ids[0])
        output = net(image)
        loss = criterion(output, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) == 30:
        learning_rate /= 10
        update_lr(optimizer, learning_rate)
        
    t = time.time() - t
    print("Epoch [{}/{}], train loss: {:.4f}, time: {:.1f}s".format(epoch+1, num_epoch, loss.item(), t))
    train_loss.append(loss.item())
    
    # validation
    net.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        running_loss = 0
        for image, label in valid_loader:
            image = image.cuda(device=gpu_ids[0])
            label = label.cuda(device=gpu_ids[0])
            output = net(image)
            running_loss += criterion(output, label)
            _, predict = torch.max(output.data, 1)
            total += label.size(0)
            correct += (predict==label).sum().item()

        print("Epoch [{}/{}], valid loss: {:.4f}, valid acc: {:.4f} %".format(epoch+1, num_epoch, running_loss/len(valid_loader), 100*(correct/total)))
        valid_acc.append(100*(correct/total))
        
torch.save(net.state_dict(), "./mobilenet.pth")


# In[2]:


# visualization
import matplotlib.pyplot as plt

x = list(range(1, 51))
y = valid_acc

fig = plt.figure()
plt.plot(x, y, 'r')
plt.xlabel('epochs')
plt.ylabel('accuracy')
fig.savefig('ImageNet Accuracy.png')
plt.show()


# In[ ]:




