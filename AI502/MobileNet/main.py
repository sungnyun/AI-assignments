#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
from os.path import join
from scipy import io
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image

import import_ipynb
from model import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(1)


# In[2]:


class StanfordDogsDataset(data.Dataset):
    
    def __init__(self, root='../data/StanfordDogs', train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform
        self.image_dir = join(self.root, 'Images')
        self.list_dir = join(self.root, 'lists')
        self.list = self.load_list()
        
    def __len__(self):
        return len(self.list)
    
    def __getitem__(self, idx):
        imgname, target = self.list[idx]
        img_dir = join(self.image_dir, imgname)
        image = Image.open(img_dir).convert('RGB')
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, target
    
    def load_list(self):
        if self.train:
            imgname = io.loadmat(join(self.list_dir, 'train_list.mat'))['file_list']
            label = io.loadmat(join(self.list_dir, 'train_list.mat'))['labels']
        else:
            imgname = io.loadmat(join(self.list_dir, 'test_list.mat'))['file_list']
            label = io.loadmat(join(self.list_dir, 'test_list.mat'))['labels']
        imgname = [i[0][0] for i in imgname]
        label = [i[0]-1 for i in label]
        
        return list(zip(imgname, label))


# In[3]:


# change size depending on resolution
train_transform = transforms.Compose([transforms.RandomResizedCrop(224, ratio=(1,1.3)), 
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor()])
test_transform = transforms.Compose([transforms.RandomResizedCrop(224, ratio=(1,1.3)), 
                                     transforms.ToTensor()])

trainset = StanfordDogsDataset(root='../data/StanfordDogs', train=True, transform=train_transform)
testset = StanfordDogsDataset(root='../data/StanfordDogs', train=False, transform=test_transform)

train_loader = data.DataLoader(dataset=trainset, batch_size=64 , shuffle=True, num_workers=5)
test_loader = data.DataLoader(dataset=testset, batch_size=64 , shuffle=False, num_workers=5)


# In[4]:


# 1.0 MobileNet-224
net = MobileNet(alpha=1, resolution=224).to(device)

# 0.75 MobileNet-224
#net = MobileNet(alpha=0.75, resolution=224).to(device)

# 0.75 MobileNet-192
#net = MobileNet(alpha=0.75, resolution=192).to(device)


# In[5]:


# hyperparameters - no info. in the paper
learning_rate = 0.1
num_epoch = 5000
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
#scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=)

test_acc = []

# train
for epoch in range(num_epoch):
    for image, label in train_loader:
        image = image.to(device)
        label = label.to(device)
        output = net(image)
        loss = criterion(output, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    #scheduler.step()        
    print("Epoch [{}/{}], loss: {:.4f}".format(epoch+1, num_epoch, loss.item()))
    
    ### validation
    if (epoch+1) % 50 == 0:
        with torch.no_grad():
            correct = 0
            total = 0
            for image, label in test_loader:
                image = image.to(device)
                label = label.to(device)
                output = net(image)
                _, predict = torch.max(output.data, 1)
                total += label.size(0)
                correct += (predict==label).sum().item()
            test_acc.append(100*(correct/total))

            #print('Test Accuracy: {} %'.format(100*(correct/total)))
    
print(test_acc)
    
'''
# test
net.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for image, label in test_loader:
        image = image.to(device)
        label = label.to(device)
        output = net(image)
        _, predict = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predict==label).sum().item()

    print('Test Accuracy: {} %'.format(100*(correct/total)))


# In[ ]:
'''