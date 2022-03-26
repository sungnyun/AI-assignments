#!/usr/bin/env python
# coding: utf-8

# In[1]:


import import_ipynb
from model import *
torch.cuda.set_device(3)


# In[2]:


# followed "Deeply supervised nets." arXiv:1409.5185, 2014.
transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(32, padding=4),
                                transforms.ToTensor()])

train_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                             train=True, 
                                             transform=transform,
                                             download=True)
# validation - in experiment, no need
'''
valid_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                            train=True, 
                                            transform=transforms.ToTensor(),
                                            download=True)
'''
test_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                            train=False, 
                                            transform=transforms.ToTensor(),
                                            download=True)

# validation - in experiment, no need
'''
num_trainset = len(train_dataset)
splitidx = int(0.1 * num_trainset)    # val. ratio from the paper
indices = list(range(num_trainset))

random_seed = 10
np.random.seed(random_seed)
np.random.shuffle(indices)

train_idx = indices[splitidx:]
valid_idx = indices[:splitidx]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

# validation - in experiment, no need

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=128,
                                           sampler = train_sampler)                                           

valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset,
                                           batch_size=128,
                                           sampler = valid_sampler)
'''
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=128,
                                           shuffle=True) 

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=128, 
                                          shuffle=False)


# In[3]:


net = ResNet32().to(device)


# In[5]:


learning_rate=0.1    
num_epochs = 200
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
# training
total_step = len(train_loader)
iteration=0

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        output = net(images)
        loss = criterion(output, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        iteration += 1

    # validation - in experiment, no need
    '''
    with torch.no_grad():
        correct = 0
        total = 0
        for j, (vimages, vlabels) in enumerate(valid_loader):
            vimages = vimages.to(device)
            vlabels = vlabels.to(device)
            output = net(vimages)
            _, predict = torch.max(output.data, 1)
            total += vlabels.size(0)
            correct += (predict==vlabels).sum().item()
        print("Epoch [{}/{}], Validation Accuracy: {:.2f}%".format(epoch+1, num_epochs, (correct/total)*100))
    '''
    
    #debugging
    if (epoch+1) % 10 == 0:
            print ("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, num_epochs, loss.item()))
    
    # lr decay at 32k & 48k iter, according to the paper
    if epoch == 80 or epoch == 120:
        learning_rate /= 10
        update_lr(optimizer, learning_rate)  
    # termination at 64k iter, according to the paper
    if iteration > 64000:
        torch.save(net.state_dict(), "./resnet.pth")
        break
        
# test
net.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        output = net(images)
        _, predict = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predict==labels).sum().item()

    print('Test Accuracy: {} %'.format(100*(correct/total)))


# In[ ]:




