#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import import_ipynb
from model import *
torch.cuda.set_device(3)


# In[ ]:


transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                transforms.RandomCrop(32, padding=4),
                                transforms.ToTensor()])

train_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                             train=True, 
                                             transform=transform,
                                             download=True)
test_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                            train=False, 
                                            transform=transforms.ToTensor(),
                                            download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=128,
                                           shuffle=True,
                                           num_workers=4) 

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=128, 
                                          shuffle=False,
                                          num_workers=4)


# In[ ]:


# For the result of resnets, see the document.
net1 = Plain20().to(device)
net2 = Plain32().to(device)
net3 = Plain44().to(device)
net4 = Plain56().to(device)

#net5 = ResNet20().to(device)
#net6 = ResNet32().to(device)
#net7 = ResNet44().to(device)
#net8 = ResNet56().to(device)


# In[ ]:


net_group = [net1, net2, net3, net4]
#net_group = [net5, net6, net7, net8]


# In[ ]:


def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

        
total_step = len(train_loader)
train_error = []       
test_error = []
        
for net in net_group:
    
    model_train_error = []
    model_test_error = []
    train_total = 0
    train_correct = 0
    test_total = 0
    test_correct = 0
    learning_rate=0.1    
    num_epochs = 200
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)

    # training
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
            _, predict = torch.max(output.data, 1)
            train_total += labels.size(0)
            train_correct += (predict != labels).sum().item()
            
            # error evaluation
            if iteration % 300 == 0:
                model_train_error.append(100*train_correct/train_total)
                train_total = 0
                train_correct = 0
                
                with torch.no_grad():
                    for j, (image, label) in enumerate(test_loader):
                        image = image.to(device)
                        label = label.to(device)
                        outputs = net(image)
                        _, prediction = torch.max(outputs.data, 1)
                        test_total += label.size(0)
                        test_correct += (prediction != label).sum().item()
                    model_test_error.append(100*test_correct/test_total)       
                    test_total = 0
                    test_correct = 0
            
        #debugging
        if (epoch+1) % 10 == 0:
                print ("Epoch [{}/{}], Loss: {:.4f}".format(epoch+1, num_epochs, loss.item()))
        
        # lr decay at 32k & 48k iter, according to the paper
        if epoch == 80 or epoch == 120:
            learning_rate /= 10
            update_lr(optimizer, learning_rate)
        # termination at 64k iter, according to the paper
        if iteration > 64000:
            print("Test Error: {:.2f}%".format(model_test_error[-1]))
            #torch.save(net.state_dict(), "./resnet.pth")
            break
    
    
    train_error.append(model_train_error)
    test_error.append(model_test_error)
    
    del net


# In[ ]:


it = [i*300 for i in list(range(1, 214))]

# I did not use for-loop, just to clearly show there are 8 networks in experiment.
net1_train_error = train_error[0][:213]
net2_train_error = train_error[1][:213]
net3_train_error = train_error[2][:213]
net4_train_error = train_error[3][:213]
net1_test_error = test_error[0][:213]
net2_test_error = test_error[1][:213]
net3_test_error = test_error[2][:213]
net4_test_error = test_error[3][:213]

#net5_train_error = train_error[0][:213]
#net6_train_error = train_error[1][:213]
#net7_train_error = train_error[2][:213]
#net8_train_error = train_error[3][:213]
#net5_test_error = test_error[0][:213]
#net6_test_error = test_error[1][:213]
#net7_test_error = test_error[2][:213]
#net8_test_error = test_error[3][:213]


# In[ ]:


fig = plt.figure(figsize=(12,8))

plt.plot(it, net1_train_error, 'b', linewidth=0.5)
plt.plot(it, net1_test_error, 'b', linewidth=2, label='plain-20')
plt.plot(it, net2_train_error, 'g',linewidth=0.5)
plt.plot(it, net2_test_error, 'g', linewidth=2, label='plain-32')
plt.plot(it, net3_train_error, 'r', linewidth=0.5)
plt.plot(it, net3_test_error, 'r', linewidth=2, label='plain-44')
plt.plot(it, net4_train_error, 'c', linewidth=0.5)
plt.plot(it, net4_test_error, 'c', linewidth=2, label='plain-56')

#plt.plot(it, net5_train_error, 'b', linewidth=0.5)
#plt.plot(it, net5_test_error, 'b', linewidth=2, label='ResNet-20')
#plt.plot(it, net6_train_error, 'g',linewidth=0.5)
#plt.plot(it, net6_test_error, 'g', linewidth=2, label='ResNet-32')
#plt.plot(it, net7_train_error, 'r', linewidth=0.5)
#plt.plot(it, net7_test_error, 'r', linewidth=2, label='ResNet-44')
#plt.plot(it, net8_train_error, 'c', linewidth=0.5)
#plt.plot(it, net8_test_error, 'c', linewidth=2, label='ResNet-56')

plt.ylim(0, 25)
plt.legend(loc='lower left')
fig.savefig('error.png')
plt.show()


# In[ ]:




