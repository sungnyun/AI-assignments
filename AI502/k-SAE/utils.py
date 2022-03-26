#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import time
from model import SparseAutoEncoder, AutoEncoder


def train(args, trainloader, net):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate)

    print("Training...")
    for epoch in range(args.num_epoch):
        t = time.time()
        for image, label in trainloader:
            for params in net.parameters():
                params.requires_grad = True
            image = image.view(image.size(0), -1).cuda()
            output = net(image)
            loss = criterion(output, image)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        t = time.time() - t
        print("Epoch [{}/{}] training loss: {:.4f} time: {:.1f}s".format(epoch+1, args.num_epoch, loss.item(), t))
        
    torch.save(net.state_dict(), './save/%s.pth' %(net.name))

    
def CIFAR_visualize(args, testloader):
    autoencoder = AutoEncoder(args.res, args.hidden, args.res).cuda()
    net = SparseAutoEncoder(args.res, args.hidden, args.res, sparsity=args.sparsity).cuda()
    autoencoder.load_state_dict(torch.load('./save/AutoEncoder.pth'))
    net.load_state_dict(torch.load('./save/SparseAutoEncoder50.pth'))
    
    with torch.no_grad():
        for image, label in testloader:
            image = image.view(image.size(0), -1).cuda()
            ae_output = autoencoder(image)
            sae_output = net(image)
            original = image.view(image.size(0), 3, 32, 32).cpu()
            ae_output = ae_output.view(ae_output.size(0), 3, 32, 32).cpu()
            sae_output = sae_output.view(sae_output.size(0), 3, 32, 32).cpu()
            
            break
            
        grid = torch.cat((original, ae_output, sae_output), dim=0)
        grid = vutils.make_grid(grid, nrow=args.test_batch_size, padding=5)
        vutils.save_image(grid, './save/CIFAR10_result.png', nrow=args.test_batch_size, padding=5)
        
        
def MNIST_visualize(args):
    net70 = SparseAutoEncoder(args.res, args.hidden, args.res, sparsity=70).cuda()
    net40 = SparseAutoEncoder(args.res, args.hidden, args.res, sparsity=40).cuda()
    net25 = SparseAutoEncoder(args.res, args.hidden, args.res, sparsity=25).cuda()
    net10 = SparseAutoEncoder(args.res, args.hidden, args.res, sparsity=10).cuda()
    net70.load_state_dict(torch.load('./save/SparseAutoEncoder70.pth'))
    net40.load_state_dict(torch.load('./save/SparseAutoEncoder40.pth'))
    net25.load_state_dict(torch.load('./save/SparseAutoEncoder25.pth'))
    net10.load_state_dict(torch.load('./save/SparseAutoEncoder10.pth'))

    for net in [net70, net40, net25, net10]:
        filters = net.encoder.weight.cpu()
        filters = filters[:120]
        filters = filters.view(120, 1, 28, 28) + 0.2*torch.ones((120, 1, 28, 28))
        
        grid = vutils.make_grid(filters, nrow=30, padding=2)
        vutils.save_image(grid, './save/MNIST_result_%s.png' %(net.name), nrow=30, padding=2)
        

