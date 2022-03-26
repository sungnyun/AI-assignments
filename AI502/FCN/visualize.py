#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import torchvision.transforms as transforms
import torchvision.utils as utils
from models import *
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
from utility import *

gpu_ids = [1, 0]    
visual_transform = transforms.Compose([transforms.ToTensor()])

def visualize(args):
    # loading saved models
    vgg16 = VGG16()
    fcn32 = nn.DataParallel(FCN32s(pretrained_net=vgg16, n_class=args.n_class).cuda(device=gpu_ids[0]), device_ids=gpu_ids)
    fcn32.load_state_dict(torch.load('./save/FCN32s.pth'))
    fcn16 = nn.DataParallel(FCN16s(pretrained_net=vgg16, n_class=args.n_class).cuda(device=gpu_ids[0]), device_ids=gpu_ids)
    fcn16.load_state_dict(torch.load('./save/FCN16s.pth'))
    fcn8 = nn.DataParallel(FCN8s(pretrained_net=vgg16, n_class=args.n_class).cuda(device=gpu_ids[0]), device_ids=gpu_ids)
    fcn8.load_state_dict(torch.load('./save/FCN8s.pth'))
    
    transform = transforms.Compose([Rescale(16),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    target_transform = transforms.Compose([Rescale(16), MaskToTensor()])
    validset = VOCSegmentation(root='../data', year='2011', image_set='val', download=True, transform=transform, target_transform=target_transform)
    validloader = DataLoader(validset, batch_size=args.test_batch_size, shuffle=False)
    
    fcn32.eval()
    fcn16.eval()
    fcn8.eval()
    with torch.no_grad():
        for i, (data, target) in enumerate(validloader):
            data.cuda(device=gpu_ids[0])
            output32 = fcn32(data)
            output16 = fcn16(data)
            output8 = fcn8(data)
            
            ground_truth = visual_transform(colorize_mask(target[0].data.numpy()).convert('RGB'))
            predict32 = visual_transform(mask_output(output32).convert('RGB'))
            predict16 = visual_transform(mask_output(output16).convert('RGB'))
            predict8 = visual_transform(mask_output(output8).convert('RGB'))
            
            grid = torch.stack([predict32, predict16, predict8, ground_truth], dim=0)
            grid = utils.make_grid(grid, nrow=4)
            utils.save_image(grid, './save/image%d.png' % (i+1), nrow=4)
            
            if i == 100:
                break