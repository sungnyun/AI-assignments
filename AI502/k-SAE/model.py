#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import numpy as np

class SparseAutoEncoder(nn.Module):
    def __init__(self, nin, nhidden, nout, sparsity=10):
        super(SparseAutoEncoder, self).__init__()
        self.encoder = nn.Linear(nin, nhidden)
        self.decoder = nn.Linear(nhidden, nout)
        self.sparsity = sparsity
        self.name = 'SparseAutoEncoder%d' %(sparsity)
        
    def forward(self, x):
        output = self.encoder(x)
        
        # leave k largest activations of z and set the rest to zero
        values, indices = torch.topk(output, self.sparsity, dim=1)
        for i in range(output.size(0)):
            mask = np.zeros(output.size(1))
            for j in indices[i]:
                mask[j] = 1
            output[i] = torch.mul(output[i], torch.from_numpy(mask).cuda())
            
        output = self.decoder(output)
        
        return output

    
class AutoEncoder(nn.Module):
    def __init__(self, nin, nhidden, nout):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Linear(nin, nhidden)
        self.decoder = nn.Linear(nhidden, nout)
        self.sigmoid = nn.Sigmoid()
        self.name = 'AutoEncoder'
    
    def forward(self, x):
        output = self.sigmoid(self.encoder(x))
        output = self.sigmoid(self.decoder(output))
        return output
