#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch.nn as nn


class SimpleRNN(nn.Module):
    def __init__(self, num_features, name='LSTM', seq_length=100, hidden_size=128):
        super(SimpleRNN, self).__init__()
        self.num_features = num_features
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.name = name
        if self.name == 'RNN':
            self.rnn = nn.RNN(input_size=self.num_features, hidden_size=self.hidden_size, num_layers=2, dropout=0.1, batch_first=True)
        elif self.name == 'LSTM':
            self.rnn = nn.LSTM(input_size=self.num_features, hidden_size=self.hidden_size, batch_first=True)
        #self.linear = nn.Linear(self.hidden_size*2, 1)
        #self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        if self.name == 'LSTM':
            output, (h_n, c_n) = self.rnn(x)
        else:
            output, h_n = self.rnn(x)
                   
        # h_n = h_n.reshape(h_n.size(1), 1, -1).squeeze()
        # output = self.sigmoid(self.linear(h_n))
	#output = output.reshape(output.size(0), output.size(1), -1)
        #output = self.linear(output)
        
        return output


# In[ ]:




