import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from lstm import SimpleRNN


x_dim = 45
y_dim = 1
r_dim = 8
z_dim = 2
h_dim = 64
portion = 0.2
num_points = 32
n_epoch = 200
batch_size = 64
class_num = 5
random_seed = 10
seq_length = 50
hidden_size = x_dim


data_points = batch_size * num_points

import torch
import pandas as pd
import torch.utils.data as data
import numpy as np
import random


class OrderDataset(data.Dataset):
    
    # sparse -> data augmentation (randomly pop out)
    def __init__(self, seq_length=100, mode='train', sparse=False):
        self.feature_group = ['WIP', 'rev_no', 'BOM_days', 'LT1', 'LT2', 'LT3', 'problem1', 'problem2', 'problem3', 'problem4', 'problem5', 'problem6',
                 'problem7', 'problem8', 'problem9', 'material1', 'material2', 'material3', 'material4', 'material5', 'material6', 'material7',
                'material8', 'material9', 'dia', 'joint_no', 'weight', 'thickness', 'corp1', 'corp2', 'corp3', 'corp4', 'corp5', 'corp6', 'corp7',
                'corp8','corp9', 'corp10', 'corp11', 'corp12', 'corp13', 'corp14','corp15', 'corp16' 'deliver_date']
        self.target_group = ['process_date']

        xy = np.loadtxt('191126_data.csv', delimiter=',', skiprows=1, dtype=np.float32, encoding='CP949')[:22592]
        self.mode = mode
        if self.mode == 'train':
            xy = xy[:-data_points]
        elif self.mode == 'test':
            xy = xy[-data_points:]
        normalize = [0, 2, 24, 25, 26, 27]
        for i in normalize:
            mean = np.mean(xy[:, i])
            std = np.std(xy[:, i])
            xy[:, i] = (xy[:, i] - mean) / std

        x_data = torch.Tensor(xy[:, :-1].reshape(-1, num_points, x_dim))
        y_data = xy[:, -1]
        y_data = torch.Tensor(np.reshape(y_data, (-1, num_points, y_dim)))

        self.seq_length = seq_length
        self.train = train
        self.sparse = sparse
        self.dataset = x_data
        self.label = y_data
        self.num_features = x_dim
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        
        return self.dataset[idx], self.label[idx]

    def get_features(self):
        return len(self.feature_group)
    

    
class NeuralProcess(nn.Module):
    def __init__(self, mode='train'):    
        super(NeuralProcess, self).__init__()
        self.mode = mode
        self.xy_to_r = nn.Sequential(nn.Linear(x_dim + y_dim, h_dim),
                                     nn.BatchNorm1d(h_dim),
                                     nn.ReLU(),
                                     nn.Linear(h_dim, r_dim))
        
        self.xz_to_y_mu = nn.Sequential(nn.Linear(x_dim + z_dim, h_dim),
                                        nn.BatchNorm1d(h_dim),
                                        nn.ReLU(),
                                        nn.Linear(h_dim, y_dim))

        self.xz_to_y_log_var = nn.Sequential(nn.Linear(x_dim + z_dim, h_dim),
                                             nn.BatchNorm1d(h_dim),
                                             nn.ReLU(),
                                             nn.Linear(h_dim, y_dim))

        self.r_to_z_mu = nn.Sequential(nn.Linear(r_dim, h_dim),
                                       nn.BatchNorm1d(h_dim),
                                       nn.ReLU(),
                                       nn.Linear(h_dim, z_dim))

        self.r_to_z_log_var = nn.Sequential(nn.Linear(r_dim, h_dim),
                                            nn.BatchNorm1d(h_dim),
                                            nn.ReLU(),
                                            nn.Linear(h_dim, z_dim))


    def reparametrize(self, z_mu, z_std):
        eps = torch.randn(z_std.size()).cuda()
        return z_mu + z_std * eps

    def aggregate(self, r):
        return torch.mean(r, dim=1)

    def encoderNet(self, x, y):
        new_batch_size, new_num_points, _ = x.size()
        x_flat = x.view(new_batch_size * new_num_points, x_dim)
        y_flat = y.view(new_batch_size * new_num_points, y_dim)
        xy = torch.cat((x_flat, y_flat), dim=1)
        r_flat = self.xy_to_r(xy)
        
        r = r_flat.view(new_batch_size, new_num_points, r_dim)
        r_mean = self.aggregate(r)
        z_mu = self.r_to_z_mu(r_mean)
        z_log_var = self.r_to_z_log_var(r_mean)
        z_std = torch.exp(0.5 * z_log_var)
        return z_mu, z_std


    def decoderNet(self, x, z):
        new_batch_size, new_num_points, _ = x.size()
        x_flat = x.view(new_batch_size * new_num_points, x_dim)
        z = torch.unsqueeze(z, dim=1).repeat(1, new_num_points, 1)
        z_flat = z.view(new_batch_size * new_num_points, z_dim)
        xz = torch.cat((x_flat, z_flat), dim=1)
        y_mu = self.xz_to_y_mu(xz).view(new_batch_size, new_num_points, y_dim)
        y_log_var = self.xz_to_y_log_var(xz).view(new_batch_size, new_num_points, y_dim)
        y_std = torch.exp(0.5 * y_log_var)
        return y_mu, y_std

    def forward(self, x_context, y_context, x_target, y_target=None):
        if self.mode == 'train':
            z_context_mu, z_context_std = self.encoderNet(x_context, y_context)
            x_all = torch.cat((x_context, x_target), 1)
            y_all = torch.cat((y_context, y_target), 1)
            z_all_mu, z_all_std = self.encoderNet(x_all, y_all)
            z_context_dist = Normal(loc=z_context_mu, scale=z_context_std)
            z_all_dist = Normal(loc=z_all_mu, scale=z_all_std)

            z = self.reparametrize(z_all_mu, z_all_std)
            y_mu, y_std = self.decoderNet(x_target, z)
            y_dist = Normal(loc=y_mu, scale=y_std)

            self.recon = -y_dist.log_prob(y_target).mean(dim=0).mean(dim=0).sum()
            self.kl = kl_divergence(z_all_dist, z_context_dist).mean(dim=0).sum()
            return self.recon, self.kl

        elif self.mode == 'test':
            z_context_mu, z_context_std = self.encoderNet(x_context, y_context)

            z = self.reparametrize(z_context_mu, z_context_std)
            y_mu, y_std = self.decoderNet(x_target, z)
            #y_dist = Normal(loc=y_mu, scale=y_std)

            #self.y_pred = y_dist.sample()
            #return self.y_pred
            return y_mu, y_std


def context_target_split(x, y):
    num_context = int(num_points * portion)
    num_extra_target = num_points - num_context
    locations = np.random.choice(num_points, size=num_context + num_extra_target, replace=False)
    #locations = np.arange(num_points)
    x_context = x[:, locations[:num_context], :]
    y_context = y[:, locations[:num_context], :]
    x_target = x[:, locations[num_context:], :]
    y_target = y[:, locations[num_context:], :]
    return x_context, y_context, x_target, y_target


def train():
    model = NeuralProcess().cuda()
    learning_rate = 1e-2
    #for name, param in model.named_parameters():
        #if param.requires_grad:
            #print(name)
    scai_data = OrderDataset(seq_length=seq_length, mode='train')
    total = len(scai_data)
    bound = int(total / (batch_size * num_points))
    dataloader = torch.utils.data.DataLoader(dataset=scai_data, batch_size=batch_size, shuffle=True)
    train_list, recon_list, kl_list = [], [], []
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(n_epoch):
        model.train()
        kl_annealing_factor = 1/(np.exp(1)-1) * (np.exp((epoch+1)/n_epoch)-1) 
        #if (epoch+1) % 30 == 0:
        #    for param_group in optimizer.param_groups:
        #        param_group['lr'] = 0.1*learning_rate
        
        train_loss, recon_err, kl_div = 0.0, 0.0, 0.0
        cnt = 0
        for data in dataloader:
            cnt += 1
            if cnt == bound - 1:
                break
            x, y = data

            x_context, y_context, x_target, y_target = context_target_split(x, y)
            recon, kl = model(x_context.cuda(), y_context.cuda(), x_target.cuda(), y_target.cuda())
            loss = recon + kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            target_batch_size, target_num_points, _ = x_target.size()
            train_loss += loss * target_batch_size / batch_size
            recon_err += recon * target_batch_size / batch_size
            kl_div += kl * target_batch_size / batch_size

        train_loss = train_loss / cnt
        recon_err = recon_err / cnt
        kl_div = kl_div / cnt

        train_list.append(train_loss)
        recon_list.append(recon_err)
        kl_list.append(kl_div)

        print('[Epoch %d] train_loss: %.3f, recon_err: %.3f, kl_div: %.3f'
              % (epoch + 1, train_loss, recon_err, kl_div))

    torch.save(model.state_dict(), 'save/vanillaNeuralProcess.pt')

    fig = plt.figure()
    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    ax1.plot(train_list, 'r', label='total loss')
    ax1.legend()
    ax1.set_ylim([0, 5])
    ax2.plot(recon_list, 'g', label='recon error')
    ax2.legend()
    ax2.set_ylim([0, 5])
    ax3.plot(kl_list, 'b', label='kl div')
    ax3.legend()
    
    plt.ylim((0.0, 5.0))
    plt.savefig('save/vanilla_loss_curve.png')

    print('train ends')


def test():
    model = NeuralProcess(mode='test').cuda()
    model.load_state_dict(torch.load('save/vanillaNeuralProcess.pt'))
    
    test_scai_data = OrderDataset(seq_length=seq_length, mode='test')
    dataloader = torch.utils.data.DataLoader(dataset=test_scai_data, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        prediction, target = [], []
        MSE = []
        for data in dataloader:
            x, y = data
            x_context, y_context, x_target, y_target = context_target_split(x, y)
            y_pred, std = model(x_context.cuda(), y_context.cuda(), x_target.cuda())
            y_pred = y_pred.cpu().numpy().reshape(-1, 1)
            std = std.cpu().numpy().reshape(-1, 1)
            y_target = y_target.numpy().reshape(-1, 1)
            
            y_pred = y_pred[(y_target!=0)]
            std = std[(y_target!=0)]
            y_target = y_target[y_target!=0]
            mse = np.mean(np.square(y_pred - y_target))
            MSE.append(mse)
            np.savetxt('van_y_pred.csv', y_pred)
            np.savetxt('van_y_std.csv', std)
            np.savetxt('van_y_target.csv', y_target)
        RMSE = np.sqrt(np.mean(np.array(MSE)))
        
    print('test ends')
    print('RMSE: ', RMSE)


if __name__ == '__main__':
    train()
    test()
