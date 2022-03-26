import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence
import torch.nn as nn
import torch.optim as optim


x_dim = 37
y_dim = 1
r_dim = 8
z_dim = 2
h_dim = 16
portion = 0.5
num_points = 32
n_epoch = 100
batch_size = 32
class_num = 2

data_points = batch_size * num_points

class ScaiDataset(Dataset):
    def __init__(self, csv_file='191112_data.csv', mode='train'):
        raw_data = pd.read_csv(csv_file).iloc[:14272]
        label = raw_data[raw_data.columns[-1]]
        criteria = [-1]

        for i in range(class_num):
            quan = label.quantile(1 / class_num * (i + 1))
            criteria.append(quan)
            mask1 = label > criteria[i]
            mask2 = label <= criteria[-1]
            mask = mask1 * mask2
            label.loc[mask] = i
        raw_data[raw_data.columns[-1]] = label


        self.mode = mode
        if self.mode == 'train':
            raw_data = raw_data.iloc[:-data_points]
        elif self.mode == 'test':
            raw_data = raw_data.iloc[-data_points:]
        self.dataset = np.array(raw_data[raw_data.columns[:]], dtype=np.float32)
        x = self.dataset[:, :-1].reshape(-1, num_points, x_dim)
        y = self.dataset[:, -1].reshape(-1, num_points, y_dim)
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.dataset)


class NeuralProcess(nn.Module):
    def __init__(self, mode='train'):
        super(NeuralProcess, self).__init__()
        self.mode = mode
        self.xy_to_r = nn.Sequential(nn.Linear(x_dim + y_dim, h_dim),
                                     nn.BatchNorm1d(h_dim),
                                     nn.ReLU(),
                                     nn.Linear(h_dim, r_dim))

        self.r_to_z_mu = nn.Sequential(nn.Linear(r_dim, h_dim),
                                       nn.BatchNorm1d(h_dim),
                                       nn.ReLU(),
                                       nn.Linear(h_dim, z_dim))

        self.r_to_z_log_var = nn.Sequential(nn.Linear(r_dim, h_dim),
                                            nn.BatchNorm1d(h_dim),
                                            nn.ReLU(),
                                            nn.Linear(h_dim, z_dim))

        self.xz_to_y = nn.Sequential(nn.Linear(x_dim + z_dim, h_dim),
                                     nn.BatchNorm1d(h_dim),
                                     nn.ReLU(),
                                     nn.Linear(h_dim, class_num),
                                     nn.Softmax())

    def reparametrize(self, z_mu, z_std):
        eps = torch.randn(z_std.size())
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
        y_prob = self.xz_to_y(xz).view(new_batch_size, new_num_points, class_num)
        return y_prob

    def forward(self, x_context, y_context, x_target, y_target=None):
        if self.mode == 'train':
            z_context_mu, z_context_std = self.encoderNet(x_context, y_context)
            z_target_mu, z_target_std = self.encoderNet(x_target, y_target)
            z_context_dist = Normal(loc=z_context_mu, scale=z_context_std)
            z_target_dist = Normal(loc=z_target_mu, scale=z_target_std)

            z = self.reparametrize(z_target_mu, z_target_std)
            y_prob = self.decoderNet(x_target, z)

            y_log_prob = torch.log(y_prob)
            y_log_prob = y_log_prob.view(-1, class_num)
            y_target = y_target.view(-1, y_dim)
            y_target = one_hot(y_target, class_num).type(torch.float32)

            recon = y_log_prob * y_target
            self.recon = - recon.mean(dim=0).sum()
            self.kl = kl_divergence(z_target_dist, z_context_dist).sum()
            return self.recon, self.kl

        elif self.mode == 'test':
            z_context_mu, z_context_std = self.encoderNet(x_context, y_context)

            z = self.reparametrize(z_context_mu, z_context_std)
            y_prob = self.decoderNet(x_target, z)
            dist = Categorical(y_prob)
            self.y_pred = dist.sample()
            return self.y_pred


def context_target_split(x, y):
    num_context = int(num_points * portion)
    num_extra_target = num_points - num_context
    locations = np.random.choice(num_points, size=num_context + num_extra_target, replace=False)
    x_context = x[:, locations[:num_context], :]
    y_context = y[:, locations[:num_context], :]
    x_target = x[:, locations[num_context:], :]
    y_target = y[:, locations[num_context:], :]
    return x_context, y_context, x_target, y_target


def one_hot(y, class_num):
    row, _ = y.size()
    matrix = torch.zeros([row, class_num], dtype=torch.int64)
    for i in range(row):
        matrix[i][int(y[i])] = 1
    return matrix


def train():
    model = NeuralProcess()
    scai_data = ScaiDataset()
    total = len(scai_data)
    bound = int(total / (batch_size * num_points))
    dataloader = torch.utils.data.DataLoader(dataset=scai_data, batch_size=batch_size)
    lr = 1e-4
    for epoch in range(n_epoch):
        model.train()
        if (epoch+1) % 30 == 0:
            lr = lr * 0.1
            print('current learning rate is', lr)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        train_loss, recon_err, kl_div = 0.0, 0.0, 0.0
        cnt = 0
        for data in dataloader:
            cnt += 1
            if cnt == bound - 1:
                break
            x, y = data
            x_context, y_context, x_target, y_target = context_target_split(x, y)
            recon, kl = model(x_context, y_context, x_target, y_target)
            loss = recon + kl

            target_batch_size, target_num_points, _ = x_target.size()
            train_loss += loss * target_batch_size / batch_size
            recon_err += recon * target_batch_size / batch_size
            kl_div += kl * target_batch_size / batch_size

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss = train_loss / cnt
        recon_err = recon_err / cnt
        kl_div = kl_div / cnt

        print('[Epoch %d] train_loss: %.3f, recon_err: %.3f, kl_div: %.3f'
              % (epoch + 1, train_loss, recon_err, kl_div))

    torch.save(model.state_dict(), 'save/ConditionalNeuralProcess.pt')

    print('train ends')


def test():
    model = NeuralProcess(mode='test')
    model.load_state_dict(torch.load('save/ConditionalNeuralProcess.pt'))
    test_scai_data = ScaiDataset(mode='test')
    x = test_scai_data.x.clone()
    y = test_scai_data.y.clone()
    x_context, y_context, x_target, y_target = context_target_split(x, y)
    y_pred = model(x_context, y_context, x_target).detach().numpy().reshape(-1, 1)
    y_target = y_target.numpy().reshape(-1, 1)
    np.savetxt('y_pred.csv', y_pred)
    np.savetxt('y_target.csv', y_target)

    print('test ends')


if __name__ == '__main__':
    train()
    test()