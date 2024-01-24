import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import scipy.ndimage as ndimage
from scipy.stats import multivariate_normal
import numpy as np

def get_shapes(layer, dataset='cifar10'):
    if dataset == 'visdaC':
        if layer == 1:
            return 256, 56
        if layer == 2:
            return 512, 28
        if layer == 3:
            return 1024, 14
        if layer == 4:
            return 2048, 7
    elif dataset in ['cifar10','cifar100']:
        if layer == 1:
            return 256, 32
        if layer == 2:
            return 512, 16
        if layer == 3:
            return 1024, 8
        if layer == 4:
            return 2048, 1

def conv_out_size(c, k):
    return (c + k) + 1

def score(x):
    was_numpy = False
    if isinstance(x, np.ndarray):
        was_numpy = True
        x = torch.from_numpy(x)
    while x.ndim > 2:
        x = torch.max(x, dim=-1).values
    if x.ndim == 2:
        x = torch.max(x, dim=1).values
    if was_numpy:
        return x.numpy()
    return x

def init_weight(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)

class Projection(torch.nn.Module):
    def __init__(self, in_planes, out_planes=None, n_layers=1, layer_type=0, stride=1):
        super(Projection, self).__init__()

        if out_planes is None:
            out_planes = in_planes
        self.layers = torch.nn.Sequential()
        _in = None
        _out = None
        for i in range(n_layers):
            _in = in_planes if i == 0 else _out
            _out = out_planes
            if stride == 2:
                self.layers.add_module(f"{i}fc",
                                       torch.nn.Conv2d(_in, _out, kernel_size=1, stride=2))
            else:
                self.layers.add_module(f"{i}fc",
                                   torch.nn.Conv2d(_in, _out, kernel_size=1))
            if i < n_layers - 1:
                # if layer_type > 0:
                #     self.layers.add_module(f"{i}bn",
                #                            torch.nn.BatchNorm1d(_out))
                if layer_type > 1:
                    self.layers.add_module(f"{i}relu",
                                           torch.nn.LeakyReLU(.2))
        self.apply(init_weight)

    def forward(self, x):

        # x = .1 * self.layers(x) + x
        x = self.layers(x)
        return x

class Discriminator(torch.nn.Module):
    def __init__(self, in_planes, n_layers=1, hidden=None):
        super(Discriminator, self).__init__()

        _hidden = in_planes if hidden is None else hidden
        self.body = torch.nn.Sequential()
        for i in range(n_layers-1):
            _in = in_planes if i == 0 else _hidden
            _hidden = int(_hidden // 1.5) if hidden is None else hidden
            self.body.add_module('block%d'%(i+1),
                                 torch.nn.Sequential(
                                     torch.nn.Conv2d(_in, _hidden, kernel_size=1),
                                     torch.nn.BatchNorm2d(_hidden),
                                     torch.nn.LeakyReLU(0.2)
                                 ))
        self.tail = torch.nn.Conv2d(_hidden, 1, kernel_size=1)
        self.apply(init_weight)

    def forward(self,x):
        x = self.body(x)
        x = self.tail(x)
        return x

class SimpleNet(nn.Module):
    def __init__(self, layer, embed_dimension, std1=0.015, std2=0.05, dataset='cifar10', device='cpu'):
        super(SimpleNet, self).__init__()
        self.device = device
        self.layer = layer[0]
        self.std1 = std1
        self.std2 = std2
        self.projector = []
        in_planes = get_shapes(self.layer, dataset)[0]
        self.projector = Projection(in_planes, embed_dimension, 1, 0).to(self.device)
        self.discriminator = Discriminator(embed_dimension, n_layers=2, hidden=4)

    def forward(self, features, train=False):
        # Getting features
        features = features[self.layer]

        # Projection (adapter)
        true_feats = self.projector(features)
        crossentropy = nn.BCELoss().to(self.device)

        if train:
            # Noise addition
            B = true_feats.shape[0]
            D = true_feats.shape[1]
            H = W = true_feats.shape[2]
            mean = torch.zeros(D).to(self.device)
            I = torch.eye(D).to(self.device)
            if self.std1 != 0:
                gaussian1 = torch.distributions.MultivariateNormal(mean, (self.std1 ** 2) * I)
            gaussian2 = torch.distributions.MultivariateNormal(mean, (self.std2 ** 2) * I)
            if self.std1 != 0:
                N1 = gaussian1.sample((B * W * H,)).reshape(B, D, H * W).transpose(2, 1).reshape(B, D, W, H)
            N2 = gaussian2.sample((B * W * H,)).reshape(B, D, H * W).transpose(2, 1).reshape(B, D, W, H)
            noise_idxs = torch.randint(0, 1, torch.Size([true_feats.shape[0]]))
            noise_one_hot = torch.nn.functional.one_hot(noise_idxs, num_classes=1).to(self.device)
            N2 = (N2 * noise_one_hot.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).sum(1)
            if self.std1 != 0:
                N1 = (N1 * noise_one_hot.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)).sum(1)
                N = torch.cat([N1, N2], dim=0)
                z = 0.5 * ((1 / self.std2) ** 2 - (1 / (self.std1 + 1e-6)) ** 2) * (N * N).sum(dim=1) - D * np.log(
                    self.std1 / self.std2)
                Y = 1 / (1 + torch.exp(-z))
                Y = torch.where(Y < 1e-6, 0, Y)
                X1 = true_feats + N1.to(self.device)
            else:
                X1 = true_feats
                Y1 = torch.zeros(X1.shape[0]).to(self.device)
                Y2 = torch.ones(N2.shape[0]).to(self.device)
                Y = torch.cat([Y1, Y2]).unsqueeze(1).float()
            X2 = true_feats + N2.to(self.device)
            X = torch.cat([X1, X2], dim=0)
            scores = self.discriminator(X)
            scores = F.sigmoid(scores.squeeze(1))
            loss = crossentropy(scores, Y.to(self.device))

            return loss

        else:
            scores = -self.discriminator(true_feats)
            scores = F.sigmoid(scores.squeeze(1))
            Y = torch.ones_like(scores).to(self.device)
            loss = crossentropy(scores, Y)

            return loss