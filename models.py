import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch


class Generator(nn.Module):
    def __init__(self, latent_dim, channels, img_size):
        super(Generator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, channels * img_size * img_size),
            nn.Tanh()
        )
        self.channels = channels
        self.img_size = img_size

    def forward(self, z):
        img = self.fc(z)
        img = img.view(img.size(0), self.channels, self.img_size, self.img_size)
        return img


class Discriminator(nn.Module):
    def __init__(self, channels, img_size):
        super(Discriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels * img_size * img_size, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.channels = channels
        self.img_size = img_size

    def forward(self, img):
        img = img.view(img.size(0), -1)
        validity = self.fc(img)
        return validity

    
# class Generator(nn.Module):
#     # initializers
#     def __init__(self,latent_dim,C,H,W):
#         super(Generator, self).__init__()
#         self.fc1_1 = nn.Linear(latent_dim, 256)
#         self.fc1_1_bn = nn.BatchNorm1d(256)
#         self.fc1_2 = nn.Linear(C, 256)
#         self.fc1_2_bn = nn.BatchNorm1d(256)
#         self.fc2 = nn.Linear(512, 512)
#         self.fc2_bn = nn.BatchNorm1d(512)
#         self.fc3 = nn.Linear(512, 1024)
#         self.fc3_bn = nn.BatchNorm1d(1024)
#         self.fc4 = nn.Linear(1024, H*W)


#     # forward method
#     def forward(self, input, label):
#         x = F.relu(self.fc1_1_bn(self.fc1_1(input)))
#         y = F.relu(self.fc1_2_bn(self.fc1_2(label)))
#         x = torch.cat([x, y], 1)
#         x = F.relu(self.fc2_bn(self.fc2(x)))
#         x = F.relu(self.fc3_bn(self.fc3(x)))
#         x = F.tanh(self.fc4(x))
#         return x

# class Discriminator(nn.Module):
#     # initializers
#     def __init__(self,C,H,W):
#         super(Discriminator, self).__init__()
#         self.fc1_1 = nn.Linear(H*W, 1024)
#         self.fc1_2 = nn.Linear(C, 1024)
#         self.fc2 = nn.Linear(2048, 512)
#         self.fc2_bn = nn.BatchNorm1d(512)
#         self.fc3 = nn.Linear(512, 256)
#         self.fc3_bn = nn.BatchNorm1d(256)
#         self.fc4 = nn.Linear(256, 1)


#     # forward method
#     def forward(self, input, label):
#         x = F.leaky_relu(self.fc1_1(input.view(input.size(0),-1)), 0.2)
#         y = F.leaky_relu(self.fc1_2(label), 0.2)
#         x = torch.cat([x, y], 1)
#         x = F.leaky_relu(self.fc2_bn(self.fc2(x)), 0.2)
#         x = F.leaky_relu(self.fc3_bn(self.fc3(x)), 0.2)
#         x = F.sigmoid(self.fc4(x))
#         return x