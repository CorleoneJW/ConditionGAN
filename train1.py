import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.transforms import Compose
from skimage.transform import resize

from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from tqdm import tqdm

from utils import *
from models import *
from datasets import *

os.environ['CUDA_VISIBLE_DEVICES']='2'
device = torch.device('cuda')

if __name__ == "__main__":


    img_save_path = 'images'
    os.makedirs(img_save_path, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
    parser.add_argument('--batch_size', type=int, default=32 , help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
    parser.add_argument('--beta1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--beta2', type=float, default=0.999, help='adam: decay of second order momentum of gradient')
    parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
    parser.add_argument('--n_classes', type=int, default=1, help='number of classes for dataset')
    parser.add_argument('--img_size', type=int, default=256, help='size of each image dimension')
    parser.add_argument('--channels', type=int, default=1, help='number of image channels')
    parser.add_argument('--sample_interval', type=int, default=200, help='interval between image sampling')
    args = parser.parse_args()
    print(args)

    C,H,W = args.channels, args.img_size, args.img_size
    img_size = args.img_size
    latent_dim = args.latent_dim

    # Loss function
    criterion = torch.nn.MSELoss()
    # Initialize Generator and discriminator
    generator = Generator(latent_dim,C,img_size)
    discriminator = Discriminator(C,img_size)

    if torch.cuda.is_available():
        # generator.cuda()
        # discriminator.cuda()
        # criterion.cuda()
        generator = generator.to(device)
        discriminator = discriminator.to(device)
        criterion = criterion.to(device)

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Define your transformations
    transform = Compose([
        Resize((256, 256)),
        ToTensor()
        # transforms.Normalize(mean=[0.5], std=[0.5])  # 标准化到[-1, 1]
    ])

    # prepare dataset
    root = "./data/cbct2ct02/"
    full_dataset = PaireDataset(root, transform)

    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    batch_size = args.batch_size
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Optimizers
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    #Start training
    batches_done=0
    num_epochs = args.n_epochs
    for epoch in range(num_epochs):
        disc_loss_list = []
        gene_loss_list = []
        for i, (imgs, labels) in enumerate(tqdm(train_dataloader)):
            real_images = imgs
            real_images = real_images.to(device)
            labels = labels.to(device)

            disc_optimizer.zero_grad()

            # 真实图像的标签为1
            valid = torch.ones(real_images.size(0), 1).to(device)

            # 生成器生成图像
            z = torch.randn(real_images.size(0), latent_dim).to(device)
            gen_images = generator(z)

            # 生成图像的标签为0
            fake = torch.zeros(real_images.size(0), 1).to(device)

            # 判别器对真实图像的损失
            real_loss = criterion(discriminator(real_images), valid)
            # 判别器对生成图像的损失
            fake_loss = criterion(discriminator(gen_images.detach()), fake)
            # 判别器总损失
            disc_loss = 0.5 * (real_loss + fake_loss)

            disc_loss.backward()
            disc_optimizer.step()

             # 训练生成器
            gen_optimizer.zero_grad()

            # 生成器生成图像并将其输入判别器
            gen_images = generator(z)
            gen_loss = criterion(discriminator(gen_images), valid)

            gen_loss.backward()
            gen_optimizer.step()

            disc_loss_list.append(disc_loss.cpu().detach().numpy())
            gene_loss_list.append(gen_loss.cpu().detach().numpy())

            # 将数据展平为 [batch_size, 1*256*256]
            # real_images = real_images.view(-1, 1 * img_size * img_size)     # torch.Size([batch_size, 65536])

            # 创建条件向量 [batch_size, num_classes]，并将其与输入连接
            # label_one_hot = torch.zeros(batch_size, C)
            # label_one_hot[range(batch_size), labels] = 1.0
            # label_one_hot = label_one_hot.float()
            # input_data = torch.cat([real_images, label_one_hot], dim=1)

            # 训练判别器
            # disc_optimizer.zero_grad()
            # real_labels = torch.ones(batch_size, 1)
            # fake_labels = torch.zeros(batch_size, 1)

            # 使用真实图像计算判别器损失
            # real_outputs = discriminator(input_data)
            # real_loss = criterion(real_outputs, real_labels)
            # real_loss.backward()

            # 使用生成的图像计算判别器损失
            # z = torch.randn(batch_size, latent_dim)
            # fake_images = generator(torch.cat([z, label_one_hot], dim=1))
            # fake_inputs = torch.cat([fake_images, label_one_hot], dim=1)
            # fake_outputs = discriminator(fake_inputs.detach())
            # fake_loss = criterion(fake_outputs, fake_labels)
            # fake_loss.backward()

            # 更新判别器权重
            # disc_optimizer.step()
            # total_disc_loss = real_loss + fake_loss

            # 训练生成器
            # gen_optimizer.zero_grad()
            # fake_outputs = discriminator(torch.cat([fake_images, label_one_hot], dim=1))
            # gen_loss = criterion(fake_outputs, real_labels)
            # gen_loss.backward()

            # 更新生成器权重
            # gen_optimizer.step()

        # 打印训练信息
        print(f"Epoch [{epoch + 1}/{num_epochs}] | d_loss: {np.mean(disc_loss_list):.4f} | g_loss: {np.mean(gene_loss_list):.4f}")

    # 保存生成器模型
    torch.save(generator.state_dict(), 'generator.pth')
    torch.save(discriminator.state_dict(), 'discriminator.pth')