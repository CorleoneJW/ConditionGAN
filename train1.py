import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision.transforms import Compose
from skimage.transform import resize
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error, mean_absolute_error

from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torch.autograd import variable
import cv2

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
        print("GPU loaded.")

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
    total_disc_loss = []
    total_gene_loss = []
    total_mse_trian = []
    total_mse_val = []
    image_total_size = img_size*img_size
    for epoch in range(num_epochs):

        """
        Trian part
        """
        disc_loss_list = []
        gene_loss_list = []
        ssim_list = []  # 存储每个batch的SSIM值
        mse_list = []   # 存储每个batch的MSE值
        mae_list = []   # 存储每个batch的MAE值
        psnr_list = []  # 存储每个batch的PSNR值
        generator.train()
        for i, (cbct, ct) in enumerate(tqdm(train_dataloader)):
            cbct = cbct.to(device)
            ct = ct.to(device)

            """
            Train discriminator
            """
            disc_optimizer.zero_grad()
            # 真实图像的标签为1
            valid = torch.ones(ct.size(0), 1).to(device)
            # 生成图像的标签为0
            fake = torch.zeros(ct.size(0), 1).to(device)
            # 生成器生成图像
            z = torch.randn(ct.size(0), latent_dim).to(device)
            gen_images = generator(z, cbct)
            # 判别器对真实图像的损失
            real_loss = criterion(discriminator(ct), valid)
            # 判别器对生成图像的损失
            fake_loss = criterion(discriminator(gen_images.detach()), fake)
            # 判别器总损失
            disc_loss = 0.5 * (real_loss + fake_loss)

            disc_loss.backward()
            disc_optimizer.step()

            """
            Train generator
            """
            gen_optimizer.zero_grad()

            # 生成器生成图像并将其输入判别器
            gen_images = generator(z, cbct)
            gen_loss = criterion(discriminator(gen_images), valid)

            gen_loss.backward()
            gen_optimizer.step()

            disc_loss_list.append(disc_loss.cpu().detach().numpy())
            gene_loss_list.append(gen_loss.cpu().detach().numpy())

            # 计算 SSIM、MSE、MAE 和 PSNR
            for j in range(ct.size(0)):
                ct_image = ct[j].squeeze().cpu().detach().numpy()
                gen_image = gen_images[j].squeeze().cpu().detach().numpy()
                # 计算 SSIM
                ssim_value = ssim(ct_image, gen_image, data_range=ct_image.max() - ct_image.min())
                # 计算 MSE
                mse_value = mean_squared_error(ct_image, gen_image)
                # 计算 MAE
                mae_value = mean_absolute_error(ct_image, gen_image)
                # 计算 PSNR
                psnr_value = cv2.PSNR(ct_image, gen_image)
                # 添加到对应list中
                ssim_list.append(ssim_value)
                mse_list.append(mse_value)
                mae_list.append(mae_value)
                psnr_list.append(psnr_value)

        # 计算每个epoch的平均值
        avg_disc_loss = np.mean(disc_loss_list)
        avg_gene_loss = np.mean(gene_loss_list)
        avg_ssim = np.mean(ssim_list)
        avg_mse = np.mean(mse_list)/(image_total_size)
        avg_mae = np.mean(mae_list)
        avg_psnr = np.mean(psnr_list)

        total_disc_loss.append(avg_disc_loss)
        total_gene_loss.append(avg_gene_loss)
        total_mse_trian.append(avg_mse)

        # 打印训练信息，包括SSIM、MSE、MAE和PSNR
        print(f"Epoch [{epoch + 1}/{num_epochs}] | d_loss: {avg_disc_loss:.4f} | g_loss: {avg_gene_loss:.4f} | SSIM: {avg_ssim:.4f} | MSE: {avg_mse:.4f} | MAE: {avg_mae:.4f} | PSNR: {avg_psnr:.4f}")

        """
        validate part
        """
        ssim_list_val = []  # 存储每个batch的SSIM值
        mse_list_val = []   # 存储每个batch的MSE值
        mae_list_val = []   # 存储每个batch的MAE值
        psnr_list_val = []  # 存储每个batch的PSNR值
        # set eval mode
        generator.eval()
        with torch.no_grad():
            for i, (cbct, ct) in enumerate(tqdm(val_dataloader)):
                cbct = cbct.to(device)
                ct = ct.to(device)

                """
                validate generator
                """
                # 生成器生成图像
                z = torch.randn(ct.size(0), latent_dim).to(device)
                gen_images = generator(z, cbct)

                # 计算 SSIM、MSE、MAE 和 PSNR
                for j in range(ct.size(0)):
                    ct_image = ct[j].squeeze().cpu().detach().numpy()
                    gen_image = gen_images[j].squeeze().cpu().detach().numpy()
                    # 计算 SSIM
                    ssim_value = ssim(ct_image, gen_image, data_range=ct_image.max() - ct_image.min())
                    # 计算 MSE
                    mse_value = mean_squared_error(ct_image, gen_image)
                    # 计算 MAE
                    mae_value = mean_absolute_error(ct_image, gen_image)
                    # 计算 PSNR
                    psnr_value = cv2.PSNR(ct_image, gen_image)
                    # 添加到对应list中
                    ssim_list_val.append(ssim_value)
                    mse_list_val.append(mse_value)/(image_total_size)
                    mae_list_val.append(mae_value)
                    psnr_list_val.append(psnr_value)

            avg_val_ssim = np.mean(ssim_list_val)
            avg_val_mse = np.mean(mse_list_val)
            avg_val_mae = np.mean(mae_list_val)
            avg_val_psnr = np.mean(psnr_list_val)
            # 打印训练和验证信息，包括SSIM、MSE、MAE和PSNR
            print(f"Epoch [{epoch + 1}/{num_epochs}] | Validation SSIM: {avg_val_ssim:.4f} | Validation MSE: {avg_val_mse:.4f} | Validation MAE: {avg_val_mae:.4f} | Validation PSNR: {avg_val_psnr:.4f}")

        # 保存生成器模型
        # if 
        #     torch.save(generator.state_dict(), './models/generator.pth')
        #     torch.save(discriminator.state_dict(), './models/discriminator.pth')