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


from utils import *
from models import *
from datasets import *

os.environ['CUDA_VISIBLE_DEVICES']='3'
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
    parser.add_argument('--n_classes', type=int, default=10, help='number of classes for dataset')
    parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
    parser.add_argument('--channels', type=int, default=1, help='number of image channels')
    parser.add_argument('--sample_interval', type=int, default=200, help='interval between image sampling')
    args = parser.parse_args()
    print(args)

    C,H,W = args.channels, args.img_size, args.img_size

    # Loss function
    adversarial_loss = torch.nn.MSELoss()
    # Initialize Generator and discriminator
    generator = Generator(H,W)
    discriminator = Discriminator(H,W)

    if torch.cuda.is_available():
        # generator.cuda()
        # discriminator.cuda()
        # adversarial_loss.cuda()
        generator = generator.to(device)
        discriminator = discriminator.to(device)
        adversarial_loss = adversarial_loss.to(device)

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Define your transformations
    transform = Compose([
        Resize((256, 256)),
        ToTensor()
    ])

    # prepare dataset
    root = "./data/cbct2ct02/"
    full_dataset = PaireDataset(root, transform)

    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.15 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    batch_size = 16
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    #Start training
    batches_done=0
    for epoch in range(args.n_epochs):
        for i, (imgs, labels) in enumerate(train_dataloader):

            Batch_Size = args.batch_size
            N_Class = args.n_classes
            # Adversarial ground truths
            valid = Variable(torch.ones(Batch_Size).cuda(), requires_grad=False)
            fake = Variable(torch.zeros(Batch_Size).cuda(), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(torch.FloatTensor).cuda())

            real_y = torch.zeros(Batch_Size, N_Class)
            real_y = Variable(real_y.scatter_(1, labels.view(Batch_Size, 1), 1).cuda())
            #y = Variable(y.cuda())

            # Sample noise and labels as generator input
            noise = Variable(torch.randn((Batch_Size, args.latent_dim)).cuda())
            gen_labels = (torch.rand(Batch_Size, 1) * N_Class).type(torch.LongTensor)
            gen_y = torch.zeros(Batch_Size, N_Class)
            gen_y = Variable(gen_y.scatter_(1, gen_labels.view(Batch_Size, 1), 1).cuda())

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            # Loss for real images
            d_real_loss = adversarial_loss(discriminator(real_imgs, real_y).squeeze(), valid)
            # Loss for fake images
            gen_imgs = generator(noise, gen_y)
            d_fake_loss = adversarial_loss(discriminator(gen_imgs.detach(),gen_y).squeeze(), fake)
            # Total discriminator loss
            d_loss = (d_real_loss + d_fake_loss)

            d_loss.backward()
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            #gen_imgs = generator(noise, gen_y)
            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs,gen_y).squeeze(), valid)

            g_loss.backward()
            optimizer_G.step()


            print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, args.n_epochs, i, len(train_dataloader),
                                                                d_loss.data.cpu(), g_loss.data.cpu()))

            batches_done = epoch * len(train_dataloader) + i
            if batches_done % args.sample_interval == 0:
                noise = Variable(torch.FloatTensor(np.random.normal(0, 1, (N_Class**2, args.latent_dim))).cuda())
                #fixed labels
                y_ = torch.LongTensor(np.array([num for num in range(N_Class)])).view(N_Class,1).expand(-1,N_Class).contiguous()
                y_fixed = torch.zeros(N_Class**2, N_Class)
                y_fixed = Variable(y_fixed.scatter_(1,y_.view(N_Class**2,1),1).cuda())

                gen_imgs = generator(noise, y_fixed).view(-1,C,H,W)

                save_image(gen_imgs.data, img_save_path + '/%d-%d.png' % (epoch,batches_done), nrow=N_Class, normalize=True)