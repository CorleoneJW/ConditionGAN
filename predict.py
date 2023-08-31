import SimpleITK
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import argparse
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from torch import nn
from Src import models
import os
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.cuda.amp import GradScaler
from utils.fit01 import Fit
from utils import utils
import data_loader

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
device = 'cuda'


def psnr(target, ref):
    target_data = np.array(target, dtype=np.float64)
    ref_data = np.array(ref, dtype=np.float64)
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    eps = np.finfo(np.float64).eps
    if rmse == 0:
        rmse = eps
    return 20 * math.log10(255.0 / rmse)


def SSIM(imageA, imageB):
    imageA = np.array(imageA, dtype=np.uint8)
    imageB = np.array(imageB, dtype=np.uint8)

    (grayScore, diff) = ssim(imageA, imageB, full=True)

    return grayScore


def onezero(x):
    x -= np.min(x)
    x = x / np.max(x)
    return x


if __name__ == "__main__":
    args = utils.get_parse()
    args.training = False
    args.device = [device]
    resize = False
    show = True
    all_test = False
    args.sampling_timesteps = 60

    model_data = torch.load('./weights/weights0726.pth',
                            map_location=device)

    model = models.model_T(image_size=args.image_size)

    try:
        model.load_state_dict(model_data['model_dict'])
    except:
        model = nn.DataParallel(model)
        model.load_state_dict(model_data['model_dict'])
    model = model.to(device)

    if device == 'cuda':
        model = nn.DataParallel(model)
    # print(model)
    gen = Fit(
        model,
        args,
        None,
        None,
        None,
    )

    *_, val_data = data_loader.get_data_path()
    dataloaders_val = DataLoader(data_loader.Dataset(val_data, (args.image_size, args.image_size),
                                                     return_roi=True, transform=False),
                                 batch_size=1,
                                 shuffle=True,
                                 num_workers=1)
    print(len(dataloaders_val.dataset))
    mse, mae, s, p = [], [], [], []
    num = 0
    for iteration, batch in enumerate(dataloaders_val):
        label, imgs, mask, roi = batch
        img = imgs.cpu().numpy()
        roi = roi.cpu().numpy()
        label = label.cpu().numpy()
        with torch.no_grad():
            cond_img = imgs.to(device)
            pre, pres_all = gen.sample(cond_img)
            pres_all = torch.stack(pres_all, dim=0)[:, 0, 0].cpu().numpy()
            print(pres_all.shape)
            if not os.path.exists('./sample'):
                os.mkdir('./sample')
            SimpleITK.WriteImage(SimpleITK.GetImageFromArray(pres_all), './sample/denoise03.nii')
            if show:
                # print(f"CBCT image Min: {cond_img.min()}, Max: {cond_img.max()}")
                # print(f"CT image Min: {label.min()}, Max: {label.max()}")
                plt.figure(figsize=(12, 12))
                plt.subplot(221)
                plt.title('cbct')
                plt.imshow(cond_img[0, 0].cpu().numpy(), 'gray')

                plt.subplot(222)
                plt.title('ct')
                plt.imshow(label[0, 0], 'gray')

                plt.subplot(223)
                plt.title('pre_ct')
                plt.imshow(pre[0, 0].cpu().numpy(), 'gray')

                plt.show()
                if not os.path.exists('./sample/figures'):
                    os.makedirs('./sample/figures')
                plt.savefig(f'./sample/figures/figure_{iteration}.png')
            mse.append(mean_squared_error(label[0, 0], pre[0, 0].cpu().numpy()))
            mae.append(mean_absolute_error(label[0, 0], pre[0, 0].cpu().numpy()))
            s.append(SSIM(label[0, 0], pre[0, 0].cpu().numpy()))
            p.append(psnr(label[0, 0], pre[0, 0].cpu().numpy()))
        if all_test:
            continue
        else:
            break

    print('mean mse is : ', np.mean(mse))
    print('mean mae is : ', np.mean(mae))
    print('mean ssim is : ', np.mean(s))
    print('mean psnr is : ', np.mean(p))