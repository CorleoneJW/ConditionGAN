import os
import sys
import torch
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Compose
import numpy as np
import collections
from PIL import Image
import csv
import random
import shutil
sys.path.append('../')
from skimage.transform import resize

class Resize(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, array):
        array = resize(array, self.output_size, mode='reflect', anti_aliasing=True)
        return array

class ToTensor(object):
    def __call__(self, array):
        tensor = torch.from_numpy(array.astype(np.float32))
        return tensor.unsqueeze(0)

class PaireDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.files = []

        patients = os.listdir(root)
        for patient in patients:
            root_A = os.path.join(root, patient, "cbct")
            root_B = os.path.join(root, patient, "ct")
            files_A = {os.path.splitext(file)[0]: file for file in os.listdir(root_A)
                       if os.path.isfile(os.path.join(root_A, file)) and file.endswith(".npy")}
            files_B = {os.path.splitext(file)[0]: file for file in os.listdir(root_B)
                       if os.path.isfile(os.path.join(root_B, file)) and file.endswith(".npy")}
            files = {f: (os.path.join(root_A, files_A[f]), os.path.join(root_B, files_B[f])) for f in files_A if f in files_B}
            self.files.extend(list(files.values()))

    def __getitem__(self, index):
        try:
            file_A, file_B = self.files[index]
            img_A = np.load(file_A).astype(np.float32)
            img_B = np.load(file_B).astype(np.float32)

            if self.transform:
                img_A = self.transform(img_A)
                img_B = self.transform(img_B)
        except:
            print("file_A",file_A)
            print("Aimg",img_A)
            # print("B",file_B)

        return img_A, img_B

    def __len__(self):
        return len(self.files)
