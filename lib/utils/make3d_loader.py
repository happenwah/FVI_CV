import matplotlib.pyplot as plt
import h5py

import cv2
from PIL import Image
import os

import torch
from torch.utils import data
import glob
import numpy as np
import time
import torchvision.transforms.functional as TF
from torchvision import transforms
import random
from PIL import Image

import gzip
import os

class Make3dDataset(data.Dataset):
    def __init__(self, train, dir=None):
            'Initialization'
            self._img_coords_cache = None
            self.train = train

            tmp = np.load(dir)
            self.depth = tmp['depth'].astype(np.float32)/100.0
            self.rgb = tmp['img'].astype(np.float32)

    def __len__(self):
            'Denotes the total number of samples'
            return self.depth.shape[0]

    def transform(self, X, y):
        if self.train:
            X = Image.fromarray(np.uint8(X))
            #Random horizontal flipping
            if random.random() > 0.5:
                X = TF.hflip(X)
                y = y[:,::-1]

            #ColorJitter transforms
            colorjitter_transform = transforms.ColorJitter(brightness=0.1, saturation=0.1,
                                                           contrast=0.1,hue=0.1)
            X = colorjitter_transform(X)


        X = np.array(X).astype(np.float32)
        y = np.array(y).astype(np.float32)

        return X, y

    def __getitem__(self, index):
            'Generates one sample of data'
            # Select sample
            data_depth = self.depth[index,:,:]
            data_rgb = self.rgb[index,:,:,:]

            data_rgb, data_depth = self.transform(data_rgb, data_depth)
            data_rgb = cv2.resize(data_rgb,(224,168),interpolation=cv2.INTER_LINEAR)
            data_depth = cv2.resize(data_depth,(224,168),interpolation=cv2.INTER_NEAREST)
            X = data_rgb.astype(np.float32).transpose(2, 0, 1) / 255.
            y = data_depth.astype(np.float32) / 70.
            
            
            return X, y

if __name__ == '__main__':
    dir = '/rdsgpfs/general/user/etc15/home/datasets/make3d/make3d_train.npz'
    dataset = Make3dDataset(train=False, dir=dir)

    N_test = dataset.__len__()
    for k, (X, y) in enumerate(list(dataset)):
        print(X.shape)
        print(y.shape)
        print(X.flatten().min(), X.flatten().max(), y.flatten().min(), y.flatten().max())
