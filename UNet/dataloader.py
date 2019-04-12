import os
from os.path import isdir, exists, abspath, join
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from torchvision import transforms

class DataLoader():
    def __init__(self, root_dir='inpainting_set', batch_size=16, augmentation=True):
        self.batch_size = batch_size
        self.augmentation = augmentation
        self.root_dir = abspath(root_dir)
        self.train_files = join(self.root_dir, 'train.png')
        self.test_files = join(self.root_dir, 'test.png')

    def __iter__(self):
        if self.mode == 'train':
            self.data_files = self.train_files
        elif self.mode == 'test':
            self.data_files = self.test_files
        current, counter = 0, 0
        data_transform = transforms.Compose([
            transforms.RandomRotation(45),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(),
            transforms.RandomResizedCrop(128, scale=(0.022, 0.032)),
        ])
        data_transform_test = transforms.Compose([
            transforms.RandomResizedCrop(128, scale=(0.044, 0.054)),
        ])
        while current < self.batch_size:
            current += 1
            data_image = Image.open(self.data_files)
            if self.mode == 'train':
                image_ = data_transform(data_image)
            elif self.mode == 'test':
                image_ = data_transform_test(data_image)
            image_ = np.array(image_)
            image_ = image_ / image_.max()
            data_image = self.applymask(image_)
            label_image = image_
            imgnp = data_image
            labelnp = label_image
            data_image = data_image.transpose((2, 0, 1))
            label_image = label_image.transpose((2, 0, 1))
            yield (imgnp, labelnp, data_image, label_image)

    def setMode(self, mode):
        self.mode = mode

    def generateMask(self, size=128, num=5):
        mask = np.ones((size, size))
        for i in range(num):
            xlength, ylength = np.random.choice([8, 64], 2, replace=False)
            x_coor = np.random.choice(size - xlength)
            y_coor = np.random.choice(size - 64)
            mask[x_coor:x_coor + xlength, y_coor:y_coor + ylength] = 0
        return mask

    def applymask(self, img):
        mask = self.generateMask(128, 5)
        height, width, channels = img.shape
        newimg = np.zeros((height, width, channels + 1))
        for i in range(channels):
            newimg[:, :, i] = np.multiply(img[:, :, i], mask)
        newimg[:, :, 3] = mask
        return newimg