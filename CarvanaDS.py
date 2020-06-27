import os
import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import logging
import sys

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split
from torch.autograd import Function

from torchvision import transforms, datasets
import torchvision.transforms.functional as TF

from albumentations import (Compose, RandomCrop, Resize, HorizontalFlip, ShiftScaleRotate, RandomResizedCrop, RandomBrightnessContrast, ElasticTransform, IAAAffine, IAAPerspective, OneOf)







class CarvanaData(Dataset):
    def __init__(self, root_dir= r"C:\Users\Indy-Windows\Desktop\carvana\carvana\data\train", transform=None, image_size=(512, 512)):

        #Initialize Directory Tree from current working directory if no directory is provided
        self.root_dir = root_dir
        self.img_dir = os.path.join(self.root_dir, 'images')
        self.mask_dir = os.path.join(self.root_dir, 'masks')

        self.img_transform = transform
        self.num_img = len(os.listdir(self.img_dir))
        self.num_mask = len(os.listdir(self.mask_dir))

        self.img_list = os.listdir(self.img_dir)
        self.mask_list = os.listdir(self.mask_dir)

        self.image_height = image_size[1]
        self.image_width = image_size[0]

        self.transform = transform

        self.album_transform = Compose([
            HorizontalFlip(),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            OneOf([
                ElasticTransform(p=.2),
                IAAPerspective(p=.35),
            ], p=.35)
        ])

        if self.transform == None:
            self.transform = self.album_transform



    def segment_transform(self, image, mask):

        seed_top = np.random.randint(0, 568)
        seed_left = np.random.randint(0, 1408)

        image = TF.resized_crop(image, seed_top, seed_left, 512, 512, (self.image_height, self.image_width))

        mask = TF.resized_crop(mask, seed_top, seed_left, 512, 512, (self.image_height, self.image_width))
        #cropped = transforms.RandomCrop(size=(self.image_height, self.image_width))

        #
        #
        # # # Random crop
        # i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(int(1920*self.scale), int(1080*self.scale)))
        # image = TF.crop(image, i, j, h, w)
        # mask = TF.crop(mask, i, j, h, w)
        #
        # # Random horizontal flipping
        # if random.random() > 0.5:
        #     image = TF.hflip(image)
        #     mask = TF.hflip(mask)

        # if random.random() > 0.5:
        #     angle = random.randint(-30, 30)
        #     image = TF.rotate(image, angle)
        #     mask = TF.rotate(mask, angle)

        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        TF.normalize(image, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        return image, mask


    def __len__(self):

        if self.num_img == self.num_mask:
            return self.num_img
        else:
            raise Exception("Number of Images & GT Masks is NOT equal")

    def __getitem__(self, item):

        img_path = os.path.join(self.img_dir, self.img_list[item])
        mask_path = os.path.join(self.mask_dir, self.mask_list[item])

        # img = Image.open(img_path)
        # mask = Image.open(mask_path).convert('L')

        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)

        if self.transform:
            # augment = self.transform(image=image, mask=mask)
            # image, mask = augment['image'], augment['mask']
            #
            image, mask = Image.fromarray(image), Image.fromarray(mask)
            image, mask = self.segment_transform(image, mask)

        else:
            image, mask = Image.fromarray(image), Image.fromarray(mask)
            image, mask = self.segment_transform(image, mask)

        return image, mask