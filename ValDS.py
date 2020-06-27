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






# class ValidationData(Dataset):
#     def __init__(self, root_dir= r"C:\Users\Indy-Windows\Desktop\carvana\carvana\data\val", transform=None, image_size=(512, 512)):
#
#         #Initialize Directory Tree from current working directory if no directory is provided
#         self.root_dir = root_dir
#         self.img_dir = os.path.join(self.root_dir, 'images')
#         self.mask_dir = os.path.join(self.root_dir, 'masks')
#
#         self.img_transform = transform
#         self.num_img = len(os.listdir(self.img_dir))
#         self.num_mask = len(os.listdir(self.mask_dir))
#
#         self.img_list = os.listdir(self.img_dir)
#         self.mask_list = os.listdir(self.mask_dir)
#
#         self.image_height = image_size[1]
#         self.image_width = image_size[0]
#
#         self.transform = transform
#
#
#
#     def segment_transform(self, image, mask):
#         # Resize
#         resize = transforms.Resize(size=(self.image_height, self.image_width))
#
#         image = resize(image)
#         mask = resize(mask)
#
#         image = TF.to_tensor(image)
#         mask = TF.to_tensor(mask)
#
#         TF.normalize(image, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
#         return image, mask
#
#
#     def __len__(self):
#
#         if self.num_img == self.num_mask:
#             return self.num_img
#         else:
#             raise Exception("Number of Images & GT Masks is NOT equal")
#
#     def __getitem__(self, item):
#
#         img_path = os.path.join(self.img_dir, self.img_list[item])
#         mask_path = os.path.join(self.mask_dir, self.mask_list[item])
#
#         image = cv2.imread(img_path)
#         mask = cv2.imread(mask_path, 0)
#
#         image, mask = Image.fromarray(image), Image.fromarray(mask)
#         image, mask = self.segment_transform(image, mask)
#
#         return image, mask


class ValidationData(Dataset):
    def __init__(self, root_dir=r"C:\Users\Indy-Windows\Desktop\carvana\carvana\data\val", transform=None,
                 image_size=(512, 512)):

        # Initialize Directory Tree from current working directory if no directory is provided
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


    def segment_transform(self, image, mask):

        image = TF.resize(image, 512)
        mask = TF.resize(mask, 512)


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

        image, mask = Image.fromarray(image), Image.fromarray(mask)
        image, mask = self.segment_transform(image, mask)

        return image, mask