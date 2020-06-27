import segmentation_models_pytorch as smp
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
import random
from tqdm import tqdm
# from unet import UNet

from pytorch_lightning import _logger as log

from unet_new import UNet
from albumentations import (Compose, Resize, HorizontalFlip, ShiftScaleRotate, RandomResizedCrop, RandomBrightnessContrast, ElasticTransform, IAAAffine, IAAPerspective, OneOf)
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule
from CarvanaDS import CarvanaData


class PL_Class(LightningModule):

    def __init__(self,

                 drop_prob: float = 0.2,
                 batch_size: int = 5,
                 learning_rate: float = 0.0003,
                 ):

        super().__init__()
        self.model = UNet()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.drop_prob = drop_prob


    def forward(self, x):
        return self.model.forward(x)

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        x, y = batch
        y_hat = self(x)
        loss = F.binary_cross_entropy_with_logits(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}


    def configure_optimizers(self):
        """
        Return whatever optimizers and learning rate schedulers you want here.
        At least one optimizer is required.
        """
        optimizer = optim.Adam(self.parameters(), lr=(self.learning_rate))
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

    def prepare_data(self):

        self.ds = CarvanaData()


    def train_dataloader(self):
        log.info('Training data loader called.')
        return DataLoader(self.ds, batch_size=self.batch_size, num_workers=4)



if __name__ == "__main__":

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = PL_Class().load_from_checkpoint(checkpoint_path=r"checkpoints/name=0-epoch=10-val_loss=0.00.ckpt")
    # model.load_state_dict(torch.load(path_to_model_file))
    # model = mod.load_from_checkpoint(check_point_path="")

    model.to(device)

    image_h = 512
    image_w =512

    image_h = image_h
    image_w = image_w




    data_transform = transforms.Compose([
        transforms.Resize(size=(image_h, image_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])



    path_to_data_folder = r"C:\Users\Indy-Windows\Desktop\test_fold"
    path_to_output_folder = r"C:\Users\Indy-Windows\Desktop\test_fold\output"


    model.eval()

    test_ds = datasets.ImageFolder(path_to_data_folder, transform=data_transform)
    test_dataLoader = DataLoader(test_ds, batch_size=1, num_workers=1)

    for test_img, _ in test_dataLoader:
        test_img = test_img.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            mask_pred = model(test_img)

            prediction = torch.sigmoid(mask_pred)
            print(f"max: {prediction.max()}, min {prediction.min()}, mean:{prediction.mean()}")

            # val = ((prediction.max()-prediction.min())/2.0
            # prediction = (prediction > .5 ).float()

            # prediction = mask_pred[0, 0, :, :].to('cpu')
            prediction = prediction[0, 0, :, :].to('cpu')
            real = test_img[0,0,:,:].to('cpu')

            # print(prediction)

            plt.figure()
            plt.subplot(2,1, 1)
            plt.imshow(prediction)
            plt.subplot(2,1,2)
            plt.imshow(real)
            plt.show()





