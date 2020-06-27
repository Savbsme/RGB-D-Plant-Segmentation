import os
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLossMetrics(nn.Module):
    def __init__(self):
        super().__init__()

    class Dice:
        def __init__(self):
            super().__init__()

        def DiceCoef(self, pred, target):
            smooth = 1.
            iflat = pred.contiguous().view(-1)
            tflat = target.contiguous().view(-1)
            intersection = (iflat * tflat).sum()
            A_sum = torch.sum(iflat * iflat)
            B_sum = torch.sum(tflat * tflat)
            return (2. * intersection + smooth) / (A_sum + B_sum + smooth)

        def forward(self, pred, target):
            num = target.size(0)  # Number of batches
            dc = self.DiceCoef(pred, target)
            return dc

    class DiceLoss:
        def __init__(self):
            super().__init__()

        def DiceCoef(self, pred, target):
            smooth = 1.
            iflat = pred.contiguous().view(-1)
            tflat = target.contiguous().view(-1)
            intersection = (iflat * tflat).sum()
            A_sum = torch.sum(iflat * iflat)
            B_sum = torch.sum(tflat * tflat)
            return (2. * intersection + smooth) / (A_sum + B_sum + smooth)

        def forward(self, pred, target):
            num = target.size(0)  # Number of batches
            dc = self.DiceCoef(pred, target)
            dc_loss = (1 - (dc / num))
            return dc_loss

    class SoftDiceLoss:
        def __init__(self):
            super().__init__()

        def DiceCoef(self, pred, target):
            smooth = 1.
            iflat = pred.contiguous().view(-1)
            tflat = target.contiguous().view(-1)
            intersection = (iflat * tflat).sum()
            A_sum = torch.sum(iflat * iflat)
            B_sum = torch.sum(tflat * tflat)
            return (2. * intersection + smooth) / (A_sum + B_sum + smooth)

        def forward(self, logits, target):
            probs = torch.sigmoid(logits)
            num = target.size(0)  # Number of batches
            dc = self.DiceCoef(probs, target)
            dc_loss = (1 - (dc / num))
            return dc_loss

    class BceDiceLoss:
        def __init__(self):
            super().__init__()
            self.bce_loss = nn.BCEWithLogitsLoss()
            # self.bce_loss = nn.BCELoss()

            self.dice_loss = CustomLossMetrics.DiceLoss()

        def forward(self, pred, target):

            bce = self.bce_loss(pred, target)
            dice = self.dice_loss.forward(pred, target)
            combined_loss = (2.*bce + 1.*dice)/3.0

            return combined_loss

    class BceDice:
        def __init__(self):
            super().__init__()
            self.bce_loss = nn.BCEWithLogitsLoss()
            # self.bce_loss = nn.BCELoss()

            self.dice = CustomLossMetrics.Dice()

        def forward(self, pred, target):

            bce = self.bce_loss(pred, target)
            dice = self.dice.forward(pred, target)

            combined_loss = bce - dice

            return combined_loss

