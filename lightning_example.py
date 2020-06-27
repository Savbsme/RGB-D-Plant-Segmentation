"""
Example template for defining a system.
"""
import os
from argparse import ArgumentParser

from unet_new import UNet
from CarvanaDS import CarvanaData
from ValDS import ValidationData
from LossMetrics import CustomLossMetrics


import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch import optim
import torchvision
from torch.utils.tensorboard import SummaryWriter


from torch.utils.data import DataLoader, random_split

from pytorch_lightning import _logger as log
from pytorch_lightning.core import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler
import pytorch_lightning as pl
from pytorch_lightning.logging import TensorBoardLogger
from matplotlib import pyplot as plt
import numpy as np

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks.model_checkpoint import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from pytorch_lightning.

class LightningTemplateModel(LightningModule):
    # """
    # Sample model to show how to define a template.
    # Example:
    #     >>> # define simple Net for MNIST dataset
    #     >>> params = dict(
    #     ...     drop_prob=0.2,
    #     ...     batch_size=2,
    #     ...     in_features=28 * 28,
    #     ...     learning_rate=0.001 * 8,
    #     ...     optimizer_name='adam',
    #     ...     data_root='./datasets',
    #     ...     out_features=10,
    #     ...     hidden_dim=1000,
    #     ... )
    #     >>> model = LightningTemplateModel(**params)
    # """
    def __init__(self,
                 drop_prob: float = 0.35,
                 batch_size: int = 5,
                 learning_rate: float = 0.0003,
                 ):
                 # drop_prob: float = 0.2,
                 # batch_size: int = 2,
                 # in_features: int = 28 * 28,
                 # learning_rate: float = 0.001 * 8,
                 # optimizer_name: str = 'adam',
                 # data_root: str = './datasets',
                 # out_features: int = 10,
                 # hidden_dim: int = 1000,
                 # **kwargs
                 # ):
        # init superclass
        super().__init__()
        self.model = UNet()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.drop_prob = drop_prob
        self.ds = None
        self.vs = None

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

        criterion = CustomLossMetrics.BceDiceLoss()
        dice_score = CustomLossMetrics.Dice()
        dice = dice_score.forward(y_hat, y)

        loss = criterion.forward(y_hat, y)


        grid_image = torchvision.utils.make_grid(x)
        grid_pred = torchvision.utils.make_grid(y_hat)
        grid_mask = torchvision.utils.make_grid(y)


        tb_logger.experiment.add_image("Input_Image", grid_image, 0)
        tb_logger.experiment.add_image("GT_Mask", grid_mask, 0)
        tb_logger.experiment.add_image("Pred_Mask", grid_pred, 0)

        tensorboard_logs = {'train_loss': loss, 'lr': self.learning_rate, 'train_dice': dice}

        output = {
            "loss": loss,
            "training_loss": loss,
            "progress_bar": tensorboard_logs,
            "log": tensorboard_logs
        }
        return output


    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        x, y = batch
        y_hat = self(x)

        criterion = CustomLossMetrics.BceDiceLoss()
        DL = CustomLossMetrics.DiceLoss()

        val_loss = criterion.forward(y_hat, y)
        dice_loss = DL.forward(y_hat, y)

        val_tensorboard_logs = {'val_loss': val_loss, "val_dice_loss": dice_loss}

        output = {
            "val_loss": val_loss,
            "progress_bar": val_tensorboard_logs,
            "log": val_tensorboard_logs
        }
        return output

    # def test_step(self, batch, batch_idx):
    #     x, y = batch
    #     y_hat = self(x)
    #     test_loss = F.cross_entropy(y_hat, y)
    #     labels_hat = torch.argmax(y_hat, dim=1)
    #     n_correct_pred = torch.sum(y == labels_hat).item()
    #     return {'test_loss': test_loss, "n_correct_pred": n_correct_pred, "n_pred": len(x)}
    #
    # def validation_epoch_end(self, outputs):
    #     """
    #     Called at the end of validation to aggregate outputs.
    #     :param outputs: list of individual outputs of each validation step.
    #     """
    #     avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #     avg_dice = torch.stack([x['dice_loss'] for x in outputs]).mean()
    #
    #     tensorboard_logs = {'val_loss': avg_loss, '    val_dice_score': avg_dice}
    #     print(f" \n Validation INFO: Val Loss {avg_loss.item()} Val DiceLoss {avg_dice.item()}  Val BCE {avg_loss.item()-avg_dice.item()}")
    #     return {'val_loss': avg_loss, "dice_score": avg_dice, 'log': tensorboard_logs}

    def configure_optimizers(self):
        """
        Return whatever optimizers and learning rate schedulers you want here.
        At least one optimizer is required.
        """
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, threshold_mode='rel', verbose=True)
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1 )

        return [optimizer], [{'scheduler': scheduler, 'reduce_on_plateau': True, 'monitor': 'training_loss'}]

    def prepare_data(self):
        self.ds = CarvanaData()
        self.vs = ValidationData()

    def train_dataloader(self):
        log.info('Training data loader called.')
        return DataLoader(self.ds, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        log.info('Validation data loader called.')
        return DataLoader(self.vs, batch_size=self.batch_size, shuffle=False, num_workers=4)

    # @staticmethod
    # def add_model_specific_args(parent_parser, root_dir):  # pragma: no-cover
    #     """
    #     Define parameters that only apply to this model
    #     """
    #     parser = ArgumentParser(parents=[parent_parser])
    #
    #     # param overwrites
    #     # parser.set_defaults(gradient_clip_val=5.0)
    #
    #     # network params
    #     parser.add_argument('--in_features', default=28 * 28, type=int)
    #     parser.add_argument('--out_features', default=10, type=int)
    #     # use 500 for CPU, 50000 for GPU to see speed difference
    #     parser.add_argument('--hidden_dim', default=50000, type=int)
    #     parser.add_argument('--drop_prob', default=0.2, type=float)
    #     parser.add_argument('--learning_rate', default=0.001, type=float)
    #
    #     # data
    #     parser.add_argument('--data_root', default=os.path.join(root_dir, 'mnist'), type=str)
    #
    #     # training params (opt)
    #     parser.add_argument('--epochs', default=20, type=int)
    #     parser.add_argument('--optimizer_name', default='adam', type=str)
    #     parser.add_argument('--batch_size', default=64, type=int)
    #     return parser


if __name__ == "__main__":
    # early_stop_callback = EarlyStopping(
    #     monitor='val_loss',
    #     min_delta=0.00,
    #     patience=50,
    #     verbose=False,
    #     mode='max'
    # )


    tb_logger = TensorBoardLogger('tb_logs', name='my_model')

    cp_cb = ModelCheckpoint(filepath='checkpoints/{epoch}-{val_loss:.3f}', save_top_k=-1)
    model = LightningTemplateModel()
    learner = Trainer(fast_dev_run=False, logger=[tb_logger],  accumulate_grad_batches=2, check_val_every_n_epoch=1,
                      min_epochs=80, max_epochs=200, gpus=1, default_save_path=os.path.join(os.getcwd(), 'checkpoints'),
                      checkpoint_callback=cp_cb)

    # # # Run learning rate finder
    # lr_finder = learner.lr_find(model)
    # new_lr = lr_finder.suggestion()
    # model.learning_rate = new_lr
    # print(new_lr)

    model.learning_rate = 0.0003
    learner.fit(model)