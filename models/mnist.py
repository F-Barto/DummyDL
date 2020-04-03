import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from argparse import ArgumentParser

import pytorch_lightning as pl



class SimpleClassifier(pl.LightningModule):
    def __init__(self, train_dataset, val_dataset, test_dataset, hparams):
        super(self).__init__()

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.hparams = hparams

        # models images are (1, 28, 28) (channels, width, height)
        # just a very very simple model
        self.l1 = torch.nn.Linear(28 * 28, hparams.channels[0])
        self.l2 = torch.nn.Linear(hparams.channels[0], hparams.channels[1])
        self.l3 = torch.nn.Linear(hparams.channels[1], hparams.channels[2])

    def forward(self, x):
        x = torch.relu(self.l1(x.view(x.size(0), -1)))
        x = torch.relu(self.l2(x))
        return torch.relu(self.l3(x))

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        y_hat = self(x)
        return {'loss': F.cross_entropy(y_hat, y)}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {'val_loss': F.cross_entropy(y_hat, y)}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss_mean': val_loss_mean}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        return {'test_loss': F.cross_entropy(y_hat, y)}

    def test_epoch_end(self, outputs):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        return {'test_loss_mean': test_loss_mean}


    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    def train_dataloader(self):
        # REQUIRED
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        # REQUIRED
        return DataLoader(self.mnist_val, batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        # REQUIRED
        return DataLoader(self.mnist_test, batch_size=self.hparams.batch_size)