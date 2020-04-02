import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from argparse import ArgumentParser

import pytorch_lightning as pl



class SimpleClassifier(pl.LightningModule):
    def __init__(self, hparams):
        super(CoolSystem, self).__init__()

        self.hparams = hparams

        # mnist images are (1, 28, 28) (channels, width, height)
        # just a very very simple model
        self.l1 = torch.nn.Linear(28 * 28, 128)
        self.l2 = torch.nn.Linear(128, 256)
        self.l3 = torch.nn.Linear(256, 10)

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

    def prepare_data(self):
        # download
        mnist_train = MNIST(os.getcwd(), train=True, download=True,
                            transform=transforms.ToTensor())
        mnist_test = MNIST(os.getcwd(), train=False, download=True,
                           transform=transforms.ToTensor())

        # train/val split
        mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])

        # assign to use in dataloaders
        self.train_dataset = mnist_train
        self.val_dataset = mnist_val
        self.test_dataset = mnist_test

    def train_dataloader(self):
        # REQUIRED
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        # REQUIRED
        return DataLoader(self.mnist_val, batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        # REQUIRED
        return DataLoader(self.mnist_test, batch_size=self.hparams.batch_size)


    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser])
        parser.add_argument('--learning_rate', default=0.02, type=float)
        parser.add_argument('--batch_size', default=32, type=int)

        # training specific (for this model)
        parser.add_argument('--max_nb_epochs', default=2, type=int)

        return parser