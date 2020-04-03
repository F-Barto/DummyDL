"""
This file runs the main training/val loop, etc... using Lightning Trainer
"""

from argparse import ArgumentParser
from torchvision.datasets import MNIST
from torch.utils.data import random_split
import torchvision.transforms as transforms


from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from models.mnist import SimpleClassifier
from config_utils import YParams


def prepare_data(data_basedir):
    # download
    mnist_train = MNIST(data_basedir, train=True, download=True,
                        transform=transforms.ToTensor())
    mnist_test = MNIST(data_basedir, train=False, download=True,
                       transform=transforms.ToTensor())

    # train/val split
    mnist_train, mnist_val = random_split(mnist_train, [55000, 5000])

    return mnist_train, mnist_val, mnist_test

def main(gpus, nodes, project_config, hparams):
    train_dataset, val_dataset, test_dataset = prepare_data(project_config.input_dir)

    # init module
    model = SimpleClassifier(hparams, train_dataset, val_dataset, test_dataset)

    # default used by the Trainer
    checkpoint_callback = ModelCheckpoint(
        filepath=project_config.output_dir,
        save_top_k=True,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=''
    )

    wandb_logger = WandbLogger(
        name='simpleMNIST',
        save_dir=project_config.output_dir,
        #log_model=True
    )

    # most basic trainer, uses good defaults
    trainer = Trainer(
        max_epochs =hparams.max_nb_epochs,
        gpus=gpus,
        nb_gpu_nodes=nodes,
        checkpoint_callback=checkpoint_callback,
        logger=wandb_logger,
        check_val_every_n_epoch=project_config.check_val_every_n_epoch
    )
    trainer.fit(model)
    trainer.test(model)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--model_config_file', type=str)
    parser.add_argument('--model_config_profile', type=str)
    parser.add_argument('--project_config_file', type=str)
    parser.add_argument('--project_config_profile', type=str)

    # parse params
    args = parser.parse_args()

    hparams = YParams(args.model_config_file, args.model_config_profile)
    project_config = YParams(args.project_config_file, args.project_config_profile)

    main(args.gpus, args.nodes, project_config, hparams)