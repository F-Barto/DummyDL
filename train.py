"""
This file runs the main training/val loop, etc... using Lightning Trainer
"""

from argparse import ArgumentParser
from pathlib import Path
from torchvision.datasets import MNIST
from torch.utils.data import random_split
import torchvision.transforms as transforms
import torch
import numpy as np

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

def main(gpus, nodes, fast_dev_run, project_config, hparams):
    torch.manual_seed(0)
    np.random.seed(0)

    train_dataset, val_dataset, test_dataset = prepare_data(project_config.input_dir)

    # init module
    model = SimpleClassifier(hparams, train_dataset, val_dataset, test_dataset)

    base_output_dir = Path(project_config.output_dir)
    experiment_output_dir = base_output_dir / 'SimpleClassifier-Mnist'
    experiment_output_dir.mkdir(parents=True, exist_ok=True)

    wandb_output_dir = str(experiment_output_dir)
    wandb_logger = WandbLogger(
        save_dir=wandb_output_dir,
        #log_model=True
    )

    run_output_dir = experiment_output_dir / f'{wandb_logger.experiment.id}'
    run_output_dir.mkdir(parents=True, exist_ok=True)
    run_output_dir = str(run_output_dir)

    checkpoint_callback = ModelCheckpoint(
        filepath=run_output_dir,
        save_top_k=True,
        verbose=True,
        monitor='val_loss',
        mode='min',
        prefix=f'{wandb_logger.experiment.id}-'
    )

    print(wandb_logger)

    # most basic trainer, uses good defaults
    trainer = Trainer(
        max_epochs =hparams.max_nb_epochs,
        gpus=gpus,
        nb_gpu_nodes=nodes,
        checkpoint_callback=checkpoint_callback,
        logger=wandb_logger,
        check_val_every_n_epoch=project_config.check_val_every_n_epoch,
        fast_dev_run=fast_dev_run
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
    parser.add_argument("--fast_dev_run", default=False, action="store_true", help="if flag given, runs 1 batch of train, test and val to find any bugs")

    # parse params
    args = parser.parse_args()

    hparams = YParams(args.model_config_file, args.model_config_profile)
    project_config = YParams(args.project_config_file, args.project_config_profile)

    main(args.gpus, args.nodes, args.fast_dev_run, project_config, hparams)