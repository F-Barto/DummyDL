"""
This file runs the main training/val loop, etc... using Lightning Trainer
"""
from pytorch_lightning import Trainer
from argparse import ArgumentParser
from mnist.mnist import SimpleClassifier

from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger()

def main(hparams):
    # init module
    model = SimpleClassifier(hparams)

    # most basic trainer, uses good defaults
    trainer = Trainer(
        max_nb_epochs=hparams.max_nb_epochs,
        gpus=hparams.gpus,
        nb_gpu_nodes=hparams.nodes,
    )
    trainer.fit(model, logger=wandb_logger)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--gpus', type=str, default=None)
    parser.add_argument('--nodes', type=int, default=1)

    # give the module a chance to add own params
    # good practice to define LightningModule speficic params in the module
    parser = SimpleClassifier.add_model_specific_args(parser)

    # parse params
    hparams = parser.parse_args()

    main(hparams)