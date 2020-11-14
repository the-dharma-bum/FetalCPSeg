""" Main Python file to start routines """

from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from model import LightningModel
from data import DataModule
from verbose import VerboseCallback
import config as cfg


# +---------------------------------------------------------------------------------------------+ #
# |                                                                                             | #
# |                                             INIT                                            | #
# |                                                                                             | #
# +---------------------------------------------------------------------------------------------+ #

def init_trainer():
    """ Init a Lightning Trainer using from_argparse_args
        Thus every CLI command (--gpus, distributed_backend, ...) become available.
    """
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args   = parser.parse_args()
    lr_logger      = LearningRateMonitor()
    verbose        = VerboseCallback()
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.001, 
                                   patience=100, verbose=True)
    return Trainer.from_argparse_args(args, callbacks = [lr_logger, verbose, early_stopping])




# +---------------------------------------------------------------------------------------------+ #
# |                                                                                             | #
# |                                             RUN                                             | #
# |                                                                                             | #
# +---------------------------------------------------------------------------------------------+ #

def run_training(config):
    """ Instanciate a datamodule, a model and a trainer and run trainer.fit(model, data). """
    data    = DataModule.from_config(config.datamodule)
    model   = LightningModel.from_config(config)
    trainer = init_trainer()
    trainer.fit(model, data)


if __name__ == '__main__':
    config = cfg.Config(cfg.DataModule(), cfg.Train())
    run_training(config)