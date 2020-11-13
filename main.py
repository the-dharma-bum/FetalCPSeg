""" Main Python file to start routines """

from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from model import LightningModel
from data import DataModule
import config as cfg



# +-------------------------------------------------------------------------------------+ #
# |                                                                                     | #
# |                                          INIT                                       | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #


dm_config = cfg.DataModule(
    input_root        = "/homes/l17vedre/Bureau/Sanssauvegarde/patnum_data/train/",
    target_resolution = (1.5*4, 1.5*4, 8),
    target_shape      = (64, 64, 26),
    class_indexes     = [1],
    patch_size        = None,
    train_batch_size  = 2,
    val_batch_size    = 2,
    num_workers       = 4,
)


train_config = cfg.Train(
    lr           = 1e-3,
    weight_decay = 5e-4,
    milestones   = [500, 750],
    gamma        = 0.1,
    verbose      = True,
)


def init_trainer():
    """ Init a Lightning Trainer using from_argparse_args
        Thus every CLI command (--gpus, distributed_backend, ...) become available.
    """
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args   = parser.parse_args()
    lr_logger      = LearningRateMonitor()
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', min_delta=0.001, 
                                   patience=100, verbose=True)
    return Trainer.from_argparse_args(args, callbacks = [lr_logger, early_stopping])




# +-------------------------------------------------------------------------------------+ #
# |                                                                                     | #
# |                                          RUN                                        | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

def run_training(config):
    """ Instanciate a datamodule, a model and a trainer and run trainer.fit(model, data) """
    data    = DataModule.from_config(config.datamodule)
    model   = LightningModel.from_config(config)
    trainer = init_trainer()
    trainer.fit(model, data)


if __name__ == '__main__':
    config = cfg.Config(dm_config, train_config)
    run_training(config)