import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from model import LightningModel
from data import DataModule
import config as cfg


def init_trainer():
  lr_logger      = LearningRateMonitor()
  early_stopping = EarlyStopping(monitor   = 'val_loss',
                                 mode      = 'min', 
                                 min_delta = 0.001,
                                 patience  = 100,
                                 verbose   = True)
  return Trainer(gpus=1, callbacks = [lr_logger, early_stopping])


def run_colab_training(dm_config, train_config):
  print('Instancing model...')
  config = cfg.Config(dm_config, train_config)
  data = DataModule.from_config(config.datamodule)
  model = LightningModel.from_config(config)
  try: 
    trainer = init_trainer()
    print('Ready. Training will start !')
    trainer.fit(model, data)
  except MisconfigurationException:
    print('Did you forget to setup a GPU runtime ?')


def download_outputs(file_module):
  os.system('zip -r /content/output.zip /content/FetalCPSeg/lightning_logs/version_0/')
  file_module.download("/content/output.zip")



