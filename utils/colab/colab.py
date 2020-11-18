import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from model import LightningModel
from data import DataModule
import config as cfg
from google.colab import files


def get_data():
  os.system("./download 'https://mega.nz/#!tFNGkLQS!mpq8s6gK2SH6xJOBeYsw62yQlZAN9of4_nHnMjQjfMQ'")


def setup():
  print('Downloading githug repository...')
  os.system('git clone https://github.com/the-dharma-bum/FetalCPSeg/')
  os.system('unzip -q patnum_data.zip')
  os.chdir('FetalCPSeg')
  os.system('git checkout -b rewrite_network')
  os.system('git branch --set-upstream-to=origin/rewrite_network rewrite_network')
  os.system('git pull -q')
  print('Downloading requirements...')
  os.system('pip install -q -r requirements.txt')


def init_trainer():
  lr_logger      = LearningRateMonitor()
  early_stopping = EarlyStopping(monitor   = 'val_loss',
                                 mode      = 'min', 
                                 min_delta = 0.001,
                                 patience  = 100,
                                 verbose   = True)
  return Trainer(gpus=1, callbacks = [lr_logger, early_stopping])


def run_training(dm_config, train_config):
  config = cfg.Config(dm_config, train_config)
  data = DataModule.from_config(config.datamodule)
  model = LightningModel.from_config(config)
  trainer = init_trainer()
  trainer.fit(model, data)


def download_outputs():
    os.system('zip -r /content/output.zip /content/FetalCPSeg/lightning_logs/version_0/')
    files.download("/content/output.zip")


def colab_training(dm_config, train_config):
    setup()
    run_training(dm_config, train_config)
    download_outputs()


