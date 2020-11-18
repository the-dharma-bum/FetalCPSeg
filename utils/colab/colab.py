import os
import subprocess
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from model import LightningModel
from data import DataModule
import config as cfg


def get_data():
  print("Downloading dataset ...")
  os.system('apt install jq pv')
  os.system('chmod 755 /content/FetalCPSeg/utils/colab/download.sh')
  subprocess.check_call(
    ['/content/FetalCPSeg/utils/colab/download.sh', 
     'https://mega.nz/#!tFNGkLQS!mpq8s6gK2SH6xJOBeYsw62yQlZAN9of4_nHnMjQjfMQ'])
  print("Extracting dataset...")
  os.system('unzip -q patnum_data.zip')


def init_trainer():
  lr_logger      = LearningRateMonitor()
  early_stopping = EarlyStopping(monitor   = 'val_loss',
                                 mode      = 'min', 
                                 min_delta = 0.001,
                                 patience  = 100,
                                 verbose   = True)
  return Trainer(gpus=1, callbacks = [lr_logger, early_stopping])


def run_training(dm_config, train_config):
  print('Instancing model...')
  config = cfg.Config(dm_config, train_config)
  data = DataModule.from_config(config.datamodule)
  model = LightningModel.from_config(config)
  try: 
    trainer = init_trainer()
    trainer.fit(model, data)
  except MisconfigurationException:
    print('Did you forget to setup a GPU runtime ?')
  print('Ready. Training will start !')
  


def download_outputs():
  os.system('zip -r /content/output.zip /content/FetalCPSeg/lightning_logs/version_0/')
  files.download("/content/output.zip")


def colab_training(dm_config, train_config):
  get_data()
  run_training(dm_config, train_config)
  download_outputs()


def setup():
  print('Downloading github repository...')
  os.system('git clone https://github.com/the-dharma-bum/FetalCPSeg/')
  os.chdir('FetalCPSeg')
  os.system('git checkout -b rewrite_network')
  os.system('git branch --set-upstream-to=origin/rewrite_network rewrite_network')
  os.system('git pull -q')
  print('Downloading requirements...')
  os.system('pip install -q -r requirements.txt')
