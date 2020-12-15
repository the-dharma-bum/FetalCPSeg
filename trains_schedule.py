from copy import deepcopy
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from model import LightningModel
from data import DataModule
import config as cfg


ROOTDIR = "/homes/l17vedre/Bureau/Sanssauvegarde/patnum_data/train/"
LOGDIR  = "/homes/l17vedre/Bureau/Sanssauvegarde/patnum_output/monoclass_logs"


# +------------------------------------------------------------------------------------------+ #
# |                                                                                          | #
# |                                           BASELINE                                       | #
# |                                                                                          | #
# +------------------------------------------------------------------------------------------+ #

dm_baseline = cfg.DataModule(
    input_root         = ROOTDIR,
    target_resolution  = (1.5*2, 1.5*2, 8),
    target_shape       = (128, 128, 26),
    class_indexes      = (1, ),
    patch_size         = None,
    train_batch_size   = 2,
    val_batch_size     = 2,
    num_workers        = 4
)

train_baseline = cfg.Train(
    in_channels  = 1,
    supervision  = False,
    attention    = False,
    depth        = 3,
    activation   = nn.ReLU,
    se           = False,
    dropout      = 0.,
    lr           = 1e-3,
    weight_decay = 5e-4
)

baseline = cfg.Config(dm_baseline, train_baseline)




# +------------------------------------------------------------------------------------------+ #
# |                                                                                          | #
# |                                         MODEL EXPERIMENTS                                | #
# |                                                                                          | #
# +------------------------------------------------------------------------------------------+ #

trains_schedule = [baseline]                    # VERSION 0

# +-------------------------------------------+ #
# |                  ACTIVATION               | #
# +-------------------------------------------+ #

celu = deepcopy(baseline)
celu.train.activation = nn.CELU
trains_schedule.append(celu)                    # VERSION 1

selu = deepcopy(baseline)
selu.train.activation = nn.SELU
trains_schedule.append(selu)                    # VERSION 2

gelu = deepcopy(baseline)
gelu.train.activation = nn.GELU
trains_schedule.append(gelu)                    # VERSION 3

prelu = deepcopy(baseline)
prelu.train.activation = nn.PReLU
trains_schedule.append(prelu)                   # VERSION 4

baseline.train.activation = nn.PReLU


# +-------------------------------------------+ #
# |                    DEPTH                  | #
# +-------------------------------------------+ #

small = deepcopy(baseline)
small.train.depth = 2
trains_schedule.append(small)                   # VERSION 5

large = deepcopy(baseline)
large.train.depth = 4
trains_schedule.append(large)                   # VERSION 6

xtralarge = deepcopy(baseline)
xtralarge.train.depth = 5
trains_schedule.append(xtralarge)               # VERSION 7


# +-------------------------------------------+ #
# |                  ATTENTION                | #
# +-------------------------------------------+ #

attentive = deepcopy(baseline)
attentive.train.attention = True
trains_schedule.append(attentive)               # VERSION 8

supervised = deepcopy(baseline)
supervised.train.supervision = True
trains_schedule.append(supervised)              # VERSION 9

attentive_and_supervised = deepcopy(baseline)
attentive_and_supervised.train.attention = True
attentive_and_supervised.train.supervision = True
trains_schedule.append(attentive_and_supervised)# VERSION 10

baseline.train.attention = True
baseline.train.supervision = True


# +-------------------------------------------+ #
# |              SQUEEZE AND EXCITE           | #
# +-------------------------------------------+ #

se = deepcopy(baseline)
se.train.se = True
trains_schedule.append(se)                      # VERSION 11


# +-------------------------------------------+ #
# |                    DROPOUT                | #
# +-------------------------------------------+ #

dropout_1 = deepcopy(baseline)
dropout_1.train.dropout = 0.1
trains_schedule.append(dropout_1)               # VERSION 12

dropout_2 = deepcopy(baseline)
dropout_2.train.dropout = 0.2
trains_schedule.append(dropout_2)               # VERSION 13

dropout_3 = deepcopy(baseline)
dropout_3.train.dropout = 0.3
trains_schedule.append(dropout_3)               # VERSION 14

dropout_4 = deepcopy(baseline)
dropout_4.train.dropout = 0.4
trains_schedule.append(dropout_4)               # VERSION 15

dropout_5 = deepcopy(baseline)
dropout_5.train.dropout = 0.5
trains_schedule.append(dropout_5)               # VERSION 16

dropout_6 = deepcopy(baseline)
dropout_6.train.dropout = 0.6
trains_schedule.append(dropout_6)               # VERSION 17

dropout_7 = deepcopy(baseline)
dropout_7.train.dropout = 0.7
trains_schedule.append(dropout_7)               # VERSION 18

dropout_8 = deepcopy(baseline)
dropout_8.train.dropout = 0.8
trains_schedule.append(dropout_8)               # VERSION 19

dropout_9 = deepcopy(baseline)
dropout_9.train.dropout = 0.9
trains_schedule.append(dropout_9)               # VERSION 20




# +------------------------------------------------------------------------------------------+ #
# |                                                                                          | #
# |                                          RUN EXPERIMENTS                                 | #
# |                                                                                          | #
# +------------------------------------------------------------------------------------------+ #

def init_trainer(dev=False):
    lr_logger      = LearningRateMonitor()
    early_stopping = EarlyStopping(monitor   = 'val_loss',
                                    mode      = 'min', 
                                    min_delta = 0.001,
                                    patience  = 100,
                                    verbose   = True)
    return Trainer(default_root_dir=LOGDIR, gpus=1, fast_dev_run=dev,
                   callbacks = [lr_logger, early_stopping])


def run_training(trainer, config):
    """ Instanciate a datamodule, a model and a trainer and run trainer.fit(model, data). """
    data    = DataModule.from_config(config.datamodule)
    model   = LightningModel.from_config(config)
    trainer.fit(model, data)


def sanity_check(trains_schedule):
    trainer = init_trainer(dev=True)
    for config in trains_schedule:
        run_training(trainer, config)


def run_experiments(trains_schedule):
    for i, config in enumerate(trains_schedule):
        print(80*'_')
        print(f'TRAIN {i}/{len(trains_schedule)}')
        trainer = init_trainer(dev=False)
        run_training(trainer, config)


#sanity_check(trains_schedule)
run_experiments(trains_schedule)
