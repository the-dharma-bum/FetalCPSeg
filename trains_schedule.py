from copy import deepcopy
import torch.nn as nn
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from model import LightningModel
from data import DataModule
from utils import VerboseCallback
import config as cfg


ROOTDIR = "/homes/l17vedre/Bureau/Sanssauvegarde/patnum_data/train/"
    

# +------------------------------------------------------------------------------------------+ #
# |                                                                                          | #
# |                                           BASELINE                                       | #
# |                                                                                          | #
# +------------------------------------------------------------------------------------------+ #

dm_baseline = cfg.DataModule(
    input_root         = ROOTDIR,
    target_resolution  = (1.5*2, 1.5*2, 8),
    target_shape       = (128, 128, 26),
    class_indexes      = (1, 2, 3, 4),
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

trains_schedule = []

# +-------------------------------------------+ #
# |                  ACTIVATION               | #
# +-------------------------------------------+ #

celu = deepcopy(baseline)
celu.train.activation = nn.CELU
trains_schedule.append(celu)

selu = deepcopy(baseline)
selu.train.activation = nn.SELU
trains_schedule.append(selu)

gelu = deepcopy(baseline)
gelu.train.activation = nn.GELU
trains_schedule.append(gelu)

prelu = deepcopy(baseline)
prelu.train.activation = nn.PReLU
trains_schedule.append(prelu)

baseline.train.activation = nn.PReLU


# +-------------------------------------------+ #
# |                    DEPTH                  | #
# +-------------------------------------------+ #

small = deepcopy(baseline)
small.train.depth = 2
trains_schedule.append(small)

large = deepcopy(baseline)
large.train.depth = 4
trains_schedule.append(large)

xtralarge = deepcopy(baseline)
xtralarge.train.depth = 5
trains_schedule.append(xtralarge)

# +-------------------------------------------+ #
# |                  ATTENTION                | #
# +-------------------------------------------+ #

attentive = deepcopy(baseline)
attentive.train.attention = True
trains_schedule.append(attentive)

supervised = deepcopy(baseline)
supervised.train.supervision = True
trains_schedule.append(supervised)

attentive_and_supervised = deepcopy(baseline)
attentive_and_supervised.train.attention = True
attentive_and_supervised.train.supervision = True
trains_schedule.append(attentive_and_supervised)

baseline.train.attention = True
baseline.train.supervision = True

# +-------------------------------------------+ #
# |              SQUEEZE AND EXCITE           | #
# +-------------------------------------------+ #

se = deepcopy(baseline)
se.train.se = True
trains_schedule.append(se)


# +-------------------------------------------+ #
# |                    DROPOUT                | #
# +-------------------------------------------+ #

dropout_1 = deepcopy(baseline)
dropout_1.train.dropout = 0.1
trains_schedule.append(dropout_1)

dropout_2 = deepcopy(baseline)
dropout_2.train.dropout = 0.2
trains_schedule.append(dropout_2)

dropout_3 = deepcopy(baseline)
dropout_3.train.dropout = 0.3
trains_schedule.append(dropout_3)

dropout_4 = deepcopy(baseline)
dropout_4.train.dropout = 0.4
trains_schedule.append(dropout_4)

dropout_5 = deepcopy(baseline)
dropout_5.train.dropout = 0.5
trains_schedule.append(dropout_5)

dropout_6 = deepcopy(baseline)
dropout_6.train.dropout = 0.6
trains_schedule.append(dropout_6)

dropout_7 = deepcopy(baseline)
dropout_7.train.dropout = 0.7
trains_schedule.append(dropout_7)

dropout_8 = deepcopy(baseline)
dropout_8.train.dropout = 0.8
trains_schedule.append(dropout_8)

dropout_9 = deepcopy(baseline)
dropout_9.train.dropout = 0.9
trains_schedule.append(dropout_9)




# +------------------------------------------------------------------------------------------+ #
# |                                                                                          | #
# |                                          RUN EXPERIMENTS                                 | #
# |                                                                                          | #
# +------------------------------------------------------------------------------------------+ #

def init_trainer(dev=False):
    lr_logger      = LearningRateMonitor()
    verbose        = VerboseCallback()
    early_stopping = EarlyStopping(monitor   = 'val_loss',
                                    mode      = 'min', 
                                    min_delta = 0.001,
                                    patience  = 100,
                                    verbose   = True)
    return Trainer(gpus=1, fast_dev_run=dev, callbacks = [lr_logger, verbose, early_stopping])


def run_training(trainer, config):
    """ Instanciate a datamodule, a model and a trainer and run trainer.fit(model, data). """
    data    = DataModule.from_config(config.datamodule)
    model   = LightningModel.from_config(config)
    trainer.fit(model, data)


def sanity_check(runs_configs):
    trainer = init_trainer(dev=True)
    for config in runs_configs:
        run_training(trainer, config)


sanity_check(trains_schedule)