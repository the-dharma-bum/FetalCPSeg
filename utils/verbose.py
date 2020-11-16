""" Verbose callback to be used by a Trainer object.
    The whole idea of this file is to display a responsive table in terminal.
"""

import os
import numpy as np
from dataclasses import dataclass
from tqdm import tqdm
from pytorch_lightning import Callback


# +-------------------------------------------------------------------------------------+ #
# |                                                                                     | #
# |                                     FANCY DISPLAY                                   | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

@dataclass
class FancyDisplay():

    current_loss:        str  = '| Current training Loss......:'
    current_dice:         str = '| Current training Dice......:'
    current_lr:          str  = '| Current Learning Rate......:'
    last_avg_train_loss: str  = '| Training loss..............:'
    last_avg_val_loss:   str  = '| Validation loss............:'
    last_avg_train_dice:  str = '| Training dice score........:'
    last_avg_val_dice:    str = '| Validation dice score......:'
    best_avg_train_loss: str  = '| Training Loss..............:'
    best_avg_val_loss:   str  = '| Validation Loss............:'
    best_avg_train_dice:  str = '| Training dice score........:'
    best_avg_val_dice:    str = '| Validation dice score......:'




# +-------------------------------------------------------------------------------------+ #
# |                                                                                     | #
# |                                    TQDM DESCRIPTOR                                  | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

class Descriptor:

    def __init__(self, width, position, string):
        self.tqdm   = tqdm(total=0, position=position, bar_format='{desc}')
        self.string = string
        self.offset = width - 2 - 8 - len(string)

    def update(self, value):
        status = self.string + ' {:4f}'.format(value) + self.offset*' ' + '|'
        self.tqdm.set_description_str(status)

    def close(self):
        self.tqdm.close()

# +-------------------------------------------------------------------------------------+ #
# |                                                                                     | #
# |                                       TQDM TITLE                                    | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

class Title:

    def __init__(self, position, width, string=None, top_border_only=False):
        self.top_border_only = top_border_only
        self.top_border = tqdm(total=0, position=position-1, bar_format='{desc}')
        if not self.top_border_only:
            self.title      = tqdm(total=0, position=position,   bar_format='{desc}')
            self.bot_border = tqdm(total=0, position=position+1, bar_format='{desc}')
            self.string     = '|' + string + (width-2-len(string))*' ' + '|'
        self.str_border = '+' + (width-2)*'-' + '+'

    def display(self):
        self.top_border.set_description_str(self.str_border)
        if not self.top_border_only:
            self.title.set_description_str(self.string)
            self.bot_border.set_description_str(self.str_border)

    def close(self):
        self.top_border.close()
        if not self.top_border_only:
            self.title.close()
            self.bot_border.close()




# +-------------------------------------------------------------------------------------+ #
# |                                                                                     | #
# |                                          TABLE                                      | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

class Table:
    """
        Table to be display in terminal, showing current training infos.
        Updated every batch.
    """
    def __init__(self, width=42, best_only=False):
        self.width   = width
        self.strings = FancyDisplay()
        if not best_only:
            self.top_title = Title(4,  width, 'CURRENT EPOCH')
            self.current_loss = Descriptor(width, 6, self.strings.current_loss)
            self.current_dice = Descriptor(width, 7, self.strings.current_dice)
            self.current_lr   = Descriptor(width, 8, self.strings.current_lr)
            self.mid_title = Title(10, width, 'LAST EPOCH (average)')
            self.last_epoch_avg_train_loss = Descriptor(width, 12, self.strings.last_avg_train_loss)
            self.last_epoch_avg_val_loss   = Descriptor(width, 13, self.strings.last_avg_val_loss)
            self.last_epoch_avg_train_dice = Descriptor(width, 14, self.strings.last_avg_train_dice)
            self.last_epoch_avg_val_dice   = Descriptor(width, 15, self.strings.last_avg_val_dice)
            self.bot_title = Title(17, width, 'BEST SO FAR (one epoch average)')
            self.best_train_loss = Descriptor(width, 19, self.strings.best_avg_train_loss)
            self.best_val_loss   = Descriptor(width, 20, self.strings.best_avg_val_loss)
            self.best_train_dice = Descriptor(width, 21, self.strings.best_avg_train_dice)
            self.best_val_dice   = Descriptor(width, 22, self.strings.best_avg_val_dice)
            self.last_line = Title(24, width, top_border_only=True)
        else:
            self.bot_title = Title(2, width, 'BEST SO FAR (one epoch average)')
            self.best_train_loss = Descriptor(width, 4, self.strings.best_avg_train_loss)
            self.best_val_loss   = Descriptor(width, 5, self.strings.best_avg_val_loss)
            self.best_train_dice = Descriptor(width, 6, self.strings.best_avg_train_dice)
            self.best_val_dice   = Descriptor(width, 7, self.strings.best_avg_val_dice)
            self.last_line = Title(9, width, top_border_only=True)

    def update_current(self, loss, dice, lr):
        self.top_title.display()
        self.current_loss.update(loss)
        self.current_dice.update(dice)
        self.current_lr.update(lr)

    def update_last_average(self, loss, val_loss, dice, val_dice):
        self.mid_title.display()
        self.last_epoch_avg_train_loss.update(loss)
        self.last_epoch_avg_val_loss.update(val_loss)
        self.last_epoch_avg_train_dice.update(dice)
        self.last_epoch_avg_val_dice.update(val_dice)

    def update_best_average(self, train_loss, val_loss, train_dice, val_dice):
        self.bot_title.display()
        self.best_train_loss.update(train_loss)
        self.best_val_loss.update(val_loss)
        self.best_train_dice.update(train_dice)
        self.best_val_dice.update(val_dice)
        self.last_line.display()

    def close(self):
        self.top_title.close()
        self.mid_title.close()
        self.bot_title.close()
        self.last_line.close()
        self.current_loss.close()
        self.current_dice.close()
        self.current_lr.close()
        self.last_epoch_avg_train_loss.close()
        self.last_epoch_avg_val_loss.close()
        self.last_epoch_avg_train_dice.close()
        self.last_epoch_avg_val_dice.close()
        self.best_train_loss.close()
        self.best_val_loss.close()
        self.best_train_dice.close()
        self.best_val_dice.close()


# +-------------------------------------------------------------------------------------+ #
# |                                                                                     | #
# |                                         STATE                                       | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

class State():

    def __init__(self, table_width=42):
        self.last_avg_train_loss = 9.999
        self.last_avg_val_loss   = 9.9999
        self.last_avg_train_dice = 0.
        self.last_avg_val_dice   = 0.
        self.best_avg_train_loss = 9.9999
        self.best_avg_val_loss   = 9.9999
        self.best_avg_train_dice = 0.
        self.best_avg_val_dice   = 0.
        self.epoch_train_losses  = []
        self.epoch_train_dices   = []
        self.epoch_val_losses    = []
        self.epoch_val_dices     = []
        self.table               = Table(table_width)

    def update_current_train(self, output):
        current_loss = output['Loss/Train']
        current_dice  = output['Dice Score/Train']
        current_lr   = 0.1
        self.epoch_train_losses.append(current_loss.item())
        self.epoch_train_dices.append(current_dice)
        self.table.update_current(current_loss, current_dice, current_lr)

    def update_current_val(self, output):
        current_loss = output['val_loss']
        current_dice = output['val_dice']
        self.epoch_val_losses.append(current_loss.item())
        self.epoch_val_dices.append(current_dice)

    def update_best_average(self):
        self.best_avg_train_loss = min(self.best_avg_train_loss, self.last_avg_train_loss)
        self.best_avg_val_loss   = min(self.best_avg_val_loss,   self.last_avg_val_loss)
        self.best_avg_train_dice = max(self.best_avg_train_dice,  self.last_avg_train_dice)
        self.best_avg_val_dice   = max(self.best_avg_val_dice,    self.last_avg_val_dice)
        self.table.update_best_average(self.best_avg_train_loss, self.best_avg_val_loss,
                                       self.best_avg_train_dice, self.best_avg_val_dice)

    def update_last_average(self):
        self.last_avg_train_loss = np.asarray(self.epoch_train_losses).mean()
        self.last_avg_train_dice = np.asarray(self.epoch_train_dices).mean()
        self.last_avg_val_loss   = np.asarray(self.epoch_val_losses).mean()
        self.last_avg_val_dice   = np.asarray(self.epoch_val_dices).mean()
        self.table.update_last_average(self.last_avg_train_loss, self.last_avg_val_loss,
                                       self.last_avg_train_dice,  self.last_avg_val_dice)




# +-------------------------------------------------------------------------------------+ #
# |                                                                                     | #
# |                                    VERBOSE CALLBACK                                 | #
# |                                                                                     | #
# +-------------------------------------------------------------------------------------+ #

class VerboseCallback(Callback):

    def __init__(self):
        self.state = State()

    @staticmethod
    def clear_terminal():
        os.system('cls' if os.name == 'nt' else 'clear')

    def on_train_batch_end(self,  trainer, pl_module, *args):
        if pl_module.current_epoch==0:
            return
        output = trainer.callback_metrics   
        self.state.update_current_train(output)

    def on_validation_batch_end(self, trainer, pl_module, *args):
        if pl_module.current_epoch==0: 
            return
        output = trainer.callback_metrics
        if output:
            self.state.update_current_val(output)

    def on_epoch_end(self, pl_module, *args):
        if pl_module.current_epoch==0: 
            return
        self.state.update_last_average()
        self.state.update_best_average()

    def on_fit_end(self, *args):
        print(2*'\n')

    def on_keyboard_interrupt(self, *args):
        self.state.table.close()
        self.clear_terminal()
        print("Keyboard interrupt. Pytorch Lightning attempted a graceful shutdown.")
        print("So far, the training results were the following:\n")
        table = Table(best_only=True)
        table.bot_title.display()
        table.update_best_average(self.state.best_avg_train_loss, self.state.best_avg_val_loss,
                                  self.state.best_avg_train_dice,  self.state.best_avg_val_dice)
        table.last_line.display()