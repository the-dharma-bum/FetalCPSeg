import os
import numpy as np
from tqdm import tqdm
import config as cfg
from data import DataModule
from model import LightningModel


WEIGHTS_DIR = "/homes/l17vedre/Bureau/Sanssauvegarde/patnum_output/final_logs/best/muticlass/checkpoints"
DATA_DIR    = "/homes/l17vedre/Bureau/Sanssauvegarde/patnum_data/test/to_predict"


checkpoints = sorted(list(map(lambda x: os.path.join(WEIGHTS_DIR, x), os.listdir(WEIGHTS_DIR))))

models = {}
for i, checkpoint in tqdm(enumerate(checkpoints)):
    models[i] = LightningModel.load_from_checkpoint(checkpoint)

datamodule_config = cfg.DataModule(
    input_root        = DATA_DIR,
    target_resolution = (1.5*2, 1.5*2, 8),
    target_shape      = (128, 128, 26),
    class_indexes     = (1, 2, 3, 4),
    patch_size        = None,
    train_batch_size  = 1,
    val_batch_size    = 1,
    num_workers       = 4
)
datamodule = DataModule.from_config(datamodule_config)
datamodule.setup(stage='test')

inputs, targets = next(iter(datamodule.test_dataloader()))
inputs = inputs.cuda()
targets = targets.cuda()

def dice_score(outputs, targets, ratio=0.5):
    outputs = outputs.flatten()
    targets = targets.flatten()
    outputs[outputs > ratio] = np.float32(1)
    outputs[outputs < ratio] = np.float32(0)    
    return float(2 * (targets * outputs).sum())/float(targets.sum() + outputs.sum())

dices = []

model = models[0].cuda()
model = model.eval()
outputs = model(inputs)
outputs = outputs[0]


for i in tqdm(range(4)):    
    predicted_mask_array = outputs[:,i,:,:,:].cpu().detach().numpy()
    target_mask_array    = targets[:,i,:,:,:].cpu().detach().numpy()
    dices.append(dice_score(predicted_mask_array, target_mask_array))

print(dices)
