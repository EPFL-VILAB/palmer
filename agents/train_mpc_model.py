import os
import shutil
import numpy as np
import logging
import pdb
import time
import random
from PIL import Image
import wandb

import torch
from torch import nn
from torch.utils.data import DataLoader

import dataloaders.datasets as datasets
from agents.mpc_model import MPCModel, MPCModelTrainer

#-------------------------------------------------------------------------------------------------------------
# Manually enter experiment parameters
#-------------------------------------------------------------------------------------------------------------

experiment_name = None # TODO: Set name.
group = None # TODO: Set name.
EXP_DIR = None # TODO: Set path.
LOG_DIR = os.path.join(EXP_DIR, 'logs')
MODEL_DIR = os.path.join(EXP_DIR, 'models')

checkpoint_path = None

dataset_name = 'MPCDatasetSlow'
OBS_DIR= None # TODO: Set path.
DATASET_LOG_DIR= None # TODO: Set path.
np_coin_flip_seed=57
np_rng_seed=42

dataset_size=299999
max_lookahead=6
trainer_lr = 0.00005

agent_config = {
    'max_lookahead': max_lookahead,
    'num_actions': 4,
    'input_channels': 12,
    'spatial_size': (120,160),
    'backbone_name': 'resnet18',
    'backbone_output_dim': 512,
    'fc_dim': 512
}

train_val_split=0.8
epochs = 1000
batch_size = 16
num_workers = 4

config = globals().copy()
config['project'] = None # TODO: Set wandb project name.
config['notes'] = None # TODO: Set wandb project notes.

#-------------------------------------------------------------------------------------------------------------
# The main experiment script
#-------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    
    print('Setting up experiment...')
    os.mkdir(EXP_DIR)
    print("EXP_DIR: ", EXP_DIR)
    os.mkdir(LOG_DIR)
    print("LOG_DIR: ", LOG_DIR)
    os.mkdir(MODEL_DIR)
    print("MODEL_DIR: ", MODEL_DIR)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Start logging
    logging.basicConfig(filename=os.path.join(LOG_DIR, experiment_name + '.log'), filemode="a", level=logging.DEBUG)
    logger = logging.getLogger("main_logger")
    for key, value in config.items(): logger.info(key + ": {}".format(value))
    for key, value in config.items(): print(key + ": {}".format(value))
    wandb_config = {key:value for key,value in config.items() if isinstance(value, str)}
        
    # Create the dataloaders
    print('Creating the dataset.')
    t_0 = time.time()
    dataset = getattr(datasets, dataset_name)(OBS_DIR=OBS_DIR, LOG_DIR=DATASET_LOG_DIR, np_coin_flip_seed=np_coin_flip_seed, np_rng_seed=np_rng_seed, 
                                              dataset_size=dataset_size, max_lookahead=max_lookahead, action_space_dim=agent_config['num_actions'])
    t_1 = time.time()
    train_size, val_size = round(dataset_size*train_val_split), round(dataset_size*(1-train_val_split))
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    t_2 = time.time()
    print('Creating the dataloader.')
    train_dl = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers)
    valid_dl = DataLoader(val_set, batch_size=batch_size, num_workers=num_workers)
    t_3 = time.time()
    
    logger.info("Dataset creation time: {}".format(t_1 - t_0))
    logger.info("Dataset split time: {}".format(t_2 - t_1))
    logger.info("Dataloader creation time: {}".format(t_3 - t_2))
    
    print("Dataset creation time: {}".format(t_1 - t_0))
    print("Dataset split time: {}".format(t_2 - t_1))
    print("Dataloader creation time: {}".format(t_3 - t_2))
    
    # Create the model 
    print('Creating the model.')
    model = MPCModel(config=agent_config)
        
    print(model)
    # model = nn.DataParallel(model)
    model.to(device)
    
    # Create the trainer
    print('Creating the trainer.')
    trainer = MPCModelTrainer(model, trainer_lr, wandb_config, dataset, checkpoint_path=checkpoint_path)
    
    # Start Training
    print('Start training.')
    trainer.fit(epochs, train_dl, valid_dl, device, MODEL_DIR)


