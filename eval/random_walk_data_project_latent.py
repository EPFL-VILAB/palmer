import os
import sys
sys.path.append('../')
import shutil
import numpy as np
import logging
import pdb
import time
from PIL import Image
import torch
from tqdm import tqdm

from agents.mpc_model import MPCModel
from agents.mpc_agent import MPCAgent # Don't forget to pick the agent here

#-------------------------------------------------------------------------------------------------------------
# Manually enter experiment parameters
#-------------------------------------------------------------------------------------------------------------
experiment_name = None # TODO: Set name.
SAVE_DIR = None # TODO: Set path.
EXP_DIR = os.path.join(SAVE_DIR, experiment_name)
OBS_DIR = os.path.join(EXP_DIR, 'obs')
POS_DIR = os.path.join(EXP_DIR, 'pos')
MAP_DIR = os.path.join(EXP_DIR, 'maps')
LOG_DIR = os.path.join(EXP_DIR, 'logs')
LATENT_DIR = os.path.join(EXP_DIR, 'latents')

MODEL_PATH = None # TODO: Set path.
agent_config = {
    'max_lookahead': 6,
    'num_actions': 4,
    'input_channels': 12,
    'spatial_size': (120,160),
    'backbone_name': 'resnet18',
    'backbone_output_dim': 512,
    'fc_dim': 512,
    'horizon': 6,
    'max_num_mpc_samples': 100000,
    'allT_or_lastT': 'lastT',
    'cons_or_nocons': 'nocons',
    'statH_or_dynH': 'dynH',
    'allH_or_lastH': 'lastH',
    'consistency_threshold': np.log(0.3),
}


num_timesteps = 300000
batch_size = 5000
# batch_size = 30000

#-------------------------------------------------------------------------------------------------------------
# The main experiment script
#-------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    
    print('Setting up experiment...')
    # Create directories
    os.mkdir(LATENT_DIR)
    print("LATENT_DIR: ", LATENT_DIR)
    
    # Start logging
    logging.basicConfig(filename=os.path.join(LOG_DIR, experiment_name + '.log'), level=logging.DEBUG)
    logging.info(EXP_DIR)
    logging.info(LATENT_DIR)

    # Create agent
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MPCModel(config=agent_config)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.to(device)
    model.eval()

    start = time.time()
          
    print('Starting projection...')
    all_latents_list = []
    for batch_num in tqdm(range(num_timesteps // batch_size)):
        
        print("Batch_num: ", batch_num)
        batch_list = []
        for sample_num in range(batch_size):
            sample_idx = batch_num * batch_size + sample_num
            sample_path = os.path.join(OBS_DIR, 'state_{:06d}.npy'.format(sample_idx))
            sample_np = np.load(sample_path)
            sample = (torch.from_numpy(sample_np).view(1, -1, *sample_np.shape[2:]) / 255).float().to(device)
            batch_list.append(sample)
                      
        batch = torch.cat(batch_list) # shape (B,C,H,W)
        with torch.no_grad():
            latent_batch = model.get_latent(batch).cpu().numpy() # shape (B,D)
        latent_list = np.split(latent_batch, batch_size) # shape [(1,D)]
        all_latents_list.extend(latent_list)
        
        for sample_num, latent in enumerate(latent_list):
            sample_idx = batch_num * batch_size + sample_num
            sample_path = os.path.join(LATENT_DIR, 'latent_{:06d}.npy'.format(sample_idx))
            
            with open(sample_path, 'wb') as f:
                np.save(f, latent)
        
    print('Saving all states into npz...')
    with open(os.path.join(LATENT_DIR, 'latents.npz'), 'wb') as f:
            np.savez(f, *all_latents_list)
            
    latents_block = np.concatenate(all_latents_list)
    with open(os.path.join(LATENT_DIR, 'latents_block.npy'), 'wb') as f:
        np.save(f, latents_block)
    

