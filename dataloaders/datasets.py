import os
import torch
from torch.utils.data import Dataset
import glob
import time
import pdb
from tqdm import tqdm 
import numpy as np
from torch.nn import functional as F
    
#-----------------------------------------------------------------------------------------------
# Reads a sequence of states for detailed accuracy evaluation
#-----------------------------------------------------------------------------------------------

class DCDatasetSeq(Dataset):

    def __init__(self, obs_path, LOG_DIR, np_coin_flip_seed=None, np_rng_seed=42, dataset_size=300000, max_lookahead=3, far_sampling_ratio=0.3):

        # Attributes
        self.LOG_DIR = LOG_DIR
        print("Loading observation dict...")
        t_0 = time.time()
        self.obs = np.load(obs_path)
        print("Loading took {} seconds".format(time.time() - t_0))
        print("Reading the observation dict to RAM...")
        t_1 = time.time()
        self.obs_list = [self.obs[key] for key in tqdm(list(self.obs.keys())[:dataset_size])]
        print("Reading took {} seconds".format(time.time() - t_1))
        self.action_list = np.load(os.path.join(self.LOG_DIR, 'action_list.npy'))
        self.rng = np.random.default_rng(np_rng_seed)
        self.dataset_size = dataset_size
        self.max_lookahead = max_lookahead
        self.far_sampling_ratio = far_sampling_ratio
        
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx, verbose=False):
        ''' Returns an item of the dataset.
        Args:
            idx (int): ID of data point
        '''
        
        if self.max_lookahead + 1 < self.dataset_size - 1 - idx:
            target_index_delta_near = np.arange(self.max_lookahead + 1) # Sample near observation
            far_sample_size = int(target_index_delta_near.shape[0] * self.far_sampling_ratio)
            target_index_delta_far = self.rng.integers(self.max_lookahead + 1, self.dataset_size - idx, far_sample_size)  # Sample far observation
        else:
            target_index_delta_near = np.clip(np.arange(self.max_lookahead + 1), 0, self.dataset_size - 1 - idx) # Sample near observation
            far_sample_size = int(target_index_delta_near.shape[0] * self.far_sampling_ratio)
            target_index_delta_far = self.rng.integers(0, self.dataset_size - idx, far_sample_size)  # Sample far observation
            
        target_index_delta = np.concatenate([target_index_delta_near, target_index_delta_far]) 
            
        # Set the target timesteps
        time_difference = np.clip(target_index_delta, 0, self.max_lookahead)
        
        # Set the target actions
        action = np.repeat(self.action_list[idx], len(time_difference))
        action[0] = 0
        
        # Read the observations
        t_0 = time.time()
        target_indices = target_index_delta + idx
        current_state = np.repeat(self.obs_list[idx][np.newaxis, ...], len(time_difference), axis=0)
        target_state = np.stack([self.obs_list[index] for index in target_indices])
        
        # Pass to torch tensors
        t_1 = time.time()
        current_state = torch.from_numpy(current_state).view(current_state.shape[0], -1, *current_state.shape[3:]) / 255 # normalization
        target_state = torch.from_numpy(target_state).view(target_state.shape[0], -1, *target_state.shape[3:]) / 255 # normalization
        action = torch.from_numpy(np.asarray(action))
        time_difference = torch.from_numpy(np.asarray(time_difference))
        t_2 = time.time()
        
        if verbose:
            print("State reading time: {}".format(t_1 - t_0))
            print("Torch tensor creation time: {}".format(t_2 - t_1))

        return current_state, target_state, action, time_difference
    

#-----------------------------------------------------------------------------------------------
# Reads a sequence of states for detailed accuracy evaluation
#-----------------------------------------------------------------------------------------------

class MPCDatasetSlow(Dataset):

    def __init__(self, OBS_DIR, LOG_DIR, np_coin_flip_seed=57, np_rng_seed=42, dataset_size=300000, max_lookahead=6, action_space_dim=5, near_sampling_threshold=0.25):

        # Attributes
        self.OBS_DIR = OBS_DIR
        self.LOG_DIR = LOG_DIR
        self.action_list = np.load(os.path.join(self.LOG_DIR, 'action_list.npy'))
        self.rng = np.random.default_rng(np_rng_seed)
        self.coin_flip_rng = np.random.default_rng(np_coin_flip_seed)
        self.dataset_size = dataset_size
        self.max_lookahead = max_lookahead
        self.action_space_dim = action_space_dim
        self.near_sampling_threshold = near_sampling_threshold
        
    def __len__(self):
        return self.dataset_size # Discard the last observation as the next-state should be returned for p_env

    def __getitem__(self, idx, verbose=False):
        ''' Returns an item of the dataset.
        Args:
            idx (int): ID of data point
        '''
        
        # Do a coin flip to decide whether near or far states will be sampled
        if self.max_lookahead + 1 < self.dataset_size - idx:
            if self.coin_flip_rng.uniform() > self.near_sampling_threshold:
                target_index_delta = self.rng.integers(0, self.max_lookahead + 1) # Sample near observation
            else:
                target_index_delta = self.rng.integers(self.max_lookahead + 1, self.dataset_size - idx)  # Sample far observation
        else:
            target_index_delta = self.rng.integers(0, self.dataset_size - idx) # Sample near observation
            
        # Set the target timesteps
        time_difference = np.clip(target_index_delta, 0, self.max_lookahead)
        
        # Set the target actions
        action = self.action_list[idx]
        if time_difference == 0: action = 0
        
        # Read the observations
        t_0 = time.time()
        target_index = target_index_delta + idx
        
        current_state_path = os.path.join(self.OBS_DIR, 'state_{:06d}.npy'.format(idx))
        next_state_path = os.path.join(self.OBS_DIR, 'state_{:06d}.npy'.format(idx + 1))
        target_state_path = os.path.join(self.OBS_DIR, 'state_{:06d}.npy'.format(target_index))
        
        current_state = np.load(current_state_path)
        next_state = np.load(next_state_path)
        target_state = np.load(target_state_path)
        
        # Pass to torch tensors
        t_1 = time.time()
        current_state = torch.from_numpy(current_state).view(-1, *current_state.shape[2:]) / 255 # normalization
        next_state = torch.from_numpy(next_state).view(-1, *next_state.shape[2:]) / 255 # normalization
        target_state = torch.from_numpy(target_state).view(-1, *target_state.shape[2:]) / 255 # normalization
        action = torch.tensor(action)
        one_hot_action = F.one_hot(action, num_classes=self.action_space_dim)
        time_difference = torch.tensor(time_difference)
        t_2 = time.time()
        
        if verbose:
            print("State reading time: {}".format(t_1 - t_0))
            print("Torch tensor creation time: {}".format(t_2 - t_1))

        return current_state, next_state, target_state, time_difference, action, one_hot_action
    
#--------------------------------------------------------------------------------------------------
# Reads a sequence of states for detailed accuracy evaluation, without reading the dataset to RAM
#--------------------------------------------------------------------------------------------------

class DCDatasetSeqSlow(Dataset):

    def __init__(self, OBS_DIR, LOG_DIR, np_coin_flip_seed=None, np_rng_seed=42, dataset_size=300000, max_lookahead=3, far_sampling_ratio=0.3):

        # Attributes
        self.OBS_DIR = OBS_DIR
        self.LOG_DIR = LOG_DIR
        self.action_list = np.load(os.path.join(self.LOG_DIR, 'action_list.npy'))
        self.rng = np.random.default_rng(np_rng_seed)
        self.dataset_size = dataset_size
        self.max_lookahead = max_lookahead
        self.far_sampling_ratio = far_sampling_ratio
        
    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx, verbose=False):
        ''' Returns an item of the dataset.
        Args:
            idx (int): ID of data point
        '''
        
        if self.max_lookahead + 1 < self.dataset_size - 1 - idx:
            target_index_delta_near = np.arange(self.max_lookahead + 1) # Sample near observation
            far_sample_size = int(target_index_delta_near.shape[0] * self.far_sampling_ratio)
            target_index_delta_far = self.rng.integers(self.max_lookahead + 1, self.dataset_size - idx, far_sample_size)  # Sample far observation
        else:
            target_index_delta_near = np.clip(np.arange(self.max_lookahead + 1), 0, self.dataset_size - 1 - idx) # Sample near observation
            far_sample_size = int(target_index_delta_near.shape[0] * self.far_sampling_ratio)
            target_index_delta_far = self.rng.integers(0, self.dataset_size - idx, far_sample_size)  # Sample far observation
            
        target_index_delta = np.concatenate([target_index_delta_near, target_index_delta_far]) 
            
        # Set the target timesteps
        time_difference = np.clip(target_index_delta, 0, self.max_lookahead)
        
        # Set the target actions
        action = np.repeat(self.action_list[idx], len(time_difference))
        action[0] = 0
        
        # Read the observations
        t_0 = time.time()
        target_indices = target_index_delta + idx
        target_state_paths = [os.path.join(self.OBS_DIR, 'state_{:06d}.npy'.format(index)) for index in target_indices]
        current_state_path = os.path.join(self.OBS_DIR, 'state_{:06d}.npy'.format(idx))
        
        current_state = np.repeat(np.load(current_state_path)[np.newaxis, ...], len(time_difference), axis=0)
        target_state = np.stack([np.load(path) for path in target_state_paths])
        
        # Pass to torch tensors
        t_1 = time.time()
        current_state = torch.from_numpy(current_state).view(current_state.shape[0], -1, *current_state.shape[3:]) / 255 # normalization
        target_state = torch.from_numpy(target_state).view(target_state.shape[0], -1, *target_state.shape[3:]) / 255 # normalization
        action = torch.from_numpy(np.asarray(action))
        time_difference = torch.from_numpy(np.asarray(time_difference))
        t_2 = time.time()
        
        if verbose:
            print("State reading time: {}".format(t_1 - t_0))
            print("Torch tensor creation time: {}".format(t_2 - t_1))
            print("current_state_path")
            print(current_state_path)
            print("target_state_paths")
            print(target_state_paths)

        return current_state, target_state, action, time_difference