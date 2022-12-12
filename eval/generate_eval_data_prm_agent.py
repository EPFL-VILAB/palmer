import os
import sys
sys.path.append('../')
import shutil
import numpy as np
import logging
import pdb
import time
from PIL import Image
from tqdm import tqdm
import cv2
import torch
import random

from envs.game_registry import GAMES, GAME_NAMES
from envs.vizdoom_env import VizdoomEnv
from agents.mpc_model import MPCModel
from mpc_agent_graph_PRM_retrieval import MPCAgent


#-------------------------------------------------------------------------------------------------------------
# Manually enter experiment parameters
#-------------------------------------------------------------------------------------------------------------
# Paths to save newly generated data
experiment_name = None # TODO: Set name.
SAVE_DIR = None # TODO: Set path.
EXP_DIR = os.path.join(SAVE_DIR, experiment_name)
OBS_DIR = os.path.join(EXP_DIR, 'obs')
POS_DIR = os.path.join(EXP_DIR, 'pos')
ACT_DIR = os.path.join(EXP_DIR, 'act')
MAP_DIR = os.path.join(EXP_DIR, 'maps')
LOG_DIR = os.path.join(EXP_DIR, 'logs')

# Paths to load old random walk data
LOAD_DIR = None # TODO: Set path.
LOAD_OBS_DIR = os.path.join(LOAD_DIR, 'obs')
LOAD_POS_DIR = os.path.join(LOAD_DIR, 'pos')
LOAD_MAP_DIR = os.path.join(LOAD_DIR, 'maps')
LOAD_LOG_DIR = os.path.join(LOAD_DIR, 'logs')
LOAD_LATENT_DIR = os.path.join(LOAD_DIR, 'latents')

# Experiment and agent parameters
total_num_timesteps = 300000
map_save_interval = 1000
num_goal_close_actions = 10

MODEL_PATH = None # TODO: Set path.

default_config = {
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
    'consistency_threshold': np.log(0.4),
    'GRAPH_EXP_DIR': None, # TODO: Set path.
    'num_nodes_graph': 1000,
    'pT_distance_threshold': 5.5,
    'latent_distance_threshold': 1.5,
    'overlap_pT_distance_threshold': 4,
    'overlap_latent_distance_threshold': 2,
    'overlap_retrieval_distance_threshold': 1,
    'pT_search_threshold': 5.5,
    'graph_rng_seed': 285,
    'num_batches': 100,
}



#-------------------------------------------------------------------------------------------------------------
# The main experiment script
#-------------------------------------------------------------------------------------------------------------

EXP_DIR = os.path.join(SAVE_DIR, experiment_name)

if __name__ == "__main__":
    
    print('Setting up experiment...')
    # Create directories
    os.mkdir(EXP_DIR)
    print("EXP_DIR: ", EXP_DIR)
    os.mkdir(OBS_DIR)
    print("OBS_DIR: ", OBS_DIR)
    os.mkdir(POS_DIR)
    print("POS_DIR: ", POS_DIR)
    os.mkdir(ACT_DIR)
    print("ACT_DIR: ", ACT_DIR)
    os.mkdir(MAP_DIR)
    print("MAP_DIR: ", MAP_DIR)
    os.mkdir(LOG_DIR)
    print("LOG_DIR: ", LOG_DIR)
    
    # Start logging
    logging.basicConfig(filename=os.path.join(LOG_DIR, experiment_name + '.log'), level=logging.DEBUG)
    logging.info(EXP_DIR)
    logging.info(OBS_DIR)
    logging.info(MAP_DIR)
    logging.info(LOG_DIR)
    
    # Create the environment
    env = VizdoomEnv('SimpleExplorationGame')
    env.reset()
    
    # Create agent
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MPCModel(device, config=default_config)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    agent = MPCAgent(model, device, env, config=default_config)
    
    # Data Creation
    num_env_steps = 0
    agent.flush_action_buffer()
    
    goal_id_list = []
    goal_position_list = []
    goal_latent_list = []
    
    goal_episode_distances_list = []
    goal_episode_positions_list = []
    goal_episode_timesteps_list = []
    
    goal_close_countdown = 50 # -1 denotes the goal isn't near
    
    for timestep in range(total_num_timesteps):
        
        print()
        t_0 = time.time()
        # Sample and set goal if the previous goal was reached
        if goal_close_countdown == 0 or timestep == 0:
            while True:
                try:
                    agent.flush_action_buffer()
                    goal_id, goal_position, goal_latent = agent.graph_memory.sample_goal()
                    agent.set_goal_latent(goal_latent, goal_position)
                except:
                    print('agent.set_goal_latent() failed')
                    continue
                else:
                    goal_id_list.append(goal_id)
                    goal_position_list.append(goal_position)
                    goal_latent_list.append(goal_latent)

                    with open(os.path.join(LOG_DIR, 'goal_episode_distances_list_{:06d}.npy'.format(timestep)), 'wb') as f:
                        np.save(f, goal_episode_distances_list)
                        
                    with open(os.path.join(LOG_DIR, 'goal_episode_distances_list_{:06d}.npy'.format(timestep)), 'wb') as f:
                        np.save(f, goal_episode_positions_list)
                        
                    with open(os.path.join(LOG_DIR, 'goal_episode_distances_list_{:06d}.npy'.format(timestep)), 'wb') as f:
                        np.save(f, goal_episode_timesteps_list)
                        
                    goal_episode_distances_list = []
                    goal_episode_positions_list = []
                    goal_episode_timesteps_list = []

                    goal_close_countdown = 50
                    goal_close_first_flag = False
                    
                    print('----------------------')
                    print('Changing goal')
                    print('----------------------')
                    print()
                    break 
            
        # Check if the current goal is close, start a countdown if it is
        if goal_close_first_flag == False and agent.goal_close_flag:
            goal_close_countdown = num_goal_close_actions
            goal_close_first_flag = True
        
        # Decrement the count
        goal_close_countdown = goal_close_countdown - 1

        # Take action
        current_state = env.game.imgs
        current_position = env.game.agent_positions[-1]
        goal_distance = np.linalg.norm(current_position - goal_position)

        goal_episode_distances_list.append(goal_distance)
        goal_episode_positions_list.append(current_position)
        goal_episode_timesteps_list.append(timestep)

        try:
            retrieval_actions, retrieval_positions, action = agent.navigate_to_goal_retrieval_PRM(current_state, current_position)
        except:
            retrieval_actions, retrieval_positions = None, None
            action = random.randint(0,3)

        t_1 = time.time()
        # Save state and action for current timestep
        with open(os.path.join(OBS_DIR, 'state_{:06d}.npy'.format(timestep)), 'wb') as f:
            np.save(f, env.game.imgs)

        with open(os.path.join(POS_DIR, 'pos_{:06d}.npy'.format(timestep)), 'wb') as f:
            np.save(f, env.game.agent_positions[-1])
            
        with open(os.path.join(ACT_DIR, 'action_{:06d}.npy'.format(timestep)), 'wb') as f:
            np.save(f, np.array(action))
        t_2 = time.time()

        if timestep % map_save_interval == 0:
            print("Drawing map.")
            logging.info("Drawing map.")
            im = Image.fromarray(env.game.map)
            im.save(os.path.join(MAP_DIR, 'state_{:06d}_action_{}.png'.format(timestep, action)))
        t_3 = time.time()

        # Implement the action and get the next state
        env.step(action)
        t_4 = time.time()
        
        print("Timestep: {}".format(timestep))
        print("Goal close countdown: {}".format(goal_close_countdown))
        print("Total step time: {}".format(t_4 - t_0))
        print("Action time: {}".format(t_1 - t_0))
        print("Action: {}".format(action))
        print("Img save time: {}".format(t_2 - t_1))
        print("Draw map time: {}".format(t_3 - t_2))
        print("Env step time: {}".format(t_4 - t_3))

    with open(os.path.join(POS_DIR, 'position_list.npy'), 'wb') as f:
        np.save(f, env.game.agent_positions)
        
    with open(os.path.join(LOG_DIR, 'goal_id_list.npy'), 'wb') as f:
        np.save(f, goal_id_list)
        
    with open(os.path.join(LOG_DIR, 'goal_position_list.npy'), 'wb') as f:
        np.save(f, goal_position_list)
        
    with open(os.path.join(LOG_DIR, 'goal_latent_list.npy'), 'wb') as f:
        np.save(f, goal_latent_list)