import os
import sys
sys.path.append('../')
import shutil
import numpy as np
import logging
import pdb
import time
from PIL import Image

from envs.vizdoom_env import VizdoomEnv
from agents.random_agent import RandomAgent

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

action_space_dim = 5
seed = 75
num_timesteps = 300000
map_save_interval = 1000

#-------------------------------------------------------------------------------------------------------------
# The main experiment script
#-------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    
    print('Setting up experiment...')
    # Create directories
    os.mkdir(EXP_DIR)
    print("EXP_DIR: ", EXP_DIR)
    os.mkdir(OBS_DIR)
    print("OBS_DIR: ", OBS_DIR)
    os.mkdir(POS_DIR)
    print("POS_DIR: ", POS_DIR)
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

    # Create the agent and the environment
    env = VizdoomEnv('SimpleExplorationGame')
    agent = RandomAgent(action_space_dim=action_space_dim, seed=seed, num_of_actions=num_timesteps)
    agent.export_action_list(os.path.join(LOG_DIR, 'action_list.npy'))
    
    logging.info('ENV: SimpleExplorationGame')
    logging.info('AGENT: RandomAgent')
    logging.info('action_space_dim: {}'.format(action_space_dim))
    logging.info('seed: {}'.format(seed))
    logging.info('num_timesteps: {}'.format(num_timesteps))

    start = time.time()
          
    print('Starting experiment...')
    env.reset()
    img_list = []          
    for i in range(num_timesteps):
        print("---------------------------------")
        logging.info("---------------------------------")
        print("Timestep: ", i)
        print("Runtime: ", time.time() - start)
        logging.info("Timestep: {}".format(i))
        logging.info("Runtime: {}".format(time.time() - start))
        t_0 = time.time()
        
        # Take action
        action = agent.act()
        t_1 = time.time()
        
        # Save state and action for current timestep
        with open(os.path.join(OBS_DIR, 'state_{:06d}.npy'.format(i)), 'wb') as f:
            np.save(f, env.game.imgs)
            
        with open(os.path.join(POS_DIR, 'pos_{:06d}.npy'.format(i)), 'wb') as f:
            np.save(f, env.game.agent_positions[-1])
        t_2 = time.time()
        
        if i % map_save_interval == 0:
            print("Drawing map.")
            logging.info("Drawing map.")
            im = Image.fromarray(env.game.map)
            im.save(os.path.join(MAP_DIR, 'state_{:06d}_action_{}.png'.format(i, action)))
        t_3 = time.time()
        
        # Implement the action and get the next state
        env.step(action)
        t_4 = time.time()
        
        print("Total step time: {}".format(t_4 - t_0))
        logging.info("Total step time: {}".format(t_4 - t_0))
        print("Action time: {}".format(t_1 - t_0))
        logging.info("Action time: {}".format(t_1 - t_0))
        print("Action: {}".format(action))
        logging.info("Action: {}".format(action))
        print("Img save time: {}".format(t_2 - t_1))
        logging.info("Img save time: {}".format(t_2 - t_1))
        print("Draw map time: {}".format(t_3 - t_2))
        logging.info("Draw map time: {}".format(t_3 - t_2))
        print("Env step time: {}".format(t_4 - t_3))
        logging.info("Env step time: {}".format(t_4 - t_3))


