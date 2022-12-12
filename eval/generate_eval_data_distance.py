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

from envs.game_registry import GAMES, GAME_NAMES
from envs.vizdoom_env import VizdoomEnv
from agents.random_agent import RandomAgent

#-------------------------------------------------------------------------------------------------------------
# Manually enter experiment parameters
#-------------------------------------------------------------------------------------------------------------
experiment_name = None # TODO: Set name.
SAVE_DIR = None # TODO: Set path.
EXP_DIR = os.path.join(SAVE_DIR, experiment_name)
rng_seed = 153
num_spawn_points = 100
num_random_samples = 20
spawn_radii = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
spawn_radii_pairs = list(zip(spawn_radii[:-1], spawn_radii[1:]))

#-------------------------------------------------------------------------------------------------------------
# The main experiment script
#-------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    
    print('Setting up experiment...')
    # Create directories
    os.mkdir(EXP_DIR)
    print("EXP_DIR: ", EXP_DIR)
    
    # Start logging
    logging.basicConfig(filename=os.path.join(EXP_DIR, experiment_name + '.log'), level=logging.DEBUG)
    logging.info(EXP_DIR)
    
    rng = np.random.default_rng(rng_seed)

    # Create the environment
    env = VizdoomEnv('SimpleExplorationGame')
    env.reset()
    for spawn_point_num in tqdm(range(0, num_spawn_points)):
        ORIGIN_DIR = os.path.join(EXP_DIR, 'origin_{:06d}'.format(spawn_point_num))
        os.mkdir(ORIGIN_DIR)
        print("ORIGIN_DIR: ", ORIGIN_DIR)

        start = time.time()
        
        env.game.reset_game_random_spawn()
        origin_position = env.game.agent_positions[0]
        origin_image = env.game.imgs
        
        with open(os.path.join(ORIGIN_DIR, 'origin_position.npy'), 'wb') as f:
            np.save(f, origin_position)
            
        with open(os.path.join(ORIGIN_DIR, 'origin_image.npy'), 'wb') as f:
            np.save(f, origin_image)
        
        for pair_num, spawn_radii_pair in enumerate(spawn_radii_pairs):
            
            INTERVAL_DIR = os.path.join(ORIGIN_DIR, 'interval_{}_{}'.format(spawn_radii_pair[0], spawn_radii_pair[1]))
            os.mkdir(INTERVAL_DIR)
            print()
            print("INTERVAL_DIR: ", INTERVAL_DIR)
        
            img_list = []
            position_list = []
            
            print("---------------------------------")
            logging.info("---------------------------------")
            print("Pair Num: ", pair_num)
            print("Runtime: ", time.time() - start)
            logging.info("Pair num: {}".format(pair_num))
            logging.info("Runtime: {}".format(time.time() - start))
            t_0 = time.time()

            # Sample target positions
            for j in range(num_random_samples):
                position_list.append(env.game.sample_random_point_within(origin_position, spawn_radii_pair[0], spawn_radii_pair[1]))
                
            print("Drawing map.")
            logging.info("Drawing map.")
            _ = env.reset()
            env.game.draw_vis_map()
            env.game.draw_points([origin_position], colors=[[0,255,0]])
            env.game.draw_points(position_list)
            im = Image.fromarray(env.game.vis_map)
            im.save(os.path.join(INTERVAL_DIR, 'target_positions.png'))
            t_3 = time.time()

            # Save state and action for current timestep
            for starting_position in position_list:
                env.game.reset_game_spawn_at(starting_position)
                img = env.game.imgs
                img_list.append(img) 

            with open(os.path.join(INTERVAL_DIR, 'position_list.npy'), 'wb') as f:
                np.save(f, position_list)

            with open(os.path.join(INTERVAL_DIR, 'obs.npz'), 'wb') as f:
                np.savez(f, states=img_list)

