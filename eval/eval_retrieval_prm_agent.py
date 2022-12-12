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
experiment_name = None # TODO: Set name.
SAVE_DIR = None # TODO: Set path.
LOAD_DIR = None # TODO: Set path.
EXP_DIR = os.path.join(SAVE_DIR, experiment_name)

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

num_spawn_points = 100
num_random_samples = [10, 20, 30, 40, 40, 40, 40, 40, 40, 40, 10, 10, 10, 10, 10]
spawn_radii = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
spawn_radii_pairs = list(zip(spawn_radii[:-1], spawn_radii[1:]))


#-------------------------------------------------------------------------------------------------------------
# The main experiment script
#-------------------------------------------------------------------------------------------------------------

EXP_DIR = os.path.join(SAVE_DIR, experiment_name)

if __name__ == "__main__":
    
    print('Setting up experiment...')
    # Create directories
    os.mkdir(EXP_DIR)
    print("EXP_DIR: ", EXP_DIR)
    
    # Start logging
    logging.basicConfig(filename=os.path.join(EXP_DIR, experiment_name + '.log'), level=logging.DEBUG)
    logging.info(EXP_DIR)
    
    # Create the environment
    env = VizdoomEnv('SimpleExplorationGame')
    env.reset()
    
    # Create agent
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MPCModel(device, config=default_config)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    agent = MPCAgent(model, device, env, config=default_config)
    
    # Read the goal images
    spawn_goal_obss_list = []
    spawn_goal_positions_list = []
    spawn_origin_image_list = []
    spawn_origin_position_list = []
    spawn_target_steps_list = []
    for spawn_point_num in tqdm(range(num_spawn_points)):
        ORIGIN_DIR = os.path.join(LOAD_DIR, 'origin_{:06d}'.format(spawn_point_num))
        
        origin_image= np.load(os.path.join(ORIGIN_DIR, 'origin_image.npy'))
        origin_position = np.load(os.path.join(ORIGIN_DIR, 'origin_position.npy'))
        
        goal_obss_list = []
        goal_positions_list = []
        target_steps_list = []
        for pair_num, spawn_radii_pair in enumerate(spawn_radii_pairs):
            INTERVAL_DIR = os.path.join(ORIGIN_DIR, 'interval_{}_{}'.format(spawn_radii_pair[0], spawn_radii_pair[1]))
            goal_obss = np.load(os.path.join(INTERVAL_DIR, 'obs.npz'))['states']
            goal_position_list = np.load(os.path.join(INTERVAL_DIR, 'position_list.npy'))
            target_step = spawn_radii_pair[-1] // 50
            goal_obss_list.append(goal_obss)
            goal_positions_list.append(goal_position_list)
            target_steps_list.append(target_step)
            
        spawn_goal_obss_list.append(goal_obss_list)
        spawn_goal_positions_list.append(goal_positions_list)
        spawn_origin_image_list.append(origin_image)
        spawn_origin_position_list.append(origin_position)
        spawn_target_steps_list.append(target_steps_list)
    
    # Start evaluation
    print("Starting evaluation...")
    for spawn_point_num in range(0, num_spawn_points):
        print("---------------------------------")
        print("spawn_point_num: ", spawn_point_num)
        
        origin_image = spawn_origin_image_list[spawn_point_num]
        origin_position = spawn_origin_position_list[spawn_point_num]
        goal_obss_list = spawn_goal_obss_list[spawn_point_num]
        goal_positions_list = spawn_goal_positions_list[spawn_point_num]
        target_steps_list = spawn_target_steps_list[spawn_point_num]
        
        for interval_num, goal_obss in tqdm(enumerate(goal_obss_list)):
            print("interval_num: ", interval_num)
            
            spawn_radii_pair = spawn_radii_pairs[interval_num]
            goal_position_list = goal_positions_list[interval_num]
            target_step = target_steps_list[interval_num]
            start = time.time()

            for goal_num in range(goal_obss.shape[0]):
                print("goal_num: ", goal_num)
                print()
                
                RANDOM_WALK_SAVE_DIR = os.path.join(EXP_DIR, 'origin_{:06d}'.format(spawn_point_num), 'interval_{}_{}'.format(spawn_radii_pair[0], spawn_radii_pair[1]), 'goal_{}'.format(goal_num))
                OBS_SAVE_DIR = os.path.join(RANDOM_WALK_SAVE_DIR, 'obs')
                MAP_SAVE_DIR = os.path.join(RANDOM_WALK_SAVE_DIR, 'maps')
                
                os.makedirs(RANDOM_WALK_SAVE_DIR)
                print("RANDOM_WALK_SAVE_DIR: ", RANDOM_WALK_SAVE_DIR)
                os.mkdir(OBS_SAVE_DIR)
                print("OBS_SAVE_DIR: ", OBS_SAVE_DIR)
                os.mkdir(MAP_SAVE_DIR)
                print("MAP_SAVE_DIR: ", MAP_SAVE_DIR)
            
                env.game.reset_game_spawn_at(origin_position)
                goal_state = goal_obss[goal_num]
                goal_position = goal_position_list[goal_num]

                with open(os.path.join(RANDOM_WALK_SAVE_DIR, 'goal_state.npy'), 'wb') as f:
                    np.save(f, goal_state)

                with open(os.path.join(RANDOM_WALK_SAVE_DIR, 'goal_position.npy'), 'wb') as f:
                    np.save(f, goal_position)

                img_list = []
                action_list = []
                retrieval_positions_list = []
                retrieval_actions_list = []
                goal_distance_list = []
                success = np.array(0)
                success_timestep = np.array(target_step * 2)
                
                try:
                    agent.flush_action_buffer()
                    agent.set_goal(goal_state, goal_position)

                    for time_step in range(target_step * 2):
                        print("---------------------------------")
                        logging.info("---------------------------------")
                        print("Timestep: ", time_step)
                        print("Runtime: ", time.time() - start)
                        logging.info("Timestep: {}".format(time_step))
                        logging.info("Runtime: {}".format(time.time() - start))
                        t_0 = time.time()

                        # Take action
                        current_state = env.game.imgs
                        current_position = env.game.agent_positions[-1]
                        goal_distance = np.linalg.norm(current_position[:2] - goal_position)
                        goal_distance_list.append(goal_distance)

                        try:
                            retrieval_actions, retrieval_positions, action = agent.navigate_to_goal_retrieval_PRM(current_state, current_position)
                        except:
                            retrieval_actions, retrieval_positions = None, None
                            action = random.randint(0,3)
                        t_1 = time.time()

                        # Save state and action for current timestep
                        img_list.append(env.game.imgs)
                        action_list.append(action)
                        retrieval_positions_list.append(retrieval_positions)
                        retrieval_actions_list.append(retrieval_actions)
                        t_2 = time.time()

                        print("Drawing map.")
                        logging.info("Drawing map.")
                        im = Image.fromarray(cv2.circle(env.game.map, env.game.map_to_pixel(goal_position), 4, (0, 200, 0), thickness=-1))
                        im.save(os.path.join(MAP_SAVE_DIR, 'state_{:06d}_action_{}.png'.format(time_step, action)))
                        t_3 = time.time()

                        # Implement the action and get the next state
                        env.step(action)
                        t_4 = time.time()

                        print()
                        print("Time Limit: {}".format(target_step * 2))
                        logging.info("Time Limit: {}".format(target_step * 2))
                        print("Time Target: {}".format(target_step))
                        logging.info("Time Target: {}".format(target_step))
                        print("Current Time: {}".format(time_step))
                        logging.info("Current Time: {}".format(time_step))
                        print("Goal Distance: {}".format(goal_distance_list[-1]))
                        logging.info("Goal Distance: {}".format(goal_distance_list[-1]))

                        print()
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

                        if goal_distance <= 100:
                            success = np.array(1)
                            success_timestep = np.array(time_step)
                            print("!!!! SUCCESS !!!!")
                            break
                            
                except:
                    img_list = np.array(-1)
                    action_list = np.array(-1)
                    retrieval_positions_list = np.array(-1)
                    retrieval_actions_list = np.array(-1)
                    goal_distance_list = np.array(-1)
                    success = np.array(-1)
                    success_timestep = np.array(-1)

                with open(os.path.join(RANDOM_WALK_SAVE_DIR, 'position_list.npy'), 'wb') as f:
                    np.save(f, env.game.agent_positions)
                    
                with open(os.path.join(RANDOM_WALK_SAVE_DIR, 'retrieval_positions_list.npy'), 'wb') as f:
                    np.save(f, retrieval_positions_list)
                    
                with open(os.path.join(RANDOM_WALK_SAVE_DIR, 'retrieval_actions_list.npy'), 'wb') as f:
                    np.save(f, retrieval_actions_list)

                with open(os.path.join(RANDOM_WALK_SAVE_DIR, 'action_list.npy'), 'wb') as f:
                    np.save(f, action_list)

                with open(os.path.join(RANDOM_WALK_SAVE_DIR, 'goal_distance_list.npy'), 'wb') as f:
                    np.save(f, goal_distance_list)

                with open(os.path.join(RANDOM_WALK_SAVE_DIR, 'success.npy'), 'wb') as f:
                    np.save(f, success)
                    
                with open(os.path.join(RANDOM_WALK_SAVE_DIR, 'success_timestep.npy'), 'wb') as f:
                    np.save(f, success_timestep)

                with open(os.path.join(OBS_SAVE_DIR, 'obs.npz'), 'wb') as f:
                    np.savez(f, states=img_list)

