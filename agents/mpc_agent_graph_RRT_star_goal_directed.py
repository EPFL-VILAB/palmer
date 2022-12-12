import logging
import os
import sys
sys.path.append('../')
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import wandb
from .backbone_models import ResNetEncoder, resnet18
from itertools import product
import pdb
import math
from envs.vizdoom_env import VizdoomEnv
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import networkx as nx
from tqdm import tqdm
import random

    
#-----------------------------------------------------------------------------------------------            
#-----------------------------------------------------------------------------------------------
# Agent config
#-----------------------------------------------------------------------------------------------            
#-----------------------------------------------------------------------------------------------    

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
    'num_nodes_graph': 500,
    'pT_distance_threshold': 5.5,
    'latent_distance_threshold': 2,
    'overlap_pT_distance_threshold': 4,
    'overlap_latent_distance_threshold': 2,
    'overlap_retrieval_distance_threshold': 2,
    'pT_search_threshold': 4.5,
    'graph_rng_seed': 285,
    'RRT_steer_steps': 3,
}



#-----------------------------------------------------------------------------------------------            
#-----------------------------------------------------------------------------------------------
# Graph Memory
#-----------------------------------------------------------------------------------------------            
#-----------------------------------------------------------------------------------------------

class GraphMemory():

    def __init__(self, model, device, config=default_config):
        '''
        A simple optimizer that samples random actions, and chooses the best performing sequence above a consistency threshold
        
        Args:
            - optimization_objective: a function that takes as input latent_current_states (B,D) and latent_goal_states (B,D), and returns a scalar score
        '''
        self.model = model
        self.device = device
        self.config = config
        self.env = VizdoomEnv('SimpleExplorationGame')
        self.env.reset()
        
        
        # Graph Configuration
        self.GRAPH_EXP_DIR = self.config['GRAPH_EXP_DIR']
        self.OBS_DIR = os.path.join(self.GRAPH_EXP_DIR, 'obs')
        self.POS_DIR = os.path.join(self.GRAPH_EXP_DIR, 'pos')
        self.MAP_DIR = os.path.join(self.GRAPH_EXP_DIR, 'maps')
        self.LOG_DIR = os.path.join(self.GRAPH_EXP_DIR, 'logs')
        self.LATENT_DIR = os.path.join(self.GRAPH_EXP_DIR, 'latents')
        
        latent_block_path = os.path.join(self.LATENT_DIR, 'latents_block.npy')
        position_block_path = os.path.join(self.POS_DIR, 'pos_block.npy')
        self.latents_block = np.load(latent_block_path) # shape (num_random_walk_steps, D)
        self.latents_block_gpu = torch.from_numpy(self.latents_block).to(self.device) # shape (num_random_walk_steps, D)
        self.positions_block = np.load(position_block_path) # shape (num_random_walk_steps, 3)
        
        self.latent_distance_threshold = self.config['latent_distance_threshold']
        self.overlap_retrieval_distance_threshold = self.config['overlap_retrieval_distance_threshold']
        self.graph_rng_seed = self.config['graph_rng_seed']
        self.rng = np.random.default_rng(self.graph_rng_seed)
        self.RRT_steer_steps = self.config['RRT_steer_steps']
        
        self.latents_block = np.load(latent_block_path) # shape (num_random_walk_steps, D)
        self.latents_block_gpu = torch.from_numpy(self.latents_block).to(self.device) # shape (num_random_walk_steps, D)
        self.positions_block = np.load(position_block_path) # shape (num_random_walk_steps, 3)
        
        # Fields to be initialized later
        self.tree_node_walk_ids = [] # shape [num_nodes], to be expanded
        self.tree_node_parent_tree_ids = [] # shape [num_nodes], to be expanded
        self.tree_node_positions = [] # shape [num_nodes], to be expanded
        self.tree_node_latents = [] # shape [num_nodes], to be expanded
        self.tree_node_costs = [] # shape [num_nodes], to be expanded
        self.tree = None # networkX graph object to be initialized later
        
        self.goal_latent = None
        self.goal_position = None
        self.latent_distances_walk_to_goal = None
        self.latent_distance_masks_walk_to_goal = None
        
        # Fields for graph extension
        self.latent_distances_nodes_to_walk = None # shape (num_nodes, num_samples_random_walk), to be expanded
        self.latent_distance_masks_nodes_to_walk = None # shape (num_nodes, num_samples_random_walk), to be expanded
        
        self.point_latent = None # shape (1, D)
        self.point_position = None # shape (1, 3)
        
        self.latent_distances_walk_to_point = None 
        self.latent_distance_masks_walk_to_point = None 
        self.retrieval_distances_nodes_to_point = None # shape (num_new_samples, num_samples)
        
        self.latent_distances_new_point_to_walk = None 
        self.latent_distance_masks_new_point_to_walk = None
        
        self.retrieval_distances_nodes_to_new_point = None
        self.retrieval_distances_new_point_to_nodes = None
        
        self.overlap_node_tree_id = None
        
        # Fields for visualization
        self.random_point_ids = []
        self.random_point_positions = []
        self.retrieval_trajectory_positions = []
        self.overlap_retrieval_distances = []
        self.tree_node_parent_tree_ids_list = []
        self.tree_node_positions_list = []

    #-----------------------------------------------------------------------------------------------
    # Tree initialization
    #-----------------------------------------------------------------------------------------------
    
    def compute_latent_distances_to_walk(self, point_latent):
        
        # Compute the latent distances and masks to the tree root
        point_latent_torch = torch.from_numpy(point_latent).to(self.device) # shape (1, D)
        latents_block_expanded_torch = self.latents_block_gpu # shape (num_samples_random_walk, D)
        latent_distances_nodes_to_walk = torch.norm(latents_block_expanded_torch - point_latent_torch, dim=-1).unsqueeze(0).cpu().numpy() # shape (1, num_samples_random_walk)
        latent_distance_masks_nodes_to_walk = (latent_distances_nodes_to_walk < self.latent_distance_threshold) # shape (1, num_samples_random_walk)
        
        return latent_distances_nodes_to_walk, latent_distance_masks_nodes_to_walk
    
    def initialize_tree(self, root_latent, root_position):
        '''
        - num_nodes: int
        '''

        self.tree_node_walk_ids.append(0.5) # Assign an invalid index as the root
        self.tree_node_parent_tree_ids.append(0.5) # Assign an invalid index as the root
        self.tree_node_positions.append(root_position) 
        self.tree_node_latents.append(root_latent)
        self.tree_node_costs.append(0)
        
        # Compute the latent distances and masks to the tree root
        self.latent_distances_nodes_to_walk, self.latent_distance_masks_nodes_to_walk =  self.compute_latent_distances_to_walk(root_latent)
    
    def register_goal(self, goal_latent, goal_position):
        '''
        - num_nodes: int
        '''

        self.goal_latent = goal_latent
        self.goal_position = goal_position
        
        # Compute the latent distances and masks to the tree root
        self.latent_distances_walk_to_goal, self.latent_distance_masks_walk_to_goal =  self.compute_latent_distances_to_walk(goal_latent)
    
    #-----------------------------------------------------------------------------------------------
    # Sample Free
    #-----------------------------------------------------------------------------------------------
    
    def sample_free_point(self):
        '''
        - num_nodes: int
        '''

        random_point_id = self.rng.integers(0, self.positions_block.shape[0])
        random_point_position = self.positions_block[random_point_id]
        random_point_latent = self.latents_block[random_point_id]
        
        return random_point_id, random_point_position, random_point_latent
    
    #-----------------------------------------------------------------------------------------------
    # Find the nearest point
    #-----------------------------------------------------------------------------------------------
                
    def get_retrieval_distance_nodes_to_point(self, sample_id):
    
        mask_current = self.latent_distance_masks_nodes_to_walk[sample_id]
        indices_current = np.array(mask_current.nonzero())

        mask_target = self.latent_distance_masks_walk_to_point.squeeze()
        indices_target = np.array(mask_target.nonzero())

        time_steps_inbetween = indices_target.transpose(1,0) - indices_current
        
        # Only take trajectories that go current -> target, not the inverse direction
        time_steps_inbetween[time_steps_inbetween < 0 ] = self.latents_block.shape[0] * 5 
        
        min_distance = time_steps_inbetween.min()

        return min_distance
    
    def find_nearest_node(self, random_point_latent):
        '''
        - goal_latent: shape (1, D)
        '''
        
        num_tree_nodes = len(self.tree_node_walk_ids)
        
        # Compute the latent distances and masks to the sampled point
        self.latent_distances_walk_to_point, self.latent_distance_masks_walk_to_point = self.compute_latent_distances_to_walk(random_point_latent) # shape (1, num_samples_random_walk)
        
        # Compute retrieval distances to the sampled point
        self.retrieval_distances_nodes_to_point = np.ones(num_tree_nodes) * -1 # shape (N)
        for node_id in range(num_tree_nodes):
            self.retrieval_distances_nodes_to_point[node_id] = self.get_retrieval_distance_nodes_to_point(node_id) # shape (num_nodes)
            
        # Find the closest tree node to the point, and steer towards random point
        nearest_node_tree_id = self.retrieval_distances_nodes_to_point.argmin()
        nearest_node_walk_id = self.tree_node_walk_ids[nearest_node_tree_id]
        nearest_node_latent = self.tree_node_latents[nearest_node_tree_id]
        nearest_node_position = self.tree_node_positions[nearest_node_tree_id]
        
        return nearest_node_tree_id, nearest_node_walk_id, nearest_node_latent, nearest_node_position
                
    #-----------------------------------------------------------------------------------------------
    # Steer towards random point
    #-----------------------------------------------------------------------------------------------
     
    def get_retrieval_trajectory_node_to_point(self, node_tree_id):
        
        mask_current = self.latent_distance_masks_nodes_to_walk[node_tree_id]
        indices_current = np.array(mask_current.nonzero())
        
        mask_target = self.latent_distance_masks_walk_to_point.squeeze()
        indices_target = np.array(mask_target.nonzero())

        time_steps_inbetween = indices_target.transpose(1,0) - indices_current
        
        # Only take trajectories that go current -> target, not the inverse direction
        time_steps_inbetween[time_steps_inbetween < 0 ] = self.latents_block.shape[0] * 5 

        min_distance = time_steps_inbetween.min()

        argmin_distance = (time_steps_inbetween == min_distance).nonzero()
        
        id = random.randint(0,len(argmin_distance[1]) - 1)
        current_id = indices_current.squeeze()[argmin_distance[1]][id]
        target_id = indices_target.squeeze()[argmin_distance[0]][id]
        sorted_idx = sorted([target_id.item(), current_id.item()])
        retrieval_trajectory = list(range(sorted_idx[0], sorted_idx[1] + 1))

        return retrieval_trajectory
    
    def get_retrieval_distance_new_point_to_goal(self):
    
        mask_current = self.latent_distance_masks_new_point_to_walk.squeeze()
        indices_current = np.array(mask_current.nonzero())

        mask_target = self.latent_distance_masks_walk_to_goal.squeeze()
        indices_target = np.array(mask_target.nonzero())

        time_steps_inbetween = indices_target.transpose(1,0) - indices_current
        
        # Only take trajectories that go current -> target, not the inverse direction
        time_steps_inbetween[time_steps_inbetween < 0 ] = self.latents_block.shape[0] * 5 
        
        min_distance = time_steps_inbetween.min()

        return min_distance
    
    def steer_nearest_node_to_point(self, nearest_node_tree_id, nearest_node_walk_id):
        
        retrieval_trajectory = self.get_retrieval_trajectory_node_to_point(nearest_node_tree_id)
        self.retrieval_trajectory_positions.append(self.positions_block[retrieval_trajectory])
        cut_off = min(len(retrieval_trajectory), self.RRT_steer_steps + 1)
            
        retrieval_trajectory_steer_segment = retrieval_trajectory[:cut_off]
        retrieval_trajectory_steer_segment_latents = self.latents_block[retrieval_trajectory_steer_segment]
        overlap_latent_distances = torch.norm(torch.from_numpy(self.goal_latent) - torch.from_numpy(retrieval_trajectory_steer_segment_latents), dim=-1)
        
        new_point_walk_id = retrieval_trajectory_steer_segment[overlap_latent_distances.argmin().item()]
        new_point_position = self.positions_block[new_point_walk_id]
        new_point_latent = self.latents_block[new_point_walk_id]
        line_cost = overlap_latent_distances.argmin().item() + 1
        
        # Expand the latent distances and masks
        self.latent_distances_new_point_to_walk, self.latent_distance_masks_new_point_to_walk =  self.compute_latent_distances_to_walk(new_point_latent)
        self.latent_distances_nodes_to_walk = np.concatenate([self.latent_distances_nodes_to_walk, self.latent_distances_new_point_to_walk])
        self.latent_distance_masks_nodes_to_walk = np.concatenate([self.latent_distance_masks_nodes_to_walk, self.latent_distance_masks_new_point_to_walk])
        
        self.tree_node_walk_ids.append(new_point_walk_id) # Assign an invalid index as the root
        self.tree_node_parent_tree_ids.append(nearest_node_tree_id)
        self.tree_node_positions.append(new_point_position) 
        self.tree_node_latents.append(new_point_latent)
        self.tree_node_costs.append(line_cost + self.tree_node_costs[nearest_node_tree_id])
        
        # Check overlap
        retrieval_distance_new_point_to_goal = self.get_retrieval_distance_new_point_to_goal()
        overlap_flag = retrieval_distance_new_point_to_goal <= self.overlap_retrieval_distance_threshold
        if overlap_flag and self.overlap_node_tree_id is None:
            self.overlap_node_tree_id = len(self.tree_node_walk_ids) - 1
        
        return overlap_flag, retrieval_distance_new_point_to_goal, retrieval_trajectory
    
    #-----------------------------------------------------------------------------------------------
    # Rewire the neighborhood of the newly added point
    #-----------------------------------------------------------------------------------------------
    
    def get_retrieval_distance_nodes_to_new_point(self, sample_id):
    
        mask_current = self.latent_distance_masks_nodes_to_walk[sample_id]
        indices_current = np.array(mask_current.nonzero())

        mask_target = self.latent_distance_masks_new_point_to_walk.squeeze()
        indices_target = np.array(mask_target.nonzero())

        time_steps_inbetween = indices_target.transpose(1,0) - indices_current
        
        # Only take trajectories that go current -> target, not the inverse direction
        time_steps_inbetween[time_steps_inbetween < 0 ] = self.latents_block.shape[0] * 5 
        
        min_distance = time_steps_inbetween.min()

        return min_distance
    
    def get_retrieval_distance_new_point_to_nodes(self, sample_id):
    
        mask_current = self.latent_distance_masks_new_point_to_walk.squeeze()
        indices_current = np.array(mask_current.nonzero())

        mask_target = self.latent_distance_masks_nodes_to_walk[sample_id]
        indices_target = np.array(mask_target.nonzero())

        time_steps_inbetween = indices_target.transpose(1,0) - indices_current
        
        # Only take trajectories that go current -> target, not the inverse direction
        time_steps_inbetween[time_steps_inbetween < 0 ] = self.latents_block.shape[0] * 5 
        
        min_distance = time_steps_inbetween.min()

        return min_distance
    
    def rewire_new_point(self):
        
        # Compute retrieval distances to and from the newly added point
        num_tree_nodes = len(self.tree_node_walk_ids) - 1 # Don't count the new point
        self.retrieval_distances_nodes_to_new_point = np.ones(num_tree_nodes) * -1 # shape (num_nodes)
        self.retrieval_distances_new_point_to_nodes = np.ones(num_tree_nodes) * -1 # shape (num_nodes)
        for node_id in range(num_tree_nodes):
            self.retrieval_distances_nodes_to_new_point[node_id] = self.get_retrieval_distance_nodes_to_new_point(node_id) # shape (num_nodes)
            self.retrieval_distances_new_point_to_nodes[node_id] = self.get_retrieval_distance_new_point_to_nodes(node_id) # shape (num_nodes)
            
        # Find the neigborhood of near points
        near_mask = (self.retrieval_distances_nodes_to_new_point < self.RRT_steer_steps + 1) * (self.retrieval_distances_new_point_to_nodes < self.RRT_steer_steps + 1)
        near_indices = near_mask.nonzero()[0]
        
        # Rewire the edge towards the new point
        cost_min = self.tree_node_costs[-1]
        parent_tree_id_min = self.tree_node_parent_tree_ids[-1]
        for near_index in near_indices:
            near_index = near_index.item()
            rewired_cost = self.retrieval_distances_nodes_to_new_point[near_index] + self.tree_node_costs[near_index] + 1
            if rewired_cost < cost_min:
                cost_min = rewired_cost
                parent_tree_id_min = near_index
                
        self.tree_node_parent_tree_ids[-1] = parent_tree_id_min
        self.tree_node_costs[-1] = cost_min
        
        # Rewire the edges from the new point
        for near_index in near_indices:
            near_index = near_index.item()
            rewired_cost = self.retrieval_distances_new_point_to_nodes[near_index] + self.tree_node_costs[-1] + 1
            if rewired_cost < self.tree_node_costs[near_index]:
                self.tree_node_costs[near_index] = rewired_cost
                self.tree_node_parent_tree_ids[near_index] = len(self.tree_node_walk_ids) - 1 # Index of the new point
    
    #-----------------------------------------------------------------------------------------------
    # Build RRT
    #-----------------------------------------------------------------------------------------------
    
    def build_RRT(self, num_iterations, root_latent, root_position, goal_latent, goal_position, ignore_overlap=False):
        
        self.initialize_tree(root_latent, root_position)
        self.register_goal(goal_latent, goal_position)
        
        for iter_num in tqdm(range(num_iterations)):
            random_point_id, random_point_position, random_point_latent = self.sample_free_point()
            
            # Fields for visualization
            self.random_point_ids.append(random_point_id)
            self.random_point_positions.append(random_point_position)
            self.tree_node_parent_tree_ids_list.append(self.tree_node_parent_tree_ids.copy())
            self.tree_node_positions_list.append(self.tree_node_positions.copy())
            
            nearest_node_tree_id, nearest_node_walk_id, nearest_node_latent, nearest_node_position = self.find_nearest_node(random_point_latent)
            overlap_flag, overlap_retrieval_distance, retrieval_trajectory = self.steer_nearest_node_to_point(nearest_node_tree_id, nearest_node_walk_id)
            
            self.tree_node_parent_tree_ids_list.append(self.tree_node_parent_tree_ids.copy())
            self.tree_node_positions_list.append(self.tree_node_positions.copy())
            self.retrieval_trajectory_positions.append(self.positions_block[retrieval_trajectory])
            self.overlap_retrieval_distances.append(overlap_retrieval_distance)
            
            self.rewire_new_point()
            
            if overlap_flag and not ignore_overlap: break
        
    def flush_graph(self):
        
        # Fields to be initialized later
        self.tree_node_walk_ids = [] # shape [num_nodes], to be expanded
        self.tree_node_parent_tree_ids = [] # shape [num_nodes], to be expanded
        self.tree_node_positions = [] # shape [num_nodes], to be expanded
        self.tree_node_latents = [] # shape [num_nodes], to be expanded
        self.tree_node_costs = [] # shape [num_nodes], to be expanded
        self.tree = None # networkX graph object to be initialized later
        
        self.goal_latent = None
        self.goal_position = None
        self.latent_distances_walk_to_goal = None
        self.latent_distance_masks_walk_to_goal = None
        
        # Fields for graph extension
        self.latent_distances_nodes_to_walk = None # shape (num_nodes, num_samples_random_walk), to be expanded
        self.latent_distance_masks_nodes_to_walk = None # shape (num_nodes, num_samples_random_walk), to be expanded
        
        self.point_latent = None # shape (1, D)
        self.point_position = None # shape (1, 3)
        
        self.latent_distances_walk_to_point = None 
        self.latent_distance_masks_walk_to_point = None 
        self.retrieval_distances_nodes_to_point = None # shape (num_new_samples, num_samples)
        
        self.latent_distances_new_point_to_walk = None 
        self.latent_distance_masks_new_point_to_walk = None
        
        self.retrieval_distances_nodes_to_new_point = None
        self.retrieval_distances_new_point_to_nodes = None
        
        self.overlap_node_tree_id = None
        
        # Fields for visualization
        self.random_point_ids = []
        self.random_point_positions = []
        self.retrieval_trajectory_positions = []
        self.overlap_retrieval_distances = []
        self.tree_node_parent_tree_ids_list = []
        self.tree_node_positions_list = []
        
    def render_RRT_graph(self):

        self.env.game.draw_vis_map()
        self.env.game.draw_RRT_graph(self.goal_position, self.tree_node_positions, self.tree_node_parent_tree_ids)
        figure(figsize=(10, 10))
        plt.imshow(self.env.game.vis_map)
        
    def render_RRT_graph_animated(self):

        self.env.game.draw_vis_map()
        frames = self.env.game.draw_RRT_star_graph_animated(self.goal_position, self.tree_node_positions_list, self.tree_node_parent_tree_ids_list, self.random_point_positions, self.retrieval_trajectory_positions)
        return frames
    
    #-----------------------------------------------------------------------------------------------
    # Path Stitching
    #-----------------------------------------------------------------------------------------------
        
    def get_RRT_trajectory(self):
        
        RRT_trajectory_walk_ids = []
        RRT_trajectory_positions = []
        RRT_trajectory_walk_ids = [self.tree_node_walk_ids[self.overlap_node_tree_id]] + RRT_trajectory_walk_ids
        RRT_trajectory_positions = [self.tree_node_positions[self.overlap_node_tree_id]] + RRT_trajectory_positions
        parent_id = self.tree_node_parent_tree_ids[self.overlap_node_tree_id]
        
        while parent_id != 0.5:
            RRT_trajectory_walk_ids = [self.tree_node_walk_ids[parent_id]] + RRT_trajectory_walk_ids
            RRT_trajectory_positions = [self.tree_node_positions[parent_id]] + RRT_trajectory_positions
            parent_id = self.tree_node_parent_tree_ids[parent_id]
            
        RRT_trajectory_walk_ids = ['start'] + RRT_trajectory_walk_ids + ['goal']
        RRT_trajectory_positions = [self.tree_node_positions[0]] + RRT_trajectory_positions + [self.goal_position]

        return RRT_trajectory_walk_ids, RRT_trajectory_positions
    
    def render_RRT_trajectory(self):
        
        _, RRT_trajectory_positions = self.get_RRT_trajectory()
        self.env.game.draw_vis_map()
        self.env.game.draw_trajectory(RRT_trajectory_positions)
        figure(figsize=(10, 10))
        plt.imshow(self.env.game.vis_map)


        
#-----------------------------------------------------------------------------------------------            
#-----------------------------------------------------------------------------------------------
# RandomShootingOptimizer with a consistency constraint
#-----------------------------------------------------------------------------------------------            
#-----------------------------------------------------------------------------------------------

class RandomShootingOptimizer():

    def __init__(self, model, device, config=default_config):
        '''
        A simple optimizer that samples random actions, and chooses the best performing sequence above a consistency threshold
        
        Args:
            - optimization_objective: a function that takes as input latent_current_states (B,D) and latent_goal_states (B,D), and returns a scalar score
        '''
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.num_actions = self.config['num_actions']
        
        # MPC Configuration
        self.optimization_objective = self.average_time_objective
        self.max_num_mpc_samples = self.config['max_num_mpc_samples']
        self.allT_or_lastT = self.config['allT_or_lastT']
        self.cons_or_nocons = self.config['cons_or_nocons']
        self.allH_or_lastH = self.config['allH_or_lastH']
        self.consistency_threshold = self.config['consistency_threshold']
        
        self.action_buffer = None 
        
        
    #-----------------------------------------------------------------------------------------------
    # Optimization Objectives
    #-----------------------------------------------------------------------------------------------
    
    def average_time_objective(self, latent_current_states, latent_goal_states):
        '''
        - latent_current_states, latent_goal_states: numpy arrays of shape (B,D)
        '''

        with torch.no_grad():
            joint_latent = torch.cat([latent_current_states, latent_goal_states], dim=-1) # shape (B,2D)
            time_logits = self.model.time_logits(joint_latent) # shape (B,self.num_classes_time)
            time_probs = torch.nn.functional.softmax(time_logits, dim=-1) # shape (B,self.num_classes_time)
            time_values = torch.arange(time_probs.shape[-1], dtype=torch.float).unsqueeze(0).to(self.device) # shape (1, self.num_classes_time)
            time_average = torch.sum(time_values * time_probs, dim=-1).squeeze() # shape (B,)

        return time_average
    
    def evaluate_objective(self, latent_states, latent_goal_state):
        '''
        Evaluates the optimization objective on N trajectories.
        
        Args:
            - latent_states: shape(self.horizon, self.num_samples, D)
            - goal_state: shape(1, 12, H, W)
        '''
        
        horizon = latent_states.shape[0] - 1
        latent_states = latent_states[1:,:,:] # shape (self.horizon, self.num_samples, D)
        
        #-----------------------------------------------------------------------------------------------
        # *** All T or Last T
        #-----------------------------------------------------------------------------------------------
    
        if self.allT_or_lastT == 'lastT': 
            latent_goal_states = torch.cat(self.num_samples*[latent_goal_state]) # shape (self.num_samples,D)
            latent_last_states = latent_states[-1, :, :] # shape (self.num_samples,D)
            scores = self.optimization_objective(latent_last_states, latent_goal_states) # shape (self.num_samples)
                
        elif self.allT_or_lastT == 'allT':
            latent_goal_states = torch.cat(self.num_samples * horizon * [latent_goal_state]) # shape (self.num_samples * self.horizon, D)
            latent_states = latent_states.view(horizon * self.num_samples, *latent_states.shape[2:]) # shape (self.num_samples * self.horizon, D)
            scores = self.optimization_objective(latent_states, latent_goal_states).view(horizon, self.num_samples).mean(dim=0) # shape (self.num_samples)

        return scores 

    
    #-----------------------------------------------------------------------------------------------
    # Latent rollout related functions
    #-----------------------------------------------------------------------------------------------
    
    def simulate_one_step(self, latent_current_state, one_hot_action):
        '''
        Simulates a forward rollout.
        
        Args:
            - latent_current_states: shape(self.num_samples, D)
            - one_hot_actions: shape(self.num_samples, self.num_actions)
        '''
        
        action_state_latent = torch.cat([latent_current_state, one_hot_action], dim=-1) # shape (self.num_samples,D+K)
        latent_next_state_delta_preds = self.model.forward_model_delta(action_state_latent) # shape (self.num_samples,D)
        latent_next_state_pred = latent_next_state_delta_preds + latent_current_state # shape (self.num_samples,D)
        
        return latent_next_state_pred
        
    def simulate_rollout(self, latent_current_states, one_hot_actions, horizon=None):
        '''
        Simulates a forward rollout.
        
        Args:
            - latent_current_states: shape(self.num_samples, D)
            - one_hot_actions: shape(self.horizon, self.num_samples, self.num_actions)
        '''
        if horizon == None: horizon = self.horizon
        latent_states = [latent_current_states]
        for time_step in range(horizon):
            action_state_latent = torch.cat([latent_states[-1], one_hot_actions[time_step]], dim=-1) # shape (self.num_samples,D+K)
            latent_next_state_delta_preds = self.model.forward_model_delta(action_state_latent) # shape (self.num_samples,D)
            latent_next_state_pred = latent_next_state_delta_preds + latent_states[-1] # shape (self.num_samples,D)
            latent_states.append(latent_next_state_pred)

        latent_states = torch.stack(latent_states)
                
        return latent_states
    
    
    #-----------------------------------------------------------------------------------------------
    # Cycle consistency related functions
    #-----------------------------------------------------------------------------------------------
    
    def evaluate_consistency_score(self, latent_states, forward_action_sequences, horizon=None):
        '''
        Computes the agreement score on the rollouts.
        
        Args:
            - latent_states: shape(self.horizon, self.num_samples, D)
            - forward_action_sequences: shape(self.horizon, self.num_samples) non-one-hot version of one-hot actions
        '''
        horizon = latent_states.shape[0] - 1
        log_prob_sequences = []
        for time_step in range(horizon):
            latent_current_states = latent_states[time_step] # shape (self.num_samples,D)
            latent_goal_states = latent_states[time_step + 1] # shape (self.num_samples,D)
            forward_actions = forward_action_sequences[time_step] # shape (self.num_samples)

            joint_latent = torch.cat([latent_current_states, latent_goal_states], dim=-1) # shape (self.num_samples,2D)
            action_logits = self.model.action_logits(joint_latent) # shape (self.num_samples, self.num_actions)
            action_log_probs = F.log_softmax(action_logits, dim=-1) # shape (self.num_samples, self.num_actions)
            forward_action_log_probs = torch.gather(action_log_probs, -1, forward_actions.unsqueeze(-1)).squeeze() # shape (self.num_samples)
            
            log_prob_sequences.append(forward_action_log_probs)
            
        log_prob_sequences = torch.stack(log_prob_sequences).transpose(1, 0) # shape (self.num_samples, self.horizon)
        consistency_scores = log_prob_sequences.min(dim=-1)[0] # shape (self.num_samples)

        return consistency_scores
    
    def evaluate_consistency_score_one_step(self, latent_current_state, latent_next_state_pred, action):
        '''
        Simulates a forward rollout.
        
        Args:
            - latent_current_states: shape(self.num_samples, D)
            - actions: shape(self.num_actions)
        '''
        
        joint_latent = torch.cat([latent_current_state, latent_next_state_pred], dim=-1) # shape (self.num_samples,2D)
        action_logits = self.model.action_logits(joint_latent) # shape (self.num_samples, self.num_actions)
        action_log_probs = F.log_softmax(action_logits, dim=-1) # shape (self.num_samples, self.num_actions)
        forward_action_log_probs = torch.gather(action_log_probs, -1, action.view(-1,1)).squeeze() # shape (self.num_samples)
        
        return forward_action_log_probs
    
    def evaluate_consistency_violation_one_step(self, latent_current_state, latent_next_state_pred, action):
        '''
        Simulates a forward rollout.
        
        Args:
            - latent_current_states: shape(self.num_samples, D)
            - actions: shape(self.num_actions)
        '''
        
        joint_latent = torch.cat([latent_current_state, latent_next_state_pred], dim=-1) # shape (self.num_samples,2D)
        action_logits = self.model.action_logits(joint_latent) # shape (self.num_samples, self.num_actions)
        action_log_probs = F.log_softmax(action_logits, dim=-1) # shape (self.num_samples, self.num_actions)
        violation_flags = (torch.argmax(action_log_probs, dim=-1) != action.view(-1,1)).squeeze() # shape (self.num_samples)
        
        return violation_flags
    
    
    #-----------------------------------------------------------------------------------------------
    # Main act method to do MPC
    #-----------------------------------------------------------------------------------------------
    
    def get_action(self, latent_current_state, latent_goal_state, horizon=None):
        '''
        - current_states, goal_states: numpy arrays of shape (4,3,H,W)
        '''
        
        with torch.no_grad():
            
            #-----------------------------------------------------------------------------------------------
            # *** Last H vs All H
            #----------------------------------------------------------------------------------------------- 
            
            if self.allH_or_lastH == 'lastH':
                if self.action_buffer is None: 
                    if (self.num_actions ** horizon * horizon) < self.max_num_mpc_samples:
                        random_actions_list = list(product(range(self.num_actions), repeat=horizon)) # shape (self.num_samples, horizon)
                    else: 
                        rng = np.random.default_rng()
                        random_actions_list = rng.integers(0, self.num_actions, size=(self.max_num_mpc_samples // horizon, horizon)) # shape (self.num_samples, horizon)

                    forward_action_sequences = torch.tensor(random_actions_list).transpose(1, 0).to(self.device) # shape (horizon, self.num_samples)
            
                elif horizon <= len(self.action_buffer):
                    previous_actions = self.action_buffer[:horizon-1]
                    forward_action_sequences = torch.tensor([previous_actions.copy() + [i] for i in range(self.num_actions)]).transpose(1, 0).to(self.device) # shape (horizon, self.num_samples)

                else:
                    num_actions_to_optimize = horizon - len(self.action_buffer)
                    previous_actions = self.action_buffer
                    if (self.num_actions ** num_actions_to_optimize * num_actions_to_optimize) < self.max_num_mpc_samples:
                        new_actions_list = list(product(range(self.num_actions), repeat=num_actions_to_optimize)) # shape (self.num_samples, horizon)
                    else: 
                        rng = np.random.default_rng()
                        new_actions_list = rng.integers(0, self.num_actions, size=(self.max_num_mpc_samples // num_actions_to_optimize, num_actions_to_optimize)) # shape (self.num_samples, horizon)
                        
                    forward_action_sequences = torch.tensor([previous_actions + list(new_actions) for new_actions in new_actions_list]).transpose(1, 0).to(self.device) # shape (horizon, self.num_samples)
                
            elif self.allH_or_lastH == 'allH':
                forward_action_sequences = torch.tensor(list(product(range(self.num_actions), repeat=horizon))).transpose(1, 0).to(self.device) # shape (horizon, self.num_samples)
                
            self.num_samples = forward_action_sequences.shape[-1]
            one_hot_actions = F.one_hot(forward_action_sequences, num_classes=self.num_actions) # shape (horizon, self.num_samples, self.num_actions)
            one_hot_actions = one_hot_actions.to(self.device)

            latent_current_states = torch.cat(forward_action_sequences.shape[-1]*[latent_current_state]) # shape (self.num_samples,D)

            # Simulate a forward rollout
            latent_states = self.simulate_rollout(latent_current_states, one_hot_actions, horizon) # shape (horizon + 1, self.num_samples, D)

            # Consistency based filtering
            consistency_scores = self.evaluate_consistency_score(latent_states, forward_action_sequences) # shape (self.num_samples)
            consistency_mask = consistency_scores >= self.consistency_threshold
            consistency_ratio = (consistency_mask.sum() / self.num_samples).item()

            #-----------------------------------------------------------------------------------------------
            # *** Cons vs NoCons
            #-----------------------------------------------------------------------------------------------
            
            # Compute the objective on resulting trajectories
            obj_scores = self.evaluate_objective(latent_states, latent_goal_state) # shape (self.num_samples)
            if self.cons_or_nocons == 'cons': obj_scores[~ consistency_mask] = float("inf")

            forward_action_sequences = forward_action_sequences.permute(1,0) # shape (self.num_samples, horizon)
            obj_scores, forward_action_sequences = list(zip(*sorted(zip(obj_scores.cpu().numpy(), forward_action_sequences.cpu().numpy()), key=lambda x: x[0])))
            obj_scores = np.array(obj_scores)
            forward_action_sequences = np.array(forward_action_sequences)

            best_sequence = list(forward_action_sequences[0])
            self.action_buffer = best_sequence
            
        return self.action_buffer.pop(0), forward_action_sequences, obj_scores, consistency_ratio
    

    
#-----------------------------------------------------------------------------------------------            
#-----------------------------------------------------------------------------------------------
# Agent
#-----------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------

class MPCAgent():

    def __init__(self, model, device, env, config=default_config):
        '''
        A simple agent that acts according to the predictions of a classification model
        '''
        self.model = model.to(device)
        self.device = device
        self.env = env
        self.config = config
        
        self.horizon = self.config['horizon']
        self.statH_or_dynH = self.config['statH_or_dynH']
        self.mpc_optimizer = RandomShootingOptimizer(self.model, self.device, self.config)
        self.graph_memory = GraphMemory(self.model, self.device, config=self.config)
        self.num_actions = self.model.num_classes_action
     
    #-----------------------------------------------------------------------------------------------
    # Main graph navigation methods
    #-----------------------------------------------------------------------------------------------
    
    def set_goal(self, goal_state, goal_position):
        '''
        - goal_state: an image of shape(1, 12, H, W)
        - goal_position: shape(3,)
        '''
        
        with torch.no_grad():
            
            goal_state = (torch.from_numpy(goal_state).view(1, -1, *goal_state.shape[2:]) / 255).float().to(self.device) # normalization, shape(1, 12, H, W)
            latent_goal_state = self.model.get_latent(goal_state).cpu().numpy()
            self.graph_memory.insert_goal_node(latent_goal_state, goal_position)
    
    def navigate_to_goal(self, current_state, current_position):
        '''
        - current_state: an image of shape(1, 12, H, W)
        - current_position: shape(3,)
        '''
        
        with torch.no_grad():
            
            mpc_action, forward_action_sequences, obj_scores, consistency_ratio = None, None, None, None

            # Insert start node
            current_state = (torch.from_numpy(current_state).view(1, -1, *current_state.shape[2:]) / 255).float().to(self.device) # normalization, shape(1, 12, H, W)
            latent_current_state = self.model.get_latent(current_state).cpu().numpy()
            self.graph_memory.insert_start_node(latent_current_state, current_position)

            # Check if the goal is reached
            success_flag, distances = self.graph_memory.check_overlap('start', 'goal')
            # if success_flag:
            #     return success_flag, distances, mpc_action, forward_action_sequences, obj_scores, consistency_ratio

            # Get the shortest path and stitch it from memory
            path = self.graph_memory.shortest_path(self.graph_memory.filtered_graph_expanded, 'start', 'goal')
            stitched_path = self.graph_memory.stitch_path(path)

            # Find the furthest reachable state in the stitched path
            stitched_path_latents = self.graph_memory.latents_block[stitched_path[1:-1]]
            stitched_path_latents = np.concatenate([stitched_path_latents, self.graph_memory.goal_latent])
            stitched_path_pT = self.average_time_latent(torch.cat(stitched_path_latents.shape[0]*[torch.from_numpy(latent_current_state)]), torch.from_numpy(stitched_path_latents))
            arg_max_pT = (stitched_path_pT < self.graph_memory.pT_search_threshold).nonzero()[0].max()
            max_pT = stitched_path_pT[arg_max_pT]
            latent_target_state_pT =  stitched_path_latents[arg_max_pT]

            if self.statH_or_dynH == 'dynH':
                horizon = max(round(max_pT), 1)

            elif self.statH_or_dynH == 'statH':
                horizon = self.horizon

            latent_current_state = torch.from_numpy(latent_current_state).to(self.device)
            latent_target_state_pT = torch.from_numpy(latent_target_state_pT).unsqueeze(0).to(self.device)
            mpc_action, forward_action_sequences, obj_scores, consistency_ratio = self.mpc_optimizer.get_action(latent_current_state, latent_target_state_pT, horizon)
            
        return success_flag, distances, mpc_action, forward_action_sequences, obj_scores, consistency_ratio
    
    def navigate_to_goal_inv(self, current_state, current_position):
        '''
        - current_state: an image of shape(1, 12, H, W)
        - current_position: shape(3,)
        '''
        
        with torch.no_grad():
            
            mpc_action, forward_action_sequences, obj_scores, consistency_ratio = None, None, None, None

            # Insert start node
            current_state = (torch.from_numpy(current_state).view(1, -1, *current_state.shape[2:]) / 255).float().to(self.device) # normalization, shape(1, 12, H, W)
            latent_current_state = self.model.get_latent(current_state).cpu().numpy()
            self.graph_memory.insert_start_node(latent_current_state, current_position)

            # Check if the goal is reached
            success_flag, distances = self.graph_memory.check_overlap('start', 'goal')
            # if success_flag:
            #     return success_flag, distances, mpc_action, forward_action_sequences, obj_scores, consistency_ratio

            # Get the shortest path and stitch it from memory
            path = self.graph_memory.shortest_path(self.graph_memory.filtered_graph_expanded, 'start', 'goal')
            stitched_path = self.graph_memory.stitch_path(path)

            # Find the furthest reachable state in the stitched path
            stitched_path_latents = self.graph_memory.latents_block[stitched_path[1:-1]]
            stitched_path_latents = np.concatenate([stitched_path_latents, self.graph_memory.goal_latent])
            stitched_path_pT = self.average_time_latent(torch.cat(stitched_path_latents.shape[0]*[torch.from_numpy(latent_current_state)]), torch.from_numpy(stitched_path_latents))
            arg_max_pT = (stitched_path_pT < self.graph_memory.pT_search_threshold).nonzero()[0].max()
            max_pT = stitched_path_pT[arg_max_pT]
            latent_target_state_pT =  stitched_path_latents[1]

            latent_current_state = torch.from_numpy(latent_current_state).to(self.device)
            latent_target_state_pT = torch.from_numpy(latent_target_state_pT).unsqueeze(0).to(self.device)
            mpc_action, _ = self.inverse_model_act_latent(latent_current_state, latent_target_state_pT)
            
        return success_flag, distances, mpc_action, forward_action_sequences, obj_scores, consistency_ratio
    
    #-----------------------------------------------------------------------------------------------
    # Main act() methods
    #-----------------------------------------------------------------------------------------------
    
    def act(self, current_state, target_state):
        '''
        - current_states, goal_states: numpy arrays of shape (4,3,H,W)
        '''
        #-----------------------------------------------------------------------------------------------
        # *** Stat H vs Dyn H
        #-----------------------------------------------------------------------------------------------
        
        if self.statH_or_dynH == 'dynH':
            horizon = max(round(self.average_time(current_state, target_state)), 1)
            
        elif self.statH_or_dynH == 'statH':
            horizon = self.horizon
            
        with torch.no_grad():
            current_state = (torch.from_numpy(current_state).view(1, -1, *current_state.shape[2:]) / 255).float().to(self.device)  # normalization, shape(1, 12, H, W)
            target_state = (torch.from_numpy(target_state).view(1, -1, *target_state.shape[2:]) / 255).float().to(self.device) # normalization, shape(1, 12, H, W)
            latent_current_state = self.model.get_latent(current_state)
            latent_target_state = self.model.get_latent(target_state)
            mpc_action, forward_action_sequences, obj_scores, consistency_ratio = self.mpc_optimizer.get_action(latent_current_state, latent_target_state, horizon)
            
        return mpc_action, forward_action_sequences, obj_scores, consistency_ratio
    
    def act_latent(self, latent_current_state, latent_target_state):
        '''
        - current_states, goal_states: numpy arrays of shape (4,3,H,W)
        '''
        #-----------------------------------------------------------------------------------------------
        # *** Stat H vs Dyn H
        #-----------------------------------------------------------------------------------------------
        
        if self.statH_or_dynH == 'dynH':
            horizon = max(round(self.average_time_latent(latent_current_state, latent_target_state)), 1)
            
        elif self.statH_or_dynH == 'statH':
            horizon = self.horizon
            
        with torch.no_grad():
            mpc_action, forward_action_sequences, obj_scores, consistency_ratio = self.mpc_optimizer.get_action(latent_current_state, latent_target_state, horizon)
            
        return mpc_action, forward_action_sequences, obj_scores, consistency_ratio
    
    def inverse_model_act(self, current_state, target_state, return_probs=False, return_logits=False):
        '''
        - current_states, goal_states: numpy arrays of shape (4,3,H,W)
        '''
        
        current_state = (torch.from_numpy(current_state).view(1, -1, *current_state.shape[2:]) / 255).float().to(self.device)  # normalization, shape(1, 12, H, W)
        target_state = (torch.from_numpy(target_state).view(1, -1, *target_state.shape[2:]) / 255).float().to(self.device) # normalization, shape(1, 12, H, W)
            
        with torch.no_grad():
            time_logits, action_logits = self.model.inverse_model_forward(current_state, target_state)
            time_prediction, action_prediction = torch.argmax(time_logits, dim=1), torch.argmax(action_logits, dim=1)
            action_probs, time_probs = torch.nn.functional.softmax(action_logits, dim=1).squeeze(), torch.nn.functional.softmax(time_logits, dim=1).squeeze()
        
        return action_prediction.item(), action_probs.cpu().numpy()
    
    def inverse_model_act_latent(self, latent_current_state, latent_target_state, return_probs=False, return_logits=False):
        '''
        - current_states, goal_states: numpy arrays of shape (4,3,H,W)
        '''

        with torch.no_grad():
            time_logits, action_logits = self.model.inverse_model_forward_latent(latent_current_state, latent_target_state)
            time_prediction, action_prediction = torch.argmax(time_logits, dim=1), torch.argmax(action_logits, dim=1)
            action_probs, time_probs = torch.nn.functional.softmax(action_logits, dim=1).squeeze(), torch.nn.functional.softmax(time_logits, dim=1).squeeze()
        
        return action_prediction.item(), action_probs.cpu().numpy()
        
    def inverse_model_sample(self, current_state, target_state, return_probs=False, return_logits=False):
        '''
        - current_states, goal_states: numpy arrays of shape (4,3,H,W)
        '''
        
        current_state = (torch.from_numpy(current_state).view(1, -1, *current_state.shape[2:]) / 255).float().to(self.device)  # normalization, shape(1, 12, H, W)
        target_state = (torch.from_numpy(target_state).view(1, -1, *target_state.shape[2:]) / 255).float().to(self.device) # normalization, shape(1, 12, H, W)
        
        with torch.no_grad():
            time_logits, action_logits = self.model.inverse_model_forward(current_state, target_state)
            action = torch.distributions.Categorical(logits = action_logits).sample()
            time_prediction, action_prediction = torch.argmax(time_logits, dim=1), torch.argmax(action_logits, dim=1)
            action_probs, time_probs = torch.nn.functional.softmax(action_logits, dim=1).squeeze(), torch.nn.functional.softmax(time_logits, dim=1).squeeze()
        
        return action.item(), action_probs.cpu().numpy()
    
    #-----------------------------------------------------------------------------------------------
    # act() method helpers
    #-----------------------------------------------------------------------------------------------
    
    def flush_action_buffer(self):
        self.mpc_optimizer.action_buffer = None
        
    def average_time(self, current_state, target_state):
        '''
        - current_states, goal_states: numpy arrays of shape (4,3,H,W)
        '''
        return self.model.average_time(current_state, target_state)
    
    def average_time_latent(self, latent_current_states, latent_target_states):
        '''
        - current_states, goal_states: numpy arrays of shape (4,3,H,W)
        '''
        return self.model.average_time_latent(latent_current_states, latent_target_states)

    #-----------------------------------------------------------------------------------------------
    # Utility functions for debugging and evaluation
    #-----------------------------------------------------------------------------------------------
    
    def get_latent(self, current_state):
        '''
        - current_states, goal_states: numpy arrays of shape (4,3,H,W)
        '''
        
        current_state = (torch.from_numpy(current_state).view(1, -1, *current_state.shape[2:]) / 255).float().to(self.device)  # normalization, shape(1, 12, H, W)
        
        with torch.no_grad():
            current_latent = self.model.get_latent(current_state)
            
        return current_latent
    
    def get_preds(self, current_state, action, next_state, target_state):
        '''
        - current_states, goal_states: numpy arrays of shape (4,3,H,W)
        '''
        
        current_state = (torch.from_numpy(current_state).view(1, -1, *current_state.shape[2:]) / 255).float().to(self.device)  # normalization, shape(1, 12, H, W)
        next_state = (torch.from_numpy(next_state).view(1, -1, *next_state.shape[2:]) / 255).float().to(self.device)  # normalization, shape(1, 12, H, W)
        target_state = (torch.from_numpy(target_state).view(1, -1, *target_state.shape[2:]) / 255).float().to(self.device) # normalization, shape(1, 12, H, W)
        one_hot_action = F.one_hot(torch.tensor(action), num_classes=self.num_actions).unsqueeze(0) # shape (1, self.num_actions)
        one_hot_action = one_hot_action.to(self.device)
        
        with torch.no_grad():
            time_logits, action_logits, latent_next_state_deltas, latent_next_state_delta_preds, latent_current_states = self.model.forward(current_state, next_state, target_state, one_hot_action)
            
        return time_logits, action_logits, latent_next_state_deltas, latent_next_state_delta_preds, latent_current_states
    
    def get_distances(self, current_state, target_state):
        '''
        - current_states, goal_states: numpy arrays of shape (4,3,H,W)
        '''
        
        current_state = (torch.from_numpy(current_state).view(1, -1, *current_state.shape[2:]) / 255).float().to(self.device)  # normalization, shape(1, 12, H, W)
        target_state = (torch.from_numpy(target_state).view(1, -1, *target_state.shape[2:]) / 255).float().to(self.device) # normalization, shape(1, 12, H, W)
        
        with torch.no_grad():
            current_latent = self.model.get_latent(current_state)
            target_latent = self.model.get_latent(target_state)
        
        with torch.no_grad():
            time_logits, action_logits = self.model.inverse_model_forward(current_state, target_state)
            time_prediction, action_prediction = torch.argmax(time_logits, dim=1), torch.argmax(action_logits, dim=1)
            time_probs = torch.nn.functional.softmax(time_logits, dim=1).squeeze()
            time_values = torch.arange(time_probs.shape[-1], dtype=torch.float).to(self.device)
            
            time_average = torch.sum(time_values * time_probs)
            action_entropy = torch.distributions.Categorical(logits = action_logits).entropy()
            time_entropy = torch.distributions.Categorical(logits = time_logits).entropy()
            latent_distance = torch.norm(target_latent - current_latent)
            
        return time_average, action_entropy, time_entropy, latent_distance
    
    
    