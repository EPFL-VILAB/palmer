import logging
import os
import sys
sys.path.append('../')
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms.functional as T
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
from PIL import Image

    
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
    'num_nodes_graph': 1000,
    'pT_distance_threshold': 5.5,
    'latent_distance_threshold': 1.5,
    'overlap_pT_distance_threshold': 4,
    'overlap_latent_distance_threshold': 2,
    'overlap_retrieval_distance_threshold': 1,
    'pT_search_threshold': 5.5,
    'graph_rng_seed': 285,
    'num_batches': 100,
    'q_value_distance_threshold': 8.7,
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
        self.model = model.to(device)
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
        action_block_path = os.path.join(self.LOG_DIR, 'action_list.npy')
        
        self.num_nodes_graph = self.config['num_nodes_graph']
        self.pT_distance_threshold = self.config['pT_distance_threshold']
        self.latent_distance_threshold = self.config['latent_distance_threshold']
        self.q_value_distance_threshold = self.config['q_value_distance_threshold']
        
        self.overlap_pT_distance_threshold = self.config['overlap_pT_distance_threshold']
        self.overlap_latent_distance_threshold = self.config['overlap_latent_distance_threshold']
        self.overlap_retrieval_distance_threshold = self.config['overlap_retrieval_distance_threshold']
        self.pT_search_threshold = self.config['pT_search_threshold']
        self.num_batches = self.config['num_batches']
        
        self.graph_rng_seed = self.config['graph_rng_seed']
        self.rng = np.random.default_rng(self.graph_rng_seed)
        
        # Fields to be initialized later
        self.latents_block = np.load(latent_block_path) # shape (num_random_walk_steps, D)
        self.latents_block_gpu = torch.from_numpy(self.latents_block).to(self.device) # shape (num_random_walk_steps, D)
        self.positions_block = np.load(position_block_path) # shape (num_random_walk_steps, 3)
        self.actions_block = np.load(action_block_path)
        
        self.sample_ids = None # shape [num_samples]
        self.sampled_positions = None # shape (num_samples, 3)
        self.sampled_latents = None # shape (num_samples, D)
        self.pT_distances = None # shape (num_samples, num_samples)
        self.q_value_distances = None # shape (num_samples, num_samples)
        self.latent_distances = None # shape (num_samples, num_samples_random_walk)
        self.latent_distance_masks = None # shape (num_samples, num_samples_random_walk)
        self.retrieval_distances = None # shape (num_samples, num_samples)
        self.unfiltered_graph = None
        self.filtered_graph = None
        
        # Fields for graph extension
        self.goal_latent = None # shape (1, D)
        self.start_latent = None # shape (1, D)
        
        self.pT_distances_samples_to_goal = None # shape (num_new_samples, num_samples)
        self.pT_distances_start_to_samples = None # shape (num_new_samples, num_samples)
        self.pT_distance_start_to_goal = None # shape (num_new_samples, num_samples)
        
        self.latent_distances_walk_to_goal = None # shape (num_new_samples, num_samples_random_walk)
        self.latent_distances_start_to_walk = None # shape (num_new_samples, num_samples_random_walk)
        self.latent_distance_start_to_goal = None # shape (num_new_samples, num_samples_random_walk)
        
        self.latent_distance_masks_walk_to_goal = None # shape (num_new_samples, num_samples_random_walk)
        self.latent_distance_masks_start_to_walk = None # shape (num_new_samples, num_samples_random_walk)
        self.latent_distance_mask_start_to_goal = None # shape (num_new_samples, num_samples_random_walk)
        
        self.retrieval_distances_samples_to_goal = None # shape (num_new_samples, num_samples)
        self.retrieval_distances_start_to_samples = None # shape (num_new_samples, num_samples)
        self.retrieval_distance_start_to_goal = None # shape (num_new_samples, num_samples)
        
        # Build the graphs
        self.sample_graph_nodes()
        self.set_pT_distances()
        self.set_latent_distances()
        self.set_retrieval_distances()
        self.build_unfiltered_graph()
        self.build_filtered_graph()
        self.filtered_graph_expanded = self.filtered_graph.copy()
        
    #-----------------------------------------------------------------------------------------------
    # Utilities to build the graph
    #-----------------------------------------------------------------------------------------------
    
    def sample_graph_nodes(self, num_nodes=None):
        '''
        - num_nodes: int
        '''

        if num_nodes is None: num_nodes = self.num_nodes_graph
        self.sample_ids = self.rng.integers(0, self.positions_block.shape[0], size = num_nodes)
        self.sampled_positions = self.positions_block[self.sample_ids]
        self.sampled_latents = self.latents_block[self.sample_ids]
        
        print('Sampling states...')
        self.sampled_state_paths = [os.path.join(self.OBS_DIR, 'state_{:06d}.npy'.format(idx)) for idx in self.sample_ids]
        state_list = []
        for current_state_path in tqdm(self.sampled_state_paths):
            current_state = np.load(current_state_path)
            current_state = torch.from_numpy(current_state).view(-1, *current_state.shape[2:]) / 255 # normalization
            current_state = T.resize(current_state, [64, 64])
            state_list.append(current_state)
            
        self.sampled_states = torch.stack(state_list).to(self.device) # shape (N, 12, 64, 64)
        
    def render_graph_nodes(self, ):

        self.env.game.draw_vis_map()
        self.env.game.draw_points(self.sampled_positions)
        figure(figsize=(10, 10))
        plt.imshow(self.env.game.vis_map)
        # Image.fromarray(self.env.game.vis_map).save('graph_nodes.png')
        
    def set_pT_distances(self, ):

        num_nodes = self.sample_ids.shape[0]
        self.pT_distances = np.zeros([num_nodes, num_nodes])
        
        print('Setting pT distances...')
        for sample_num in tqdm(range(num_nodes)):
            current_latent_rep = torch.stack(num_nodes*[torch.from_numpy(self.sampled_latents[sample_num])])
            target_latent_rep = torch.from_numpy(self.sampled_latents)
            pT_current_to_target = self.model.average_time_latent(current_latent_rep, target_latent_rep)
            self.pT_distances[sample_num, :] = pT_current_to_target
            
    def set_latent_distances(self):

        num_batches = self.num_batches
        num_samples = self.sampled_latents.shape[0]
        num_samples_random_walk = self.latents_block.shape[0]
        current_latents_torch = torch.from_numpy(self.sampled_latents).unsqueeze(1).to(self.device) # shape (2N, 1, D)
        latents_block_expanded_torch = self.latents_block_gpu.unsqueeze(0) # shape (1, M, D)
        
        self.latent_distances = np.zeros([num_samples, num_samples_random_walk]) # shape (2N, M)
        batch_size = num_samples*2 // num_batches
        
        print('Setting latent distances...')
        for batch_num in tqdm(range(num_batches)):
            start_id = batch_num * batch_size 
            end_id = (batch_num + 1) * batch_size 
            latent_distances_batch = torch.norm(latents_block_expanded_torch - current_latents_torch[start_id:end_id], dim=-1).cpu().numpy()
            self.latent_distances[start_id:end_id, :] = latent_distances_batch
            
        self.latent_distance_masks = (self.latent_distances < self.latent_distance_threshold)
      
    def get_retrieval_distance(self, sample_id_current, sample_id_target):
    
        mask_current = self.latent_distance_masks[sample_id_current]
        indices_current = np.array(mask_current.nonzero())

        mask_target = self.latent_distance_masks[sample_id_target]
        indices_target = np.array(mask_target.nonzero())

        time_steps_inbetween = indices_target.transpose(1,0) - indices_current
        
        # Only take trajectories that go current -> target, not the inverse direction
        time_steps_inbetween[time_steps_inbetween < 0 ] = self.latents_block.shape[0] * 5 
        
        min_distance = time_steps_inbetween.min()

        return min_distance

    def set_retrieval_distances(self, num_batches=100):

        num_samples = self.sampled_latents.shape[0]
        self.retrieval_distances = np.ones([num_samples, num_samples]) * -1
        
        print('Setting retrieval distances...')
        for i in tqdm(range(num_samples)):
            for j in range(num_samples):
                if self.pT_distances[i, j] < self.pT_distance_threshold:
                    retrieval_distance = self.get_retrieval_distance(i, j)
                    self.retrieval_distances[i, j] = retrieval_distance
                    
    def build_unfiltered_graph(self, ):

        num_samples = self.sampled_latents.shape[0]
        self.unfiltered_graph = nx.DiGraph()
        
        print('Building unfiltered graph...')
        for i in tqdm(range(num_samples)):
            for j in range(num_samples):
                pT_distance = self.pT_distances[i, j]
                if pT_distance < self.pT_distance_threshold:
                    self.unfiltered_graph.add_edge(i, j, weight=pT_distance)
                    
    def render_unfiltered_graph(self, ):

        self.env.game.draw_vis_map()
        self.env.game.draw_graph(self.sampled_positions, self.pT_distances, self.pT_distance_threshold)
        figure(figsize=(10, 10))
        plt.imshow(self.env.game.vis_map)
        # Image.fromarray(self.env.game.vis_map).save('pT_graph.png')
        
    def build_filtered_graph(self, ):

        num_samples = self.sampled_latents.shape[0]
        self.filtered_graph = nx.DiGraph()
        
        print('Building filtered graph...')
        for i in tqdm(range(num_samples)):
            for j in range(num_samples):
                pT_distance = self.pT_distances[i, j]
                retrieval_distance = self.retrieval_distances[i, j]
                if (retrieval_distance != -1) and (pT_distance < self.pT_distance_threshold) and (retrieval_distance < pT_distance):
                    self.filtered_graph.add_edge(i, j, weight=pT_distance)
                    
    def render_filtered_graph(self, ):

        self.env.game.draw_vis_map()
        self.env.game.draw_graph_filtered(self.sampled_positions, self.pT_distances, self.retrieval_distances, self.pT_distance_threshold)
        figure(figsize=(10, 10))
        plt.imshow(self.env.game.vis_map)
        # Image.fromarray(self.env.game.vis_map).save('retrieval_graph.png')
        
    #-----------------------------------------------------------------------------------------------
    # Graph expansion
    #-----------------------------------------------------------------------------------------------
    
    def get_retrieval_distance_to_goal(self, sample_id):
    
        mask_current = self.latent_distance_masks[sample_id]
        indices_current = np.array(mask_current.nonzero())

        mask_target = self.latent_distance_masks_walk_to_goal
        indices_target = np.array(mask_target.nonzero())

        time_steps_inbetween = indices_target.transpose(1,0) - indices_current
        
        # Only take trajectories that go current -> target, not the inverse direction
        time_steps_inbetween[time_steps_inbetween < 0 ] = self.latents_block.shape[0] * 5 
        
        min_distance = time_steps_inbetween.min()

        return min_distance
    
    def insert_goal_node(self, goal_latent, goal_position):
        '''
        - goal_latent: shape (1, D)
        '''

        try:
            self.filtered_graph_expanded.remove_node('goal')
        except:
            pass
        
        self.goal_latent = goal_latent
        self.goal_position = goal_position
        num_samples = self.sampled_latents.shape[0]
        
        # Compute the pT distances to the goal
        goal_latent_rep = torch.cat(num_samples*[torch.from_numpy(goal_latent)]) # shape (num_samples, D)
        sampled_latent_rep = torch.from_numpy(self.sampled_latents) # shape (num_samples, D)
        self.pT_distances_samples_to_goal = self.model.average_time_latent(sampled_latent_rep, goal_latent_rep) # shape (num_samples)
        
        # Compute the latent distances and masks to the goal
        goal_latent_torch = torch.from_numpy(goal_latent).to(self.device) # shape (1, D)
        latents_block_expanded_torch = self.latents_block_gpu # shape (M, D)
        self.latent_distances_walk_to_goal = torch.norm(latents_block_expanded_torch - goal_latent_torch, dim=-1).cpu().numpy() # shape (M)
        self.latent_distance_masks_walk_to_goal = (self.latent_distances_walk_to_goal < self.latent_distance_threshold) # shape (M)
        
        # Compute retrieval distances to the goal
        self.retrieval_distances_samples_to_goal = np.ones(num_samples) * -1 # shape (N)
        for sample_id in range(num_samples):
            if self.pT_distances_samples_to_goal[sample_id] < self.pT_distance_threshold:
                self.retrieval_distances_samples_to_goal[sample_id] = self.get_retrieval_distance_to_goal(sample_id)
                
        # Add edges to the goal
        for sample_id in range(num_samples):
            pT_distance = self.pT_distances_samples_to_goal[sample_id]
            retrieval_distance = self.retrieval_distances_samples_to_goal[sample_id]
            if (retrieval_distance != -1) and (pT_distance < self.pT_distance_threshold) and (retrieval_distance < pT_distance):
                self.filtered_graph_expanded.add_edge(sample_id, 'goal', weight=pT_distance)
                
    def get_retrieval_distance_from_start(self, sample_id):
    
        mask_current = self.latent_distance_masks_start_to_walk
        indices_current = np.array(mask_current.nonzero())
        
        mask_target = self.latent_distance_masks[sample_id]
        indices_target = np.array(mask_target.nonzero())

        time_steps_inbetween = indices_target.transpose(1,0) - indices_current
        
        # Only take trajectories that go current -> target, not the inverse direction
        time_steps_inbetween[time_steps_inbetween < 0 ] = self.latents_block.shape[0] * 5 
        
        min_distance = time_steps_inbetween.min()

        return min_distance
        
    def insert_start_node(self, start_latent, start_position):
        '''
        - start_latent: shape (1, D)
        '''

        try:
            self.filtered_graph_expanded.remove_node('start')
        except:
            pass
        
        self.start_latent = start_latent
        self.start_position = start_position
        num_samples = self.sampled_latents.shape[0]
        
        # Compute the pT distances from the start
        start_latent_rep = torch.cat(num_samples*[torch.from_numpy(start_latent)]) # shape (num_samples, D)
        sampled_latent_rep = torch.from_numpy(self.sampled_latents) # shape (num_samples, D)
        self.pT_distances_start_to_samples = self.model.average_time_latent(start_latent_rep, sampled_latent_rep) # shape (num_samples)
        self.pT_distance_start_to_goal = self.model.average_time_latent(torch.from_numpy(start_latent), torch.from_numpy(self.goal_latent)) # shape (1)
        
        # Compute the latent distances and masks from the start
        start_latent_torch = torch.from_numpy(start_latent).to(self.device) # shape (1, D)
        latents_block_expanded_torch = self.latents_block_gpu # shape (M, D)
        self.latent_distances_start_to_walk = torch.norm(latents_block_expanded_torch - start_latent_torch, dim=-1).cpu().numpy() # shape (M)
        self.latent_distance_masks_start_to_walk = (self.latent_distances_start_to_walk < self.latent_distance_threshold) # shape (M)
        self.latent_distance_start_to_goal = torch.norm(torch.from_numpy(self.goal_latent).to(self.device) - start_latent_torch, dim=-1).cpu().numpy() # shape (1)
        self.latent_distance_mask_start_to_goal = (self.latent_distance_start_to_goal < self.latent_distance_threshold) # shape (1)
        
        # Compute retrieval distances from the start
        self.retrieval_distances_start_to_samples = np.ones(num_samples) * -1 # shape (N)
        for sample_id in range(num_samples):
            if self.pT_distances_start_to_samples[sample_id] < self.pT_distance_threshold:
                self.retrieval_distances_start_to_samples[sample_id] = self.get_retrieval_distance_from_start(sample_id)
                
        # Compute retrieval distances from start to goal
        mask_current = self.latent_distance_masks_start_to_walk
        indices_current = np.array(mask_current.nonzero())
        mask_target = self.latent_distance_masks_walk_to_goal
        indices_target = np.array(mask_target.nonzero())
        time_steps_inbetween = indices_target.transpose(1,0) - indices_current
        # Only take trajectories that go current -> target, not the inverse direction
        time_steps_inbetween[time_steps_inbetween < 0 ] = self.latents_block.shape[0] * 5
        self.retrieval_distance_start_to_goal = time_steps_inbetween.min()
        
        argmin_distance = (time_steps_inbetween == self.retrieval_distance_start_to_goal).nonzero()
        
        id = random.randint(0,len(argmin_distance[1]) - 1)
        
        current_id = indices_current.squeeze(0)[argmin_distance[1]][id]
        target_id = indices_target.squeeze(0)[argmin_distance[0]][id]
        sorted_idx = sorted([target_id.item(), current_id.item()])
        self.retrieval_trajectory_start_to_goal = list(range(sorted_idx[0], sorted_idx[1] + 1))
                
        # Add edges from the start
        for sample_id in range(num_samples):
            pT_distance = self.pT_distances_start_to_samples[sample_id]
            retrieval_distance = self.retrieval_distances_start_to_samples[sample_id]
            if (retrieval_distance != -1) and (pT_distance < self.pT_distance_threshold) and (retrieval_distance < pT_distance):
                self.filtered_graph_expanded.add_edge('start', sample_id, weight=pT_distance)
                
        pT_distance = self.pT_distance_start_to_goal
        retrieval_distance = self.retrieval_distance_start_to_goal
        if (retrieval_distance != -1) and (pT_distance < self.pT_distance_threshold) and (retrieval_distance < pT_distance):
            self.filtered_graph_expanded.add_edge('start', 'goal', weight=pT_distance)
    
    #-----------------------------------------------------------------------------------------------
    # Shortest path queries
    #-----------------------------------------------------------------------------------------------
    
    def shortest_path(self, g, start_idx, end_idx):
        '''
        - shortest paths return id's corresponding to self.sampled_positions
        '''
        
        path = nx.shortest_path(g, start_idx, end_idx)
        
        return path
    
    def render_path(self, path):
        
        if path[0] == 'start':
            path_positions = [self.start_position] + self.sampled_positions[path[1:-1]].tolist() + [self.goal_position]
            
        else:
            path_positions = self.sampled_positions[path]
        
        self.env.game.draw_vis_map()
        self.env.game.draw_trajectory_marked(path_positions)
        figure(figsize=(10, 10))
        plt.imshow(self.env.game.vis_map)
        
    #-----------------------------------------------------------------------------------------------
    # Path Stitching
    #-----------------------------------------------------------------------------------------------
        
    def get_retrieval_trajectory(self, sample_id_current, sample_id_target):
        
        if sample_id_current == 'start':
            mask_current = self.latent_distance_masks_start_to_walk
        else:
            mask_current = self.latent_distance_masks[sample_id_current]

        indices_current = np.array(mask_current.nonzero())
        
        if sample_id_target == 'goal':
            mask_target = self.latent_distance_masks_walk_to_goal
        else:
            mask_target = self.latent_distance_masks[sample_id_target]

        indices_target = np.array(mask_target.nonzero())

        time_steps_inbetween = indices_target.transpose(1,0) - indices_current
        
        # Only take trajectories that go current -> target, not the inverse direction
        time_steps_inbetween[time_steps_inbetween < 0 ] = self.latents_block.shape[0] * 5 

        min_distance = time_steps_inbetween.min()

        argmin_distance = (time_steps_inbetween == min_distance).nonzero()
        
        id = random.randint(0,len(argmin_distance[1]) - 1)
        current_id = indices_current.squeeze(0)[argmin_distance[1]][id]
        target_id = indices_target.squeeze(0)[argmin_distance[0]][id]
        sorted_idx = sorted([target_id.item(), current_id.item()])
        retrieval_trajectory = list(range(sorted_idx[0], sorted_idx[1] + 1))

        return retrieval_trajectory
        
    def stitch_path(self, path):
        '''
        - stiched paths return id's corresponding to self.positions_block
        '''
        
        stitched_path = []
        stitched_path_actions = []
        for path_segment in zip(path[:-1], path[1:]):
            retrieval_trajectory = self.get_retrieval_trajectory(path_segment[0], path_segment[1])
            
            if path_segment[0] == 'start':
                stitched_path.append('start')
            else:
                stitched_path.append(self.sample_ids[path_segment[0]])
                
            stitched_path.extend(retrieval_trajectory)
            stitched_path_actions.extend(self.actions_block[retrieval_trajectory[:-1]])
        
        if path_segment[1] == 'goal':
            stitched_path.append('goal')
        else:
            stitched_path.append(self.sample_ids[path_segment[1]])
            
        path_positions = self.positions_block[path[1:-1]]
        stitched_path_positions = self.positions_block[stitched_path[1:-1]]

        return stitched_path, path, stitched_path_actions, stitched_path_positions, path_positions
    
    def render_stitched_path(self, stitched_path, path):
        
        if stitched_path[0] == 'start':
            stitched_path_positions = [self.start_position] + self.positions_block[stitched_path[1:-1]].tolist() + [self.goal_position]
            
        else:
            stitched_path_positions = self.positions_block[stitched_path]
        
        path_positions = self.sampled_positions[path[1:-1]].tolist()
        colors = [(0, 255, 0) for i in range(len(path_positions))]
        
        self.env.game.draw_vis_map()
        self.env.game.draw_trajectory(stitched_path_positions)
        self.env.game.draw_points_marked(path_positions)
        figure(figsize=(10, 10))
        plt.imshow(self.env.game.vis_map)
        
    #-----------------------------------------------------------------------------------------------
    # Localization
    #-----------------------------------------------------------------------------------------------
        
    def check_overlap(self, sample_id_current, sample_id_target):
    
        if sample_id_current == 'start' and sample_id_target == 'goal':
            pT_distance = self.pT_distance_start_to_goal # shape (num_new_samples, num_samples)
            latent_distance = self.latent_distance_start_to_goal # shape (num_new_samples, num_samples_random_walk)
            retrieval_distance = self.retrieval_distance_start_to_goal # shape (num_new_samples, num_samples)
            
        elif sample_id_current == 'start':
            pT_distance = self.pT_distance_start_to_samples[sample_id_target]
            latent_distance = self.latent_distance_start_to_samples[sample_id_target]
            retrieval_distance = self.retrieval_distance_start_to_samples[sample_id_target]
            
        elif sample_id_target == 'goal':
            pT_distance = self.pT_distance_samples_to_goal[sample_id_current]
            latent_distance = self.latent_distance_samples_to_goal[sample_id_current]
            retrieval_distance = self.retrieval_distance_samples_to_goal[sample_id_current]
            
        else:
            pT_distance = self.pT_distances[sample_id_current, sample_id_target]
            latent_distance = self.latent_distances[sample_id_current, sample_id_target]
            retrieval_distance = self.retrieval_distances[sample_id_current, sample_id_target]
            
        if pT_distance < self.overlap_pT_distance_threshold and latent_distance < self.overlap_latent_distance_threshold and retrieval_distance < self.overlap_retrieval_distance_threshold:
            return True, (pT_distance, latent_distance, retrieval_distance)
        else:
            return False, (pT_distance, latent_distance, retrieval_distance)
        
    #-----------------------------------------------------------------------------------------------
    # Retrieval Policy Actions
    #-----------------------------------------------------------------------------------------------
        
    def retrieval_actions(self):
        retrieval_trajectory_start_to_goal = self.retrieval_trajectory_start_to_goal
        retrieval_actions = self.actions_block[retrieval_trajectory_start_to_goal[:-1]]
        retrieval_positions = self.positions_block[retrieval_trajectory_start_to_goal]
        
        return retrieval_actions.tolist(), retrieval_positions
    
    def retrieval_actions_PRM(self):
        path = self.shortest_path(self.filtered_graph_expanded, 'start', 'goal')
        stitched_path, path, stitched_path_actions, stitched_path_positions, path_positions = self.stitch_path(path)
        
        return stitched_path_actions, stitched_path_positions, path_positions
    
    #-----------------------------------------------------------------------------------------------
    # Utilities for memory optimization
    #-----------------------------------------------------------------------------------------------
    
    def sample_goal(self):
        
        sample_id = self.rng.integers(0, self.positions_block.shape[0])
        sample_position = self.positions_block[sample_id]
        sample_latent = np.expand_dims(self.latents_block[sample_id], axis=0)
        
        return sample_id, sample_position, sample_latent
    
    
    
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

    def __init__(self, model, dqn_model, device, env, config=default_config):
        '''
        A simple agent that acts according to the predictions of a classification model
        '''
        self.model = model.to(device)
        self.dqn_model = dqn_model.to(device)
        self.device = device
        self.env = env
        self.config = config
        
        self.horizon = self.config['horizon']
        self.statH_or_dynH = self.config['statH_or_dynH']
        self.mpc_optimizer = RandomShootingOptimizer(self.model, self.device, self.config)
        self.graph_memory = GraphMemory(self.model, self.dqn_model, self.device, config=self.config)
        self.num_actions = self.model.num_classes_action
        self.PRM_action_buffer = None
        self.goal_close_flag = False
     
    #-----------------------------------------------------------------------------------------------
    # Main graph navigation methods
    #-----------------------------------------------------------------------------------------------
    
    def set_goal(self, goal_state, goal_position):
        '''
        - goal_state: an image of shape(1, 12, H, W)
        - goal_position: shape(3,)
        '''
        
        self.goal_close_flag = False
        
        with torch.no_grad():
            
            goal_state = (torch.from_numpy(goal_state).view(1, -1, *goal_state.shape[2:]) / 255).float().to(self.device) # normalization, shape(1, 12, H, W)
            self.latent_goal_state_torch = self.model.get_latent(goal_state)
            latent_goal_state = self.latent_goal_state_torch.cpu().numpy()
            
            # Line search to insert goal in the graph
            for i in range(20):
                try:
                    self.graph_memory.insert_goal_node(latent_goal_state, goal_position)
                    self.graph_memory.latent_distance_threshold = self.graph_memory.config['latent_distance_threshold']
                    print('Latent distance threshold: {}'.format(self.graph_memory.latent_distance_threshold ))
                    break

                except:
                    self.graph_memory.latent_distance_threshold = self.graph_memory.latent_distance_threshold + 0.1
                    print('Latent distance threshold is adjusted to: {}'.format(self.graph_memory.latent_distance_threshold))
    
    def navigate_to_goal_retrieval(self, current_state, current_position):
        '''
        - current_state: an image of shape(1, 12, H, W)
        - current_position: shape(3,)
        '''
        
        with torch.no_grad():
            
            # Insert start node
            current_state = (torch.from_numpy(current_state).view(1, -1, *current_state.shape[2:]) / 255).float().to(self.device) # normalization, shape(1, 12, H, W)
            latent_current_state_torch = self.model.get_latent(current_state)
            latent_current_state = latent_current_state_torch.cpu().numpy()
            self.graph_memory.insert_start_node(latent_current_state, current_position)

            # Check if the goal is reached
            success_flag, distances = self.graph_memory.check_overlap('start', 'goal')
            # if success_flag:
            #     return success_flag, distances, mpc_action, forward_action_sequences, obj_scores, consistency_ratio

            retrieval_actions, retrieval_positions = self.graph_memory.retrieval_actions()
            
            # Action smoothing to prevent switchings due to MPC (i.e., planning at every step)
            if self.PRM_action_buffer == None:
                self.PRM_action_buffer = retrieval_actions
            elif len(retrieval_actions) < len(self.PRM_action_buffer):
                # self.PRM_action_buffer[len(retrieval_actions) - 1] = retrieval_actions[-1]
                self.PRM_action_buffer[:len(retrieval_actions)] = retrieval_actions
            else:
                temp_list = retrieval_actions
                temp_list[:len(self.PRM_action_buffer)] = self.PRM_action_buffer
                self.PRM_action_buffer = temp_list
               
            # Just in case    
            random_action = random.randint(0, self.num_actions - 1)
            if self.PRM_action_buffer is None or len(self.PRM_action_buffer) == 0: 
                print('rand')
                self.PRM_action_buffer = [random_action]
            
            # if len(retrieval_actions) <= 2:
            #     print('inv')
            #     action, action_probs = self.inverse_model_act_latent(latent_current_state_torch, self.latent_goal_state_torch)
            #     action = torch.distributions.Categorical(probs = torch.from_numpy(action_probs)).sample().item()
            #     if self.PRM_action_buffer is None or len(self.PRM_action_buffer) == 0: self.PRM_action_buffer = [action]
            
        return self.PRM_action_buffer, retrieval_positions, self.PRM_action_buffer.pop(0)
    
    def navigate_to_goal_retrieval_PRM(self, current_state, current_position):
        '''
        - current_state: an image of shape(1, 12, H, W)
        - current_position: shape(3,)
        '''
        
        with torch.no_grad():
            
            # Insert start node
            current_state = (torch.from_numpy(current_state).view(1, -1, *current_state.shape[2:]) / 255).float().to(self.device) # normalization, shape(1, 12, H, W)
            latent_current_state_torch = self.model.get_latent(current_state)
            latent_current_state = latent_current_state_torch.cpu().numpy()
            self.graph_memory.insert_start_node(latent_current_state, current_position)

            # Check if the goal is reached
            success_flag, distances = self.graph_memory.check_overlap('start', 'goal')
            # if success_flag:
            #     return success_flag, distances, mpc_action, forward_action_sequences, obj_scores, consistency_ratio

            stitched_path_actions, stitched_path_positions, path_positions = self.graph_memory.retrieval_actions_PRM()
            
            # Action smoothing to prevent switchings due to MPC (i.e., planning at every step)
            if self.PRM_action_buffer == None:
                self.PRM_action_buffer = stitched_path_actions
            elif len(stitched_path_actions) < len(self.PRM_action_buffer):
                # self.PRM_action_buffer[len(stitched_path_actions) - 1] = stitched_path_actions[-1]
                self.PRM_action_buffer[:len(stitched_path_actions)] = stitched_path_actions
            else:
                temp_list = stitched_path_actions
                temp_list[:len(self.PRM_action_buffer)] = self.PRM_action_buffer
                self.PRM_action_buffer = temp_list
                
            # Just in case    
            random_action = random.randint(0, self.num_actions - 1)
            if self.PRM_action_buffer is None or len(self.PRM_action_buffer) == 0: 
                print('rand')
                self.PRM_action_buffer = [random_action]
            
            if len(stitched_path_actions) <= 2:
                self.goal_close_flag = True
                
            # if len(stitched_path_actions) <= 2:
            #     print('inv')
            #     action, action_probs = self.inverse_model_act_latent(latent_current_state_torch, self.latent_goal_state_torch)
            #     action = torch.distributions.Categorical(probs = torch.from_numpy(action_probs)).sample().item()
            #     if self.PRM_action_buffer is None or len(self.PRM_action_buffer) == 0: self.PRM_action_buffer = [action]
            #     self.PRM_action_buffer[0] = action
            
        return self.PRM_action_buffer, stitched_path_positions, self.PRM_action_buffer.pop(0)
    
    #-----------------------------------------------------------------------------------------------
    # Graph navigation methods for memory optimization
    #-----------------------------------------------------------------------------------------------
    
    def set_goal_latent(self, latent_goal_state, goal_position):
        '''
        - goal_state: an image of shape(1, 12, H, W)
        - goal_position: shape(3,)
        '''
        
        self.goal_close_flag = False
        
        with torch.no_grad():
            self.latent_goal_state_torch = torch.from_numpy(latent_goal_state).to(self.device)
            
            # Line search to insert goal in the graph
            for i in range(20):
                try:
                    self.graph_memory.insert_goal_node(latent_goal_state, goal_position)
                    self.graph_memory.latent_distance_threshold = self.graph_memory.config['latent_distance_threshold']
                    print('Latent distance threshold: {}'.format(self.graph_memory.latent_distance_threshold ))
                    break

                except:
                    self.graph_memory.latent_distance_threshold = self.graph_memory.latent_distance_threshold + 0.1
                    print('Latent distance threshold is adjusted to: {}'.format(self.graph_memory.latent_distance_threshold))
                    
                    
    #-----------------------------------------------------------------------------------------------
    # Main act() methods
    #-----------------------------------------------------------------------------------------------
    
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
        self.PRM_action_buffer = None
        self.goal_close_flag = False
    
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
    
    
    