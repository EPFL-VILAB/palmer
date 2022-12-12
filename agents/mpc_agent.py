import logging
import os
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
    'statH_or_dynH': 'statH',
    'allH_or_lastH': 'lastH',
    'consistency_threshold': np.log(0.4),
}



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
                    if (self.num_actions ** horizon) < self.max_num_mpc_samples:
                        random_actions_list = list(product(range(self.num_actions), repeat=horizon)) # shape (self.num_samples, horizon)
                    else: 
                        rng = np.random.default_rng()
                        random_actions_list = rng.integers(0, self.num_actions, size=(self.max_num_mpc_samples // horizon, horizon)) # shape (self.num_samples, horizon)

                    forward_action_sequences = torch.tensor(random_actions_list).transpose(1, 0).to(self.device) # shape (horizon, self.num_samples)
            
                elif horizon <= len(self.action_buffer) + 1:
                    print('hey')
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

    def __init__(self, model, device, config=default_config):
        '''
        A simple agent that acts according to the predictions of a classification model
        '''
        self.model = model.to(device)
        self.device = device
        self.config = config
        
        self.horizon = self.config['horizon']
        self.statH_or_dynH = self.config['statH_or_dynH']
        self.mpc_optimizer = RandomShootingOptimizer(self.model, self.device, self.config)
        self.num_actions = self.model.num_classes_action
        
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
            horizon_estimate = max(round(self.average_time(current_state, target_state)), 1)
            if horizon_estimate == 6:
                horizon = self.horizon
            else:
                horizon = horizon_estimate
            
        elif self.statH_or_dynH == 'statH':
            horizon = self.horizon
            
        with torch.no_grad():
            current_state = (torch.from_numpy(current_state).view(1, -1, *current_state.shape[2:]) / 255).float().to(self.device)  # normalization, shape(1, 12, H, W)
            target_state = (torch.from_numpy(target_state).view(1, -1, *target_state.shape[2:]) / 255).float().to(self.device) # normalization, shape(1, 12, H, W)
            latent_current_state = self.model.get_latent(current_state)
            latent_target_state = self.model.get_latent(target_state)
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
        
        current_state = (torch.from_numpy(current_state).view(1, -1, *current_state.shape[2:]) / 255).float().to(self.device)  # normalization, shape(1, 12, H, W)
        target_state = (torch.from_numpy(target_state).view(1, -1, *target_state.shape[2:]) / 255).float().to(self.device) # normalization, shape(1, 12, H, W)
        
        with torch.no_grad():
            time_logits, action_logits = self.model.inverse_model_forward(current_state, target_state)
            time_prediction, action_prediction = torch.argmax(time_logits, dim=1), torch.argmax(action_logits, dim=1)
            time_probs = torch.nn.functional.softmax(time_logits, dim=1).squeeze()
            time_values = torch.arange(time_probs.shape[-1], dtype=torch.float).to(self.device)
            time_average = torch.sum(time_values * time_probs)
            
        return time_average.item()
    
    def average_time_latent(self, latent_current_states, latent_target_states):
        '''
        - current_states, goal_states: numpy arrays of shape (4,3,H,W)
        '''
        
        latent_current_states = latent_current_states.float().to(self.device)
        latent_target_states = latent_target_states.float().to(self.device)
        
        with torch.no_grad():
            joint_latent = torch.cat([latent_current_states, latent_target_states], dim=-1) # shape (B,2D)
            time_logits = self.model.time_logits(joint_latent) # shape (B,self.num_classes_time)
            time_probs = torch.nn.functional.softmax(time_logits, dim=1).squeeze()
            time_values = torch.arange(time_probs.shape[-1], dtype=torch.float).view(1, time_probs.shape[-1]).to(self.device)
            time_average = torch.sum(time_values * time_probs, dim=-1)
            
        return time_average.cpu().numpy()

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
    
    
    