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
# Model
#-----------------------------------------------------------------------------------------------            
#-----------------------------------------------------------------------------------------------

default_config = {
    'max_lookahead': 6,
    'num_actions': 5,
    'input_channels': 12,
    'spatial_size': (120,160),
    'backbone_name': 'resnet18',
    'backbone_output_dim': 512,
    'fc_dim': 512,
    'horizon': 6,
    'allT_or_lastT': 'lastT',
    'cons_or_nocons': 'nocons',
    'statH_or_dynH': 'dynH',
    'allH_or_lastH': 'lastH',
    'consistency_threshold': np.log(0.4),
}

class MPCModel(nn.Module):

    def __init__(self, device, config=default_config):
        super().__init__()

        self.config = config
        self.device = device
        self.num_classes_time = self.config['max_lookahead'] + 1
        self.num_classes_action = self.config['num_actions']
        
        # Create the backbone
        self.backbone = ResNetEncoder(resnet18, input_channels=self.config['input_channels'], spatial_size=self.config['spatial_size'], output_size=self.config['backbone_output_dim'])
        self.latent_dim = self.backbone.output_shape[-1] # This is different than self.config['backbone_output_dim'] due to rounding in maxpooling
        
        # Create the FC layers
        self.time_logits = nn.Sequential(
                                 nn.Linear(self.latent_dim * 2, self.config['fc_dim']),
                                 nn.ReLU(),
                                 nn.Linear(self.config['fc_dim'], self.config['fc_dim']),
                                 nn.ReLU(),
                                 nn.Linear(self.config['fc_dim'], self.config['fc_dim']),
                                 nn.ReLU(),
                                 nn.Linear(self.config['fc_dim'], self.config['fc_dim']),
                                 nn.ReLU(),
                                 nn.Linear(self.config['fc_dim'], self.num_classes_time),
                             )
        
        self.action_logits = nn.Sequential(
                                 nn.Linear(self.latent_dim * 2, self.config['fc_dim']),
                                 nn.ReLU(),
                                 nn.Linear(self.config['fc_dim'], self.config['fc_dim']),
                                 nn.ReLU(),
                                 nn.Linear(self.config['fc_dim'], self.config['fc_dim']),
                                 nn.ReLU(),
                                 nn.Linear(self.config['fc_dim'], self.config['fc_dim']),
                                 nn.ReLU(),
                                 nn.Linear(self.config['fc_dim'], self.num_classes_action),
                             )
        
        
        self.forward_model_delta = nn.Sequential(
                                       nn.Linear(self.latent_dim + self.num_classes_action, self.config['fc_dim']),
                                       nn.ReLU(),
                                       nn.Linear(self.config['fc_dim'], self.config['fc_dim']),
                                       nn.ReLU(),
                                       nn.Linear(self.config['fc_dim'], self.config['fc_dim']),
                                       nn.ReLU(),
                                       nn.Linear(self.config['fc_dim'], self.latent_dim),
                                   )

    def forward(self, current_states, next_states, goal_states, one_hot_actions):
        """
        Args:
            current_states: shape(B,C,H,W)
            next_states: shape(B,C,H,W)
            goal_states: shape(B,C,H,W)
        """
        latent_current_states = self.backbone(current_states) # shape (B,D)
        latent_next_states = self.backbone(next_states) # shape (B,D)
        latent_goal_states = self.backbone(goal_states) # shape (B,D)
        latent_next_state_deltas = latent_next_states - latent_current_states # shape (B,D)
        
        joint_latent = torch.cat([latent_current_states, latent_goal_states], dim=-1) # shape (B,2D)
        action_state_latent = torch.cat([latent_current_states, one_hot_actions], dim=-1) # shape (B,D+K)
        
        time_logits = self.time_logits(joint_latent) # shape (B,self.num_classes_time)
        action_logits = self.action_logits(joint_latent) # shape (B,self.num_classes_action)
        latent_next_state_delta_preds = self.forward_model_delta(action_state_latent)
        
        return  [time_logits, action_logits, latent_next_state_deltas, latent_next_state_delta_preds, latent_current_states]
    
    def inverse_model_forward(self, current_states, goal_states):
        """
        Args:
            current_states: shape(B,C,H,W)
            next_states: shape(B,C,H,W)
            goal_states: shape(B,C,H,W)
        """
        latent_current_states = self.backbone(current_states) # shape (B,D)
        latent_goal_states = self.backbone(goal_states) # shape (B,D)
        joint_latent = torch.cat([latent_current_states, latent_goal_states], dim=-1) # shape (B,2D)
        time_logits = self.time_logits(joint_latent) # shape (B,self.num_classes_time)
        action_logits = self.action_logits(joint_latent) # shape (B,self.num_classes_action)
        
        return time_logits, action_logits
    
    def inverse_model_forward_latent(self, latent_current_states, latent_goal_states):
        """
        Args:
            current_states: shape(B,C,H,W)
            next_states: shape(B,C,H,W)
            goal_states: shape(B,C,H,W)
        """
        joint_latent = torch.cat([latent_current_states, latent_goal_states], dim=-1) # shape (B,2D)
        time_logits = self.time_logits(joint_latent) # shape (B,self.num_classes_time)
        action_logits = self.action_logits(joint_latent) # shape (B,self.num_classes_action)
        
        return time_logits, action_logits
    
    def get_latent(self, current_state):
        """
        Args:
            current_state: shape(1,C,H,W)
        """
        
        return self.backbone(current_state)
    
    def average_time(self, current_state, target_state):
        '''
        - current_states, goal_states: numpy arrays of shape (4,3,H,W)
        '''
        
        current_state = (torch.from_numpy(current_state).view(1, -1, *current_state.shape[2:]) / 255).float().to(self.device)  # normalization, shape(1, 12, H, W)
        target_state = (torch.from_numpy(target_state).view(1, -1, *target_state.shape[2:]) / 255).float().to(self.device) # normalization, shape(1, 12, H, W)
        
        with torch.no_grad():
            time_logits, action_logits = self.inverse_model_forward(current_state, target_state)
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
            time_logits = self.time_logits(joint_latent) # shape (B,self.num_classes_time)
            time_probs = torch.nn.functional.softmax(time_logits, dim=1).squeeze()
            time_values = torch.arange(time_probs.shape[-1], dtype=torch.float).view(1, time_probs.shape[-1]).to(self.device)
            time_average = torch.sum(time_values * time_probs, dim=-1)
            
        return time_average.cpu().numpy()
 

#-----------------------------------------------------------------------------------------------            
#-----------------------------------------------------------------------------------------------
# Trainer
#-----------------------------------------------------------------------------------------------            
#-----------------------------------------------------------------------------------------------

class MPCModelTrainer(nn.Module):

    def __init__(self, model, trainer_lr, wandb_config, dataset, checkpoint_path=None):
        super().__init__()
        
        self.logger = logging.getLogger("main_logger")
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=trainer_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        self.epoch_shift = 0
        
        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epoch_shift = checkpoint['epoch']

        wandb.init(project=wandb_config['project'], group=wandb_config['group'], notes=wandb_config['notes'], name=wandb_config['experiment_name'], config=wandb_config)
        self.dataset = dataset
        
    def loss_function(self, time_logits, time_targets, action_logits, action_targets, latent_next_state_deltas, latent_next_state_delta_preds, lambda_mse=1e-5):
        """
        Defines the loss function to train the model
        
        Args:
            time_logits: shape(B,self.num_classes_time)
            time_targets: shape(B,)
            action_logits: shape(B,self.num_classes_action)
            action_targets: shape(B,)
        """
        CE_time = F.cross_entropy(time_logits, time_targets)
        CE_action = F.cross_entropy(action_logits, action_targets)
        MSE_latent_next_state_delta = lambda_mse * F.mse_loss(latent_next_state_deltas, latent_next_state_delta_preds)
        loss = (CE_time + CE_action) + MSE_latent_next_state_delta
        
        return {'loss': loss, 'CE_time': CE_time, 'CE_action': CE_action, 'MSE_latent_next_state_delta': MSE_latent_next_state_delta}

    def optimization_step(self, batch, device):
        """
        Bridges the optimizer step and the outputs of the dataloader and the loss function (as they can return additional info for plotting/visualization/profiling etc.)
        
        Args:
            batch: Dictionary with dict_keys(['current_states', 'target_states', 'actions', 'time_differences'])
                batch['current_states']: shape (B,C,H,W)
                batch['target_states']: shape (B,C,H,W)
                batch['actions']: shape (B,)
                batch['time_differences']: shape (B,)
        """
        current_states = batch[0].to(device)
        next_states = batch[1].to(device) 
        goal_states = batch[2].to(device)
        time_targets = batch[3].to(device)
        action_targets = batch[4].to(device)
        one_hot_actions = batch[5].to(device)
        num_samples = time_targets.shape[0]
        
        time_logits, action_logits, latent_next_state_deltas, latent_next_state_delta_preds, latent_current_states = self.model.forward(current_states, next_states, goal_states, one_hot_actions)
        
        with torch.no_grad():
            time_predictions, action_predictions = torch.argmax(time_logits, dim=1), torch.argmax(action_logits, dim=1)
            time_accuracy_total = (time_predictions == time_targets).float().sum() / time_predictions.shape[0]
            action_accuracy_total = (action_predictions == action_targets).float().sum() / action_predictions.shape[0]

            time_true_preds = np.array([(time_predictions[time_targets == i] == time_targets[time_targets == i]).float().sum().item() for i in range(self.dataset.max_lookahead+1)])
            time_total_preds = np.array([time_predictions[time_targets == i].shape[0] for i in range(self.dataset.max_lookahead+1)])
            
            action_true_preds = np.array([(action_predictions[time_targets == i] == action_targets[time_targets == i]).float().sum().item() for i in range(self.dataset.max_lookahead+1)])
            action_total_preds = np.array([action_predictions[time_targets == i].shape[0] for i in range(self.dataset.max_lookahead+1)])
        
        loss_dict = self.loss_function(time_logits, time_targets, action_logits, action_targets, latent_next_state_deltas, latent_next_state_delta_preds)
        loss = loss_dict['loss']
        CE_time = loss_dict['CE_time']
        CE_action = loss_dict['CE_action']
        MSE_latent_next_state_delta = loss_dict['MSE_latent_next_state_delta']
        
        print('Training Loss: {}'.format(loss))
        # print('Latent Current States:')
        # print(latent_current_states)
        # print()
        # self.logger.info('Training Loss: {}'.format(loss))

        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss.item(), time_true_preds, time_total_preds, action_true_preds, action_total_preds, time_accuracy_total, action_accuracy_total, MSE_latent_next_state_delta, num_samples
    
    def validation_step(self, batch, device):
        """
        Bridges the optimizer step and the outputs of the dataloader and the loss function (as they can return additional info for plotting/visualization/profiling etc.)
        
        Args:
            batch: Dictionary with dict_keys(['current_states', 'target_states', 'actions', 'time_differences'])
                batch['current_states']: shape (B,C,H,W)
                batch['target_states']: shape (B,C,H,W)
                batch['actions']: shape (B,)
                batch['time_differences']: shape (B,)
        """
        current_states = batch[0].to(device)
        next_states = batch[1].to(device) 
        goal_states = batch[2].to(device)
        time_targets = batch[3].to(device)
        action_targets = batch[4].to(device)
        one_hot_actions = batch[5].to(device)
        num_samples = time_targets.shape[0]
        
        time_logits, action_logits, latent_next_state_deltas, latent_next_state_delta_preds, latent_current_states = self.model.forward(current_states, next_states, goal_states, one_hot_actions)
        
        with torch.no_grad():
            time_predictions, action_predictions = torch.argmax(time_logits, dim=1), torch.argmax(action_logits, dim=1)
            time_accuracy_total = (time_predictions == time_targets).float().sum() / time_predictions.shape[0]
            action_accuracy_total = (action_predictions == action_targets).float().sum() / action_predictions.shape[0]

            time_true_preds = np.array([(time_predictions[time_targets == i] == time_targets[time_targets == i]).float().sum().item() for i in range(self.dataset.max_lookahead+1)])
            time_total_preds = np.array([time_predictions[time_targets == i].shape[0] for i in range(self.dataset.max_lookahead+1)])
            
            action_true_preds = np.array([(action_predictions[time_targets == i] == action_targets[time_targets == i]).float().sum().item() for i in range(self.dataset.max_lookahead+1)])
            action_total_preds = np.array([action_predictions[time_targets == i].shape[0] for i in range(self.dataset.max_lookahead+1)])
        
        loss_dict = self.loss_function(time_logits, time_targets, action_logits, action_targets, latent_next_state_deltas, latent_next_state_delta_preds)
        loss = loss_dict['loss']
        CE_time = loss_dict['CE_time']
        CE_action = loss_dict['CE_action']
        MSE_latent_next_state_delta = loss_dict['MSE_latent_next_state_delta']

        return loss.item(), time_true_preds, time_total_preds, action_true_preds, action_total_preds, time_accuracy_total, action_accuracy_total, MSE_latent_next_state_delta, num_samples
    
    def fit(self, epochs, train_dl, valid_dl, device, MODEL_DIR):
        """
        Runs multiple training epochs and handles logging
        """
        for epoch in range(epochs):
            
            # Training epoch
            t_0 = time.time()
            self.model.train()
            
            train_losses = []
            train_time_true_preds = np.zeros(self.dataset.max_lookahead+1)
            train_time_total_preds = np.zeros(self.dataset.max_lookahead+1)
            train_action_true_preds = np.zeros(self.dataset.max_lookahead+1)
            train_action_total_preds = np.zeros(self.dataset.max_lookahead+1)
            train_time_accuracy_total = []
            train_action_accuracy_total = []
            train_MSE_latent_next_state_deltas = []
            nums = []
            batch_count = 0
            
            for batch in train_dl:  
                loss_batch, time_true_preds, time_total_preds, action_true_preds, action_total_preds, time_accuracy_total, action_accuracy_total, MSE_latent_next_state_delta, num = self.optimization_step(batch, device)
                
                train_losses.append(loss_batch)
                train_time_true_preds = train_time_true_preds + time_true_preds
                train_time_total_preds = train_time_total_preds + time_total_preds
                train_action_true_preds = train_action_true_preds + action_true_preds
                train_action_total_preds = train_action_total_preds + action_total_preds
                train_time_accuracy_total.append(time_accuracy_total)
                train_action_accuracy_total.append(action_accuracy_total)
                train_MSE_latent_next_state_deltas.append(MSE_latent_next_state_delta)
                nums.append(num)
                print('Batch Count: {}'.format(batch_count))
                batch_count = batch_count + 1
                
            train_loss = np.sum(np.multiply(train_losses, nums)) / np.sum(nums)
            train_MSE_latent_next_state_delta = np.sum(np.multiply(train_MSE_latent_next_state_deltas, nums)) / np.sum(nums)
            train_time_accuracy_total = np.sum(np.multiply(train_time_accuracy_total, nums)) / np.sum(nums)
            train_action_accuracy_total = np.sum(np.multiply(train_action_accuracy_total, nums)) / np.sum(nums)
            train_time_accuracy = train_time_true_preds / train_time_total_preds
            train_action_accuracy = train_action_true_preds / train_action_total_preds

            # Validation epoch
            t_1 = time.time()
            self.model.eval()
            
            with torch.no_grad():
                val_losses = []
                val_time_true_preds = np.zeros(self.dataset.max_lookahead+1)
                val_time_total_preds = np.zeros(self.dataset.max_lookahead+1)
                val_action_true_preds = np.zeros(self.dataset.max_lookahead+1)
                val_action_total_preds = np.zeros(self.dataset.max_lookahead+1)
                val_time_accuracy_total = []
                val_action_accuracy_total = []
                val_MSE_latent_next_state_deltas = []
                nums = []
                batch_count = 0

                for batch in valid_dl:  
                    loss_batch, time_true_preds, time_total_preds, action_true_preds, action_total_preds, time_accuracy_total, action_accuracy_total, MSE_latent_next_state_delta, num = self.validation_step(batch, device)

                    val_losses.append(loss_batch)
                    val_time_true_preds = val_time_true_preds + time_true_preds
                    val_time_total_preds = val_time_total_preds + time_total_preds
                    val_action_true_preds = val_action_true_preds + action_true_preds
                    val_action_total_preds = val_action_total_preds + action_total_preds
                    val_time_accuracy_total.append(time_accuracy_total)
                    val_action_accuracy_total.append(action_accuracy_total)
                    val_MSE_latent_next_state_deltas.append(MSE_latent_next_state_delta)
                    nums.append(num)
            
            val_loss = np.sum(np.multiply(val_losses, nums)) / np.sum(nums)
            val_MSE_latent_next_state_delta = np.sum(np.multiply(val_MSE_latent_next_state_deltas, nums)) / np.sum(nums)
            val_time_accuracy_total = np.sum(np.multiply(val_time_accuracy_total, nums)) / np.sum(nums)
            val_action_accuracy_total = np.sum(np.multiply(val_action_accuracy_total, nums)) / np.sum(nums)
            val_time_accuracy = val_time_true_preds / val_time_total_preds
            val_action_accuracy = val_action_true_preds / val_action_total_preds
            
            # Log Epoch Metrics
            t_2 = time.time()
            wandb_log_dict = {"train_loss": train_loss, 
                              "train_time_accuracy_total": train_time_accuracy_total,
                              "train_action_accuracy_total": train_action_accuracy_total,
                              "train_MSE_latent_next_state_delta": train_MSE_latent_next_state_delta,
                              "val_loss": val_loss,
                              "val_time_accuracy_total": val_time_accuracy_total,
                              "val_action_accuracy_total": val_action_accuracy_total,
                              "val_MSE_latent_next_state_delta": val_MSE_latent_next_state_delta}
            
            for i in range(len(train_time_accuracy)): 
                wandb_log_dict['train_time_accuracy_{}_step'.format(i)] = train_time_accuracy[i]
                wandb_log_dict['train_action_accuracy_{}_step'.format(i)] = train_action_accuracy[i]
                wandb_log_dict['val_time_accuracy_{}_step'.format(i)] = val_time_accuracy[i]
                wandb_log_dict['val_action_accuracy_{}_step'.format(i)] = val_action_accuracy[i]
                
            wandb.log(wandb_log_dict)
            
            self.logger.info("--------------------------------------")
            self.logger.info("Epoch: {}".format(epoch))
            self.logger.info("Training epoch time: {}".format(t_1 - t_0))
            self.logger.info("Validation epoch time: {}".format(t_2 - t_1))
            self.logger.info("train_loss: {}".format(train_loss))
            self.logger.info("train_time_accuracy: {}".format(train_time_accuracy[1]))
            self.logger.info("train_action_accuracy: {}".format(train_action_accuracy[1]))
            self.logger.info("val_loss: {}".format(val_loss))
            self.logger.info("val_time_accuracy: {}".format(val_time_accuracy[1]))
            self.logger.info("val_action_accuracy: {}".format(val_action_accuracy[1]))
            self.logger.info("--------------------------------------")
            
            print("--------------------------------------")
            print("Epoch: {}".format(epoch))
            print("Training epoch time: {}".format(t_1 - t_0))
            print("Validation epoch time: {}".format(t_2 - t_1))
            print("train_loss: {}".format(train_loss))
            print("train_time_accuracy: {}".format(train_time_accuracy[1]))
            print("train_action_accuracy: {}".format(train_action_accuracy[1]))
            print("val_loss: {}".format(val_loss))
            print("val_time_accuracy: {}".format(val_time_accuracy[1]))
            print("val_action_accuracy: {}".format(val_action_accuracy[1]))
            print("--------------------------------------")
            
            # Save Model
            checkpoint = {'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': epoch + self.epoch_shift}
            torch.save(checkpoint, os.path.join(MODEL_DIR, 'epoch_{}.pt'.format(epoch + self.epoch_shift)))
    
    
    