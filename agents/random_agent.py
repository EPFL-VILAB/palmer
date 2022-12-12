import numpy as np

class RandomAgent():

    def __init__(self, action_space_dim=5, seed=42, num_of_actions=10):
        '''
        A simple agent that takes random agents according to the set seed. 

        Notes
        -----------
        - self.last_count holds the index of the last action that was completely executed in self.action_list

        '''
        self.action_space_dim = action_space_dim
        self.num_of_actions = num_of_actions
        self.rng = np.random.default_rng(seed)
        self.action_list = self.rng.integers(0,self.action_space_dim,size=self.num_of_actions)
        self.last_count = 0
         
    def act(self):
        action = self.action_list[self.last_count]
        self.last_count = self.last_count + 1
        return action 
    
    def export_action_list(self, path):
        with open(path, 'wb') as f:
            np.save(f, self.action_list)