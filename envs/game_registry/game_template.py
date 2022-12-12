from abc import ABC, abstractmethod
from os import path as osp
import numpy as np
from gym.utils import seeding
import vizdoom as vzd


BUTTONS = [] # The buttons that the agent can press, in vizdoom format
BUTTON_NAMES = [] # The buttons that the agent can press
ACTION_NAMES = [] # The actions the agent can take, which are combinations of button pushes

class GameTemplate(ABC):
    '''
    A Game class initializes a vzd.DoomGame() object from the corresponding [NAME]_game.cfg file, and wraps it to implement the step, reset and render functions.
    
    - All children must have at least the following fields. Additional game parameters should be specified in the doc string of __init__().
    - All abstract functions share information by writing into class fields that constitute the game state (which are later returned by the step() function). Such fields should be specified in the doc string of __init__().
    '''
    def __init__(self):
        '''
        
        '''
        self.cfg_path = None
        self.game = create_game(self.cfg_path)
        self.buttons = BUTTONS
        self.button_names = BUTTON_NAMES
        self.action_names = ACTION_NAMES
        self.num_actions = len(self.button_names)
        self.obs_shape = None
        
        game_buttons = [b.name for b in self.game.get_available_buttons()]
        assert set(self.button_names).issubset(set(game_buttons)), "Button name mismatch, game_buttons: {}, self.button_names: {}.".format(game_buttons, self.button_names)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        self.game.set_seed(seed)
        return [seed]

    def step(self, action):
        reward = self.implement_action(action)
        done = self.is_done()
        obs = self.get_obs(done)
        info = vars(self).copy()
        info.pop('game', None) # infos for openai baselines need to be picklable, game is not
        return obs, reward, done, info
    
    def reset(self):
        self.reset_game()
        return self.get_obs(False)
    
    def render(self):
        self.render_state()
    
    @abstractmethod
    def implement_action(self, action):
        pass
    
    @abstractmethod
    def is_done(self):
        pass
    
    @abstractmethod
    def get_obs(self, done):
        pass

    @abstractmethod
    def reset_game(self):
        pass
    
    @abstractmethod
    def render_state(self):
        pass
    
    
def create_game(cfg_path):
    '''Takes the path to a wad file, and creates a game instance.'''
    
    game = vzd.DoomGame()
    game.load_config(cfg_path)
    game.init()
    
    print("Game created. Parameters:")
    print("WAD path:", cfg_path)
    print("Screen format:", game.get_screen_format())
    print("Available buttons:", [b.name for b in game.get_available_buttons()])
    print()
    
    return game
