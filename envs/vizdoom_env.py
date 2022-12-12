import numpy as np
import gym
from gym import error, spaces, utils
from .game_registry import GAMES



class VizdoomEnv(gym.Env):
  def __init__(self, game_name):
    self.game = GAMES[game_name]

  def seed(self, seed=None):
    return self.game.seed(seed)

  def step(self, action):
    return self.game.step(action)

  def reset(self):
    return self.game.reset()

  def render(self, mode='rgb_array'):
    self.game.render()