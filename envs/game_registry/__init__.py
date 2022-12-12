import os
from .simple_exploration_game import SimpleExplorationGame
from .simple_exploration_game_stochastic import SimpleExplorationGameStochastic

GAMES = {}
GAMES['SimpleExplorationGame'] = SimpleExplorationGame(os.path.join(os.path.dirname(__file__), "simple_exploration_game.cfg"))
GAMES['SimpleExplorationGameStochastic'] = SimpleExplorationGameStochastic(os.path.join(os.path.dirname(__file__), "simple_exploration_game.cfg"))

GAME_NAMES = GAMES.keys()