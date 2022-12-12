from .game_template import GameTemplate
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpltPath
import pdb
import time
import cv2
import omg
import vizdoom as vzd
import random
from tqdm import tqdm

#-------------------------------------------------------------------------------------------------------------
# Functions to create DoomGame() objects
#-------------------------------------------------------------------------------------------------------------

def create_game(cfg_path):
    '''Takes the path to a wad file, and creates a game instance.'''
    
    game = vzd.DoomGame()
    game.load_config(cfg_path)
    game.init()
    
    # print("Game created. Parameters:")
    # print("WAD path:", cfg_path)
    # print("Screen format:", game.get_screen_format())
    # print("Available buttons:", [b.name for b in game.get_available_buttons()])
    # print()
    
    return game

def create_game_spawn_at(cfg_path, spawn_location=[-544.,   64.]):
    '''Takes the path to a wad file, and creates a game instance.'''
    
    game_index = random.randint(0, 999999)
    wad = omg.WAD('./simple_exploration_game.wad')
    wad_str = str(wad.data['TEXTMAP'].data, 'utf-8')
    wad_str_modified = wad_str.replace("thing // 37\n{\nx = -544.000;\ny = 64.000;" , "thing // 37\n{{\nx = {:.3f};\ny = {:.3f};".format(spawn_location[0], spawn_location[1]))
    wad.data['TEXTMAP'].data = bytes(wad_str_modified, 'utf-8')
    wad.to_file('./random_spawn_init_files/simple_exploration_game_random_spawn_{:06d}.wad'.format(game_index))
    
    with open (cfg_path, "r") as myfile:
        cfg_data=myfile.readlines()
        
    cfg_data[1] = 'doom_scenario_path = simple_exploration_game_random_spawn_{:06d}.wad\n'.format(game_index)
    cfg_data = ''.join(cfg_data)
    file = open('./random_spawn_init_files/simple_exploration_game_{:06d}.cfg'.format(game_index), 'w')
    file.write(cfg_data)
    file.close()
    
    game = vzd.DoomGame()
    game.load_config('./random_spawn_init_files/simple_exploration_game_{:06d}.cfg'.format(game_index))
    game.init()
    
    # print("Game created. Parameters:")
    # print("WAD path:", cfg_path)
    # print("Screen format:", game.get_screen_format())
    # print("Available buttons:", [b.name for b in game.get_available_buttons()])
    # print()
    
    return game

#-------------------------------------------------------------------------------------------------------------
# Environment dynamics and Visualization Utilities
#-------------------------------------------------------------------------------------------------------------

# The list of actions that can be communicated to vzd.DoomGame()
MOVE_FORWARD = [1, 0, 0, 0, 0]
MOVE_RIGHT = [0, 1, 0, 0, 0]
MOVE_BACKWARD = [0, 0, 1, 0, 0]
MOVE_LEFT = [0, 0, 0, 1, 0]
IDLE = [0, 0, 0, 0, 0] # The convention is that IDLE should always be the last action
TURN_RIGHT_90DEG = [0, 0, 0, 0, 90]

BUTTONS = [MOVE_FORWARD, MOVE_RIGHT, MOVE_BACKWARD, MOVE_LEFT, IDLE] # The buttons that the agent can press, in vizdoom format
BUTTON_NAMES = ['MOVE_FORWARD', 'MOVE_RIGHT', 'MOVE_BACKWARD', 'MOVE_LEFT', 'IDLE'] # The buttons that the agent can press
ACTION_NAMES = ['MOVE_FORWARD', 'MOVE_RIGHT', 'MOVE_BACKWARD', 'MOVE_LEFT', 'IDLE'] # The actions the agent can take, which are combinations of button pushes
        
class SimpleExplorationGame(GameTemplate):
    '''
    A simple exloration game with reversible actions. 
    
    Game State
    -----------
    - self.imgs: shape(4,3,H,W). => set by self.implement_action
    
    '''
    def __init__(self, cfg_path):
        self.cfg_path = cfg_path
        self.game = create_game(self.cfg_path)
        self.buttons = BUTTONS
        self.button_names = BUTTON_NAMES
        self.action_names = ACTION_NAMES
        self.num_actions = len(self.button_names)
        self.obs_shape = (4,3,120,160)
        
        # Initialize game state variables
        self.imgs = np.zeros(self.obs_shape, dtype=np.uint8)
        self.agent_positions = []
        self.total_timesteps = 0
        
        game_buttons = [b.name for b in self.game.get_available_buttons()]
        assert set(self.button_names[:-1]).issubset(set(game_buttons)), "Button name mismatch, game_buttons: {}, self.button_names: {}.".format(game_buttons, self.button_names)

    def implement_action(self, action):
        '''
        Notes:
        ---------
        - IDLE actions are necessary after any movement because the game has momentum

        '''
        # Initialize previous states
        if self.total_timesteps > 0:
            self.previous_imgs = self.imgs
       
        self.total_timesteps = self.total_timesteps + 1
            
        # Move forward
        if action == 0:
            self.game.set_action(self.buttons[0])
            self.game.advance_action(12, False)
            
            self.game.set_action(IDLE)
            self.game.advance_action(50, False)
            self.game.advance_action(1, True)
        
        # Move Right
        elif action == 1:
            self.game.set_action(self.buttons[1])
            self.game.advance_action(12, False)
            
            self.game.set_action(IDLE)
            self.game.advance_action(50, False)
            self.game.advance_action(1, True)
            
        # Move Backward
        elif action == 2:
            self.game.set_action(self.buttons[2])
            self.game.advance_action(12, False)
            
            self.game.set_action(IDLE)
            self.game.advance_action(50, False)
            self.game.advance_action(1, True)
            
        # Move Left
        elif action == 3:
            self.game.set_action(self.buttons[3])
            self.game.advance_action(12, False)
            
            self.game.set_action(IDLE)
            self.game.advance_action(50, False)
            self.game.advance_action(1, True)
            
        # Do nothing
        elif action == 4:
            self.game.set_action(self.buttons[4])
            self.game.advance_action(12, False)
            
            self.game.set_action(IDLE)
            self.game.advance_action(50, False)
            self.game.advance_action(1, True)
       
        self.update_state()
            
    def is_done(self):
        '''The agent explores perpetually.'''
        return False
    
    def get_obs(self, done):
        '''Observations only consist of images.'''
        return self.imgs
    
    def update_state(self):
        imgs=[]
        for i in range(4):
            state = self.game.get_state()
            imgs.append(state.screen_buffer)
            self.game.make_action(TURN_RIGHT_90DEG)
        
        self.imgs = np.stack(imgs)
        self.agent_positions.append(state.game_variables)
        self.update_map()

    def reset_game(self):
        '''Recreates the game and return the first observation.'''
        self.agent_positions = []
        try:
            self.game.close()
        except Exception:
            pass
        self.game = create_game_spawn_at(self.cfg_path)
        self.update_state()
        self.draw_map()
        
    def reset_game_random_spawn(self):
        '''Recreates the game and return the first observation.'''
        self.agent_positions = []
        spawn_loc = self.sample_random_point()
        try:
            self.game.close()
        except Exception:
            pass
        self.game = create_game_spawn_at(self.cfg_path, spawn_location=spawn_loc)
        self.update_state()
        self.draw_map()
        
    def reset_game_spawn_at(self, spawn_loc):
        '''Recreates the game and return the first observation.'''
        self.agent_positions = []
        try:
            self.game.close()
        except Exception:
            pass
        self.game = create_game_spawn_at(self.cfg_path, spawn_location=spawn_loc)
        self.update_state()
        self.draw_map()
    
    def render_state(self):
        '''Displays the screen buffer.'''
        print("Imgs Shape:", self.imgs.shape)
        fig, ax = plt.subplots(1, 4, figsize=(30, 15))
        ax[0].imshow(self.imgs[0].transpose(1,2,0))
        ax[1].imshow(self.imgs[1].transpose(1,2,0))
        ax[2].imshow(self.imgs[2].transpose(1,2,0))
        ax[3].imshow(self.imgs[3].transpose(1,2,0))
        
    def render_state_difference(self):
        '''A detailed render option for debugging.'''
        print("Imgs Shape:", self.imgs.shape)
        fig, ax = plt.subplots(2, 4, figsize=(30, 15))
        ax[0][0].imshow(self.imgs[0].transpose(1,2,0))
        ax[0][1].imshow(self.imgs[1].transpose(1,2,0))
        ax[0][2].imshow(self.imgs[2].transpose(1,2,0))
        ax[0][3].imshow(self.imgs[3].transpose(1,2,0))
        
        if self.total_timesteps > 1:
            ax[1][0].imshow(np.abs(self.imgs[0].transpose(1,2,0) - self.previous_imgs[0].transpose(1,2,0)))
            ax[1][1].imshow(np.abs(self.imgs[1].transpose(1,2,0) - self.previous_imgs[1].transpose(1,2,0)))
            ax[1][2].imshow(np.abs(self.imgs[2].transpose(1,2,0) - self.previous_imgs[2].transpose(1,2,0)))
            ax[1][3].imshow(np.abs(self.imgs[3].transpose(1,2,0) - self.previous_imgs[3].transpose(1,2,0)))
        
    def draw_map(self, res_x=1080):
        '''Sets up everything related to map visualizations and random point sampling.'''
        
        state = self.game.get_state()

        # Get line endpoints and compute the map bound box
        self.line_endpoints = []
        line_endpoints_x = []
        line_endpoints_y = []
        
        for s in state.sectors:
            for l in s.lines:
                if l.is_blocking:
                    self.line_endpoints.append([(l.x1, l.y1), (l.x2, l.y2)])
                    line_endpoints_x = line_endpoints_x + [l.x1, l.x2]
                    line_endpoints_y = line_endpoints_y + [l.y1, l.y2]
                    
        self.object_coordinates = []
        for o in state.objects:
            if o.name != "DoomPlayer":
                self.object_coordinates.append([o.position_x, o.position_y])
                    
        # Order the line end-points (for random starting point sampling)
        line_endpoints_unordered = np.array(self.line_endpoints)  
        line_endpoints_ordered = np.zeros_like(line_endpoints_unordered)
        for i in range(line_endpoints_unordered.shape[0]):
            for j in range(0, line_endpoints_unordered.shape[0]):
                if i == 0:
                    line_endpoints_ordered[i,:,:] = line_endpoints_unordered[i,:,:]
                    continue
                elif (line_endpoints_unordered[j,0,:] == line_endpoints_ordered[i-1,1,:]).all():
                    line_endpoints_ordered[i,:,:] = line_endpoints_unordered[j,:,:]
                    
        # Create the environment borders and bounding box (for random starting point sampling)
        self.rng = np.random.default_rng()
        self.env_borders = mpltPath.Path(line_endpoints_ordered[:,0,:])
        self.x_max, self.x_min, self.y_max, self.y_min = int(max(line_endpoints_x) * 1.1), int(min(line_endpoints_x) * 1.1), int(max(line_endpoints_y) * 1.1), int(min(line_endpoints_y) * 1.1)
        
        # Draw Map
        res_y = int(res_x * (self.y_max - self.y_min) / (self.x_max - self.x_min))
        self.map = np.ones([res_x, res_y, 3], dtype=np.uint8) * 255
        for endpoints in self.line_endpoints:
            self.map = cv2.line(self.map, self.map_to_pixel(endpoints[0]), self.map_to_pixel(endpoints[1]), (0, 0, 0), thickness=2)
            
        for object_coordinate_pair in self.object_coordinates :
            object_pixel_coord = self.map_to_pixel(object_coordinate_pair)
            # self.map = cv2.rectangle(self.map, (object_pixel_coord[0] - 2, object_pixel_coord[1] - 2), (object_pixel_coord[0] + 2, object_pixel_coord[1] + 2), (0, 0, 0), thickness=-1)
            self.map = cv2.circle(self.map, self.map_to_pixel(object_coordinate_pair), 3, (0.1,0.1,0.1), thickness=-1)
            self.map = cv2.circle(self.map, self.map_to_pixel(object_coordinate_pair), 4, (0.5,0.5,0.5), thickness=1)
        
        self.map = cv2.circle(self.map, self.map_to_pixel(self.agent_positions[-1]), 4, (255, 0, 0), thickness=-1)
        
    def sample_random_point(self):
        '''Samples a random navigable point.'''
        
        valid_sample = False
        while(not valid_sample):
            x_rand = self.rng.uniform(low=self.x_min, high=self.x_max)
            y_rand = self.rng.uniform(low=self.y_min, high=self.y_max)
            valid_sample = self.env_borders.contains_point([x_rand, y_rand], radius=75)
        
        return (x_rand, y_rand)
    
    def sample_random_point_near(self, origin, radius):
        '''Samples a random navigable point.'''
        
        valid_sample = False
        while(not valid_sample):
            x_rand = self.rng.uniform(low=self.x_min, high=self.x_max)
            y_rand = self.rng.uniform(low=self.y_min, high=self.y_max)
            within_borders = self.env_borders.contains_point([x_rand, y_rand], radius=75)
            within_radius = np.linalg.norm(origin[:2] - np.array([x_rand, y_rand])) < radius
            valid_sample = within_borders and within_radius
        
        return (x_rand, y_rand)
    
    def sample_random_point_within(self, origin, radius_low, radius_high):
        '''Samples a random navigable point.'''
        
        valid_sample = False
        while(not valid_sample):
            x_rand = self.rng.uniform(low=self.x_min, high=self.x_max)
            y_rand = self.rng.uniform(low=self.y_min, high=self.y_max)
            within_borders = self.env_borders.contains_point([x_rand, y_rand], radius=75)
            distance = np.linalg.norm(origin[:2] - np.array([x_rand, y_rand]))
            within_radius = (distance > radius_low) and (distance < radius_high)
            valid_sample = within_borders and within_radius
        
        return (x_rand, y_rand)
        
    def map_to_pixel(self, point, res_x=1080):
        '''Convert the coordinates of a point from map to pixel frame.'''
        
        x_new = (point[0] - self.x_min) / (self.x_max - self.x_min) * res_x 
        y_new = ((point[1] - self.y_min) / (self.x_max - self.x_min) * res_x) * -1 + (res_x * (self.y_max - self.y_min) / (self.x_max - self.x_min)) # OpenCV starts (0,0) from upper-left corner
                    
        return (int(x_new), int(y_new))
        
    def update_map(self):
        '''Updates the agent map.'''
        
        try:
            self.map = cv2.circle(self.map, self.map_to_pixel(self.agent_positions[-2]), 4, (0, 0, 255), thickness=-1)
            self.map = cv2.line(self.map, self.map_to_pixel(self.agent_positions[-2]), self.map_to_pixel(self.agent_positions[-1]), (0, 0, 255), thickness=1)
        except:
            pass
        
        try:
            self.map = cv2.circle(self.map, self.map_to_pixel(self.agent_positions[-1]), 4, (255, 0, 0), thickness=-1)
        except:
            pass
        
    def draw_vis_map(self, res_x=1080, draw_objects=True):
        '''Draws the current map.'''
        
        # Draw Map
        res_y = int(res_x * (self.y_max - self.y_min) / (self.x_max - self.x_min))
        self.vis_map = np.ones([res_x, res_y, 3], dtype=np.uint8) * 255
        for endpoints in self.line_endpoints:
            self.vis_map = cv2.line(self.vis_map, self.map_to_pixel(endpoints[0]), self.map_to_pixel(endpoints[1]), (0, 0, 0), thickness=4, lineType=cv2.LINE_AA)
         
        if draw_objects:
            for object_coordinate_pair in self.object_coordinates:
                # object_pixel_coord = self.map_to_pixel(object_coordinate_pair)
                # self.vis_map = cv2.rectangle(self.vis_map, (object_pixel_coord[0] - 2, object_pixel_coord[1] - 2), (object_pixel_coord[0] + 2, object_pixel_coord[1] + 2), (0, 0, 0), thickness=-1)
                self.vis_map = cv2.circle(self.vis_map, self.map_to_pixel(object_coordinate_pair), 6, (0.1,0.1,0.1), thickness=-1, lineType=cv2.LINE_AA)
                self.vis_map = cv2.circle(self.vis_map, self.map_to_pixel(object_coordinate_pair), 6, (0.5,0.5,0.5), thickness=2, lineType=cv2.LINE_AA)
        
    def draw_points(self, points, colors=None):
        '''
        Draws points on the current map.
        - points: shape (N, >2)
        - colors: shape (N, 3)
        
        '''
        if colors is None: 
            colors = [(253, 216, 112) for i in range(len(points))]
            
        for point, color in zip(points, colors):
            object_pixel_coord = self.map_to_pixel(point)
            self.vis_map = cv2.rectangle(self.vis_map, (object_pixel_coord[0] - 7, object_pixel_coord[1] - 7), (object_pixel_coord[0] + 7, object_pixel_coord[1] + 7), color, thickness=-1, lineType=cv2.LINE_AA)
            self.vis_map = cv2.rectangle(self.vis_map, (object_pixel_coord[0] - 8, object_pixel_coord[1] - 8), (object_pixel_coord[0] + 8, object_pixel_coord[1] + 8), (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            # self.vis_map = cv2.circle(self.vis_map, self.map_to_pixel(point), 2, color, thickness=-1)
            # self.vis_map = cv2.circle(self.vis_map, self.map_to_pixel(point), 3, (0,0,0), thickness=1)
            
        for object_coordinate_pair in self.object_coordinates:
                # object_pixel_coord = self.map_to_pixel(object_coordinate_pair)
                # self.vis_map = cv2.rectangle(self.vis_map, (object_pixel_coord[0] - 2, object_pixel_coord[1] - 2), (object_pixel_coord[0] + 2, object_pixel_coord[1] + 2), (0, 0, 0), thickness=-1)
                self.vis_map = cv2.circle(self.vis_map, self.map_to_pixel(object_coordinate_pair), 2, (0.1,0.1,0.1), thickness=-1, lineType=cv2.LINE_AA)
                self.vis_map = cv2.circle(self.vis_map, self.map_to_pixel(object_coordinate_pair), 2, (0.5,0.5,0.5), thickness=2, lineType=cv2.LINE_AA)
                
    def draw_points_marked(self, points, colors=None):
        '''
        Draws points on the current map.
        - points: shape (N, >2)
        - colors: shape (N, 3)
        
        '''
        if colors is None: 
            colors = [(253, 216, 112) for i in range(len(points))]
            
        for point, color in zip(points, colors):
            object_pixel_coord = self.map_to_pixel(point)
            self.vis_map = cv2.rectangle(self.vis_map, (object_pixel_coord[0] - 9, object_pixel_coord[1] - 9), (object_pixel_coord[0] + 9, object_pixel_coord[1] + 9), (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
            self.vis_map = cv2.rectangle(self.vis_map, (object_pixel_coord[0] - 10, object_pixel_coord[1] - 10), (object_pixel_coord[0] + 10, object_pixel_coord[1] + 10), (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            self.vis_map = cv2.line(self.vis_map, (object_pixel_coord[0] - 10, object_pixel_coord[1] - 10), (object_pixel_coord[0] + 10, object_pixel_coord[1] + 10), (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            self.vis_map = cv2.line(self.vis_map, (object_pixel_coord[0] + 10, object_pixel_coord[1] - 10), (object_pixel_coord[0] - 10, object_pixel_coord[1] + 10), (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            # self.vis_map = cv2.circle(self.vis_map, self.map_to_pixel(point), 2, color, thickness=-1)
            # self.vis_map = cv2.circle(self.vis_map, self.map_to_pixel(point), 3, (0,0,0), thickness=1)
            
        for object_coordinate_pair in self.object_coordinates:
                # object_pixel_coord = self.map_to_pixel(object_coordinate_pair)
                # self.vis_map = cv2.rectangle(self.vis_map, (object_pixel_coord[0] - 2, object_pixel_coord[1] - 2), (object_pixel_coord[0] + 2, object_pixel_coord[1] + 2), (0, 0, 0), thickness=-1)
                self.vis_map = cv2.circle(self.vis_map, self.map_to_pixel(object_coordinate_pair), 2, (0.1,0.1,0.1), thickness=-1, lineType=cv2.LINE_AA)
                self.vis_map = cv2.circle(self.vis_map, self.map_to_pixel(object_coordinate_pair), 2, (0.5,0.5,0.5), thickness=2, lineType=cv2.LINE_AA)
                
    def draw_point(self, point):
        '''
        Draws points on the current map.
        - points: shape (N, >2)
        - colors: shape (N, 3)
        
        '''
        color = (253, 216, 112)
        object_pixel_coord = self.map_to_pixel(point)
        self.vis_map = cv2.rectangle(self.vis_map, (object_pixel_coord[0] - 9, object_pixel_coord[1] - 9), (object_pixel_coord[0] + 9, object_pixel_coord[1] + 9), (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
        self.vis_map = cv2.rectangle(self.vis_map, (object_pixel_coord[0] - 10, object_pixel_coord[1] - 10), (object_pixel_coord[0] + 10, object_pixel_coord[1] + 10), (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        self.vis_map = cv2.line(self.vis_map, (object_pixel_coord[0] - 10, object_pixel_coord[1] - 10), (object_pixel_coord[0] + 10, object_pixel_coord[1] + 10), (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        self.vis_map = cv2.line(self.vis_map, (object_pixel_coord[0] + 10, object_pixel_coord[1] - 10), (object_pixel_coord[0] - 10, object_pixel_coord[1] + 10), (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        # self.vis_map = cv2.circle(self.vis_map, self.map_to_pixel(point), 2, color, thickness=-1)
        # self.vis_map = cv2.circle(self.vis_map, self.map_to_pixel(point), 2, (0,0,0), thickness=4)
            
        for object_coordinate_pair in self.object_coordinates:
                # object_pixel_coord = self.map_to_pixel(object_coordinate_pair)
                # self.vis_map = cv2.rectangle(self.vis_map, (object_pixel_coord[0] - 2, object_pixel_coord[1] - 2), (object_pixel_coord[0] + 2, object_pixel_coord[1] + 2), (0, 0, 0), thickness=-1)
                self.vis_map = cv2.circle(self.vis_map, self.map_to_pixel(object_coordinate_pair), 2, (0.1,0.1,0.1), thickness=-1, lineType=cv2.LINE_AA)
                self.vis_map = cv2.circle(self.vis_map, self.map_to_pixel(object_coordinate_pair), 2, (0.5,0.5,0.5), thickness=2, lineType=cv2.LINE_AA)
                
                
    def draw_start_end_points(self, start_point, end_point, draw_start=True, draw_end=True):
        '''
        Draws points on the current map.
        - points: shape (N, >2)
        - colors: shape (N, 3)
        
        '''
       
        color = (253, 216, 112)
        if draw_start:
            object_pixel_coord = self.map_to_pixel(start_point)
            self.vis_map = cv2.rectangle(self.vis_map, (object_pixel_coord[0] - 10, object_pixel_coord[1] - 10), (object_pixel_coord[0] + 10, object_pixel_coord[1] + 10), color, thickness=-1, lineType=cv2.LINE_AA)
            self.vis_map = cv2.rectangle(self.vis_map, (object_pixel_coord[0] - 10, object_pixel_coord[1] - 10), (object_pixel_coord[0] + 10, object_pixel_coord[1] + 10), (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            self.vis_map = cv2.line(self.vis_map, (object_pixel_coord[0] - 10, object_pixel_coord[1] - 10), (object_pixel_coord[0] + 10, object_pixel_coord[1] + 10), (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            
        if draw_end:
            object_pixel_coord = self.map_to_pixel(end_point)
            self.vis_map = cv2.rectangle(self.vis_map, (object_pixel_coord[0] - 10, object_pixel_coord[1] - 10), (object_pixel_coord[0] + 10, object_pixel_coord[1] + 10), (0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
        
        for object_coordinate_pair in self.object_coordinates:
                # object_pixel_coord = self.map_to_pixel(object_coordinate_pair)
                # self.vis_map = cv2.rectangle(self.vis_map, (object_pixel_coord[0] - 2, object_pixel_coord[1] - 2), (object_pixel_coord[0] + 2, object_pixel_coord[1] + 2), (0, 0, 0), thickness=-1)
                self.vis_map = cv2.circle(self.vis_map, self.map_to_pixel(object_coordinate_pair), 2, (0.1,0.1,0.1), thickness=-1, lineType=cv2.LINE_AA)
                self.vis_map = cv2.circle(self.vis_map, self.map_to_pixel(object_coordinate_pair), 2, (0.5,0.5,0.5), thickness=2, lineType=cv2.LINE_AA)
            
    def draw_trajectory(self, points, colors=None, draw_endpoints=True):
        '''
        Draws points on the current map.
        - points: shape (N, >2)
        - colors: shape (N, 3)
        
        '''
        if colors is None: 
            colors = [(253, 216, 112) for i in range(len(points))]
            
        for i in range(1, len(points)):
            self.vis_map = cv2.line(self.vis_map, self.map_to_pixel(points[i-1]), self.map_to_pixel(points[i]), (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            
        for point, color in zip(points, colors):
            object_pixel_coord = self.map_to_pixel(point)
            self.vis_map = cv2.rectangle(self.vis_map, (object_pixel_coord[0] - 7, object_pixel_coord[1] - 7), (object_pixel_coord[0] + 7, object_pixel_coord[1] + 7), color, thickness=-1, lineType=cv2.LINE_AA)
            self.vis_map = cv2.rectangle(self.vis_map, (object_pixel_coord[0] - 8, object_pixel_coord[1] - 8), (object_pixel_coord[0] + 8, object_pixel_coord[1] + 8), (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            
        if draw_endpoints:    
            object_pixel_coord = self.map_to_pixel(points[0])
            self.vis_map = cv2.rectangle(self.vis_map, (object_pixel_coord[0] - 10, object_pixel_coord[1] - 10), (object_pixel_coord[0] + 10, object_pixel_coord[1] + 10), colors[-1], thickness=-1, lineType=cv2.LINE_AA)
            self.vis_map = cv2.rectangle(self.vis_map, (object_pixel_coord[0] - 10, object_pixel_coord[1] - 10), (object_pixel_coord[0] + 10, object_pixel_coord[1] + 10), (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            self.vis_map = cv2.line(self.vis_map, (object_pixel_coord[0] - 10, object_pixel_coord[1] - 10), (object_pixel_coord[0] + 10, object_pixel_coord[1] + 10), (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)

            object_pixel_coord = self.map_to_pixel(points[-1])
            self.vis_map = cv2.rectangle(self.vis_map, (object_pixel_coord[0] - 10, object_pixel_coord[1] - 10), (object_pixel_coord[0] + 10, object_pixel_coord[1] + 10), (0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
        
        for object_coordinate_pair in self.object_coordinates:
                # object_pixel_coord = self.map_to_pixel(object_coordinate_pair)
                # self.vis_map = cv2.rectangle(self.vis_map, (object_pixel_coord[0] - 2, object_pixel_coord[1] - 2), (object_pixel_coord[0] + 2, object_pixel_coord[1] + 2), (0, 0, 0), thickness=-1)
                self.vis_map = cv2.circle(self.vis_map, self.map_to_pixel(object_coordinate_pair), 2, (0.1,0.1,0.1), thickness=-1, lineType=cv2.LINE_AA)
                self.vis_map = cv2.circle(self.vis_map, self.map_to_pixel(object_coordinate_pair), 2, (0.5,0.5,0.5), thickness=2, lineType=cv2.LINE_AA)
                
    def draw_trajectory_marked(self, points, colors=None):
        '''
        Draws points on the current map.
        - points: shape (N, >2)
        - colors: shape (N, 3)
        
        '''
        if colors is None: 
            colors = [(253, 216, 112) for i in range(len(points))]
            
        for i in range(1, len(points)):
            self.vis_map = cv2.line(self.vis_map, self.map_to_pixel(points[i-1]), self.map_to_pixel(points[i]), (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            
        for point, color in zip(points, colors):
            object_pixel_coord = self.map_to_pixel(point)
            self.vis_map = cv2.rectangle(self.vis_map, (object_pixel_coord[0] - 9, object_pixel_coord[1] - 9), (object_pixel_coord[0] + 9, object_pixel_coord[1] + 9), (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
            self.vis_map = cv2.rectangle(self.vis_map, (object_pixel_coord[0] - 10, object_pixel_coord[1] - 10), (object_pixel_coord[0] + 10, object_pixel_coord[1] + 10), (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            self.vis_map = cv2.line(self.vis_map, (object_pixel_coord[0] - 10, object_pixel_coord[1] - 10), (object_pixel_coord[0] + 10, object_pixel_coord[1] + 10), (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            self.vis_map = cv2.line(self.vis_map, (object_pixel_coord[0] + 10, object_pixel_coord[1] - 10), (object_pixel_coord[0] - 10, object_pixel_coord[1] + 10), (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            
        object_pixel_coord = self.map_to_pixel(points[0])
        self.vis_map = cv2.rectangle(self.vis_map, (object_pixel_coord[0] - 10, object_pixel_coord[1] - 10), (object_pixel_coord[0] + 10, object_pixel_coord[1] + 10), colors[-1], thickness=-1, lineType=cv2.LINE_AA)
        self.vis_map = cv2.rectangle(self.vis_map, (object_pixel_coord[0] - 10, object_pixel_coord[1] - 10), (object_pixel_coord[0] + 10, object_pixel_coord[1] + 10), (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        self.vis_map = cv2.line(self.vis_map, (object_pixel_coord[0] - 10, object_pixel_coord[1] - 10), (object_pixel_coord[0] + 10, object_pixel_coord[1] + 10), (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            
        object_pixel_coord = self.map_to_pixel(points[-1])
        self.vis_map = cv2.rectangle(self.vis_map, (object_pixel_coord[0] - 10, object_pixel_coord[1] - 10), (object_pixel_coord[0] + 10, object_pixel_coord[1] + 10), (0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
        
        for object_coordinate_pair in self.object_coordinates:
                # object_pixel_coord = self.map_to_pixel(object_coordinate_pair)
                # self.vis_map = cv2.rectangle(self.vis_map, (object_pixel_coord[0] - 2, object_pixel_coord[1] - 2), (object_pixel_coord[0] + 2, object_pixel_coord[1] + 2), (0, 0, 0), thickness=-1)
                self.vis_map = cv2.circle(self.vis_map, self.map_to_pixel(object_coordinate_pair), 2, (0.1,0.1,0.1), thickness=-1, lineType=cv2.LINE_AA)
                self.vis_map = cv2.circle(self.vis_map, self.map_to_pixel(object_coordinate_pair), 2, (0.5,0.5,0.5), thickness=2, lineType=cv2.LINE_AA)
                
    def draw_graph(self, positions, pairwise_distances, edge_distance_threshold):
        '''
        Draws a graph.
        - positions: shape (N, 3)
        - pairwise_distances: shape (N, N)
        - edge_distance_threshold: scalar
        
        '''
            
        color = (253, 216, 112)
        num_samples = positions.shape[0]
        
        for i in range(num_samples):
            for j in range(num_samples):
                pT_distance = pairwise_distances[i, j]
                if pT_distance < edge_distance_threshold:
                    self.vis_map = cv2.line(self.vis_map, self.map_to_pixel(positions[i]), self.map_to_pixel(positions[j]), (3, 44, 46), thickness=2, lineType=cv2.LINE_AA)
                    
        for i in range(num_samples):
            object_pixel_coord = self.map_to_pixel(positions[i])
            self.vis_map = cv2.rectangle(self.vis_map, (object_pixel_coord[0] - 7, object_pixel_coord[1] - 7), (object_pixel_coord[0] + 7, object_pixel_coord[1] + 7), color, thickness=-1, lineType=cv2.LINE_AA)
            self.vis_map = cv2.rectangle(self.vis_map, (object_pixel_coord[0] - 8, object_pixel_coord[1] - 8), (object_pixel_coord[0] + 8, object_pixel_coord[1] + 8), (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
                    
    def draw_q_graph(self, positions, pairwise_distances, edge_distance_threshold):
        '''
        Draws a graph.
        - positions: shape (N, 3)
        - pairwise_distances: shape (N, N)
        - edge_distance_threshold: scalar
        
        '''
            
        color = (253, 216, 112)
        num_samples = positions.shape[0]
        
        for i in range(num_samples):
            for j in range(num_samples):
                edge_distance = pairwise_distances[i, j]
                if edge_distance > edge_distance_threshold:
                    self.vis_map = cv2.line(self.vis_map, self.map_to_pixel(positions[i]), self.map_to_pixel(positions[j]), (3, 44, 46), thickness=2, lineType=cv2.LINE_AA)
                    
        for i in range(num_samples):
            object_pixel_coord = self.map_to_pixel(positions[i])
            self.vis_map = cv2.rectangle(self.vis_map, (object_pixel_coord[0] - 7, object_pixel_coord[1] - 7), (object_pixel_coord[0] + 7, object_pixel_coord[1] + 7), color, thickness=-1, lineType=cv2.LINE_AA)
            self.vis_map = cv2.rectangle(self.vis_map, (object_pixel_coord[0] - 8, object_pixel_coord[1] - 8), (object_pixel_coord[0] + 8, object_pixel_coord[1] + 8), (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
                    
    def draw_graph_filtered(self, positions, pairwise_distances, pairwise_retrieval_distances, edge_distance_threshold, pT_coefficient=1.0):
        '''
        Draws a graph.
        - positions: shape (N, 3)
        - pairwise_distances: shape (N, N)
        - edge_distance_threshold: scalar
        
        '''
        
        color = (253, 216, 112)
        num_samples = positions.shape[0]
        
        for i in range(num_samples):
            for j in range(num_samples):
                pT_distance = pairwise_distances[i, j]
                retrieval_distance = pairwise_retrieval_distances[i, j]
                if (retrieval_distance != -1) and (pT_distance < edge_distance_threshold) and (retrieval_distance < (pT_distance * pT_coefficient)):
                    self.vis_map = cv2.line(self.vis_map, self.map_to_pixel(positions[i]), self.map_to_pixel(positions[j]), (3, 44, 46), thickness=2, lineType=cv2.LINE_AA)
                    
        for i in range(num_samples):
            object_pixel_coord = self.map_to_pixel(positions[i])
            self.vis_map = cv2.rectangle(self.vis_map, (object_pixel_coord[0] - 5, object_pixel_coord[1] - 5), (object_pixel_coord[0] + 5, object_pixel_coord[1] + 5), color, thickness=-1, lineType=cv2.LINE_AA)
            self.vis_map = cv2.rectangle(self.vis_map, (object_pixel_coord[0] - 6, object_pixel_coord[1] - 6), (object_pixel_coord[0] + 6, object_pixel_coord[1] + 6), (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
                    
    def draw_RRT_graph(self, goal_position, tree_node_positions, tree_node_parent_tree_ids):
        '''
        Draws a graph.
        - tree_node_positions: shape (N, 3)
        - tree_node_parent_tree_ids: shape [N]
        '''
            
        color = (253, 216, 112)
        num_samples = len(tree_node_positions)
        for i in range(1, num_samples):
            object_pixel_coord = self.map_to_pixel(tree_node_positions[i])
            self.vis_map = cv2.line(self.vis_map, self.map_to_pixel(tree_node_positions[i]), self.map_to_pixel(tree_node_positions[tree_node_parent_tree_ids[i]]), (3, 44, 46), thickness=2, lineType=cv2.LINE_AA)
            
        for i in range(1, num_samples):
            object_pixel_coord = self.map_to_pixel(tree_node_positions[i])
            self.vis_map = cv2.rectangle(self.vis_map, (object_pixel_coord[0] - 5, object_pixel_coord[1] - 5), (object_pixel_coord[0] + 5, object_pixel_coord[1] + 5), color, thickness=-1, lineType=cv2.LINE_AA)
            self.vis_map = cv2.rectangle(self.vis_map, (object_pixel_coord[0] - 6, object_pixel_coord[1] - 6), (object_pixel_coord[0] + 6, object_pixel_coord[1] + 6), (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
            
        object_pixel_coord = self.map_to_pixel(tree_node_positions[0])
        self.vis_map = cv2.rectangle(self.vis_map, (object_pixel_coord[0] - 7, object_pixel_coord[1] - 7), (object_pixel_coord[0] + 7, object_pixel_coord[1] + 7), color, thickness=-1, lineType=cv2.LINE_AA)
        self.vis_map = cv2.rectangle(self.vis_map, (object_pixel_coord[0] - 7, object_pixel_coord[1] - 7), (object_pixel_coord[0] + 7, object_pixel_coord[1] + 7), (0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        self.vis_map = cv2.line(self.vis_map, (object_pixel_coord[0] - 8, object_pixel_coord[1] - 8), (object_pixel_coord[0] + 8, object_pixel_coord[1] + 8), (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
            
        object_pixel_coord = self.map_to_pixel(goal_position)
        self.vis_map = cv2.rectangle(self.vis_map, (object_pixel_coord[0] - 7, object_pixel_coord[1] - 7), (object_pixel_coord[0] + 7, object_pixel_coord[1] + 7), (0, 0, 0), thickness=-1, lineType=cv2.LINE_AA)
            
    def draw_RRT_graph_animated(self, goal_position, tree_node_positions, tree_node_parent_tree_ids, random_point_positions, retrieval_trajectory_positions):
        '''
        Draws a graph.
        - tree_node_positions: shape (N, 3)
        - tree_node_parent_tree_ids: shape [N]
        '''
        
        self.vis_map = cv2.circle(self.vis_map, self.map_to_pixel(tree_node_positions[0]), 4, (255, 0, 0), thickness=-1)
        self.vis_map = cv2.circle(self.vis_map, self.map_to_pixel(goal_position), 4, (0, 255, 0), thickness=-1)
        
        mid_frames = []
        num_samples = len(tree_node_positions)
        for i in tqdm(range(1, num_samples)):
            mid_frame = self.vis_map.copy()
            
            # Draw the random point
            mid_frame = cv2.circle(mid_frame, self.map_to_pixel(random_point_positions[i-1]), 4, (255, 0, 0), thickness=-1)
            mid_frame = cv2.circle(mid_frame, self.map_to_pixel(tree_node_positions[0]), 4, (0, 0, 255), thickness=-1)
            mid_frame = cv2.circle(mid_frame, self.map_to_pixel(goal_position), 4, (0, 255, 0), thickness=-1)
            mid_frames.append(mid_frame.copy())
            
            # Draw the retrieved trajectory
            for j in range(1, len(retrieval_trajectory_positions[i-1])):
                mid_frame = cv2.line(mid_frame, self.map_to_pixel(retrieval_trajectory_positions[i-1][j-1]), self.map_to_pixel(retrieval_trajectory_positions[i-1][j]), (0, 0, 0), thickness=1)
            for j in range(1, len(retrieval_trajectory_positions[i-1])):
                mid_frame = cv2.circle(mid_frame, self.map_to_pixel(retrieval_trajectory_positions[i-1][j]), 4, (0, 0, 255), thickness=-1)
                
            mid_frame = cv2.circle(mid_frame, self.map_to_pixel(tree_node_positions[0]), 4, (0, 0, 255), thickness=-1)
            mid_frame = cv2.circle(mid_frame, self.map_to_pixel(goal_position), 4, (0, 255, 0), thickness=-1)
            mid_frames.append(mid_frame.copy())
            
            # Draw the newly added edge and node
            self.vis_map = cv2.circle(self.vis_map, self.map_to_pixel(tree_node_positions[i]), 4, (0, 0, 255), thickness=-1)
            self.vis_map = cv2.circle(self.vis_map, self.map_to_pixel(tree_node_positions[0]), 4, (255, 0, 0), thickness=-1)
            self.vis_map = cv2.circle(self.vis_map, self.map_to_pixel(goal_position), 4, (0, 255, 0), thickness=-1)
            mid_frames.append(self.vis_map.copy())
            self.vis_map = cv2.line(self.vis_map, self.map_to_pixel(tree_node_positions[i]), self.map_to_pixel(tree_node_positions[tree_node_parent_tree_ids[i]]), (3, 44, 46), thickness=1)
            self.vis_map = cv2.circle(self.vis_map, self.map_to_pixel(tree_node_positions[0]), 4, (255, 0, 0), thickness=-1)
            self.vis_map = cv2.circle(self.vis_map, self.map_to_pixel(goal_position), 4, (0, 255, 0), thickness=-1)
            mid_frames.append(self.vis_map.copy())
            
        return mid_frames
    
    def draw_RRT_star_graph_animated(self, goal_position, tree_node_positions_list, tree_node_parent_tree_ids_list, random_point_positions, retrieval_trajectory_positions):
        '''
        Draws a graph.
        - tree_node_positions: shape (N, 3)
        - tree_node_parent_tree_ids: shape [N]
        '''
        
        tree_node_positions = tree_node_positions_list[-1]
        tree_node_parent_tree_ids = tree_node_parent_tree_ids_list[-1]
        self.vis_map = cv2.circle(self.vis_map, self.map_to_pixel(tree_node_positions[0]), 4, (0, 0, 255), thickness=-1)
        self.vis_map = cv2.circle(self.vis_map, self.map_to_pixel(goal_position), 4, (0, 255, 0), thickness=-1)
        
        mid_frames = []
        num_samples = len(tree_node_positions)
        draw_count = 0
        for i in tqdm(range(1, num_samples)):
            # Draw the newly added edge and node
            self.draw_vis_map()
            self.draw_RRT_graph(goal_position, tree_node_positions_list[draw_count], tree_node_parent_tree_ids_list[draw_count])
            mid_frames.append(self.vis_map.copy())
            draw_count = draw_count + 1
            
            mid_frame = self.vis_map.copy()
            
            # Draw the random point
            mid_frame = cv2.circle(mid_frame, self.map_to_pixel(random_point_positions[i-1]), 4, (255, 0, 0), thickness=-1)
            mid_frames.append(mid_frame.copy())
            
            # Draw the retrieved trajectory
            for j in range(1, len(retrieval_trajectory_positions[i-1])):
                mid_frame = cv2.line(mid_frame, self.map_to_pixel(retrieval_trajectory_positions[i-1][j-1]), self.map_to_pixel(retrieval_trajectory_positions[i-1][j]), (0, 0, 0), thickness=1)
            for j in range(1, len(retrieval_trajectory_positions[i-1])):
                mid_frame = cv2.circle(mid_frame, self.map_to_pixel(retrieval_trajectory_positions[i-1][j]), 4, (0, 0, 255), thickness=-1)
            mid_frames.append(mid_frame.copy())
 
            self.draw_vis_map()
            self.draw_RRT_graph(goal_position, tree_node_positions_list[draw_count], tree_node_parent_tree_ids_list[draw_count])
            mid_frames.append(self.vis_map.copy())
            draw_count = draw_count + 1
            
        return mid_frames
        
    def draw_agent(self):
        '''
        Draws points on the current map.
        - points: shape (N, >2)
        - colors: shape (N, 3)
        
        '''
        self.vis_map = cv2.circle(self.vis_map, self.map_to_pixel(self.agent_positions[-1]), 4, (255, 0, 0), thickness=-1)


        
    