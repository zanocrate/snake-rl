import gym
from gym import spaces
import pygame # for rendering?
import numpy as np


class SnakeEnv(gym.Env):

    metadata = {
        'render_modes' : ['human'],
        'action_space_types' : ['absolute','relative'],
        'render_fps' : 4
    }

    def __init__(
        self,
        render_mode=None,
        width = 15,
        height = 15,
        periodic = False,
        food_reward = 1,
        terminated_penalty = -1,
        step_penalty = 0,
        max_steps=100, # maximum number of steps before terminated
        action_space_type = 'absolute',
        observation_type=1,
        history_length=1):

        # game settings
        self.width = width
        self.height = height
        self.periodic = periodic # PBC

        # RL parameters
        self.food_reward = food_reward
        self.terminated_penalty = terminated_penalty
        self.step_penalty = step_penalty
        self.max_steps = max_steps

        # action space settings
        assert action_space_type in self.metadata["action_space_types"]
        self.action_space_type = action_space_type

        # observation settings
        self.observation_type = observation_type
        self.history_length = history_length

        #   #   # initializing history attributes based on observation type and filling them with zeroes

        # simple screen output
        if self.observation_type == 1:
            
            # define history as a np matrix of size (history_len,width,height)
            self.history = np.zeros((self.history_length,self.width,self.height),dtype=int)

            # define the observation space
            self.observation_space = spaces.Box(low=-1,high=1,shape=(self.width,self.height),dtype=int) 
        
        # both screen having numbered snake pieces and direction output
        elif self.observation_type == 2:

            # define history as dict of matrices
            self.history = {
                'screen' : np.zeros((self.history_length,self.width,self.height),dtype=int),
                'direction' : np.zeros((self.history_length,4),dtype=int)
            }
            # define observation space as dictspace
            self.observation_space = spaces.Dict(
                {
                    "screen" : spaces.Box(low=-1,high=width*height,shape=(width,height),dtype=int), # the snake can have at most length n x m
                    "direction" : spaces.MultiDiscrete(2*np.ones(4)) # each possible direction the snake is moving
                }
            )

        # define action space
        if self.action_space_type == 'absolute':
            self.action_space = spaces.Discrete(4) # each of the 4 direction as input
        elif self.action_space_type == 'relative':
            self.action_space = spaces.Discrete(3) # input relative to the head
            self.rotation_matrix = {
                0 : np.array([[0,-1],[1,0]]), # rotate 90 degree CCW
                1 : np.eye(2,dtype=int), # stay the same
                2 : np.array([[0,1],[-1,0]]) # rotate 90 degrees CW
            }


        # look up tables for the action index
        self._int_to_direction = {
            0 : np.array([0,1]), # UP
            1 : np.array([0,-1]), # DOWN
            2 : np.array([1,0]), # RIGHT
            3 : np.array([-1,0]) # LEFT
        }

        self._direction_to_onehot = {
            0 : np.array([1.,0,0,0]),
            1 : np.array([0,1.,0,0]),
            2 : np.array([0,0,1.,0]),
            3 : np.array([0,0,0,1.]),
        }

        # set render mode
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window_width = 1024 # pygame window?
        self.window_height = 1024
        self.window = None
        self.clock = None

    
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment.
        Returns
        -------
        if self.observation_type == 1:
            observation : is a (history_length,width,height) numpy matrix representing the screen
        if self.observation_type == 2:
            observation : dict, observation['screen'] is a (history_length,width,height) numpy matrix representing the screen
                                observation['direction'] is a (history_length,4) numpy matrix representing the direction
        """

        # We need the following line to seed self.np_random
        # it is the default rng inherited from the gym.Env class
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        # this will be an rray like [[x0,y0]]
        # then whenever we eat an apple the list will append along the 0 axis
        # like [[x1,y1],
        #       [x0,y0]]
        self._snake = self.np_random.integers((self.width,self.height),size=(1,2))
        self._current_direction = self._int_to_direction[self.np_random.integers(4)] # choose one of the four directions at random

        # reset the counter for steps without food
        self._steps_no_food = 0

        # generate food somewhere
        self._spawn_food()

        # generate observation and info
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def _get_obs(self):
        """
        Updates history with a new frame using current game state, and returns observation.
        Returns
        -------
        if self.observation_type == 1:
            observation : is a (history_length,width,height) numpy matrix representing the screen
        if self.observation_type == 2:
            observation : dict, observation['screen'] is a (history_length,width,height) numpy matrix representing the screen
                                observation['direction'] is a (history_length,4) numpy matrix representing the direction
        """

        if self.observation_type==1:
            
            # roll the matrix: the last element becomes the first
            self.history = np.roll(self.history,1,0)
            
            # blank canvas 
            self.history[0].fill(0)
            # draw food
            self.history[0][tuple(self._food)] = -1

            # draw snake
            if self._snake.size != 0:
                for coord in self._snake: self.history[0][tuple(coord)] = 1
            
            return self.history
            
        
        elif self.observation_type==2:
            
            # roll both arrays of history dict
            self.history['screen'] = np.roll(self.history['screen'],1,0)
            self.history['direction'] = np.roll(self.history['direction'],1,0)

            # blank canvas
            self.history['screen'][0].fill(0)
            self.history['direction'][0].fill(0)
            # draw food
            self.history['screen'][0][tuple(self._food)] = -1

            # draw snake
            if self._snake.size != 0:
                for i,coord in enumerate(self._snake[::-1]): self.history['screen'][tuple(coord)] = 1+i # growing numbers for other pieces of the snake
            
            self.history['direction'] = self._direction_to_onehot[self._current_direction]
            return self.history
        
    def _get_new_head_position(self,action):
        """
        Get new head coordinates from current snake given action.
        Returns [x,y] pair
        """
        if self.action_space_type == 'absolute':
            # get vector of the next step according to input int
            move = self._int_to_direction[action]
            
            if (move @ self._current_direction) == 0:
                # move is perpendicular, accept new move as new direction
                self._current_direction= self._int_to_direction[action]
                # otherwise just keep old direction
            
            # compute new head position
            new_head_position = self._snake[-1] + self._current_direction
            return new_head_position
        
        elif self.action_space_type == 'relative':
            # calculate new direction as the chosen rotation applied to current direction
            self._current_direction = self.rotation_matrix[action] @ self._current_direction

            # compute new head position
            new_head_position = self._snake[-1] + self._current_direction
            return new_head_position
  
    def _get_info(self):
        """
        Return additional information on the state.
        """
        return None

    def _check_snake_collision(self,coords):
        """
        Checks if a [x,y] pair is in the way of the snake.
        """

        return (coords == self._snake).all(axis=1).any()

    def _check_food_collision(self,coords):
        """
        Checks if a [x,y] pair lands on food.
        """

        return (self._food == np.array(coords)).all()
    
    def _check_out_of_boundary(self,coords):
        """
        Checks if a [x,y] pair is out of the map.
        """
        return (coords[0] // self.width != 0) or (coords[1] // self.height != 0)

    def _spawn_food(self):
        """
        Updates the value of self._food by spawning a new food and checking that it does not collide with the existing snake.
        """

        new_food_cords = self.np_random.integers((self.width,self.height),size=2)
        
        while self._check_snake_collision(new_food_cords): # if proposed coordinate is in the snake, resample
            new_food_cords = self.np_random.integers((self.width,self.height),size=2)
        
        self._food = new_food_cords
    
    def step(self,action):
        """
        Given an action, compute the new state of the environment, and returns the 4-tuple (observation,reward,terminated,info).
        Params
        -------
            action : int in range 0 1 2 3 (UP,DOWN,RIGHT,LEFT)
        
        Returns
        -------
            observation : dictionary of the observation of the next state. 
                          observation['screen'] is the board at state S'
                          observation['direction'] is the direction at state S'
            reward : float, reward
            terminated : bool, true if episode terminated
            info : None
        """
        # init to false
        terminated = False
        # init to zero
        reward = 0

        new_head_position = self._get_new_head_position(action)
        
        if self.periodic:
            # PBC
            new_head_position[0] = new_head_position[0] % self.width
            new_head_position[1] = new_head_position[1] % self.height
        else:
            # walls
            if self._check_out_of_boundary(new_head_position):
                terminated = True
                reward += self.terminated_penalty

                if self._snake.shape[0] == 1:
                    self._snake = np.array([])
                else:
                    self._snake = np.roll(self._snake,-1,axis=0)
                    self._snake = self._snake[:-1]
                # get observation of state S'
                observation = self._get_obs()
                info = self._get_info()
                if self.render_mode == "human":
                    self._render_frame()

                return observation, reward, terminated, info


        # if head collides with food, grab reward, add piece to the snake and spawn new one
        if self._check_food_collision(new_head_position):
            reward += self.food_reward
            self._steps_no_food = 0
            self._spawn_food()
            # add the next move as just a new piece of the snake, no need to move the rest
            self._snake = np.append(self._snake,[new_head_position],axis=0)
        else:
            # if head collided with snake, terminated
            if self._check_snake_collision(new_head_position):
                terminated = True
                reward = self.terminated_penalty
                self._steps_no_food=0

            # move the snake
            self._snake = np.roll(self._snake,-1,axis=0)
            self._snake[-1] = new_head_position

            # penalty for empty step
            reward += self.step_penalty

            # add count for steps without food
            self._steps_no_food += 1

            # check if max empty steps
            if self._steps_no_food > self.max_steps:
                terminated = True
                reward = self.terminated_penalty
            
                
        
        
        # get observation of state S'
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, info

    def render(self):

        if self.render_mode == "rgb_array":
            return self._render_frame()
    
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_width, self.window_height))
        canvas.fill((255, 255, 255)) #white
        pix_square_sizes = np.array((
            self.window_width / self.width,
            self.window_height / self.height
        ))  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.ellipse(
            canvas,
            (0, 255, 0),
            pygame.Rect(pix_square_sizes*self._food,pix_square_sizes)
        )
        if self._snake.size != 0:
            # Now we draw the head
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(pix_square_sizes*self._snake[-1],pix_square_sizes)
            )

        # And the rest of the body
            for piece in self._snake[:-1]:
                pygame.draw.rect(
                    canvas,
                    (250,0,0),
                    pygame.Rect(pix_square_sizes*piece,pix_square_sizes)
                )

        # Finally, add some gridlines
        for y in range(self.height + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_sizes[1] * y),
                (self.window_width, pix_square_sizes[1] * y),
                width=3,
            )
        for x in range(self.width+1):
            pygame.draw.line(
                canvas,
                0,
                (pix_square_sizes[0] * x, 0),
                (pix_square_sizes[0] * x, self.window_height),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()