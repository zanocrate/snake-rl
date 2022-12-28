import gym
from gym import spaces
import pygame # for rendering?
import numpy as np


class SnakeEnv(gym.Env):

    metadata = {
        'render_modes' : ['human','rgb_array'],
        'render_fps' : 4
    }

    def __init__(
        self,
        render_mode=None,
        width = 15,
        height = 15,
        periodic = False, # PBC is currently the only boundary implemented
        food_reward = 1,
        terminated_penalty = -1):

        self.width = width
        self.height = height
        self.periodic = periodic # PBC
        self.food_reward = food_reward
        self.terminated_penalty = terminated_penalty

        self.observation_space = spaces.Dict(
            {
                "screen" : spaces.Box(low=-1,high=width*height,shape=(width,height),dtype=int), # the snake can have at most length n x m
                "direction" : spaces.MultiDiscrete(2*np.ones(4)) # each possible direction the snake is moving
            }
        )

        self.action_space = spaces.Discrete(4) # each possible direction given as input

        self._action_to_direction = {
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
            observation : dict, observation['screen'] is a numpy matrix representing the screen
                                observation['direction'] is an int in Discrete(4) space representing the direction
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
        self._current_direction = self.np_random.integers(4) # random integer

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
        Return an observation for the agent from the current state.
        """
        
        screen = np.zeros((self.width,self.height),dtype=int)
        if self._snake.size != 0:
            head_coord = self._snake[-1] # is the head
            screen[tuple(head_coord)] = 1 # 1 is the value for the head
            for i,coord in enumerate(self._snake[:-1][::-1]): screen[tuple(coord)] = 2+i # growing numbers for other pieces of the snake

        screen[tuple(self._food)] = -1 # -1 to represent food

        return {"screen" : screen,"direction":self._direction_to_onehot[self._current_direction]}
        
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

        # get vector of the next step according to input int
        move = self._action_to_direction[action]
        
        if (move @ self._action_to_direction[self._current_direction]) == 0:
            # move is perpendicular, accept new move as new direction
            self._current_direction=action
        else:
            # move is parallel, keep old direction the same and specify that the new move is just the previous direction
            move = self._action_to_direction[self._current_direction] 
        

        # compute new head position
        new_head_position = self._snake[-1] + move
        
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
            self._spawn_food()
            # add the next move as just a new piece of the snake, no need to move the rest
            self._snake = np.append(self._snake,[new_head_position],axis=0)
        else:
            # if head collided with snake, terminated
            if self._check_snake_collision(new_head_position):
                terminated = True
                reward += self.terminated_penalty
            # move the snake
            self._snake = np.roll(self._snake,-1,axis=0)
            self._snake[-1] = new_head_position
            
                
        
        
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