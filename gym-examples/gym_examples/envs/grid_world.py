import gym
from gym import spaces
import pygame
import numpy as np


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4, "configure.required": ["size", "rewards", "costs"]}
    
    def __init__(self, render_mode=None, size=None, rewards=None, costs=None): 
        if size is None:
            raise ValueError("Missing argument 'size' in constructor. Please provide a value for 'size'.")
        self.size = size  # The size of the square grid
        self.window_size = 512  # The size of the PyGame window
        self.rewards = rewards
        self.costs = costs
        self.S = size*size
        self.A = 4

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        # We have 4 actions, corresponding to "down", "right", "up", "left"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]), # DOWN
            1: np.array([0, 1]), # RIGHT
            2: np.array([-1, 0]),# UP
            3: np.array([0, -1]),# LEFT
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
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {"agent": self._agent_location} #"target": self._target_location} #TODO: Remove the target here too

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None, state=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        if state is None: 
            self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        else: 
            self._agent_location = state
        # We will sample the target's location randomly until it does not coincide with the agent's location
        #self._target_location = self._agent_location
        #while np.array_equal(self._target_location, self._agent_location):
        #    self._target_location = self.np_random.integers(
        #        0, self.size, size=2, dtype=int
        #    )#TODO: No need to have target position

        observation = self._get_obs()
        info = self._get_info() #TODO: potentially not required depending a bit on what is actually returned

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )#TODO: In our case we might also want to clip it in other scenarios if it hits a wall

        # An episode is done iff the agent has reached the target
        terminated = False
        reward = self.rewards[self._agent_location[0], self._agent_location[1]] #1 if terminated else 0  # Binary sparse rewards #TODO: Here we need to assign the reward of the new state
        cost = self.costs[self._agent_location[0], self._agent_location[1]]
        observation = self._get_obs()
        info = self._get_info() 

        if self.render_mode == "human":
            self._render_frame()

        return observation, [reward, cost], terminated, False, info #TODO: Also return the constraint

    #TODO: Adjust the rendering here for the required environment

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels


        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=2,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=2,
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
