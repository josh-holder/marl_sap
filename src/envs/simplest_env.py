import logging
from collections import namedtuple, defaultdict
from enum import Enum
from itertools import product
from gym import Env
import gym
from gym.utils import seeding
import numpy as np

class SimplestEnv(Env):
    def __init__(self,
                 benefits_by_state,
                 max_episode_steps=50,):
        self.logger = logging.getLogger(__name__)
        self.seed()

        self.n_agents = benefits_by_state[0].shape[0]
        self.num_tasks = benefits_by_state[0].shape[1]
        self.num_states = len(benefits_by_state)

        self.benefits_by_state = benefits_by_state

        self._max_episode_steps = max_episode_steps

        self.curr_state = 0
        self.curr_step = 0

        #Action space is one action for each task
        self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(self.num_tasks)] * self.n_agents))

        #Observation space is just what state the game is in
        self.observation_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(self.num_states)] * self.n_agents))

    def step(self, actions):
        """ Returns reward, terminated, info """
        total_reward = 0
        for i in range(self.n_agents):
            total_reward += self.benefits_by_state[self.curr_state][i, actions[i]]

        self.curr_state = actions[0]
        self.curr_step += 1
        
        nobs = [self.curr_state for _ in range(self.n_agents)]
        rewards = [total_reward for _ in range(self.n_agents)]

        done = self.curr_step >= self._max_episode_steps
        dones = [done for _ in range(self.n_agents)]
        return nobs, rewards, dones, {}

    def reset(self):
        """ Resets environment to state 1."""
        self.curr_state = 0
        self.curr_step = 0

        return [self.curr_state for _ in range(self.n_agents)]

    def render(self):
        raise NotImplementedError

    def close(self):
        return True

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def save_replay(self):
        raise NotImplementedError

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info