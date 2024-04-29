import logging
from collections import namedtuple, defaultdict
from enum import Enum
from itertools import product
from gym import Env
import gym
from gym.utils import seeding
import numpy as np
import scipy.optimize

class MockConstellationEnv(Env):
    """
    Model a constellation, where benefits are provided to you for task in several states
    """
    def __init__(self,
                 n_agents, num_tasks, 
                 episode_step_limit, 
                 L, lambda_,
                 bids_as_actions=False,
                 seed=None,
                 sat_prox_mat=None,#if you want to train on only one specific benefit matrix
                 benefit_info=None):
        self.logger = logging.getLogger(__name__)
        self.seed(seed)

        self.n_agents = n_agents
        self.num_tasks = num_tasks
        self.episode_step_limit = episode_step_limit

        self.benefit_fn = meaningful_task_handover_pen_benefit_fn
        self.benefit_info = benefit_info

        if sat_prox_mat is None:
            self.constant_benefits = False
            self.sat_prox_mat = generate_benefits_over_time(n_agents, num_tasks, episode_step_limit, 3, 6)
        else:
            self.constant_benefits = True
            self.sat_prox_mat = sat_prox_mat

        #Number of steps to look into the future
        self.L = L
        self.lambda_ = lambda_
        self.graphs = None #both hardcoded to None for now
        self.init_assignment = None

        #State is the previous assignment
        self.curr_assignment = np.zeros((self.n_agents, self.num_tasks))

        self.k = 0

        self.bids_as_actions = bids_as_actions
        if self.bids_as_actions:
            #Action space is a bid for each task
            self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Box(low=0, high=np.inf, shape=(self.num_tasks,))] * self.n_agents))
        else:
            #Action space is one action for each task
            self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(self.num_tasks)] * self.n_agents))

        #Observation space is just the benefits per task for the next L time steps, plus the agent's previous assignment.
        self.obs_space_size = self.L * self.num_tasks + self.num_tasks
        self.observation_space = gym.spaces.Tuple(tuple([gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_space_size,))] * self.n_agents))

    def step(self, actions):
        """ 
        Input is actions in Discrete form.
        Returns reward, terminated, info.
        """
        if self.bids_as_actions:
            _, assignments = scipy.optimize.linear_sum_assignment(actions, maximize=True)
        else:
            assignments = [int(a) for a in actions]

        adj_benefits = self.benefit_fn(self.sat_prox_mat[:,:,self.k], self.curr_assignment, self.lambda_)

        num_times_tasks_completed = np.zeros(self.num_tasks)
        for i in range(self.n_agents):
            num_times_tasks_completed[assignments[i]] += 1

        total_reward = 0
        for i in range(self.n_agents):
            chosen_task = assignments[i]
            total_reward += adj_benefits[i, chosen_task] / num_times_tasks_completed[chosen_task]

        #Update the prev assignment:
        self.curr_assignment = np.zeros((self.n_agents, self.num_tasks))
        for i in range(self.n_agents):
            self.curr_assignment[i, assignments[i]] = 1

        self.k += 1        

        self._obs = [self.curr_assignment[i,:] for i in range(self.n_agents)]
        for l in range(self.L):
            if self.k + l < self.episode_step_limit:
                self._obs = [np.concatenate([self._obs[i], self.sat_prox_mat[i,:,self.k+l]]) for i in range(self.n_agents)]
            else: #if the episode has ended within your lookahead of L, just pad with zeros
                self._obs = [np.concatenate([self._obs[i], np.zeros(self.num_tasks)]) for i in range(self.n_agents)]

        done = self.k >= self.episode_step_limit
        return total_reward, done, {}

    def reset(self):
        """ Resets environment."""
        self.curr_assignment = np.zeros((self.n_agents, self.num_tasks))
        self.k = 0

        if not self.constant_benefits:
            self.sat_prox_mat = generate_benefits_over_time(self.n_agents, self.num_tasks, self.episode_step_limit, 3, 6)
        else:
            pass #if benefits are constant, do nothing because you should use the same benefit matrix

        self._obs = [self.curr_assignment[i,:] for i in range(self.n_agents)]
        for l in range(self.L):
            if self.k + l < self.episode_step_limit:
                self._obs = [np.concatenate([self._obs[i], self.sat_prox_mat[i,:,self.k+l]]) for i in range(self.n_agents)]
            else: #if the episode has ended within your lookahead of L, just pad with zeros
                self._obs = [np.concatenate([self._obs[i], np.zeros(self.num_tasks)]) for i in range(self.n_agents)]

        return self.get_obs(), self.get_state()

    def close(self):
        return True

    def seed(self, seed=None):
        if seed is None:
            self.np_random, self._seed = seeding.np_random(seed)
        else:
            self._seed = seed

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self._obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self._obs[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.obs_space_size

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.n_agents * self.get_obs_size()

    def get_avail_actions(self):
        """ All actions are available."""
        return [self.get_avail_agent_actions(agent_id) for agent_id in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id (all actions are always available)"""
        return [1] * self.get_total_actions()

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.num_tasks
    
    def get_stats(self):
        return {}

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_step_limit": self.episode_step_limit}
        return env_info

def meaningful_task_handover_pen_benefit_fn(sat_prox_mat, prev_assign, lambda_, benefit_info=None):
    """
    Adjusts a 3D benefit matrix to account for generic handover penalty (i.e. constant penalty for switching tasks).

    benefit_info is an object which can contain extra information about the benefits - it should store:
     - task_benefits is a m-length array of the baseline benefits associated with each task.
     - T_trans is a matrix which determines which transitions between TASKS are penalized.
        It is m x m, where entry ij is the state dependence multiplier that should be applied when switching from task i to task j.
        (If it is None, then all transitions between different tasks are scaled by 1.)
    Then, prev_assign @ T_trans is the matrix which entries of the benefit matrix should be adjusted.
    """
    if lambda_ is None: lambda_ = 0 #if lambda_ is not provided, then add no penalty so lambda_=0
    init_dim = sat_prox_mat.ndim
    if init_dim == 2: sat_prox_mat = np.expand_dims(sat_prox_mat, axis=2)
    n = sat_prox_mat.shape[0]
    m = sat_prox_mat.shape[1]
    L = sat_prox_mat.shape[2]

    #Generate a matrix which determines the benefits of each task at each timestep.
    if benefit_info is not None and benefit_info.task_benefits is not None:
        task_benefits = benefit_info.task_benefits
    else:
        task_benefits = np.ones(m)
    task_benefits = np.tile(task_benefits, (n,1))
    task_benefits = np.repeat(task_benefits[:,:,np.newaxis], L, axis=2)

    benefits_hat = sat_prox_mat * task_benefits

    #Determine tasks for which to apply a handover penalty.
    if prev_assign is None:
        penalty_locations = np.zeros((n,m))
    else:
        #Calculate which transitions are not penalized because of the type of task they are
        #(i.e. some tasks are dummy tasks, or correspond to the same task at a different priority, etc.)
        try:
            T_trans = benefit_info.T_trans
        except AttributeError:
            T_trans = None
        if T_trans is None:
            T_trans = np.ones((m,m)) - np.eye(m) #default to all transitions (except nontransitions) being penalized
        pen_locs_based_on_T_trans = prev_assign @ T_trans

        #Calculate which transitions are not penalized because they are not meaningful tasks (i.e. have zero benefit)
        pen_locs_based_on_task_benefits = benefits_hat.sum(-1) > 1e-12

        #Penalty is only applied when both conditions are met, so multiply them elementwise
        penalty_locations = pen_locs_based_on_T_trans * pen_locs_based_on_task_benefits

    benefits_hat[:,:,0] = benefits_hat[:,:,0]-lambda_*penalty_locations

    if init_dim == 2: 
        benefits_hat = np.squeeze(benefits_hat, axis=2)
    return benefits_hat

def generate_benefits_over_time(n, m, T, width_min, width_max, scale_min=0.25, scale_max=2):
    """
    lightweight way of generating "constellation-like" benefit matrices.
    """
    benefits = np.zeros((n,m,T))
    for i in range(n):
        for j in range(m):
            #Determine if task is active for this sat ever
            task_active = 1 if np.random.rand() > 0.75 else 0

            if task_active:
                #where is the benefit curve maximized
                time_center = np.random.uniform(0, T)

                #how wide is the benefit curve
                time_spread = np.random.uniform(width_min, width_max)
                sigma_2 = np.sqrt(time_spread**2/-8/np.log(0.05))

                #how high is the benefit curve
                benefit_scale = np.random.uniform(scale_min, scale_max)
                  
                #iterate from time zero to t_final with T steps in between
                for t in range(T):
                    #calculate the benefit at time t
                    benefits[i,j,t] = benefit_scale*np.exp(-(t-time_center)**2/sigma_2/2)
    return benefits