import logging
from collections import namedtuple, defaultdict
from enum import Enum
from itertools import product
from gym import Env
import gym
from gym.utils import seeding
import numpy as np
import scipy.optimize

import sys
sys.path.append('/Users/joshholder/code/satellite-constellation')
from envs.HighPerformanceConstellationSim import HighPerformanceConstellationSim

import time

class RealConstellationEnv(Env):
    def __init__(self,
                 num_planes, num_sats_per_plane, m,
                 N, M, L, episode_step_limit, 
                 lambda_,
                 sat_prox_mat=None,
                 graphs=None,
                 bids_as_actions=False,
                 seed=None,
                 benefit_info=None,):
        self.logger = logging.getLogger(__name__)
        self.seed(seed)

        self.n_agents = num_planes * num_sats_per_plane

        self.num_tasks = m

        self.episode_step_limit = episode_step_limit

        if sat_prox_mat is None or graphs is None:
            self.constant_benefits = False
            st = time.time()
            #Assume altitude 550, fov=60, fov_based_proximities, dt=1min, isl_dist=4000. Propagate orbits in advance.
            self.const = HighPerformanceConstellationSim(num_planes, num_sats_per_plane, T=episode_step_limit)
            self.graphs = self.const.graphs
            print(f"Time to generate constellation: {time.time()-st:.2f}s")
        else:
            self.sat_prox_mat = sat_prox_mat
            self.graphs = graphs
            self.n_agents = sat_prox_mat.shape[0]
            self.num_tasks = sat_prox_mat.shape[1]
            self.episode_step_limit = sat_prox_mat.shape[2]
            self.constant_benefits = True

        self.k = 0

        self.N = N #Number of neighbors to consider
        self.M = M #Number of tasks to consider
        self.L = min(L, self.episode_step_limit) #Number of steps to look into the future
        self.lambda_ = lambda_
        self.init_assignment = None #hardcoded to None for now

        self.benefit_fn = meaningful_task_handover_pen_benefit_fn
        self.benefit_info = benefit_info

        #Initialize to None so far
        self.curr_assignment = None
        self.curr_benefits = None #this is the benefits for this timestep, with the small amount of noise added to incentivize exploration

        self.bids_as_actions = bids_as_actions
        if self.bids_as_actions:
            #Action space is a bid for each task
            self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Box(low=0, high=np.inf, shape=(self.get_total_actions(),))] * self.n_agents))
        else:
            self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(self.get_total_actions())] * self.n_agents))

        # Define the joint observation space for all agents
        self.obs_space_size = self.get_obs_size()
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

        num_times_tasks_completed = np.zeros(self.num_tasks)
        for i in range(self.n_agents):
            num_times_tasks_completed[assignments[i]] += 1

        #Adjust the benefits matrix to account for handover penalties
        effective_L = min(self.L, self.episode_step_limit - self.k)
        adj_benefits = self.benefit_fn(self.sat_prox_mat[:,:,self.k:self.k+effective_L], self.curr_assignment, self.lambda_)

        #Update the rewards and current assignment
        rewards = []
        self.curr_assignment = np.zeros((self.n_agents, self.num_tasks))
        for i in range(self.n_agents):
            chosen_task = assignments[i]
            if adj_benefits[i, chosen_task, 0] > 0: #only split rewards if the task is worth doing, otherwise get the full handover penalty
                rewards.append(adj_benefits[i, chosen_task, 0] / num_times_tasks_completed[chosen_task])
            else:
                rewards.append(adj_benefits[i, chosen_task, 0])

            self.curr_assignment[i, chosen_task] = 1
        self.k += 1
            
        done = self.k >= self.episode_step_limit

        #Build observation for the next step
        self._obs = self._build_obs(actions, done)

        # #Update the current benefits (handover adjusted). Only if not done, or else there are no curr benefits.
        # if not done:
        #     effective_L = min(self.L, self.episode_step_limit - self.k)
        #     self.curr_benefits = self.benefit_fn(self.sat_prox_mat[:,:,self.k:self.k+effective_L], self.curr_assignment, self.lambda_)
        #     self.curr_benefits += np.random.rand(*self.curr_benefits.shape) * 1e-6 #add a small amount of noise to break ties
        return rewards, done, {}

    def reset(self):
        """ Resets environment."""
        self.curr_assignment = np.zeros((self.n_agents, self.num_tasks))
        self.k = 0

        if not self.constant_benefits:
            #Remove the tasks from the constellation and generate new ones
            self.sat_prox_mat = self.const.get_proximities_for_random_tasks(self.num_tasks)
        else:
            pass #if benefits are constant, do nothing because you should use the same benefit matrix

        self._obs = self._build_obs()

        return self.get_obs(), self.get_state()

    def _build_obs(self, actions=None, done=False):
        """
        Builds the observation for the next step.
        For agent i, observations are composed of the following components:
         - local_benefits, a M x L matrix with the benefits for the top M tasks for agent i.
         - neighboring_benefits, a N x M x L matrix with the benefits for the top M tasks for the N agents who most directly compete for those same tasks.
            (here, compete means they have the highest benefit for one of the tasks in the top M, so are highly motivated to complete it.)
         - neighboring_other_benefits, a N x M//2 x L matrix with the benefits for the OTHER top M//2 tasks for the N agents who most directly compete for the top M tasks.
            (this should give agents an intuition about how many other good options competing agents have.)
        """
        if not done:
            effective_L = min(self.L, self.episode_step_limit - self.k)

            observations = []

            self.curr_benefits = self.sat_prox_mat[:,:,self.k:self.k+effective_L] + np.random.rand(self.n_agents, self.num_tasks, effective_L) * 1e-6 #add a small amount of noise to break ties
            total_curr_benefits = self.curr_benefits.sum(axis=-1)
            for i in range(self.n_agents):
                # ~~~ Get the local benefits for agent i ~~~
                agent_benefits = self.curr_benefits[i,:,:]

                total_agent_benefits = total_curr_benefits[i,:]

                #find M max indices in total_agent_benefits
                top_agent_tasks = np.argsort(-total_agent_benefits)[:self.M]
                local_benefits = np.zeros((self.M, self.L))
                local_benefits[:,:effective_L] = agent_benefits[top_agent_tasks, :]

                #Determine the N agents who most directly compete with agent i
                # (i.e. the N agents with the highest total benefit for the top M tasks)
                total_benefits_for_top_M_tasks = total_curr_benefits[:, top_agent_tasks]
                best_task_benefit_by_agent = np.max(total_benefits_for_top_M_tasks, axis=1)
                best_task_benefit_by_agent[i] = -np.inf #set agent i to a really low value so it doesn't show up in the sort
                top_N_agents = np.argsort(-best_task_benefit_by_agent)[:self.N]

                # ~~~ Get the neighboring benefits for agent i ~~~
                neighboring_benefits = np.zeros((self.N, self.M, self.L))
                neighboring_benefits[:,:,:effective_L] = self.curr_benefits[top_N_agents[:, np.newaxis], top_agent_tasks, :]

                # ~~~ Get the global benefits for agent i ~~~
                total_benefits_without_top_M_tasks = np.copy(total_curr_benefits)
                total_benefits_without_top_M_tasks[:, top_agent_tasks] = -np.inf

                top_N_agents_total_benefits_without_top_M_tasks = total_benefits_without_top_M_tasks[top_N_agents, :]
                #take the top M//2 entries from every row
                top_N_agents_top_M_2_other_benefit_indices = np.argsort(top_N_agents_total_benefits_without_top_M_tasks, axis=-1)[:, -self.M//2:]

                neighboring_other_benefits = np.zeros((self.N, self.M//2, self.L))
                neighboring_other_benefits[:,:,:effective_L] = self.curr_benefits[top_N_agents[:, np.newaxis], top_N_agents_top_M_2_other_benefit_indices, :]

                # ~~~ Get agent previous actions ~~~
                if actions is not None:
                    agent_action_obs = np.where(top_agent_tasks == actions[i], 1, 0)
                else:
                    agent_action_obs = np.zeros(self.M)

                #flatten local_benefits, neighboring_benefits, and neighboring_other_benefits
                observations.append(np.concatenate((local_benefits.flatten(), neighboring_benefits.flatten(), neighboring_other_benefits.flatten(), agent_action_obs)))
        else: #otherwise, if done there are no benefits to observe. (and no actions either, because we can't do filtering based on them.)
            self.curr_benefits = np.zeros((self.n_agents, self.num_tasks, self.L))
            observations = [np.zeros(self.get_obs_size())] * self.n_agents

        return observations

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
        return self.M*self.L + self.N*self.M*self.L + self.N*self.M//2*self.L + self.M + 0 #for now, we have no extra state information

    def get_state(self):
        #Return the benefit matrix so we can map the top M tasks back to it.
        #Can sum over the last axis to get the total benefit for each task, because that's all we need here.
        return self.curr_benefits.sum(axis=-1)

    def get_state_size(self):
        """ Returns the shape of the state"""
        return (self.n_agents, self.num_tasks)

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