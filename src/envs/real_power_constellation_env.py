import logging
from collections import namedtuple, defaultdict
from enum import Enum
from itertools import product
from gym import Env
import gym
from gym.utils import seeding
import numpy as np
import scipy.optimize
from components.transforms import OneHot
import torch as th

from .HighPerformanceConstellationSim import HighPerformanceConstellationSim

import time

class RealPowerConstellationEnv(Env):
    def __init__(self,
                 num_planes, num_sats_per_plane, m,
                 T, N, M, L, 
                 lambda_,
                 sat_prox_mat=None,
                 graphs=None,
                 bids_as_actions=False,
                 seed=None,
                 T_trans=None,
                 task_prios=None,):
        self.logger = logging.getLogger(__name__)
        self.seed(seed)

        #~~~~~~~~~~~~~~~~~~~~~ ENVIRONMENT SETUP ~~~~~~~~~~~~~~~~~~~~~
        self.n = num_planes * num_sats_per_plane
        self.m = m
        self.T = T

        self.N = N #Number of neighbors to consider
        self.M = M #Number of tasks to consider
        self.L = min(L, self.T) #Number of steps to look into the future

        self.k = 0
        self.done = False

        self.beta = None
        self.prev_assigns = None
        self.power_states = np.ones(self.n)

        #~~~~~~~~~~~~~~~~~~~~~ CONSTELLATION SETUP ~~~~~~~~~~~~~~~~~~~~~
        if sat_prox_mat is None or graphs is None:
            self.constant_benefits = False
            st = time.time()
            #Assume altitude 550, fov=60, fov_based_proximities, dt=1min, isl_dist=4000. Propagate orbits in advance.
            self.const = HighPerformanceConstellationSim(num_planes, num_sats_per_plane, T=T)
            self.graphs = self.const.graphs
            print(f"Time to generate constellation: {time.time()-st:.2f}s")
        else:
            self.constant_benefits = True
            self.sat_prox_mat = sat_prox_mat
            self.graphs = graphs
            self.n = sat_prox_mat.shape[0]
            self.m = sat_prox_mat.shape[1]
            self.T = sat_prox_mat.shape[2]

        #~~~~~~~~~~~~~~~~~~~~~ BENEFIT MATRIX PARAMETERS ~~~~~~~~~~~~~~~~~~~~~
        self.lambda_ = lambda_

        #default to all transitions (except nontransitions) being penalized
        self.T_trans = T_trans if T_trans is not None else np.ones((m,m)) - np.eye(m)

        #default to all tasks having a 1/4 chance of being high priority
        self.task_prios = task_prios if task_prios is not None else np.random.choice([1,1,1,5], size=m, replace=True)
        #Expand the task priorities to be a n x m x L matrix from a m length vector.
        self.task_prios = np.tile(self.task_prios, (self.n,1))
        self.task_prios = np.repeat(self.task_prios[:,:,np.newaxis], self.L, axis=-1)

        # ~~~~~~~~~~~~~~~~~~~~ ACTION AND OBS SPACE SETUP ~~~~~~~~~~~~~~~~~~~~~
        self.bids_as_actions = bids_as_actions
        if self.bids_as_actions:
            #Action space is a bid for each task
            self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Box(low=0, high=np.inf, shape=(self.get_total_actions(),))] * self.n))
        else:
            self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(self.get_total_actions())] * self.n))

        # Define the joint observation space for all agents
        self.obs_space_size = self.get_obs_size()
        self.observation_space = gym.spaces.Tuple(tuple([gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_space_size,))] * self.n))

        # ~~~~~~~~~~~~~~~~~~~~ SCHEME AND PREPROCESS SETUP ~~~~~~~~~~~~~~~~~~~~~
        self.scheme, self.preprocess = self._get_scheme_and_preprocess()

    def _get_scheme_and_preprocess(self):
        """
        Contains the scheme and preprocess functions for the environment.

        If a scheme entry contains a "part_of_state" key, the corresponding data will be passed to the agents as part of the state.
        """
        scheme = { #half precision
            "obs": {"vshape": self.obs_space_size, "group": "agents", "dtype": th.float16},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.int16},
            "avail_actions": {
                "vshape": (self.m,),
                "group": "agents",
                "dtype": th.bool,
            },
            "rewards": {"vshape": (self.n,), "dtype": th.float16},
            "terminated": {"vshape": (1,), "dtype": th.bool},
            "prev_assigns": {"vshape": (self.n,), "dtype": th.int16, "part_of_state": True},
            "beta": {"vshape": (self.n, self.m, self.L), "dtype": th.float16, "part_of_state": True},
            "power_states": {"vshape": (self.n,), "dtype": th.float16, "part_of_state": True},
        }
        preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=self.m)])}

        if self.bids_as_actions:
            scheme["actions"] = {"vshape": (self.m,), "group": "agents", "dtype": th.float32}
            preprocess = {}
        
        return scheme, preprocess

    def reset(self):
        """ Resets environment."""
        self.k = 0
        self.done = False

        if not self.constant_benefits:
            #Remove the tasks from the constellation and generate new ones
            self.sat_prox_mat = self.const.get_proximities_for_random_tasks(self.m)
        else:
            pass #if benefits are constant, do nothing because you should use the same prox matrix

        self.beta = self.sat_prox_mat[:,:,self.k:self.k+self.L] * self.task_prios
        self.prev_assigns = np.random.choice(self.m, self.n, replace=False)
        self.power_states = np.ones(self.n) #reset to full power

        self._obs = self._build_obs()

        return self.get_obs()

    def step(self, actions):
        """ 
        Input is actions in Discrete form (or bids if bids_as_actions is True.)
        Returns reward, terminated, info.
        """
        if self.bids_as_actions:
            _, assignments = scipy.optimize.linear_sum_assignment(actions, maximize=True)
        else:
            assignments = np.array(actions, dtype=int)

        num_times_tasks_completed = np.zeros(self.m)
        for i in range(self.n):
            num_times_tasks_completed[assignments[i]] += 1

        #Compute the state dependent benefit matrix.
        #(This will only ever be a single batch, so we can should squeeze the batch dimension.)
        beta_hat = self.beta_hat(self.beta, self.prev_assigns, self.power_states)

        #Update the rewards and chosen assign matrix
        rewards = []
        for i in range(self.n):
            if self.power_states[i] > 0:
                chosen_task = assignments[i]
                if beta_hat[i, chosen_task, 0] > 0: #only split rewards if the task is worth doing, otherwise get the full handover penalty
                    rewards.append(beta_hat[i, chosen_task, 0] / num_times_tasks_completed[chosen_task])
                else:
                    rewards.append(beta_hat[i, chosen_task, 0])
            else:
                rewards.append(0)

        self.k += 1
            
        self.done = self.k >= self.T

        #Update agent power states
        for i in range(self.n):
            if self.power_states[i] > 0:
                chosen_task = assignments[i]
                if self.beta[i,chosen_task,0] > 1e-12:
                    self.power_states[i] -= 0.2
                else:
                    self.power_states[i] = min(self.power_states[i] + 0.1, 1)
            else:
                pass #do nothing if the agent is already dead

        #Build observation and state for the next step
        effective_L = min(self.L, self.T - self.k)
        curr_sat_prox_mat = np.zeros((self.n, self.m, self.L))
        curr_sat_prox_mat[:,:,:effective_L] = self.sat_prox_mat[:,:,self.k:self.k+effective_L]
        self.beta = curr_sat_prox_mat * self.task_prios
        self.prev_assigns = assignments

        self._obs = self._build_obs()

        return rewards, self.done, {}
    
    def _build_obs(self):
        """
        Builds the observation for the next step.
        For agent i, observations are composed of the following components:
         - local_benefits, a M x L matrix with the benefits for the top M tasks for agent i.
         - neighboring_benefits, a N x M x L matrix with the benefits for the top M tasks for the N agents who most directly compete for those same tasks.
            (here, compete means they have the highest benefit for one of the tasks in the top M, so are highly motivated to complete it.)
         - neighboring_other_benefits, a N x M//2 x L matrix with the benefits for the OTHER top M//2 tasks for the N agents who most directly compete for the top M tasks.
            (this should give agents an intuition about how many other good options competing agents have.)
        """
        if not self.done:
            observations = []

            total_beta = self.beta.sum(axis=-1)
            for i in range(self.n):
                # ~~~ Get the local benefits for agent i ~~~
                agent_benefits = self.beta[i,:,:]

                total_agent_benefits = total_beta[i,:]

                #find M max indices in total_agent_benefits
                top_agent_tasks = np.argsort(-total_agent_benefits)[:self.M]
                local_benefits = agent_benefits[top_agent_tasks, :]

                #Determine the N agents who most directly compete with agent i
                # (i.e. the N agents with the highest total benefit for the top M tasks)
                total_benefits_for_top_M_tasks = total_beta[:, top_agent_tasks]
                best_task_benefit_by_agent = np.max(total_benefits_for_top_M_tasks, axis=1)
                best_task_benefit_by_agent[i] = -np.inf #set agent i to a really low value so it doesn't show up in the sort
                top_n = np.argsort(-best_task_benefit_by_agent)[:self.N]

                # ~~~ Get the neighboring benefits for agent i ~~~
                neighboring_benefits = self.beta[top_n[:, np.newaxis], top_agent_tasks, :]

                # ~~~ Get the global benefits for agent i ~~~
                total_benefits_without_top_M_tasks = np.copy(total_beta)
                total_benefits_without_top_M_tasks[:, top_agent_tasks] = -np.inf

                top_n_total_benefits_without_top_M_tasks = total_benefits_without_top_M_tasks[top_n, :]
                #take the top M//2 entries from every row
                top_n_top_M_2_other_benefit_indices = np.argsort(top_n_total_benefits_without_top_M_tasks, axis=-1)[:, -self.M//2:]

                neighboring_other_benefits = self.beta[top_n[:, np.newaxis], top_n_top_M_2_other_benefit_indices, :]

                # ~~~ Get agent previous assigns ~~~
                agent_assigns_obs = np.where(top_agent_tasks == self.prev_assigns[i], 1, 0)

                # ~~~ Get N power states ~~~
                agent_power_state_obs = np.zeros(self.N+1)
                agent_power_state_obs[0] = self.power_states[i]
                agent_power_state_obs[1:] = self.power_states[top_n]

                #flatten local_benefits, neighboring_benefits, and neighboring_other_benefits
                observations.append(np.concatenate((local_benefits.flatten(), neighboring_benefits.flatten(), neighboring_other_benefits.flatten(), 
                                                    agent_assigns_obs, agent_power_state_obs)))
        else: #otherwise, if done there are no benefits to observe. (and no actions either, because we can't do filtering based on them.)
            self.beta = np.zeros((self.n, self.m, self.L))
            observations = [np.zeros(self.get_obs_size())] * self.n

        return observations
    
    def get_pretransition_data(self):
        """
        Creates a dict with the data that should be recorded before the transition.

        All "add_pretransition_data()" functions must at least have entries for "beta" and "obs".
        """
        pretransition_data = {
            "beta": [self.beta],
            "obs": [self._obs],
            "prev_assigns": [self.prev_assigns],
            "avail_actions": [self.get_avail_actions()],
            "power_states": [self.power_states],
        }
        return pretransition_data

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
        #Local benefits, neighboring benefits, and neighboring other benefits
        #also previous actions, and power states of nearest N agents (plus itself)
        return self.M*self.L + self.N*self.M*self.L + self.N*self.M//2*self.L + self.M + (self.N+1)

    def get_avail_actions(self):
        """ All actions are available."""
        return [self.get_avail_agent_actions(agent_id) for agent_id in range(self.n)]

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id (all actions are always available)"""
        return [1] * self.get_total_actions()

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.m
    
    def get_stats(self):
        return {}

    def beta_hat(self, beta, prev_assigns, power_states):
        """
        Given the info contained in the state, 
        returns a matrix of benefits for each agent-task pair.

        If initially has a time dimension, outputs a time_dim x n x m x L matrix.
        """
        #if beta or prev_assigns is a tensor, convert to numpy
        if isinstance(beta, th.Tensor): beta = beta.cpu().numpy()
        if isinstance(prev_assigns, th.Tensor): prev_assigns = prev_assigns.cpu().numpy()
        if isinstance(power_states, th.Tensor): power_states = power_states.cpu().numpy()
        beta_init_dim = beta.ndim
        if beta_init_dim == 3:
            beta = np.expand_dims(beta, axis=0)
        prev_assigns_init_dim = prev_assigns.ndim
        if prev_assigns_init_dim == 1: 
            prev_assigns = np.expand_dims(prev_assigns, axis=0)
        time_dim = beta.shape[0]

        prev_assign_mat = np.zeros((time_dim, self.n, self.m))
        # for batch in range(batch_dim):
        for time in range(time_dim):
            for i in range(self.n):
                prev_assign_mat[time, i, prev_assigns[time, i]] = 1

        #~~~~~~~~~~~~~~~~~~~~~~~~ CALCULATE HANDOVER PENALTY ~~~~~~~~~~~~~~~~~~~~~~~~
        #Calculate which transitions are not penalized because of the type of task they are
        #(i.e. some tasks are dummy tasks, or correspond to the same task at a different priority, etc.)
        #this info is held in T_trans.
        pen_locs_based_on_T_trans = prev_assign_mat @ self.T_trans

        #Calculate which transitions are not penalized because they are not meaningful tasks (i.e. have zero benefit)
        pen_locs_based_on_task_prios = beta.sum(-1) > 1e-12

        #Penalty is only applied when both conditions are met, so multiply them elementwise
        penalty_locations = pen_locs_based_on_T_trans * pen_locs_based_on_task_prios

        #~~~~~~~~~~~~~~~~~~~~~~~~~~ APPLY HANDOVER PENALTY ~~~~~~~~~~~~~~~~~~~~~~~~~~~
        beta_hat = beta.copy()
        beta_hat[:,:,:,0] = beta_hat[:,:,:,0]-self.lambda_*penalty_locations

        #~~~~~~~~~~~~~~~~~~~ ZERO BETA_HAT FOR OUT OF POWER SATS ~~~~~~~~~~~~~~~~~~~~~
        sats_out_of_power = power_states < 1e-12
        sats_out_of_power = np.reshape(sats_out_of_power, (time_dim, self.n, 1, 1))
        sats_out_of_power = np.tile(sats_out_of_power, (1, 1, self.m, self.L))
        beta_hat = np.where(sats_out_of_power, 0, beta_hat)

        if beta_init_dim == 3: beta_hat = np.squeeze(beta_hat, axis=0)
        if prev_assigns_init_dim == 1: prev_assigns = np.squeeze(prev_assigns, axis=0)
        return beta_hat