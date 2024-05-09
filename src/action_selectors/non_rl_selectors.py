import torch as th
import scipy.optimize
import sys
import numpy as np
from copy import deepcopy
sys.path.append('/Users/joshholder/code/satellite-constellation')

from common.methods import generate_all_time_intervals, build_time_interval_sequences

REGISTRY = {}

class HAASelector(object):
    """
    Input: batch to take NHA action on.

    Output: actions taken for the batch.
    """
    def __init__(self, args):
        self.args = args

    def select_action(self, batch):
        #Note that num_time_steps is always 1 when calling this function, so we index that axis w/ 0 later.
        (num_batches, num_time_steps, n, m, L) = batch["beta"].shape

        if self.args.use_mps_action_selection:
            picked_actions = th.zeros(num_batches, n, device=self.args.device)
        else:
            picked_actions = th.zeros(num_batches, n, device="cpu")
        
        for batch_num in range(num_batches):
            #get fields in batch that are part of the state
            state = {}
            for key in batch.scheme.keys():
                if batch.scheme[key].get("part_of_state", False):
                    state[key] = batch[key][batch_num]
            beta_hat = self.envs[batch_num].beta_hat(**state)
            #Optimize using only the beta_hat from the current timestep
            _, col_ind = scipy.optimize.linear_sum_assignment(beta_hat[0, :, :, 0], maximize=True)
            picked_actions[batch_num, :] = th.tensor(col_ind)

        print(sum(self.envs[0].power_states > 0))

        return picked_actions

REGISTRY["haa_selector"] = HAASelector

class HAALSelector(object):
    """
    Input: batch to take HAAL action on.

    Output: actions taken for the batch.
    """
    def __init__(self, args):
        self.args = args
        assert self.args.runner == "episode", "HAALSelector only works with episode runner. (parallelization issues)"
        
    def select_action(self, batch):
        def get_state(env):
            pretrans_data = env.get_pretransition_data()
            state = {}
            for key in batch.scheme.keys():
                if batch.scheme[key].get("part_of_state", False):
                    state[key] = pretrans_data[key][0] #remove from list b/c we're using it directly
            return state

        #Note that num_time_steps is always 1 when calling this function, so we index that axis w/ 0 later.
        (num_batches, num_time_steps, n, m, L) = batch["beta"].shape

        if self.args.use_mps_action_selection:
            picked_actions = th.zeros(num_batches, n, device=self.args.device)
        else:
            picked_actions = th.zeros(num_batches, n, device="cpu")
        
        effective_L = min(self.envs[0].L, self.envs[0].T - self.envs[0].k)
        all_time_intervals = generate_all_time_intervals(effective_L)
        all_time_interval_sequences = build_time_interval_sequences(all_time_intervals, effective_L)

        # print(f"Selecting HAAL action for step {self.envs[0].k}:", end='\r')
        for batch_num in range(num_batches):
            best_value = -np.inf
            best_assignment = None
            best_time_interval = None
            for tis in all_time_interval_sequences:
                total_tis_value = 0

                tis_env = deepcopy(self.envs[batch_num])
                
                for i, ti in enumerate(tis):
                    ti_len = ti[1] - ti[0] + 1
                    tis_state = get_state(tis_env)

                    beta_hat = tis_env.beta_hat(**tis_state)
                    total_beta_hat = beta_hat.sum(axis=-1)

                    _, assigns = scipy.optimize.linear_sum_assignment(total_beta_hat, maximize=True)

                    for t in range(ti_len):
                        rewards, _, _ = tis_env.step(assigns)
                        total_tis_value += sum(rewards)

                    if i == 0: #if this is the first time interval, this is the assignment we use for the tis
                        tis_assigns = assigns

                if total_tis_value > best_value:
                    best_value = total_tis_value
                    best_assignment = tis_assigns
                    best_time_interval = tis[0]

            picked_actions[batch_num, :] = th.tensor(best_assignment)

        print(sum(self.envs[0].power_states > 0))
        return picked_actions

REGISTRY["haal_selector"] = HAALSelector