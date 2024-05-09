import torch as th
import scipy.optimize

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

        return picked_actions

REGISTRY["haa_selector"] = HAASelector

# class HAALSelector(object):
#     """
#     Input: batch to take HAAL action on.

#     Output: actions taken for the batch.
#     """
#     def __init__(self, args):
#         self.args = args
#         assert self.args.runner == "episode", "HAALSelector only works with episode runner. (parallelization issues)"
        
#     def select_action(self, batch):
#         #Note that num_time_steps is always 1 when calling this function, so we index that axis w/ 0 later.
#         (num_batches, num_time_steps, n, m, L) = batch["beta"].shape

#         if self.args.use_mps_action_selection:
#             picked_actions = th.zeros(num_batches, n, device=self.args.device)
#         else:
#             picked_actions = th.zeros(num_batches, n, device="cpu")
        
#         for batch_num in range(num_batches):
#             #get fields in batch that are part of the state
#             state = {}
#             for key in batch.scheme.keys():
#                 if batch.scheme[key].get("part_of_state", False):
#                     state[key] = batch[key][batch_num]
#             beta_hat = self.envs[batch_num].beta_hat(**state)
#             #Optimize using only the beta_hat from the current timestep
#             _, col_ind = scipy.optimize.linear_sum_assignment(beta_hat[0, :, :, 0], maximize=True)
#             picked_actions[batch_num, :] = th.tensor(col_ind)

#         return picked_actions

# REGISTRY["haal_selector"] = HAALSelector