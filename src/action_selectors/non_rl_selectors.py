import torch as th
import scipy.optimize

REGISTRY = {}

class NHASelector(object):
    """
    Input: batch to take NHA action on.

    Output: actions taken for the batch.
    """
    def __init__(self, args):
        self.args = args
    def select_action(self, state):
        decoded_state = self.env.state_decoder(state)
        beta_hat = self.env.beta_hat(*decoded_state)
        num_batches = beta_hat.shape[0]
        n = beta_hat.shape[1]
        m = beta_hat.shape[2]

        if self.args.use_mps_action_selection:
            picked_actions = th.zeros(num_batches, n, device=self.args.device)
        else:
            picked_actions = th.zeros(num_batches, n, device="cpu")
        
        for batch in range(num_batches):
            #Optimize using only the beta_hat from the current timestep
            _, col_ind = scipy.optimize.linear_sum_assignment(beta_hat[batch, :, :, 0], maximize=True)
            picked_actions[batch, :] = th.tensor(col_ind)

        return picked_actions

REGISTRY["nha_selector"] = NHASelector