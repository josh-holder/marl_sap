import torch as th
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule
import scipy.optimize
import numpy as np

REGISTRY = {}


class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = Categorical(masked_policies).sample().long()

        return picked_actions


REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = self.args.evaluation_epsilon

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector

class EpsilonGreedySAPTestActionSelector():
    """
    Epsilon greedy for training, but constrained to assignment problems for testing.
    """
    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        num_batches = agent_inputs.shape[0]
        n_agents = agent_inputs.shape[1]
        n_actions = agent_inputs.shape[2]

        if test_mode:
            picked_actions = th.zeros(num_batches, n_agents)
            for batch in range(num_batches):
                # Solve the assignment problem for each batch, converting to numpy first
                benefit_matrix_from_q_values = agent_inputs[batch, :, :].detach().numpy()# + np.random.normal(0, self.epsilon, (n_agents, n_actions))

                _, col_ind = scipy.optimize.linear_sum_assignment(benefit_matrix_from_q_values, maximize=True)
                picked_actions[batch, :] = th.tensor(col_ind)
            return picked_actions
        else:
            # mask actions that are excluded from selection
            masked_q_values = agent_inputs.clone()
            masked_q_values[avail_actions == 0.0] = -float("inf")  # should never be selected!

            random_numbers = th.rand_like(agent_inputs[:, :, 0])
            pick_random = (random_numbers < self.epsilon).long()
            random_actions = Categorical(avail_actions.float()).sample().long()

            picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
            return picked_actions


REGISTRY["epsilon_greedy_sap_test"] = EpsilonGreedySAPTestActionSelector

class SoftPoliciesSelector():

    def __init__(self, args):
        self.args = args

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        m = Categorical(agent_inputs)
        picked_actions = m.sample().long()
        return picked_actions


REGISTRY["soft_policies"] = SoftPoliciesSelector

class SequentialAssignmentProblemSelector():
    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
    
    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = self.args.evaluation_epsilon

        num_batches = agent_inputs.shape[0]
        n_agents = agent_inputs.shape[1]
        n_actions = agent_inputs.shape[2]

        picked_actions = th.zeros(num_batches, n_agents)
        for batch in range(num_batches):
            if np.random.rand() < self.epsilon:
                picked_actions[batch, :] = th.randperm(n_agents)
            else:
                # Solve the assignment problem for each batch, converting to numpy first
                benefit_matrix_from_q_values = agent_inputs[batch, :, :].detach().numpy()# + np.random.normal(0, self.epsilon, (n_agents, n_actions))

                _, col_ind = scipy.optimize.linear_sum_assignment(benefit_matrix_from_q_values, maximize=True)
                picked_actions[batch, :] = th.tensor(col_ind)

        return picked_actions
    
REGISTRY["sap"] = SequentialAssignmentProblemSelector

class PassthroughActionSelector():
    def __init__(self, args):
        self.args = args

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        return agent_inputs.detach() #we no longer need gradients once we convert NN outputs to actions, so detach
    
REGISTRY["passthrough"] = PassthroughActionSelector

class ContinuousActionSelector():
    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.variance = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        if getattr(self.args, "softmax_agent_inputs", False): agent_inputs = th.softmax(agent_inputs, dim=1)

        if test_mode:
            self.variance = self.args.evaluation_epsilon
            picked_actions = th.normal(agent_inputs, self.variance)
        else:
            self.variance = self.schedule.eval(t_env)
            picked_actions = th.normal(agent_inputs, self.variance)
        return picked_actions.detach()
    
    def action_log_prob(self, actions, old_agent_inputs):
        return th.distributions.Normal(old_agent_inputs, self.variance).log_prob(actions)
    
REGISTRY["continuous"] = ContinuousActionSelector