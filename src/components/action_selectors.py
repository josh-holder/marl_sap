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

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False, state=None):
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

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False, state=None):
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

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False, state=None):
        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        num_batches = agent_inputs.shape[0]
        n_agents = agent_inputs.shape[1]
        n_actions = agent_inputs.shape[2]

        if test_mode:
            picked_actions = th.zeros(num_batches, n_agents, device=self.args.device)
            for batch in range(num_batches):
                # Solve the assignment problem for each batch, converting to numpy first
                benefit_matrix_from_q_values = agent_inputs[batch, :, :].detach().cpu()

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

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False, state=None):
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
    
    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False, state=None):
        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = self.args.evaluation_epsilon

        num_batches = agent_inputs.shape[0]
        n_agents = agent_inputs.shape[1]
        n_actions = agent_inputs.shape[2]

        if self.args.use_mps_action_selection:
            picked_actions = th.zeros(num_batches, n_agents, device=self.args.device)
        else:
            picked_actions = th.zeros(num_batches, n_agents, device="cpu")
        for batch in range(num_batches):
            if np.random.rand() < self.epsilon:
                picked_actions[batch, :] = th.randperm(n_agents)
            else:
                # Solve the assignment problem for each batch, moving to cpu first
                benefit_matrix_from_q_values = agent_inputs[batch, :, :].detach().cpu()

                _, col_ind = scipy.optimize.linear_sum_assignment(benefit_matrix_from_q_values, maximize=True)
                picked_actions[batch, :] = th.tensor(col_ind)

        return picked_actions
    
REGISTRY["sap"] = SequentialAssignmentProblemSelector

class ContinuousActionSelector():
    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.variance = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False, state=None):
        if getattr(self.args, "softmax_agent_inputs", False): agent_inputs = th.softmax(agent_inputs, dim=1)

        if test_mode:
            self.variance = self.args.evaluation_epsilon
            picked_actions = th.normal(agent_inputs, self.variance, device=self.args.device)
        else:
            self.variance = self.schedule.eval(t_env)
            picked_actions = th.normal(agent_inputs, self.variance, device=self.args.device)
        return picked_actions.detach()
    
    def action_log_prob(self, actions, old_agent_inputs):
        return th.distributions.Normal(old_agent_inputs, self.variance).log_prob(actions)
    
REGISTRY["continuous"] = ContinuousActionSelector

class RealConstellationSAPSelector():
    """
    Like SAP action selector, but for the real constellation environment, so gets the state to recover
    which were the M best tasks.
    """
    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
    
    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False, state=None):
        assert state is not None, "Need state to figure out which are the top M tasks for each agent."
        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon = self.args.evaluation_epsilon

        num_batches = state.shape[0]
        n_agents = state.shape[1]
        n_actions = state.shape[2]

        if self.args.use_mps_action_selection:
            picked_actions = th.zeros(num_batches, n_agents, device=self.args.device)
        else:
            picked_actions = th.zeros(num_batches, n_agents, device="cpu")
        
        for batch in range(num_batches):
            if np.random.rand() < self.epsilon:
                # picked_actions[batch, :] = th.randperm(n_agents)
                _, col_ind = scipy.optimize.linear_sum_assignment(state[batch, :, :].cpu(), maximize=True)
                picked_actions[batch, :] = th.tensor(col_ind)
            else:
                # Solve the assignment problem for each batch, converting to numpy first
                top_M_benefits_from_q_values = agent_inputs[batch, :, :].detach().cpu()

                #Pop the last column off of top_M_benefits_from_q_values
                baseline_action_benefit = top_M_benefits_from_q_values[:, -1]
                top_M_benefits_from_q_values = top_M_benefits_from_q_values[:, :-1]

                #Create a matrix where the default value is the baseline action benefit
                benefit_matrix_from_q_values = baseline_action_benefit.unsqueeze(1).expand(n_agents, n_actions).clone()
                benefit_matrix_from_q_values += th.rand_like(benefit_matrix_from_q_values)*1e-8 #add noise to break ties between "do-nothing" actions randomly

                #find M max indices in total_agent_benefits_by_task
                top_agent_tasks = th.topk(state[batch, :, :], k=self.args.env_args['M'], dim=-1).indices

                #find M max indices in total_agent_benefits_by_task
                indices = th.tensor(np.indices(top_agent_tasks.shape))
                benefit_matrix_from_q_values[indices[0], top_agent_tasks] = top_M_benefits_from_q_values

                _, col_ind = scipy.optimize.linear_sum_assignment(benefit_matrix_from_q_values.cpu(), maximize=True)
                picked_actions[batch, :] = th.tensor(col_ind)

        return picked_actions
    
REGISTRY["real_const_sap"] = RealConstellationSAPSelector

class RealConstEpsGrSAPTestActionSelector():
    """
    Epsilon greedy for training, but constrained to assignment problems for testing.
    """
    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False, state=None):
        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        num_batches = state.shape[0]
        n_agents = state.shape[1]
        n_actions = state.shape[2]

        if test_mode:
            if self.args.use_mps_action_selection:
                picked_actions = th.zeros(num_batches, n_agents, device=self.args.device)
            else:
                picked_actions = th.zeros(num_batches, n_agents, device="cpu")
            for batch in range(num_batches): #need to split up by batch bc of linear_sum_assignment
                # Solve the assignment problem for each batch, converting to numpy first
                top_M_benefits_from_q_values = agent_inputs[batch, :, :].detach().cpu()

                #Pop the last column off of top_M_benefits_from_q_values
                baseline_action_benefit = top_M_benefits_from_q_values[:, -1]
                top_M_benefits_from_q_values = top_M_benefits_from_q_values[:, :-1]

                #Create a matrix where the default value is the baseline action benefit
                benefit_matrix_from_q_values = baseline_action_benefit.unsqueeze(1).expand(n_agents, n_actions).clone()
                benefit_matrix_from_q_values += th.rand_like(benefit_matrix_from_q_values)*1e-8 #add noise to break ties between "do-nothing" actions randomly

                #find M max indices in total_agent_benefits_by_task
                top_agent_tasks = th.topk(state[batch,:,:], k=self.args.env_args['M'], dim=-1).indices

                #find M max indices in total_agent_benefits_by_task
                indices = th.tensor(np.indices(top_agent_tasks.shape))
                benefit_matrix_from_q_values[indices[0], top_agent_tasks] = top_M_benefits_from_q_values

                _, col_ind = scipy.optimize.linear_sum_assignment(benefit_matrix_from_q_values.cpu(), maximize=True)
                picked_actions[batch, :] = th.tensor(col_ind)
            return picked_actions
        else:
            #General strategy: build a matrix where the default value is the baseline action benefit, then replace the M best actions with the M best Q-values.
            #Then select greedy actions from this matrix, unless the random number is under epsilon.
            top_M_benefits_from_q_values = agent_inputs[:, :, :].detach().cpu()

            #Pop the last column off of top_M_benefits_from_q_values
            baseline_action_benefit = top_M_benefits_from_q_values[:, :, -1]
            top_M_benefits_from_q_values = top_M_benefits_from_q_values[:, :, :-1]

            #Create a matrix where the default value is the baseline action benefit
            benefit_matrix_from_q_values = baseline_action_benefit.unsqueeze(2).expand(num_batches, n_agents, n_actions).clone()
            benefit_matrix_from_q_values += th.rand_like(benefit_matrix_from_q_values)*1e-8 #add noise to break ties between "do-nothing" actions randomly

            #find M max indices in total_agent_benefits_by_task
            top_agent_tasks = th.topk(state, k=self.args.env_args['M'], dim=-1).indices

            #find M max indices in total_agent_benefits_by_task
            indices = th.tensor(np.indices(top_agent_tasks.shape))
            benefit_matrix_from_q_values[indices[0], indices[1], top_agent_tasks] = top_M_benefits_from_q_values

            random_numbers = th.rand_like(state[:, :, 0])
            pick_random = (random_numbers < self.epsilon).long()
            random_actions = Categorical(avail_actions.float()).sample().long()

            picked_actions = pick_random * random_actions + (1 - pick_random) * benefit_matrix_from_q_values.max(dim=2)[1]
            print(picked_actions)
            return picked_actions

REGISTRY["real_const_epsgr_sap_test"] = RealConstEpsGrSAPTestActionSelector