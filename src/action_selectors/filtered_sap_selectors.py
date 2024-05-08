import torch as th
from components.epsilon_schedules import DecayThenFlatSchedule
import scipy.optimize
import numpy as np

class FilteredSAPActionSelector():
    """
    SAP action selector, but with an exploration strategy that
    adds noise to the Q-values according to a variance given by epsilon.
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

            #Add zero-mean gaussian noise with variance epsilon to the Q-values
            benefit_matrix_from_q_values += th.randn_like(benefit_matrix_from_q_values)*self.epsilon #TODO: more intelligent scaling of epsilon

            _, col_ind = scipy.optimize.linear_sum_assignment(benefit_matrix_from_q_values.cpu(), maximize=True)
            picked_actions[batch, :] = th.tensor(col_ind)

        return picked_actions

class FilteredSAPActionSelectorOld():
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
    

class FilteredEpsGrSAPTestActionSelector():
    """
    Epsilon greedy for training, but constrained to assignment problems for testing.
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
            if np.random.rand() < self.epsilon:
                if self.args.use_mps_action_selection:
                    picked_actions = th.zeros(num_batches, n_agents, device=self.args.device)
                else:
                    picked_actions = th.zeros(num_batches, n_agents, device="cpu")
                for batch in range(num_batches):
                    _, col_ind = scipy.optimize.linear_sum_assignment(state[batch, :, :].cpu(), maximize=True)
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

                return benefit_matrix_from_q_values.max(dim=2)[1]