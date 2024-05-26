from astropy import units as u
from main import experiment_run
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import time
import cProfile
import sys
import copy
sys.path.append('/Users/joshholder/code/satellite-constellation')

from controllers.basic_controller import BasicMAC
from envs.mock_constellation_env import generate_benefits_over_time, MockConstellationEnv
from envs.power_constellation_env import PowerConstellationEnv
from envs.real_constellation_env import RealConstellationEnv
from envs.real_power_constellation_env import RealPowerConstellationEnv

from algorithms.solve_w_haal import solve_w_haal
from algorithms.solve_randomly import solve_randomly
from algorithms.solve_greedily import solve_greedily
from algorithms.solve_wout_handover import solve_wout_handover
from haal_experiments.simple_assign_env import SimpleAssignEnv
from common.methods import *

from envs.HighPerformanceConstellationSim import HighPerformanceConstellationSim
from constellation_sim.constellation_generators import get_prox_mat_and_graphs_random_tasks

def test_rl_model(alg_str, env_str, load_path, sat_prox_mat, explicit_dict_items=None, verbose=False):
    vdn_model_path = '/Users/joshholder/code/marl_sap/results/models/vdn_mock_const'
    params = [
        'src/main.py',
        f'--config={alg_str}',
        f'--env-config={env_str}',
        'with',
        f'checkpoint_path={load_path}',
        'test_nepisode=1',
        'evaluate=True',
        'use_offline_dataset=False',
        'buffer_size=1'
        ]
    if explicit_dict_items is None:
        explicit_dict_items = {
            'env_args': {'sat_prox_mat': sat_prox_mat,
                         "graphs": [1]} #placeholder
        }
    else:
        explicit_dict_items['env_args']['sat_prox_mat'] = sat_prox_mat
    
    n = sat_prox_mat.shape[0]
    m = sat_prox_mat.shape[1]

    exp = experiment_run(params, explicit_dict_items, verbose=verbose)
    val = float(exp.result[1])
    actions = exp.result[0]
    assigns = [convert_central_sol_to_assignment_mat(n, m, a) for a in actions]

    # ps = exp.result[2]
    
    return assigns, val, #ps

def test_classic_algorithms(alg_str, env_str, sat_prox_mat, explicit_dict_items=None, verbose=False):
    params = [
        'src/main.py',
        '--config=filtered_reda',
        f'--env-config={env_str}',
        'with',
        'test_nepisode=1',
        'evaluate=True',
        'use_offline_dataset=False',
        'jumpstart_evaluation_epsilon=1',
        f'jumpstart_action_selector=\"{alg_str}\"',
        'buffer_size=1'
        ]
    if explicit_dict_items is None:
        explicit_dict_items = {
            'env_args': {'sat_prox_mat': sat_prox_mat,
                        'graphs': [1], #placeholder
                        }
        }
    else:
        explicit_dict_items['env_args']['sat_prox_mat'] = sat_prox_mat

    n = sat_prox_mat.shape[0]
    m = sat_prox_mat.shape[1]

    exp = experiment_run(params, explicit_dict_items, verbose=verbose)
    val = float(exp.result[1])
    actions = exp.result[0]
    assigns = [convert_central_sol_to_assignment_mat(n, m, a) for a in actions]

    # ps = exp.result[2]

    return assigns, val, #ps

def mock_constellation_test():
    n = 10
    m = 10
    T = 15
    L = 3
    lambda_ = 0.5

    np.random.seed(44)
    sat_prox_mat = generate_benefits_over_time(n, m, T, 3, 6)
    
    #EVALUATE VDN
    print('Evaluating VDN')
    vdn_model_path = '/Users/joshholder/code/marl_sap/results/models/vdn_mock_const'
    params = [
        'src/main.py',
        '--config=vdn',
        '--env-config=mock_constellation_env',
        'with',
        f'checkpoint_path={vdn_model_path}',
        'test_nepisode=1',
        'evaluate=True',
        ]
    explicit_dict_items = {
        'env_args': {'sat_prox_mat': sat_prox_mat}
    }
    
    vdn_exp = experiment_run(params, explicit_dict_items, verbose=False)
    vdn_val = vdn_exp.result[1]
    vdm = vdn_exp.result[0]
    vdn_assigns = [convert_central_sol_to_assignment_mat(n, m, a) for a in vdm]
    
    #EVALUATE AUCTION VDN
    print('Evaluating Auction VDN')
    vdn_sap_model_path = '/Users/joshholder/code/marl_sap/results/models/vdn_sap_mock_const'
    params = [
        'src/main.py',
        '--config=vdn_sap',
        '--env-config=mock_constellation_env',
        'with',
        f'checkpoint_path={vdn_sap_model_path}',
        'test_nepisode=1',
        'evaluate=True',
        ]
    explicit_dict_items = {
        'env_args': {'sat_prox_mat': sat_prox_mat}
    }
    
    vdn_sap_exp = experiment_run(params, explicit_dict_items, verbose=False)
    vdn_sap_val = vdn_sap_exp.result[1]
    vdn_sap_actions = vdn_sap_exp.result[0]
    vdn_sap_assigns = [convert_central_sol_to_assignment_mat(n, m, a) for a in vdn_sap_actions]
    
    print('VDN:', vdn_val)
    print('VDN SAP:', vdn_sap_val)
    #EVALUATE CLASSIC ALGORITHMS
    env = MockConstellationEnv(n, m, T, L, lambda_, sat_prox_mat=sat_prox_mat)
    haal_assigns, haal_val = solve_w_haal(env, L, verbose=False)
    print('HAAL:', haal_val)

    env.reset()
    random_assigns, random_val = solve_randomly(env)
    print('Random:', random_val)

    env.reset()
    greedy_assigns, greedy_val = solve_greedily(env)
    print('Greedy:', greedy_val)

    env.reset()
    nha_assigns, nha_val = solve_wout_handover(env)
    print('Without Handover:', nha_val)

    values = [vdn_val, vdn_sap_val, haal_val, random_val, greedy_val, nha_val]
    handovers = [calc_handovers_generically(a) for a in [vdn_assigns, vdn_sap_assigns, haal_assigns, random_assigns, greedy_assigns, nha_assigns]]
    
    alg_names = ['VDN', 'VDN SAP', 'HAAL', 'Random', 'Greedy', 'Without Handover']
    plt.bar(alg_names, values)
    plt.show()

    plt.bar(alg_names, handovers)
    plt.show()

def power_constellation_test():
    n = 10
    m = 10
    T = 15
    L = 3
    lambda_ = 0.5

    np.random.seed(44)
    sat_prox_mat = generate_benefits_over_time(n, m, T, 3, 6)
    
    #EVALUATE VDN
    print('Evaluating VDN')
    vdn_model_path = '/Users/joshholder/code/marl_sap/results/models/vdn_power_const'
    params = [
        'src/main.py',
        '--config=vdn',
        '--env-config=power_constellation_env',
        'with',
        f'checkpoint_path={vdn_model_path}',
        'test_nepisode=1',
        'evaluate=True',
        'use_offline_dataset=False'
        ]
    explicit_dict_items = {
        'env_args': {'sat_prox_mat': sat_prox_mat,
                     'T': T}
    }
    
    vdn_exp = experiment_run(params, explicit_dict_items, verbose=True)
    vdn_val = vdn_exp.result[1]
    vdm = vdn_exp.result[0]
    vdn_assigns = [convert_central_sol_to_assignment_mat(n, m, a) for a in vdm]
    
    #EVALUATE AUCTION VDN
    print('Evaluating Auction VDN')
    vdn_sap_model_path = '/Users/joshholder/code/marl_sap/results/models/vdn_sap_power_const'
    params = [
        'src/main.py',
        '--config=vdn_sap',
        '--env-config=power_constellation_env',
        'with',
        f'checkpoint_path={vdn_sap_model_path}',
        'test_nepisode=1',
        'evaluate=True',
        'use_offline_dataset=False'
        ]
    explicit_dict_items = {
        'env_args': {'sat_prox_mat': sat_prox_mat,
                     'T': T}
    }
    
    vdn_sap_exp = experiment_run(params, explicit_dict_items, verbose=True)
    vdn_sap_val = vdn_sap_exp.result[1]
    vdn_sap_actions = vdn_sap_exp.result[0]
    vdn_sap_assigns = [convert_central_sol_to_assignment_mat(n, m, a) for a in vdn_sap_actions]
    
    print('VDN:', vdn_val)
    print('VDN SAP:', vdn_sap_val)
    #EVALUATE CLASSIC ALGORITHMS
    env = PowerConstellationEnv(n, m, T, L, lambda_, sat_prox_mat=sat_prox_mat)
    haal_assigns, haal_val = solve_w_haal(env, L, verbose=False)
    print('HAAL:', haal_val)

    env.reset()
    random_assigns, random_val = solve_randomly(env)
    print('Random:', random_val)

    env.reset()
    greedy_assigns, greedy_val = solve_greedily(env)
    print('Greedy:', greedy_val)

    env.reset()
    nha_assigns, nha_val = solve_wout_handover(env)
    print('Without Handover:', nha_val)

    values = [vdn_val, vdn_sap_val, haal_val, random_val, greedy_val, nha_val]
    handovers = [calc_handovers_generically(a) for a in [vdn_assigns, vdn_sap_assigns, haal_assigns, random_assigns, greedy_assigns, nha_assigns]]
    
    alg_names = ['VDN', 'VDN SAP', 'HAAL', 'Random', 'Greedy', 'Without Handover']
    plt.bar(alg_names, values)
    plt.xlabel('Algorithm')
    plt.ylabel('Value')
    plt.show()

    plt.bar(alg_names, handovers)
    plt.xlabel('Algorithm')
    plt.ylabel('Handovers')
    plt.show()

def neighborhood_benefits_test():
    sat_prox_mat = np.zeros((4,4,2))
    sat_prox_mat[:,:,0] = np.array([[5, 0, 0, 1],
                                    [2, 0, 0, 0],
                                    [3, 1, 4, 2],
                                    [1, 3, 0, 10]])
    sat_prox_mat[:,:,1] = np.ones((4,4))
    
    params = [
        'src/main.py',
        '--config=iql_sap',
        '--env-config=real_constellation_env',
        ]
    explicit_dict_items = {
        'env_args': {'sat_prox_mat': sat_prox_mat,
                     'graphs': 1, #so its not None
                     'M': 2,
                     'N': 2,
                     'L': 2,
                     'm': 4,
                     'T': 1}
    }
    
    vdn_exp = experiment_run(params, explicit_dict_items, verbose=True)

def real_power_constellation_test():
    num_planes = 10
    num_sats_per_plane = 10
    n = num_planes * num_sats_per_plane
    m = 150
    T = 100
    L = 3
    lambda_ = 0.5

    N = 10
    M = 10

    const = HighPerformanceConstellationSim(num_planes, num_sats_per_plane, T)
    sat_prox_mat = const.get_proximities_for_random_tasks(m)

    basic_params = [
        'src/main.py',
        '--config=iql_sap',
        '--env-config=real_power_constellation_env',
        'with',
        'test_nepisode=1',
        'evaluate=True',
        'use_offline_dataset=False',
        ]
    explicit_dict_items = {
        'env_args': {'sat_prox_mat': sat_prox_mat,
                     'graphs': const.graphs,
                     'T': T,
                     }
    }

    #~~~~~~~~~ EVALUATE IQL SAP ~~~~~~~~~~
    iql_sap_params = copy.copy(basic_params)

    iql_sap_model_path = '/Users/joshholder/code/marl_sap/results/models/flat_iql_real_power'
    iql_sap_params.append(f'checkpoint_path={iql_sap_model_path}')

    iql_sap_exp = experiment_run(iql_sap_params, explicit_dict_items, verbose=False)
    iql_sap_val = float(iql_sap_exp.result[1])
    iql_sap_actions = iql_sap_exp.result[0]
    iql_sap_assigns = [convert_central_sol_to_assignment_mat(n, m, a) for a in iql_sap_actions]

    #~~~~~~~~~ EVALUATE HAA ~~~~~~~~~~
    print("EVALUATING HAA")
    haa_params = copy.copy(basic_params)

    # haa_params.append(f'checkpoint_path={iql_sap_model_path}')
    haa_params.append('jumpstart_evaluation_epsilon=1')
    haa_params.append('jumpstart_action_selector=\"haa_selector\"')

    haa_exp = experiment_run(haa_params, explicit_dict_items, verbose=True)
    haa_val = float(haa_exp.result[1])
    haa_actions = haa_exp.result[0]
    haa_assigns = [convert_central_sol_to_assignment_mat(n, m, a) for a in haa_actions]

    #~~~~~~~~~ EVALUATE HAAL ~~~~~~~~~~
    print("EVALUATING HAAL")
    haal_params = copy.copy(basic_params)
    haal_params.append('jumpstart_evaluation_epsilon=1')
    haal_params.append('jumpstart_action_selector=\"haal_selector\"')

    explicit_dict_items = {
        'env_args': {'sat_prox_mat': sat_prox_mat,
                     'graphs': const.graphs,
                     'T': T,
                     }
    }

    haal_exp = experiment_run(haal_params, explicit_dict_items, verbose=True)
    haal_val = float(haal_exp.result[1])
    haal_actions = haal_exp.result[0]
    haal_assigns = [convert_central_sol_to_assignment_mat(n, m, a) for a in haal_actions]
    

    print('IQL SAP:', iql_sap_val)
    print('HAA:', haa_val)
    print('HAAL:', haal_val)

def haal_test():
    num_planes = 10
    num_sats_per_plane = 10
    n = num_planes * num_sats_per_plane
    m = 150
    T = 100
    L = 3
    lambda_ = 0.5

    N = 10
    M = 10

    const = HighPerformanceConstellationSim(num_planes, num_sats_per_plane, T)
    sat_prox_mat = const.get_proximities_for_random_tasks(m)

    params = [
        'src/main.py',
        '--config=iql_sap_custom_cnn',
        '--env-config=real_power_constellation_env',
        'with',
        'test_nepisode=1',
        'evaluate=True',
        'use_offline_dataset=False',
        'jumpstart_evaluation_epsilon=1',
        'jumpstart_action_selector=\"haal_selector\"'
        ]
    explicit_dict_items = {
        'env_args': {'sat_prox_mat': sat_prox_mat,
                     'graphs': const.graphs,
                     'T': T,
                     }
    }

    haal_exp = experiment_run(params, explicit_dict_items, verbose=True)
    haal_val = float(haal_exp.result[1])
    haal_actions = haal_exp.result[0]
    haal_assigns = [convert_central_sol_to_assignment_mat(n, m, a) for a in haal_actions]
    print('HAAL:', haal_val)

def large_real_power_test():
    num_planes = 18
    num_sats_per_plane = 18
    n = num_planes * num_sats_per_plane
    m = 450
    T = 100
    L = 3
    lambda_ = 0.5

    N = 10
    M = 10

    const = HighPerformanceConstellationSim(num_planes, num_sats_per_plane, T)
    sat_prox_mat = const.get_proximities_for_random_tasks(m)

    env_str = 'real_power_constellation_env'
    explicit_dict_items = {
        'env_args': {'sat_prox_mat': sat_prox_mat,
                     'graphs': const.graphs,
                     'T': T,
                     }
    }
    # REDA SAP
    print("EVALUATING REDA")
    alg_str = 'reda_sap'
    load_path = '/Users/joshholder/code/marl_sap/results/models/filtered_reda_seed966611133_2024-05-10 21:09:35.182580'
    reda_assigns, reda_val = test_rl_model(alg_str, env_str, load_path, sat_prox_mat, explicit_dict_items)

    #~~~~~~~~~ EVALUATE HAA ~~~~~~~~~~
    print("EVALUATING HAA")
    alg_str = 'haa_selector'
    haa_assigns, haa_val = test_classic_algorithms(alg_str, env_str, sat_prox_mat, explicit_dict_items)

    #~~~~~~~~~ EVALUATE HAAL ~~~~~~~~~~
    print("EVALUATING HAAL")
    alg_str = 'haal_selector'
    haal_assigns, haal_val = test_classic_algorithms(alg_str, env_str, sat_prox_mat, explicit_dict_items)

    #~~~~~~~~~ EVALUATE HAAL L=1 ~~~~~~~~~~
    haal1_explicit_dict_items = {
        'env_args': {'sat_prox_mat': sat_prox_mat,
                     'graphs': const.graphs,
                     'T': T,
                     'L': 1,
                     }
    }
    print("EVALUATING HAAL L=1")
    haal1_assigns, haal1_val = test_classic_algorithms(alg_str, env_str, sat_prox_mat, haal1_explicit_dict_items)

    print('REDA:', reda_val)
    print('HAA:', haa_val)
    print('HAAL:', haal_val)
    print('HAAL L=1:', haal1_val)

def large_real_power_test():
    num_planes = 18
    num_sats_per_plane = 18
    n = num_planes * num_sats_per_plane
    m = 450
    T = 90
    L = 3
    lambda_ = 0.5

    N = 10
    M = 10

    total_haal = 0
    total_haa = 0

    num_tests = 5
    for _ in range(num_tests):
        print("TEST ",_)
        env = RealPowerConstellationEnv(num_planes, num_sats_per_plane, m, T, N, M, L, lambda_)
        env.reset()
        
        haal_assigns, haal_val = test_classic_algorithms('haal_selector', 'dictator_env', env.sat_prox_mat, verbose=True)
        total_haal += haal_val

        haa_assigns, haa_val = test_classic_algorithms('haa_selector', 'dictator_env', env.sat_prox_mat)
        total_haa += haa_val
    
    print('HAAL:', total_haal / num_tests)
    print('HAA:', total_haa / num_tests)

def calc_handovers_generically(assignments, init_assign=None, benefit_info=None):
    """
    Calculate the number of handovers generically, without assuming that the handover penalty
    is the generic handover penalty, as opposed to calc_value_num_handovers above.
    """
    n = assignments[0].shape[0]
    m = assignments[0].shape[1]
    T = len(assignments)

    #If T_trans is provided, then use it, otherwise just set it to 
    try:
        T_trans = benefit_info.T_trans
    except AttributeError:
        T_trans = np.ones((m,m)) - np.eye(m)

    num_handovers = 0
    prev_assign = init_assign
    for k in range(T):
        if prev_assign is not None:
            new_assign = assignments[k]

            #iterate through agents
            for i in range(n):
                new_task_assigned = np.argmax(new_assign[i,:])
                prev_task_assigned = np.argmax(prev_assign[i,:])

                if prev_assign[i,new_task_assigned] == 0 and T_trans[prev_task_assigned,new_task_assigned] == 1:
                    num_handovers += 1
        
        prev_assign = assignments[k]

    return num_handovers

def calc_pct_conflicts(assignments):
    T = len(assignments)
    n = assignments[0].shape[0]
    m = assignments[0].shape[1]

    pct_conflicts = []
    for k in range(T):
        num_agents_w_conflicts = 0
        for i in range(n):
            assigned_task = np.argmax(assignments[k][i,:])
            if np.sum(assignments[k][:,assigned_task]) > 1:
                num_agents_w_conflicts += 1
        
        pct_conflicts.append(num_agents_w_conflicts / n)

    return pct_conflicts

def calc_pass_statistics(benefits, assigns=None):
    """
    Given a benefit array returns various statistics about the satellite passes over tasks.

    Note that we define a satellite pass as the length of time a satellite
    can obtain non-zero benefit for completing a given task.

    Specifically:
     - avg_pass_len: the average length of time a satellite is in view of a single task
            (even if the satellite is not assigned to the task)
     - avg_pass_ben: the average benefits that would be yielded for a satellite being
            assigned to a task for the whole time it is in view

    IF assigns is provided, then we also calculate:
     - avg_ass_len: the average length of time a satellite is assigned to the same task
            (only counted when the task the satellite is completing has nonzero benefit)
    """
    n = benefits.shape[0]
    m = benefits.shape[1]
    T = benefits.shape[2]

    pass_lens = []
    pass_bens = []
    task_assign_len = []
    for j in range(m):
        for i in range(n):
            pass_started = False
            task_assigned = False
            assign_len = 0
            pass_len = 0
            pass_ben = 0
            for k in range(T):
                this_pass_assign_lens = []
                if benefits[i,j,k] > 0:
                    if not pass_started:
                        pass_started = True
                    pass_len += 1
                    pass_ben += benefits[i,j,k]

                    if assigns is not None and assigns[k][i,j] == 1:
                        if not task_assigned: task_assigned = True
                        assign_len += 1
                    #If there are benefits and the task was previously assigned,
                    #but is no longer, end the streak
                    elif task_assigned:
                        task_assigned = False
                        this_pass_assign_lens.append(assign_len)
                        assign_len = 0

                elif pass_started and benefits[i,j,k] == 0:
                    if task_assigned:
                        this_pass_assign_lens.append(assign_len)
                    pass_started = False
                    task_assigned = False
                    for ass_len in this_pass_assign_lens:
                        task_assign_len.append(ass_len)
                    this_pass_assign_lens = []
                    pass_lens.append(pass_len)
                    pass_bens.append(pass_ben)
                    pass_len = 0
                    pass_ben = 0
                    assign_len = 0
    
    avg_pass_len = sum(pass_lens) / len(pass_lens)
    avg_pass_ben = sum(pass_bens) / len(pass_bens)

    if assigns is not None:
        avg_ass_len = sum(task_assign_len) / len(task_assign_len)
        return avg_pass_len, avg_pass_ben, avg_ass_len
    else:
        return avg_pass_len, avg_pass_ben

def qual_perf_compare():
    # num_planes = 18
    # num_sats_per_plane = 18
    # n = num_planes * num_sats_per_plane
    # m = 450
    # T = 90
    # L = 3
    # lambda_ = 0.5

    # reda_sat_ps = []
    # iql_sat_ps = []
    # haal_sat_ps = []
    # haa_sat_ps = []
    # ippo_sat_ps = []

    # tot_iql_conflicts = []
    # tot_ippo_conflicts = []

    # reda_ass_len = []
    # iql_ass_len = []
    # ippo_ass_len = []
    # haal_ass_len = []
    # haa_ass_len = []

    # for _ in range(5):
    #     print(_)
    #     const = HighPerformanceConstellationSim(num_planes, num_sats_per_plane, T)
    #     sat_prox_mat = const.get_proximities_for_random_tasks(m)
    #     explicit_dict_items = {
    #         'env_args': {'sat_prox_mat': sat_prox_mat,
    #                     'graphs': const.graphs,
    #                     'T': T,
    #                     }
    #     }

    #     # REDA
    #     alg_str = 'filtered_reda'
    #     env_str = 'real_power_constellation_env'
    #     load_path = '/Users/joshholder/code/marl_sap/results/models/filtered_reda_seed952807856_2024-05-15 21:26:29.301905'
    #     reda_assigns, reda_val, reda_ps = test_rl_model(alg_str, env_str, load_path, sat_prox_mat, explicit_dict_items, verbose=False)

    #     reda_sat_ps.append(np.sum(np.where(reda_ps > 0, 1, 0)) / 324)

    #     _, _, reda_al = calc_pass_statistics(sat_prox_mat, reda_assigns)
    #     print("REDA ASS LEN", reda_al)
    #     print(np.sum(np.where(reda_ps > 0, 1, 0)) / 324)
    #     reda_ass_len.append(reda_al)

    #     # IQL
    #     alg_str = 'filtered_iql'
    #     env_str = 'real_power_constellation_env'
    #     load_path = '/Users/joshholder/code/marl_sap/results/models/filtered_iql_seed814515160_2024-05-17 01:23:22.125853'
    #     iql_assigns, iql_val, iql_ps = test_rl_model(alg_str, env_str, load_path, sat_prox_mat, explicit_dict_items, verbose=False)

    #     iql_sat_ps.append(np.sum(np.where(iql_ps > 0, 1, 0)) / 324)

    #     tot_iql_conflicts.extend(calc_pct_conflicts(iql_assigns))

    #     _, _, iql_al = calc_pass_statistics(sat_prox_mat, iql_assigns)
    #     print("IQL ASS LEN", iql_al)
    #     print(np.sum(np.where(iql_ps > 0, 1, 0)) / 324)
    #     print(np.mean(calc_pct_conflicts(iql_assigns)))
    #     iql_ass_len.append(iql_al)

    #     # IPPO
    #     alg_str = 'filtered_ippo'
    #     env_str = 'real_power_constellation_env'
    #     load_path = '/Users/joshholder/code/marl_sap/results/models/ippo'
    #     ippo_assigns, ippo_val, ippo_ps = test_rl_model(alg_str, env_str, load_path, sat_prox_mat, explicit_dict_items, verbose=False)

    #     tot_ippo_conflicts.extend(calc_pct_conflicts(ippo_assigns))

    #     _, _, ippo_al = calc_pass_statistics(sat_prox_mat, ippo_assigns)
    #     print("IPPO ASS LEN", ippo_al)
    #     print(np.sum(np.where(ippo_ps > 0, 1, 0)) / 324)
    #     print(np.mean(calc_pct_conflicts(ippo_assigns)))
    #     ippo_ass_len.append(ippo_al)

    #     ippo_sat_ps.append(np.sum(np.where(ippo_ps > 0, 1, 0)) / 324)

    #     haal_assigns, haal_val, haal_ps = test_classic_algorithms('haal_selector', env_str, sat_prox_mat, verbose=True)
    #     haal_sat_ps.append(np.sum(np.where(haal_ps > 0, 1, 0)) / 324)

    #     _, _, haal_al = calc_pass_statistics(sat_prox_mat, haal_assigns)
    #     print("HAAL ASS LEN", haal_al)
    #     print(np.sum(np.where(haal_ps > 0, 1, 0)) / 324)
    #     haal_ass_len.append(haal_al)

    #     haa_assigns, haa_val, haa_ps = test_classic_algorithms('haa_selector', env_str, sat_prox_mat)
    #     haa_sat_ps.append(np.sum(np.where(haa_ps > 0, 1, 0)) / 324)

    #     _, _, haa_al = calc_pass_statistics(sat_prox_mat, haa_assigns)
    #     print("HAA ASS LEN", haa_al)
    #     print(np.sum(np.where(haa_ps > 0, 1, 0)) / 324)
    #     haa_ass_len.append(haa_al)

    # iql_mean_conflicts = np.mean(np.array(tot_iql_conflicts))
    # ippo_mean_conflicts = np.mean(np.array(tot_ippo_conflicts))

    # iql_std_conflicts = np.std(np.array(tot_iql_conflicts))
    # ippo_std_conflicts = np.std(np.array(tot_ippo_conflicts))

    # print(iql_mean_conflicts, ippo_mean_conflicts)
    # print(iql_std_conflicts, ippo_std_conflicts)

    # reda_mean_ps = np.mean(np.array(reda_sat_ps))
    # iql_mean_ps = np.mean(np.array(iql_sat_ps))
    # haal_mean_ps = np.mean(np.array(haal_sat_ps))
    # haa_mean_ps = np.mean(np.array(haa_sat_ps))
    # ippo_mean_ps = np.mean(np.array(ippo_sat_ps))

    # reda_std_ps = np.std(np.array(reda_sat_ps))
    # iql_std_ps = np.std(np.array(iql_sat_ps))
    # haal_std_ps = np.std(np.array(haal_sat_ps))
    # haa_std_ps = np.std(np.array(haa_sat_ps))
    # ippo_std_ps = np.std(np.array(ippo_sat_ps))
    # print(reda_mean_ps, iql_mean_ps, haal_mean_ps, haa_mean_ps, ippo_mean_ps)
    # print(reda_std_ps, iql_std_ps, haal_std_ps, haa_std_ps, ippo_std_ps)

    # reda_mean_al = np.mean(np.array(reda_ass_len))
    # iql_mean_al = np.mean(np.array(iql_ass_len))
    # haal_mean_al = np.mean(np.array(haal_ass_len))
    # haa_mean_al = np.mean(np.array(haa_ass_len))
    # ippo_mean_al = np.mean(np.array(ippo_ass_len))

    # reda_std_al = np.std(np.array(reda_ass_len))
    # iql_std_al = np.std(np.array(iql_ass_len))
    # haal_std_al = np.std(np.array(haal_ass_len))
    # haa_std_al = np.std(np.array(haa_ass_len))
    # ippo_std_al = np.std(np.array(ippo_ass_len))
    # print(reda_mean_al, iql_mean_al, haal_mean_al, haa_mean_al, ippo_mean_al)
    # print(reda_std_al, iql_std_al, haal_std_al, haa_std_al, ippo_std_al)

    iql_mean_conflicts = 0.7727377560710895 
    ippo_mean_conflicts = 0.7668294668294668
    iql_std_conflicts = 0.05217196313302425
    ippo_std_conflicts = 0.03738215222644939

    power_means = np.array([0.9604938271604938, 0.888888888888889, 0.7333333333333334, 0.001, 0.001])
    power_stds = np.array([0.01672372491869637, 0.01892556755155048, 0.10746060836535165, 0, 0])

    al_means = np.array([1.4324952428228677, 1.364916387740726, 1.185541216131914, 3.5774124325688197, 3.1079990196318845])
    al_stds = np.array([0.20891323851433166, 0.14792545867912268, 0.04760254420421542, 0.08361701895022966, 0.03070397013377513])

    # Define the data
    categories = ['% of Sats with no \n charge at k=100', '% Sats in Conflict', 'Avg. # Steps Assigned\n to same task (normalized)']
    bars_per_category = 5
    data = np.array([1-power_means, [0.001, iql_mean_conflicts, ippo_mean_conflicts, 0.001, 0.001], al_means/max(al_means)])  # Replace with your data
    error = np.array([power_stds, [0, iql_std_conflicts, ippo_std_conflicts, 0, 0], al_stds/max(al_means)])  # Replace with your actual standard deviations

    # Set the width of each bar and the spacing between categories
    bar_width = 0.25
    category_spacing = np.arange(len(categories))*1.5

    SMALL_SIZE = 8
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10.3, 5))

    labels = ["REDA (ours)", "IQL", "IPPO", "HAAL", r"$\alpha(\hat{\beta}(s))$"]
    colors = ['purple', 'blue', 'red', 'green', 'gray']
    # Plot the bars and error bars for each category
    for i in range(bars_per_category):
        ax.bar(category_spacing + i * bar_width, data[:, i], bar_width, 
            yerr=error[:, i], capsize=5, label=labels[i], color=colors[i])

    # Set the x-axis tick positions and labels
    ax.set_xticks(category_spacing + bar_width * (bars_per_category - 1) / 2)
    ax.set_xticklabels(categories)

    # Add a legend
    ax.legend()

    # Add labels and title

    plt.tight_layout()
    plt.savefig('qual_perf_compare.pdf')
    plt.show()

def large_real_test():
    env_str = "real_constellation_env"
    num_planes = 18
    num_sats_per_plane = 18
    n = num_planes * num_sats_per_plane
    m = 450
    T = 100
    L = 3
    lambda_ = 0.5
    init_assign = np.zeros((n,m))
    init_assign[:n, :n] = np.eye(n)

    N = 10
    M = 10

    total_haal = []
    total_haa = []
    total_old_haal = []
    total_reda_val = []
    total_greedy_val = []

    haal_als = []
    haa_als = []
    reda_als = []
    greedy_als = []

    num_tests = 5
    for _ in range(num_tests):
        print("TEST ",_)
        env = RealConstellationEnv(num_planes, num_sats_per_plane, m, T, N, M, L, lambda_,
                                        task_prios=np.ones(m))
        env.reset()
        
        old_env = SimpleAssignEnv(env.sat_prox_mat, init_assign, L, lambda_)

        old_haal_assigns, old_haal_val = solve_w_haal(old_env, L, verbose=False)
        total_old_haal.append(old_haal_val)

        old_env.reset()
        greedy_assigns, greedy_val = solve_greedily(old_env)
        total_greedy_val.append(greedy_val)
        _, _, greedy_al = calc_pass_statistics(env.sat_prox_mat, greedy_assigns)
        greedy_als.append(greedy_al)

        haal_assigns, haal_val = test_classic_algorithms('haal_selector', env_str, env.sat_prox_mat, verbose=False)
        total_haal.append(haal_val)
        _, _, haal_al = calc_pass_statistics(env.sat_prox_mat, haal_assigns)
        haal_als.append(haal_al)

        haa_assigns, haa_val = test_classic_algorithms('haa_selector', env_str, env.sat_prox_mat)
        total_haa.append(haa_val)
        _, _, haa_al = calc_pass_statistics(env.sat_prox_mat, haa_assigns)
        haa_als.append(haa_al)

        load_path = '/Users/joshholder/code/marl_sap/results/models/large_real_no_power'
        reda_assigns, reda_val = test_rl_model('filtered_reda', env_str, load_path, env.sat_prox_mat)
        total_reda_val.append(reda_val)
        _, _, reda_al = calc_pass_statistics(env.sat_prox_mat, reda_assigns)
        reda_als.append(reda_al)

    mean_haal_val = np.sum(total_haal) / num_tests
    mean_haa_val = np.sum(total_haa) / num_tests
    mean_old_haal_val = np.sum(total_old_haal) / num_tests
    mean_reda_val = np.sum(total_reda_val) / num_tests
    mean_greedy_val = np.sum(total_greedy_val) / num_tests

    std_haal_val = np.std(total_haal)
    std_haa_val = np.std(total_haa)
    std_old_haal_val = np.std(total_old_haal)
    std_reda_val = np.std(total_reda_val)
    std_greedy_val = np.std(total_greedy_val)

    mean_haal_al = np.sum(haal_als) / num_tests
    mean_haa_al = np.sum(haa_als) / num_tests
    mean_reda_al = np.sum(reda_als) / num_tests
    mean_greedy_al = np.sum(greedy_als) / num_tests

    std_haal_al = np.std(haal_als)
    std_haa_al = np.std(haa_als)
    std_reda_al = np.std(reda_als)
    std_greedy_al = np.std(greedy_als)

    print('HAAL:', mean_haal_val)
    print('HAA:', mean_haa_val)
    print('OLD HAAL:', mean_old_haal_val)
    print('REDA:', mean_reda_val)
    print('GREEDY:', mean_greedy_val)

    print('HAAL std:', std_haal_val)
    print('HAA std:', std_haa_val)
    print('OLD HAAL std:', std_old_haal_val)
    print('REDA std:', std_reda_val)
    print('GREEDY std:', std_greedy_val)

    print('HAAL AL:', mean_haal_al)
    print('HAA AL:', mean_haa_al)
    print('REDA AL:', mean_reda_al)
    print('GREEDY AL:', mean_greedy_al)

    print('HAAL AL std:', std_haal_al)
    print('HAA AL std:', std_haa_al)
    print('REDA AL std:', std_reda_al)
    print('GREEDY AL std:', std_greedy_al)

    score_data = [mean_haa_val, mean_greedy_val, mean_haal_val, mean_reda_val]
    score_error = [std_haa_val, std_greedy_val, std_haal_val, std_reda_val]

    SMALL_SIZE = 8
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    fig, axes = plt.subplots(1,2, figsize=(10,5))

    labels = [r"$\alpha(\hat{\beta}(s))$", "GA", "HAAL", "REDA"]
    colors = ['gray', 'blue', 'green', 'purple']
    # Plot the bars and error bars for each category
    axes[0].bar(labels, score_data, 
        yerr=score_error, capsize=5, color=colors)

    al_data = [mean_haa_al, mean_greedy_al, mean_haal_al, mean_reda_al]
    al_error = [std_haa_al, std_greedy_al, std_haal_al, std_reda_al]

    axes[1].bar(labels, al_data, 
        yerr=al_error, capsize=5, color=colors)

    # Add labels and title
    axes[0].set_xlabel('Value')
    axes[0].set_ylabel('Value')

    axes[1].set_xlabel('Avg. # Steps Assigned\n to same task')
    axes[1].set_ylabel('# Steps')

    plt.tight_layout()
    plt.savefig('compare_reda_on_nopower.pdf')
    plt.show()

def very_large_test():
    env_str = "real_power_constellation_env"
    num_planes = 42
    num_sats_per_plane = 24
    n = num_planes * num_sats_per_plane
    m = n
    T = 100
    L = 3
    N = 10
    M = 10
    lambda_ = 0.5
    init_assign = np.zeros((n,m))
    init_assign[:n, :n] = np.eye(n)

    env = RealPowerConstellationEnv(num_planes, num_sats_per_plane, m, T, N, M, L, lambda_,
                                        task_prios=np.ones(m))
    env.reset()

    load_path = '/Users/joshholder/code/marl_sap/results/models/filtered_reda_seed952807856_2024-05-15 21:26:29.301905'
    reda_assigns, reda_val = test_rl_model('filtered_reda', env_str, load_path, env.sat_prox_mat, verbose=False)
    # _, _, reda_al = calc_pass_statistics(env.sat_prox_mat, reda_assigns)
    print(reda_val)

    nha_assigns, nha_val = test_classic_algorithms('haa_selector', env_str, env.sat_prox_mat, verbose=False)
    print(nha_val)

def const_sim_speed_test():
    num_planes = 18
    num_sats_per_plane = 18
    n = num_planes * num_sats_per_plane
    m = 450
    T = 100
    L = 3
    lambda_ = 0.5

    N = 10
    M = 10

    ts = time.time()
    const = HighPerformanceConstellationSim(num_planes, num_sats_per_plane, T, dt=63.76469*u.second, use_graphs=False)
    print('Time to propagate orbits and generate graphs:', time.time() - ts)

    sat_prox_mat = const.get_proximities_for_random_tasks(m, seed=42)


    ts = time.time()
    const = HighPerformanceConstellationSim(num_planes, num_sats_per_plane, T, dt=63.76469*u.second, use_graphs=True)
    print('Time to generate proximities, old strat:', time.time() - ts)

    const.propagate_orbits_and_generate_graphs(T, test=False)
    
    sat_prox_mat_old = const.get_proximities_for_random_tasks(m, seed=42)
    
    total_diff = 0
    num_nonzero = 0
    for i in range(n):
        for j in range(m):
            for k in range(T):
                total_diff += abs(sat_prox_mat[i,j,k] - sat_prox_mat_old[i,j,k])
                if sat_prox_mat[i,j,k] != 0 or sat_prox_mat_old[i,j,k] != 0:
                    num_nonzero += 1
    print("Avg diff", total_diff / num_nonzero)
    print("max diff", np.max(np.abs(sat_prox_mat - sat_prox_mat_old)))

    #ensure that matrices are the same
    assert np.allclose(sat_prox_mat_old, sat_prox_mat)

def inteference_const_test():
    num_planes = 18
    num_sats_per_plane = 18
    n = num_planes * num_sats_per_plane
    m = 450
    T = 100
    L = 3
    lambda_ = 0.5

    N = 10
    M = 10

    ts = time.time()
    const = HighPerformanceConstellationSim(num_planes, num_sats_per_plane, T, dt=63.76469*u.second, use_graphs=False, inc=70)
    print('Time to propagate orbits and generate graphs:', time.time() - ts)

    ts = time.time()
    sat_prox_mat = const.get_proximities_for_coverage_tasks()
    print('Time to generate proximities:', time.time() - ts)

    ts = time.time()
    sat_prox_mat = const.get_proximities_for_random_tasks(m)
    print('Time to generate proximities:', time.time() - ts)


if __name__ == "__main__":
    inteference_const_test()