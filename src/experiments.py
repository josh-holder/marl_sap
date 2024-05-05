from main import experiment_run
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import time
import cProfile
import sys
sys.path.append('/Users/joshholder/code/satellite-constellation')

from controllers.basic_controller import BasicMAC
from envs.mock_constellation_env import generate_benefits_over_time, MockConstellationEnv
from envs.power_constellation_env import PowerConstellationEnv
from envs.real_constellation_env import RealConstellationEnv

from algorithms.solve_w_haal import solve_w_haal
from algorithms.solve_randomly import solve_randomly
from algorithms.solve_greedily import solve_greedily
from algorithms.solve_wout_handover import solve_wout_handover
from common.methods import *

from envs.HighPerformanceConstellationSim import HighPerformanceConstellationSim
from constellation_sim.constellation_generators import get_prox_mat_and_graphs_random_tasks

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
    vdn_actions = vdn_exp.result[0]
    vdn_assigns = [convert_central_sol_to_assignment_mat(n, m, a) for a in vdn_actions]
    
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
        ]
    explicit_dict_items = {
        'env_args': {'sat_prox_mat': sat_prox_mat,
                     'episode_step_limit': T}
    }
    
    vdn_exp = experiment_run(params, explicit_dict_items, verbose=True)
    vdn_val = vdn_exp.result[1]
    vdn_actions = vdn_exp.result[0]
    vdn_assigns = [convert_central_sol_to_assignment_mat(n, m, a) for a in vdn_actions]
    
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
        ]
    explicit_dict_items = {
        'env_args': {'sat_prox_mat': sat_prox_mat,
                     'episode_step_limit': T}
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
                     'episode_step_limit': 1}
    }
    
    vdn_exp = experiment_run(params, explicit_dict_items, verbose=True)

def real_constellation_test():
    num_planes = 10
    num_sats_per_plane = 10
    n = num_planes * num_sats_per_plane
    m = 150
    episode_step_limit = 100
    L = 3
    lambda_ = 0.5

    N = 10
    M = 10

    const = HighPerformanceConstellationSim(num_planes, num_sats_per_plane, episode_step_limit)
    sat_prox_mat = const.get_proximities_for_random_tasks(m)

    env = RealConstellationEnv(num_planes, num_sats_per_plane, m, N, M, L, episode_step_limit, lambda_, sat_prox_mat=sat_prox_mat, graphs=const.graphs)
    env.reset()

    #EVALUATE VDN
    print('Evaluating IQL SAP')
    iql_sap_model_path = '/Users/joshholder/code/marl_sap/results/models/iql_sap_seed555633866_2024-05-04 22:56:32.623378'
    params = [
        'src/main.py',
        '--config=iql_sap_custom_cnn',
        '--env-config=real_constellation_env',
        'with',
        f'checkpoint_path={iql_sap_model_path}',
        'test_nepisode=1',
        'evaluate=True',
        ]
    explicit_dict_items = {
        'env_args': {'sat_prox_mat': sat_prox_mat,
                     'episode_step_limit': episode_step_limit,
                     }
    }
    
    iql_sap_exp = experiment_run(params, explicit_dict_items, verbose=True)
    iql_sap_val = float(iql_sap_exp.result[1])
    iql_sap_actions = iql_sap_exp.result[0]
    iql_sap_assigns = [convert_central_sol_to_assignment_mat(n, m, a) for a in iql_sap_actions]
    print('IQL SAP:', iql_sap_val)

    env.reset()

    _, haal_val = solve_w_haal(env, L)
    print(f'HAAL, L={L}:', haal_val)

    _, haal_val = solve_w_haal(env, 1)
    print(f'HAAL, L={1}:', haal_val)

    _, nha_val = solve_wout_handover(env)
    print('Without Handover:', nha_val)

    _, greedy_val = solve_greedily(env)
    print('Greedy:', greedy_val)


if __name__ == "__main__":
    real_constellation_test()