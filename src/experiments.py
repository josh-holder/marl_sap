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

def real_constellation_test():
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

    env = RealConstellationEnv(num_planes, num_sats_per_plane, m, N, M, L, T, lambda_, sat_prox_mat=sat_prox_mat, graphs=const.graphs)
    env.reset()

    # #EVALUATE VDN
    # print('Evaluating IQL SAP')
    # iql_sap_model_path = '/Users/joshholder/code/marl_sap/results/models/iql_real_const_trained_on_nha'
    # params = [
    #     'src/main.py',
    #     '--config=iql_sap_custom_cnn',
    #     '--env-config=real_constellation_env',
    #     'with',
    #     f'checkpoint_path={iql_sap_model_path}',
    #     'test_nepisode=1',
    #     'evaluate=True',
    #     'use_offline_dataset=False'
    #     ]
    # explicit_dict_items = {
    #     'env_args': {'sat_prox_mat': sat_prox_mat,
    #                  'T': T,
    #                  }
    # }
    
    # iql_sap_exp = experiment_run(params, explicit_dict_items, verbose=True)
    # iql_sap_val = float(iql_sap_exp.result[1])
    # iql_sap_actions = iql_sap_exp.result[0]
    # iql_sap_assigns = [convert_central_sol_to_assignment_mat(n, m, a) for a in iql_sap_actions]
    # print('Old IQL SAP:', iql_sap_val)

    # env.reset()

    #EVALUATE VDN
    print('Evaluating IQL SAP')
    iql_sap_model_path = '/Users/joshholder/code/marl_sap/results/models/iql_sap_seed938465122_2024-05-07 16:30:23.198948'
    params = [
        'src/main.py',
        '--config=iql_sap_custom_cnn',
        '--env-config=real_constellation_env',
        'with',
        f'checkpoint_path={iql_sap_model_path}',
        'test_nepisode=1',
        'evaluate=True',
        'use_offline_dataset=False',
        'load_step=-1'
        ]
    explicit_dict_items = {
        'env_args': {'sat_prox_mat': sat_prox_mat,
                     'T': T,
                     }
    }
    
    iql_sap_exp = experiment_run(params, explicit_dict_items, verbose=True)
    iql_sap_val = float(iql_sap_exp.result[1])
    iql_sap_actions = iql_sap_exp.result[0]
    iql_sap_assigns = [convert_central_sol_to_assignment_mat(n, m, a) for a in iql_sap_actions]
    print('Most Recent IQL SAP:', iql_sap_val)

    env.reset()

    # #EVALUATE VDN
    # print('Evaluating IQL SAP')
    # iql_sap_model_path = '/Users/joshholder/code/marl_sap/results/models/iql_sap_seed938465122_2024-05-07 16:30:23.198948'
    # params = [
    #     'src/main.py',
    #     '--config=iql_sap_custom_cnn',
    #     '--env-config=real_constellation_env',
    #     'with',
    #     f'checkpoint_path={iql_sap_model_path}',
    #     'test_nepisode=1',
    #     'evaluate=True',
    #     'load_step=-1'
    #     'use_offline_dataset=False'
    #     ]
    # explicit_dict_items = {
    #     'env_args': {'sat_prox_mat': sat_prox_mat,
    #                  'T': T,
    #                  }
    # }
    
    # iql_sap_exp = experiment_run(params, explicit_dict_items, verbose=True)
    # iql_sap_val = float(iql_sap_exp.result[1])
    # iql_sap_actions = iql_sap_exp.result[0]
    # iql_sap_assigns = [convert_central_sol_to_assignment_mat(n, m, a) for a in iql_sap_actions]
    # print('Only pretrained IQL SAP:', iql_sap_val)

    # env.reset()

    haal3_assigns, haal3_val = solve_w_haal(env, L)
    print(f'HAAL, L={L}:', haal3_val)

    haal1_assigns, haal1_val = solve_w_haal(env, 1)
    print(f'HAAL, L={1}:', haal1_val)

    nha_assigns, nha_val = solve_wout_handover(env)
    print('Without Handover:', nha_val)

    greedy_assigns, greedy_val = solve_greedily(env)
    print('Greedy:', greedy_val)

    values = [iql_sap_val, haal3_val, haal1_val, greedy_val, nha_val]
    handovers = [calc_handovers_generically(a) for a in [iql_sap_assigns, haal3_assigns, haal1_assigns, greedy_assigns, nha_assigns]]
    
    alg_names = ['IQL SAP', 'HAAL3', 'HAAL1', 'Greedy', 'Without\nHandover']
    plt.bar(alg_names, values)
    plt.ylabel('Value')
    plt.show()

    plt.bar(alg_names, handovers)
    plt.ylabel('Handovers')
    plt.show()

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
        '--config=iql_sap_custom_cnn',
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

    iql_sap_model_path = '/Users/joshholder/code/marl_sap/results/models/iql_real_power_continue'
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
                     'L': 1,
                     }
    }

    haal_exp = experiment_run(haal_params, explicit_dict_items, verbose=True)
    haal_val = float(haal_exp.result[1])
    haal_actions = haal_exp.result[0]
    haal_assigns = [convert_central_sol_to_assignment_mat(n, m, a) for a in haal_actions]
    

    print('IQL SAP:', iql_sap_val)
    print('HAA:', haa_val)
    print('HAAL:', haal_val)

    # env.reset()

    # haal3_assigns, haal3_val = solve_w_haal(env, L)
    # print(f'HAAL, L={L}:', haal3_val)

    # haal1_assigns, haal1_val = solve_w_haal(env, 1)
    # print(f'HAAL, L={1}:', haal1_val)

    # nha_assigns, nha_val = solve_wout_handover(env)
    # print('Without Handover:', nha_val)

    # greedy_assigns, greedy_val = solve_greedily(env)
    # print('Greedy:', greedy_val)

    # values = [iql_sap_val, haal3_val, haal1_val, greedy_val, nha_val]
    # handovers = [calc_handovers_generically(a) for a in [iql_sap_assigns, haal3_assigns, haal1_assigns, greedy_assigns, nha_assigns]]
    
    # alg_names = ['IQL SAP', 'HAAL3', 'HAAL1', 'Greedy', 'Without\nHandover']
    # plt.bar(alg_names, values)
    # plt.ylabel('Value')
    # plt.show()

    # plt.bar(alg_names, handovers)
    # plt.ylabel('Handovers')
    # plt.show()

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

def const_sim_speed_test():
    num_planes = 10
    num_sats_per_plane = 10
    n = num_planes * num_sats_per_plane
    m = 150
    T = 90
    L = 3
    lambda_ = 0.5

    N = 10
    M = 10

    const = HighPerformanceConstellationSim(num_planes, num_sats_per_plane, T, dt=63.76469*u.second)
    ts = time.time()
    sat_prox_mat = const.get_proximities_for_random_tasks(m, seed=42)
    print('Time to generate proximities, new strat:', time.time() - ts)

    ts = time.time()
    const.timestep_offset = None
    sat_prox_mat_old = const.get_proximities_for_random_tasks(m, seed=42)
    print('Time to generate proximities, old strat:', time.time() - ts)

    total_diff = 0
    num_nonzero = 0
    for i in range(n):
        for j in range(n):
            for k in range(T):
                total_diff += abs(sat_prox_mat[i,j,k] - sat_prox_mat_old[i,j,k])
                if sat_prox_mat[i,j,k] != 0 or sat_prox_mat_old[i,j,k] != 0:
                    num_nonzero += 1
    print("Avg diff", total_diff / num_nonzero)
    print("max diff", np.max(np.abs(sat_prox_mat - sat_prox_mat_old)))

    #ensure that matrices are the same
    assert np.allclose(sat_prox_mat_old, sat_prox_mat)


if __name__ == "__main__":
    const_sim_speed_test()