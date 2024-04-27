import numpy as np
import os
import random
import collections
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver, MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch as th
from utils.logging import get_logger
import yaml

from gym.envs.registration import register
import wandb

from run import run

import cProfile

SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")
# results_path = "/home/ubuntu/data"

@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]

    # run the framework
    run(_run, config, _log)


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f, Loader=yaml.FullLoader)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


if __name__ == '__main__':
    params = deepcopy(sys.argv)
    # params = ['src/main.py', '--config=sap', '--env-config=gymma', 'with', 'env_args.key=simplest-env-v0', 't_max=200000']
    th.set_num_threads(1)

    # Get the default configs from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    #Add the env and algorithm config to the default configs
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    benefits_by_state = []
    benefits_by_state.append(np.array([[2, 3, 1], 
                                       [1, 2, 3],
                                       [3, 1, 2]]))
    benefits_by_state.append(np.array([[0,   0.1, 0], 
                                       [0,   0,   0.1],
                                       [0.1, 0,   0]]))
    benefits_by_state.append(np.array([[0,   0,   0.1], 
                                       [0.1, 0,   0],
                                       [0,   0.1, 0]]))
    
    # benefits_by_state = []
    # for _ in range(3):
    #     benefits_by_state.append(np.random.rand(4,5))

    register(
        id="simplest-env-v0",
        entry_point="envs.simplest_env:SimplestEnv",
        kwargs={
                "benefits_by_state": benefits_by_state,
                "episode_step_limit": config_dict['env_args']['episode_step_limit'],
            },
    )

    # register(
    #     id="benefit-obs-env-v0",
    #     entry_point="envs.benefit_obs_env:SimplestEnv",
    #     kwargs={
    #             "benefits_by_state": benefits_by_state,
    #             "episode_step_limit": config_dict['env_args']['episode_step_limit'],
    #         },
    # )

    try:
        map_name = config_dict["env_args"]["map_name"]
    except:
        map_name = config_dict["env_args"]["key"]

    # now add all the config to sacred
    ex.add_config(config_dict)

    for param in params:
        if param.startswith("env_args.map_name"):
            map_name = param.split("=")[1]
        elif param.startswith("env_args.key"):
            map_name = param.split("=")[1]

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, f"sacred/{config_dict['name']}/{map_name}")

    # ex.observers.append(MongoObserver(db_name="marlbench")) #url='172.31.5.187:27017'))
    ex.observers.append(FileStorageObserver.create(file_obs_path))
    # ex.observers.append(MongoObserver())

    ex.run_commandline(params)

