import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath
import numpy as np

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer, EpisodeBatch
from components.transforms import OneHot


def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    if args.use_mps: args.device = "mps"
    elif args.use_cuda: args.device = "cuda"
    else: args.device = "cpu"

    # setup loggers
    logger = Logger(_log)

    #Don't print config if you're passing in benefits_over_time, because its too large printed
    if _config["env_args"].get("benefits_over_time", None) is None:
        _log.info("Experiment Parameters:")
        experiment_params = pprint.pformat(_config, indent=4, width=1)
        _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    # unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    unique_token = f"{_config['name']}_seed{_config['seed']}_{datetime.datetime.now()}"

    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(
            dirname(dirname(abspath(__file__))), "results", "tb_logs"
        )
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    if args.use_wandb:
        if args.wandb_run_name == None: args.wandb_run_name = unique_token
        logger.setup_wandb(args)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    if args.evaluate:
        actions, reward = run_sequential(args=args, logger=logger)
    else:
        run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    if args.use_wandb:
        import wandb
        wandb.finish()

    # Making sure framework really exits
    # os._exit(os.EX_OK)
    if args.evaluate: 
        return actions, reward

def evaluate_sequential(args, runner):
    for _ in range(args.test_nepisode):
        batch = runner.run(test_mode=True)

    episode_actions = batch.data.transition_data['actions'][0,:,:,0]
    episode_reward = batch.data.transition_data['reward'].sum()

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

    #Return the actions taken by agents over the course of the episode
    return episode_actions, episode_reward


def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    if getattr(args, "cooperative_rewards", False):
        rewards_vshape = 1
    else:
        rewards_vshape = args.n_agents

    # Default/Base scheme
    bids_as_actions = args.env_args.get("bids_as_actions", False)
    if not bids_as_actions:
        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
            "avail_actions": {
                "vshape": (env_info["n_actions"],),
                "group": "agents",
                "dtype": th.int,
            },
            "rewards": {"vshape": (rewards_vshape,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }
        preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=env_info["n_actions"])])}

    else: #if bids_as_actions, actions space is n_actions (mu for each task)
        scheme = {
            "state": {"vshape": env_info["state_shape"]},
            "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
            "actions": {"vshape": (env_info["n_actions"],), "group": "agents"}, #change to n_tasks, float
            "avail_actions": {
                "vshape": (env_info["n_actions"],),
                "group": "agents",
                "dtype": th.int,
            },
            "rewards": {"vshape": (rewards_vshape,)},
            "terminated": {"vshape": (1,), "dtype": th.uint8},
        }
        preprocess = {}

    groups = {"agents": args.n_agents}

    buffer = ReplayBuffer(
        scheme,
        groups,
        args.buffer_size,
        env_info["episode_step_limit"] + 1, #max_seq_length
        preprocess=preprocess,
        device="cpu" if args.buffer_cpu_only else args.device,
    )
    
    # Setup multiagent controller here
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # Give runner the scheme
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_mps: learner.mps()
    elif args.use_cuda: learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info(
                "Checkpoint directory {} doesn't exist".format(args.checkpoint_path)
            )
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            runner.log_train_stats_t = runner.t_env
            actions, reward = evaluate_sequential(args, runner)
            logger.log_stat("episode", runner.t_env, runner.t_env)
            logger.print_recent_stats()
            logger.console_logger.info("Finished Evaluation")
            return actions, reward

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            #If the data from the replay buffer is on CPU, move it to GPU
            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            learner.train(episode_sample, runner.t_env, episode)

        # Execute test runs once in a while
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info(
                "t_env: {} / {}".format(runner.t_env, args.t_max)
            )
            logger.console_logger.info(
                "Estimated time left: {}. Time passed: {}".format(
                    time_left(last_time, last_test_T, runner.t_env, args.t_max),
                    time_str(time.time() - start_time),
                )
            )
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

            # test_obs = [[1, 0, 0, 2, 3, 1],
            #             [1, 0, 0, 1, 2, 3],
            #             [1, 0, 0, 3, 2, 1]]
            # test_data = {
            #     "state": [np.concatenate(test_obs, axis=0).astype(np.float32)],
            #     "avail_actions": [np.ones(args.n_actions) for i in range(args.n_agents)],
            #     "obs": test_obs
            # }
            # test_batch = EpisodeBatch(scheme, groups, 1, 1)
            # test_batch.update(test_data, ts=0)
            # qs = runner.mac.forward(test_batch, t=0)
            # print("QS state 0")
            # print(qs)

            # test_obs = [[0, 1, 0, 0, 0.1, 0],
            #             [0, 1, 0, 0, 0, 0.1],
            #             [0, 1, 0, 0.1, 0, 0]]
            # test_data = {
            #     "state": [np.concatenate(test_obs, axis=0).astype(np.float32)],
            #     "avail_actions": [np.ones(args.n_actions) for i in range(args.n_agents)],
            #     "obs": test_obs
            # }
            # test_batch = EpisodeBatch(scheme, groups, 1, 1)
            # test_batch.update(test_data, ts=0)
            # qs = runner.mac.forward(test_batch, t=0)
            # print("QS state 1")
            # print(qs)

            # test_obs = [[0, 0, 1, 0, 0, 0.1],
            #             [0, 0, 1, 0.1, 0, 0],
            #             [0, 0, 1, 0, 0.1, 0]]
            # test_data = {
            #     "state": [np.concatenate(test_obs, axis=0).astype(np.float32)],
            #     "avail_actions": [np.ones(args.n_actions) for i in range(args.n_agents)],
            #     "obs": test_obs
            # }
            # test_batch = EpisodeBatch(scheme, groups, 1, 1)
            # test_batch.update(test_data, ts=0)
            # qs = runner.mac.forward(test_batch, t=0)
            # print("QS state 2")
            # print(qs)

        if args.save_model and (
            runner.t_env - model_save_time >= args.save_model_interval
            or model_save_time == 0
        ):
            model_save_time = runner.t_env
            save_path = os.path.join(
                args.local_results_path, "models", args.unique_token, str(runner.t_env)
            )
            # "results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):
    # set MPS and CUDA flags
    if config["use_mps"] and not th.backends.mps.is_available():
        config["use_mps"] = False
        _log.warning(
            "MPS flag use_mps was switched OFF automatically because no MPS devices are available!"
        )
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!"
        )

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]
        ) * config["batch_size_run"]

    return config
