from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np
import torch as th
import scipy.optimize

import time
# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class PretrainRunner:
    def __init__(self, args, logger, buffer, scheme, groups):
        self.args = args
        self.logger = logger

        self.batch_size = self.args.pretrain_batch_size_run

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        env_args = [self.args.env_args.copy() for _ in range(self.batch_size)]
        for i in range(self.batch_size):
            env_args[i]["seed"] += i

        self.ps = [Process(target=env_worker, args=(worker_conn, CloudpickleWrapper(partial(env_fn, **env_arg))))
                            for env_arg, worker_conn in zip(env_args, self.worker_conns)]

        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.T = self.env_info["T"]

        self.t = 0
        self.b = 0

        self.buffer = buffer

        self.pretrain_fn = pretrainerREGISTRY[self.args.pretrain_fn]

        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.T + 1,
                                preprocess=self.buffer.preprocess, device="cpu")

    def get_env_info(self):
        return self.env_info

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        self.batch = self.new_batch()

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": []
        }
        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])

        self.batch.update(pre_transition_data, ts=0)

        self.t = 0

    def fill_buffer(self):
        """
        Fills buffer until full by running individual episodes with run().
        """
        while self.b < self.args.buffer_size:
            if (self.b % 100) == 0: self.logger.console_logger.info(f"Generating offline dataset: {self.b}/{self.args.buffer_size}")
            st = time.time()
            episode_batch = self.run()
            self.buffer.insert_episode_batch(episode_batch)

            self.b += self.batch_size
            print(f"Time taken: {(time.time() - st)/self.batch_size} sec per batch")

        return self.buffer

    def run(self, test_mode=False):
        self.reset()

        all_terminated = False
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]

        while True:
            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            actions = self.pretrain_fn(self.batch[:,self.t])

            # Update the actions taken
            actions_chosen = {
                "actions": actions.unsqueeze(1)
            }
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated: # We produced actions for this env
                    if not terminated[idx]: # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step", actions[action_idx]))
                    action_idx += 1 # actions is not a list over every env
                    if idx == 0 and test_mode and self.args.render:
                        parent_conn.send(("render", None))

            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {
                "rewards": [],
                "terminated": []
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": []
            }

            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    # Remaining data for this current timestep
                    post_transition_data["rewards"].append((data["rewards"],))

                    env_terminated = False
                    if data["terminated"] and not data["info"].get("T", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])

            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        return self.batch

def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            rewards, terminated, env_info = env.step(actions)
            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            remote.send({
                # Data for the next timestep needed to pick an action
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                # Rest of the data for the current timestep
                "rewards": rewards,
                "terminated": terminated,
                "info": env_info
            })
        elif cmd == "reset":
            env.reset()
            remote.send({
                "state": env.get_state(),
                "avail_actions": env.get_avail_actions(),
                "obs": env.get_obs()
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        elif cmd == "render":
            env.render()
        elif cmd == "save_replay":
            env.save_replay()
        else:
            raise NotImplementedError


class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

pretrainerREGISTRY = {}

def NHAPretrainer(batch):
    """
    Input: batch to take NHA action on.

    Output: actions taken for the batch.
    """
    state = batch["state"]
    num_batches = state.shape[0]
    n_timesteps = state.shape[1] #this should always be 1, because we are taking one step at a time
    n = state.shape[2]
    m = state.shape[3]

    picked_actions = th.zeros(num_batches, n, device="cpu")
    
    for batch in range(num_batches):
        _, col_ind = scipy.optimize.linear_sum_assignment(state[batch, 0, :, :], maximize=True)
        picked_actions[batch, :] = th.tensor(col_ind)

    return picked_actions

pretrainerREGISTRY["nha_pretrainer"] = NHAPretrainer