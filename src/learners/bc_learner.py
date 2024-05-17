# code heavily adapted from https://github.com/AnujMahajanOxf/MAVEN
import copy
from components.episode_buffer import EpisodeBatch
from modules.critics.coma import COMACritic
from modules.critics.centralV import CentralVCritic
from utils.rl_utils import build_td_lambda_targets
import torch as th
import torch.nn as nn
from torch.optim import Adam
from modules.critics import REGISTRY as critic_resigtry
from components.standarize_stream import RunningMeanStd
import numpy as np


class BCLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n = args.n
        self.m = args.m
        self.logger = logger

        self.mac = mac
        self.agent_params = list(mac.parameters())
        self.agent_optimiser = Adam(params=self.agent_params, lr=args.lr)

        self.critic = critic_resigtry[args.critic_type](scheme, args)
        self.target_critic = copy.deepcopy(self.critic)

        self.critic_params = list(self.critic.parameters())
        self.critic_optimiser = Adam(params=self.critic_params, lr=args.lr)

        self.last_target_update_step = 0
        self.critic_training_steps = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        if args.use_mps:
            device = "mps"
        elif args.use_cuda:
            device = "cuda"
        else:
            device = "cpu"
            
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n, ), device=device)

        if self.args.standardise_rewards:
            if self.args.learner == "coma_learner":
                self.rew_ms = RunningMeanStd(shape=(1,), device=device)
            else:
                self.rew_ms = RunningMeanStd(shape=(self.n,), device=device)

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        bs = batch.batch_size
        max_t = batch.max_seq_length
        # ~~~~~~~~~~~~ TRAIN ACTOR POLICY ~~~~~~~~~~~~
        actions = batch["actions"][:, :].to(th.int64)
        avail_actions = batch["avail_actions"][:, :-1].to(th.int64)

        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        #Reshape mac_out to be (batch_size, self.m, ts, self.n)
        mac_out = mac_out.permute(0, 3, 1, 2)

        actor_loss_fn = nn.CrossEntropyLoss()
        actor_loss = actor_loss_fn(mac_out, actions.squeeze(-1)) #remove last dimension on actions

        self.agent_optimiser.zero_grad()
        actor_loss.backward()

        grad_norm = th.nn.utils.clip_grad_norm_(self.agent_params, self.args.grad_norm_clip)
        self.agent_optimiser.step()

        if not self.args.use_mps_action_selection:
            self.mac.update_action_selector_agent()

        # ~~~~~~~~~~~~ TRAIN CRITIC ~~~~~~~~~~~~
        if self.args.learner == "coma_learner":
            rewards = batch["rewards"][:, :-1].float()
            rewards = rewards.sum(dim=-1, keepdim=True) #COMA, so using shared rewards. reward should then be sum of all rewards from all agents
        else:
            rewards = batch["rewards"][:, :-1].float()
        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)

        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        critic_mask = mask.clone()
        mask = mask.repeat(1, 1, self.n).view(-1)

        if self.args.learner == "coma_learner":
            advantages = self._train_critic_coma(batch, rewards, terminated, actions, avail_actions,
                                                        critic_mask, bs, max_t)
        else:
            advantages = self._train_critic_standard(self.critic, self.target_critic, batch, rewards,
                                                                        critic_mask)
        
        self.critic_training_steps += 1
        if self.args.target_update_interval_or_tau > 1 and (
                self.critic_training_steps - self.last_target_update_step) / self.args.target_update_interval_or_tau >= 1.0:
            self._update_targets_hard()
            self.last_target_update_step = self.critic_training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

    def _train_critic_standard(self, critic, target_critic, batch, rewards, mask):
        # Optimise critic
        with th.no_grad():
            target_vals = target_critic(batch)
            target_vals = target_vals.squeeze(3)

        if self.args.standardise_returns:
            target_vals = target_vals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        target_returns = self.nstep_returns(rewards, mask, target_vals, self.args.q_nstep)
        if self.args.standardise_returns:
            self.ret_ms.update(target_returns)
            target_returns = (target_returns - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        v = critic(batch)[:, :-1].squeeze(3)
        td_error = (target_returns.detach() - v)
        masked_td_error = td_error * mask
        loss = (masked_td_error ** 2).sum() / mask.sum()

        self.critic_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()

        return masked_td_error
    
    def _train_critic_coma(self, batch, rewards, terminated, actions, avail_actions, mask, bs, max_t):
        # Optimise critic
        with th.no_grad():
            target_q_vals = self.target_critic(batch)

        targets_taken = th.gather(target_q_vals, dim=3, index=actions).squeeze(3)

        if self.args.standardise_returns:
            targets_taken = targets_taken * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        targets = self.nstep_returns(rewards, mask, targets_taken, self.args.q_nstep)

        if self.args.standardise_returns:
            self.ret_ms.update(targets)
            targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)

        actions = actions[:, :-1]
        q_vals = self.critic(batch)[:, :-1]
        q_taken = th.gather(q_vals, dim=3, index=actions).squeeze(3)

        td_error = (q_taken - targets.detach())
        masked_td_error = td_error * mask

        loss = (masked_td_error ** 2).sum() / mask.sum()
        self.critic_optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.critic_params, self.args.grad_norm_clip)
        self.critic_optimiser.step()

        return q_vals

    def nstep_returns(self, rewards, mask, values, nsteps):
        nstep_values = th.zeros_like(values[:, :-1])
        for t_start in range(rewards.size(1)):
            nstep_return_t = th.zeros_like(values[:, 0])
            for step in range(nsteps + 1):
                t = t_start + step
                if t >= rewards.size(1):
                    break
                elif step == nsteps:
                    nstep_return_t += self.args.gamma ** (step) * values[:, t] * mask[:, t]
                elif t == rewards.size(1) - 1 and self.args.add_value_last_step:
                    nstep_return_t += self.args.gamma ** (step) * rewards[:, t] * mask[:, t]
                    nstep_return_t += self.args.gamma ** (step + 1) * values[:, t + 1]
                else:
                    nstep_return_t += self.args.gamma ** (step) * rewards[:, t] * mask[:, t]
            nstep_values[:, t_start, :] = nstep_return_t
        return nstep_values

    def _update_targets(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_hard(self):
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def mps(self):
        self.old_mac.agent.to("mps")
        self.mac.agent.to("mps")
        self.critic.to("mps")
        self.target_critic.to("mps")

    def cuda(self):
        self.old_mac.cuda()
        self.mac.cuda()
        self.critic.cuda()
        self.target_critic.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.critic.state_dict(), "{}/critic.th".format(path))
        th.save(self.agent_optimiser.state_dict(), "{}/agent_opt.th".format(path))
        th.save(self.critic_optimiser.state_dict(), "{}/critic_opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.critic.load_state_dict(th.load("{}/critic.th".format(path), map_location=lambda storage, loc: storage))
        # Not quite right but I don't want to save target networks
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.agent_optimiser.load_state_dict(
            th.load("{}/agent_opt.th".format(path), map_location=lambda storage, loc: storage))
        self.critic_optimiser.load_state_dict(
            th.load("{}/critic_opt.th".format(path), map_location=lambda storage, loc: storage))
