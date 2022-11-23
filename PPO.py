import os
import argparse

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union
from contextlib import ExitStack

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions
import pytorch_lightning as pl

from common.base_model import BaseModel
from common.buffers import RolloutBuffer, RolloutBufferSamples
from common.type_aliases import GymEnv, GymObs
from common.vec_env import make_vec_env, SubprocVecEnv
from common.utils import explained_variance

class OnPolicyModel(BaseModel):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param eval_env: The environment to evaluate on, must not be vectorised/parallelrised
        (if registered in Gym, can be str. Can be None for loading trained models)
    :param buffer_length: (int) Length of the buffer and the number of steps to run for each environment per update
    :param num_rollouts: Number of rollouts to do per PyTorch Lightning epoch. This does not affect any training dynamic,
        just how often we evaluate the model since evaluation happens at the end of each Lightning epoch
    :param batch_size: Minibatch size for each gradient update
    :param epochs_per_rollout: Number of epochs to optimise the loss for
    :param num_eval_episodes: The number of episodes to evaluate for at the end of a PyTorch Lightning epoch
    :param gamma: (float) Discount factor
    :param gae_lambda: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param use_sde: (bool) Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration
    :param sde_sample_freq: (int) Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param verbose: The verbosity level: 0 none, 1 training information, 2 debug
    :param seed: Seed for the pseudo random generators
    """

    def __init__(
        self,
        env: Union[GymEnv, str],
        eval_env: Union[GymEnv, str],
        buffer_length: int,
        num_rollouts: int,
        batch_size: int,
        epochs_per_rollout: int,
        num_eval_episodes: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        verbose: int = 0,
        seed: Optional[int] = None,
    ):
        super(OnPolicyModel, self).__init__(
            env=env,
            eval_env=eval_env,
            num_eval_episodes=num_eval_episodes,
            verbose=verbose,
            support_multi_env=True,
            seed=seed,
            use_sde=use_sde,
        )

        self.buffer_length = buffer_length
        self.num_rollouts = num_rollouts
        self.batch_size = batch_size
        self.epochs_per_rollout = epochs_per_rollout
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        self.rollout_buffer = RolloutBuffer(
            buffer_length,
            self.observation_space,
            self.action_space,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
    
    def forward(self, obs: GymObs) -> Tuple[torch.distributions.Distribution, torch.Tensor]:
        """
        Override this function with the forward function of your model

        :param obs: The input observations
        :return: The chosen actions
        """
        raise NotImplementedError


    def train_dataloader(self):
        """
        Create the dataloader for our OffPolicyModel
        """
        return OnPolicyDataloader(self)

    def collect_rollouts(self) -> RolloutBufferSamples:
        """
        Collect rollouts and put them into the RolloutBuffer
        """
        assert self._last_obs is not None, "No previous observation was provided"
        with torch.no_grad():
            # Sample new weights for the state dependent exploration
            if self.use_sde:
                self.reset_noise(self.env.num_envs)

            self.eval()
            for i in range(self.buffer_length):
                if self.use_sde and self.sde_sample_freq > 0 and i % self.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    self.reset_noise(self.env.num_envs)

                # Convert to pytorch tensor, let Lightning take care of any GPU transfer
                obs_tensor = torch.as_tensor(self._last_obs).to(device=self.device, dtype=torch.float32)
                dist, values = self(obs_tensor)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

                # Rescale and perform action
                clipped_actions = actions.cpu().numpy()
                # Clip the actions to avoid out of bound error
                if isinstance(self.action_space, gym.spaces.Box):
                    clipped_actions = np.clip(clipped_actions, self.action_space.low, self.action_space.high)
                elif isinstance(self.action_space, gym.spaces.Discrete):
                    clipped_actions = clipped_actions.astype(np.int32)

                new_obs, rewards, dones, infos = self.env.step(clipped_actions)

                if isinstance(self.action_space, gym.spaces.Discrete):
                    # Reshape in case of discrete action
                    actions = actions.view(-1, 1)

                if not torch.is_tensor(self._last_dones):
                    self._last_dones = torch.as_tensor(self._last_dones).to(device=obs_tensor.device)
                rewards = torch.as_tensor(rewards).to(device=obs_tensor.device)

                self.rollout_buffer.add(obs_tensor, actions, rewards, self._last_dones, values, log_probs)
                self._last_obs = new_obs
                self._last_dones = dones

            final_obs = torch.as_tensor(new_obs).to(device=self.device, dtype=torch.float32)
            dist, final_values = self(final_obs)
            samples = self.rollout_buffer.finalize(final_values, torch.as_tensor(dones).to(device=obs_tensor.device, dtype=torch.float32))

            self.rollout_buffer.reset()
        self.train()
        return samples


class OnPolicyDataloader:
    def __init__(self, model: OnPolicyModel):
        self.model = model


    def __iter__(self):
        for i in range(self.model.num_rollouts):
            experiences = self.model.collect_rollouts()
            observations, actions, old_values, old_log_probs, advantages, returns = experiences
            for j in range(self.model.epochs_per_rollout):
                k = 0
                perm = torch.randperm(observations.shape[0], device=observations.device)
                while k < observations.shape[0]:
                    batch_size = min(observations.shape[0] - k, self.model.batch_size)
                    yield RolloutBufferSamples(
                        observations[perm[k:k+batch_size]],
                        actions[perm[k:k+batch_size]],
                        old_values[perm[k:k+batch_size]],
                        old_log_probs[perm[k:k+batch_size]],
                        advantages[perm[k:k+batch_size]],
                        returns[perm[k:k+batch_size]])
                    k += batch_size
    

class PPO(OnPolicyModel):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    and Stable Baselines 3 (https://github.com/DLR-RM/stable-baselines3)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param eval_env: The environment to evaluate on, must not be vectorised/parallelrised
        (if registered in Gym, can be str. Can be None for loading trained models)
    :param buffer_length: (int) Length of the buffer and the number of steps to run for each environment per update
    :param num_rollouts: Number of rollouts to do per PyTorch Lightning epoch. This does not affect any training dynamic,
        just how often we evaluate the model since evaluation happens at the end of each Lightning epoch
    :param batch_size: Minibatch size for each gradient update
    :param epochs_per_rollout: Number of epochs to optimise the loss for
    :param num_eval_episodes: The number of episodes to evaluate for at the end of a PyTorch Lightning epoch
    :param gamma: (float) Discount factor
    :param gae_lambda: (float) Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param value_coef: Value function coefficient for the loss calculation
    :param entropy_coef: Entropy coefficient for the loss calculation
    :param use_sde: (bool) Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration
    :param sde_sample_freq: (int) Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param verbose: The verbosity level: 0 none, 1 training information, 2 debug
    :param seed: Seed for the pseudo random generators
    """
    def __init__(
        self,
        env: Union[GymEnv, str],
        eval_env: Union[GymEnv, str],
        buffer_length: int = 2048,
        num_rollouts: int = 1,
        batch_size: int = 64,
        epochs_per_rollout: int = 10,
        num_eval_episodes: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: Optional[float] = None,
        target_kl: Optional[float] = None,
        value_coef: float = 0.5,
        entropy_coef: float = 0.0,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        verbose: int = 0,
        seed: Optional[int] = None,
        lr=3e-4, 
        hidden_size=64,
    ):
        super(PPO, self).__init__(
            env=env,
            eval_env=eval_env,
            buffer_length=buffer_length,
            num_rollouts=num_rollouts,
            batch_size=batch_size,
            epochs_per_rollout=epochs_per_rollout,
            num_eval_episodes=num_eval_episodes,
            gamma=gamma,
            gae_lambda=gae_lambda,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            verbose=verbose,
            seed=seed
        )

        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.target_kl = target_kl

        self.lr = lr

        actor = [
            nn.Linear(self.observation_space.shape[0], hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()]
        if isinstance(self.action_space, gym.spaces.Discrete):
            actor += [
                nn.Linear(hidden_size, self.action_space.n),
                nn.Softmax(dim=1)]
        elif isinstance(self.action_space, gym.spaces.Box):
            actor += [nn.Linear(hidden_size, self.action_space.shape[0] * 2)]
        else:
            raise Exception("This example only supports Discrete and Box")

        self.actor = nn.Sequential(*actor)
        self.critic = nn.Sequential(
            nn.Linear(self.observation_space.shape[0], hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1))

        self.save_hyperparameters()

    def forward(
        self, x: torch.Tensor
        ) -> Tuple[distributions.Distribution, torch.Tensor]:
        """
        Runs both the actor and critic network

        :param x: The input observations
        :return: The deterministic action of the actor
        """
        out = self.actor(x)
        if isinstance(self.action_space, gym.spaces.Discrete):
            dist = distributions.Categorical(probs=out)
        elif isinstance(self.action_space, gym.spaces.Box):
            out = list(torch.chunk(out, 2, dim=1))
            out[1] = torch.diag_embed(
                torch.exp(torch.clamp(out[1], -5, 5)))
            dist = distributions.MultivariateNormal(
                loc=out[0], scale_tril=out[1])
        return dist, self.critic(x).flatten()

    def predict(self, x, deterministic=True):
        out = self.actor(x)
        if deterministic:
            if isinstance(self.action_space, gym.spaces.Discrete):
                out = torch.max(out, dim=1)[1]
            elif isinstance(self.action_space, gym.spaces.Box):
                out = torch.chunk(out, 2, dim=1)[0]
        else:
            if isinstance(self.action_space, gym.spaces.Discrete):
                out = distributions.Categorical(probs=out).sample()
            elif isinstance(self.action_space, gym.spaces.Box):
                out = list(torch.chunk(out, 2, dim=1))
                out[1] = torch.diag_embed(
                    torch.exp(0.5 * torch.clamp(out[1], -5, 5)))
                out = distributions.MultivariateNormal(
                    loc=out[0], scale_tril=out[1]).sample()
        return out.cpu().numpy()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        """
        Specifies the update step for PPO. Override this if you wish to modify the PPO algorithm
        """
        if self.use_sde:
            self.reset_noise(self.batch_size)

        dist, values = self(batch.observations)
        log_probs = dist.log_prob(batch.actions)
        values = values.flatten()

        advantages = batch.advantages.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ratio = torch.exp(log_probs - batch.old_log_probs)
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range)
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

        if self.clip_range_vf:
            values = batch.old_values.detach() + torch.clamp(values - batch.old_values.detach(), -self.clip_range_vf, self.clip_range_vf)

        value_loss = F.mse_loss(batch.returns.detach(), values)

        entropy_loss = -dist.entropy().mean()
        loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

        with torch.no_grad():
            clip_fraction = torch.mean((torch.abs(ratio - 1) > self.clip_range).float())
            approx_kl = torch.mean(batch.old_log_probs - log_probs)
            explained_var = explained_variance(batch.old_values, batch.returns)
        self.log_dict({
            'train/train_loss': loss,
            'train/policy_loss': policy_loss,
            'train/value_loss': value_loss,
            'train/entropy_loss': entropy_loss,
            'train/clip_fraction': clip_fraction,
            'train/approx_kl': approx_kl,
            'train/explained_var': explained_var},
            prog_bar=False, logger=True)

        return loss

    @staticmethod
    def add_model_specific_args(parent_parser=None):
        if parent_parser:
            parser = argparse.ArgumentParser(
                parents=[parent_parser], add_help=False)
        else:
            parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--lr', type=float, default=3e-4)
        parser.add_argument('--hidden_size', type=int, default=64)
        parser.add_argument('--buffer_length', type=int, default=2048)
        parser.add_argument('--num_rollouts', type=int, default=1)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--epochs_per_rollout', type=int, default=10)
        parser.add_argument('--num_eval_episodes', type=int, default=10)
        parser.add_argument('--gamma', type=float, default=0.99)
        parser.add_argument('--gae_lambda', type=float, default=0.97)
        parser.add_argument('--clip_range', type=float, default=0.2)
        parser.add_argument('--clip_range_vf', type=float)
        parser.add_argument('--value_coef', type=float, default=0.5)
        parser.add_argument('--entropy_coef', type=float, default=0.0)
        parser.add_argument('--use_sde', action='store_true')
        parser.add_argument('--sde_sample_freq', type=int, default=-1)
        parser.add_argument('--target_kl', type=float)
        parser.add_argument('--verbose', action='store_true')
        parser.add_argument('--seed', type=int)
        return parser


if __name__ == '__main__':
    CHECKPOINT_PATH = "./saved_models"
    # Parse args separately so we don't have to abuse **kwargs
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--num_env', type=int, default=4)
    # If set to true, load model from model_fn and don't train
    parser.add_argument('--evaluate', action='store_true', default=False)
    parser.add_argument('--model_fn', type=str, default='ppo_mlp')
    parser.add_argument('--video_fn', type=str, default='ppo_mlp.mp4')
    args, ignored = parser.parse_known_args()

    if not args.evaluate:
        model_parser = PPO.add_model_specific_args()
        model_args, ignored = model_parser.parse_known_args()
        model_args = vars(model_args)

        trainer_parser = argparse.ArgumentParser(add_help=False)
        trainer_parser.add_argument('--gpus', type=int, default=1)
        trainer_parser.add_argument('--max_epochs', type=int, default=200)
        trainer_parser.add_argument('--gradient_clip_val', type=float, default=0.5)
        trainer_parser.add_argument('--default_root_dir', type=str, default=os.path.join(CHECKPOINT_PATH, args.env+'_ppo_mlp/'))
        # trainer_parser = pl.Trainer.add_argparse_args(trainer_parser)
        trainer_args, ignored = trainer_parser.parse_known_args()
        trainer_args = vars(trainer_args)

        env = make_vec_env(
            args.env, n_envs=args.num_env, vec_env_cls=SubprocVecEnv)

        try:
            version_id = max([int(l.split('_')[1]) for l in os.listdir(trainer_args['default_root_dir'] + '/lightning_logs')]) + 1
        except:
            version_id = 0

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor='train/val_reward_mean',
            dirpath=trainer_args['default_root_dir'] + 'checkpoints',
            filename='ppo_mlp-version=' + str(version_id) + '-{epoch:02d}-{val_reward_mean:.2f}.pl',
            save_top_k=1,
            mode='max')

        model = PPO(
            env=env,
            eval_env=gym.make(args.env),
            **model_args)

        trainer = pl.Trainer(**trainer_args, callbacks=[checkpoint_callback])
        trainer.fit(model)
    else:
        env = gym.make(args.env)
        if 'Bullet' in args.env:
            env.render(mode='human')
            env.reset()
        model = PPO.load_from_checkpoint(
            args.model_fn, env=env, eval_env=env)
        model.eval()

        # Warning: PyBullet environments are hardcoded to record at 320x240
        # There seems to be no easy way to deal with this
        rewards, lengths = model.evaluate(
            num_eval_episodes=10,
            render=True,
            record=True,
            record_fn=args.video_fn)
        print('Mean rewards and length:', np.mean(rewards), np.mean(lengths))