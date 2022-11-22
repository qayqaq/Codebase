import os
import argparse
import warnings
import copy
from typing import Optional, Union, Tuple

import pybullet_envs

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
import gym

from common.base_model import BaseModel
from common.buffers import ReplayBuffer, ReplayBufferSamples
from common.type_aliases import GymEnv
from common.vec_env import VecEnv

print(pybullet_envs.getList())

class OffPolicyModel(BaseModel):
    """
    The base for Off-Policy algorithms (ex: SAC/TD3)

    :param env: The environment to learn from
        (if registered in Gym, can be str. Can be None for loading trained models)
    :param eval_env: The environment to evaluate on, must not be vectorised/parallelrised
        (if registered in Gym, can be str. Can be None for loading trained models)
    :param batch_size: Minibatch size for each gradient update
    :param buffer_length: length of the replay buffer
    :param warmup_length: how many steps of the model to collect transitions for before learning starts
    :param train_freq: Update the model every ``train_freq`` steps. Set to `-1` to disable.
    :param episodes_per_rollout: Update the model every ``episodes_per_rollout`` episodes.
        Note that this cannot be used at the same time as ``train_freq``. Set to `-1` to disable.
    :param num_rollouts: Number of rollouts to do per PyTorch Lightning epoch. This does not affect any training dynamic,
        just how often we evaluate the model since evaluation happens at the end of each Lightning epoch
    :param gradient_steps: How many gradient steps to do after each rollout
    :param num_eval_episodes: The number of episodes to evaluate for at the end of a PyTorch Lightning epoch
    :param gamma: the discount factor
    :param squashed_actions: whether the actions are squashed between [-1, 1] and need to be unsquashed
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param verbose: The verbosity level: 0 none, 1 training information, 2 debug (default: 0)
    :param seed: Seed for the pseudo random generators
    """

    def __init__(
        self,
        env: Union[GymEnv, str],
        eval_env: Union[GymEnv, str],
        batch_size: int = 256,
        buffer_length: int = int(1e6),
        warmup_length: int = 100,
        train_freq: int = -1,
        episodes_per_rollout: int = -1,
        num_rollouts: int = 1,
        gradient_steps: int = 1,
        num_eval_episodes: int = 10,
        gamma: float = 0.99,
        squashed_actions: bool = False,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        verbose: int = 0,
        seed: Optional[int] = None,
    ):
        super(OffPolicyModel, self).__init__(
            env=env,
            eval_env=eval_env,
            num_eval_episodes=num_eval_episodes,
            verbose=verbose,
            support_multi_env=True,
            seed=seed,
            use_sde=use_sde,
        )

        assert self.env.num_envs == 1, "OffPolicyModel only support single environment at this stage"
        assert train_freq > 0 or episodes_per_rollout > 0, "At least one of train_freq or episodes_per_rollout must be passed"
        if train_freq > 0 and episodes_per_rollout > 0:
            warnings.warn(
                "You passed a positive value for `train_freq` and `n_episodes_rollout`."
                "Please make sure this is intended. "
                "The agent will collect data by stepping in the environment "
                "until both conditions are true: "
                "`number of steps in the env` >= `train_freq` and "
                "`number of episodes` > `n_episodes_rollout`"
            )

        self.batch_size = batch_size
        self.buffer_length = buffer_length
        self.warmup_length = warmup_length
        self.train_freq = train_freq
        self.episodes_per_rollout = episodes_per_rollout
        self.num_rollouts = num_rollouts
        self.gradient_steps = gradient_steps
        self.gamma = gamma
        self.squashed_actions = squashed_actions
    
        self.replay_buffer = ReplayBuffer(
            buffer_length,
            batch_size,
            self.observation_space,
            self.action_space,
            n_envs=self.n_envs,
        )

    def reset(self):
        """
        Reset the environment and set the num_timesteps to 0
        """
        super(OffPolicyModel, self).reset()
        self.num_timesteps = 0
    
    def on_step(self):
        """
        Simple callback for each step we take in the environment
        """
        pass

    def train_dataloader(self):
        """
        Create the dataloader for our OffPolicyModel
        """
        return OffPolicyDataloader(self)

    def scale_actions(
        self, actions: np.ndarray, squashed=False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scale the action appropriately for spaces.Box based on whether they
        are squashed between [-1, 1]

        :param action: The input action
        :return: The action to step the environment with and the action to buffer with
        """
        high, low = self.action_space.high, self.action_space.low
        center = (high + low) / 2
        if squashed:
            actions = center + actions * (high - low) / 2.0
        else:
            actions = np.clip(
                actions, 
                self.action_space.low, 
                self.action_space.high)
        return actions

    def sample_action(
        self, obs: np.ndarray, deterministic: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Samples an action from the environment or from our model

        :param obs: The input observation
        :param deterministic: Whether we are sampling deterministically.
        :return: The action to step with, and the action to store in our buffer
        """
        with torch.no_grad():
            # Convert to pytorch tensor
            obs_tensor = torch.as_tensor(obs).to(device=self.device, dtype=torch.float32)
            actions = self.predict(obs_tensor, deterministic=True)

        # Clip and scale actions appropriately
        if isinstance(self.action_space, gym.spaces.Box):
            actions = self.scale_actions(actions, self.squashed_actions)
        elif isinstance(self.action_space, (gym.spaces.Discrete,
                                            gym.spaces.MultiDiscrete,
                                            gym.spaces.MultiBinary)):
            actions = actions.astype(np.int32)
        return actions
    
    def collect_rollouts(self):
        """
        Collect rollouts and put them into the ReplayBuffer
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.reset_noise(self.env.num_envs)

        i = 0
        total_episodes = 0

        self.eval()
        while i < self.train_freq or total_episodes < self.episodes_per_rollout:
            if self.use_sde and self.sde_sample_freq > 0 and i % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.reset_noise(self.env.num_envs)
            
            if self.num_timesteps < self.warmup_length:
                actions = np.array([self.action_space.sample()])
            else:
                actions = self.sample_action(self._last_obs, deterministic=False)

            new_obs, rewards, dones, infos = self.env.step(actions)

            # If we are squashing actions, make sure the buffered actions are all squashed
            if isinstance(self.action_space, gym.spaces.Box) and self.squashed_actions:
                high, low = self.action_space.high, self.action_space.low
                center = (high + low) / 2
                actions = (actions - center) / (high - low) * 2
            elif isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            self.replay_buffer.add(self._last_obs, new_obs, actions, rewards, dones)

            self._last_obs = new_obs
            i += 1
            self.num_timesteps += 1
            # Note: VecEnv might not return None, it might return [None] or something, remember to double check this!
            if dones:
                total_episodes += 1

            self.on_step()
        self.train()

        # self.log('num_timesteps', self.num_timesteps, on_step=True, prog_bar=True, logger=True)

        if self.gradient_steps < 1:
            return i
        else:
            return self.gradient_steps

    def training_epoch_end(self, outputs) -> None:
        """
        Run the evaluation function at the end of the training epoch
        Override this if you also wish to do other things at the end of a training epoch
        """
        if self.num_timesteps >= self.warmup_length:
            self.eval()
            rewards, lengths = self.evaluate(self.num_eval_episodes)
            self.train()
            self.log_dict({
                'train/val_reward_mean': np.mean(rewards),
                'train/val_reward_std': np.std(rewards),
                'train/val_lengths_mean': np.mean(lengths),
                'train/val_lengths_std': np.std(lengths)},
                prog_bar=True, logger=True)


class OffPolicyDataloader:
    def __init__(self, model: OffPolicyModel):
        self.model = model


    def __iter__(self):
        for i in range(self.model.num_rollouts):
            gradient_steps = self.model.collect_rollouts()
            for j in range(gradient_steps):
                yield self.model.replay_buffer.sample()


class DQN(OffPolicyModel):
    """
    Deep Q-Network (DQN)

    Paper: https://arxiv.org/abs/1312.5602, https://www.nature.com/articles/nature14236
    Default hyperparameters are taken from the nature paper,
    except for the optimizer and learning rate that were taken from Stable Baselines defaults.

    :param env: The environment to learn from
        (if registered in Gym, can be str. Can be None for loading trained models)
    :param eval_env: The environment to evaluate on, must not be vectorised/parallelrised
        (if registered in Gym, can be str. Can be None for loading trained models)
    :param batch_size: Minibatch size for each gradient update
    :param buffer_length: length of the replay buffer
    :param warmup_length: how many steps of the model to collect transitions for before learning starts
    :param train_freq: Update the model every ``train_freq`` steps. Set to `-1` to disable.
    :param episodes_per_rollout: Update the model every ``episodes_per_rollout`` episodes.
        Note that this cannot be used at the same time as ``train_freq``. Set to `-1` to disable.
    :param num_rollouts: Number of rollouts to do per PyTorch Lightning epoch. This does not affect any training dynamic,
        just how often we evaluate the model since evaluation happens at the end of each Lightning epoch
    :param gradient_steps: How many gradient steps to do after each rollout
    :param target_update_interval: How many environment steps to wait between updating the target Q network
    :param num_eval_episodes: The number of episodes to evaluate for at the end of a PyTorch Lightning epoch
    :param gamma: the discount factor
    :param verbose: The verbosity level: 0 none, 1 training information, 2 debug (default: 0)
    :param seed: Seed for the pseudo random generators
    """
    
    def __init__(
        self,
        env: Union[GymEnv, str],
        eval_env: Union[GymEnv, str],
        batch_size: int = 256,
        buffer_length: int = int(1e6),
        warmup_length: int = 100,
        train_freq: int = 4,
        episodes_per_rollout: int = -1,
        num_rollouts: int = 1000,
        gradient_steps: int = 1,
        target_update_interval: int = 10000,
        num_eval_episodes: int = 10,
        gamma: float = 0.99,
        verbose: int = 0,
        seed: Optional[int] = None,
        lr=3e-4,
        hidden_size=64,
        eps_init=1.0,
        eps_decay=10000,
        eps_final=0.05,
    ):
        super(DQN, self).__init__(
            env=env,
            eval_env=eval_env,
            batch_size=batch_size,
            buffer_length=buffer_length,
            warmup_length=warmup_length,
            train_freq=train_freq,
            episodes_per_rollout=episodes_per_rollout,
            num_rollouts=num_rollouts,
            gradient_steps=gradient_steps,
            num_eval_episodes=num_eval_episodes,
            gamma=gamma,
            verbose=verbose,
            seed=seed,
            use_sde=False, # DQN Does not support SDE since DQN only supports Discrete actions spaces
            use_sde_at_warmup=False,
            sde_sample_freq=-1
        )

        assert isinstance(self.action_space, gym.spaces.Discrete), "DQN only supports environments with Discrete action spaces"

        self.target_update_interval = target_update_interval

        self.lr = lr

        self.qnet = nn.Sequential(
            nn.BatchNorm1d(self.observation_space.shape[0]),
            nn.Linear(self.observation_space.shape[0], hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, self.action_space.n)
        )

        self.eps = eps_init
        self.eps_init = eps_init
        self.eps_decay = eps_decay
        self.eps_final = eps_final

        self.qnet_target = copy.deepcopy(self.qnet)

        self.save_hyperparameters()

    def reset(self):
        """
        Resets the environment and the counter to keep track of target network updates
        """
        super(DQN, self).reset()
        self.update_timestep = 0

    def forward(self, x):
        return self.qnet(x)

    def forward_target(self, x):
        return self.qnet_target(x)

    def update_target(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())

    def on_step(self):  # Linearly decay our epsilon for epsilon greedy
        k = max(self.eps_decay - self.num_timesteps, 0) / self.eps_decay
        self.eps = self.eps_final + k * (self.eps_init - self.eps_final)
    
    def predict(self, x, deterministic=True):
        out = self.qnet(x)
        if deterministic:
            out = torch.max(out, dim=1)[1]
        else:
            eps = torch.rand_like(out[:, 0])
            eps = (eps < self.eps).float()
            out = eps * torch.max(torch.rand_like(out), dim=1)[1] +\
                (1 - eps) * torch.max(out, dim=1)[1]
        return out.long().cpu().numpy()
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.qnet.parameters(), lr=self.lr)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        """
        Specifies the update step for DQN. Override this if you wish to modify the DQN algorithm
        """
        if self.num_timesteps < self.warmup_length:
            return # This will make the loss a NaN but things are still working
        
        if float(self.num_timesteps - self.update_timestep) / self.target_update_interval > 1:
            self.update_target()
            self.update_timestep = self.num_timesteps
        
        with torch.no_grad():
            target_q = self.forward_target(batch.next_observations)
            target_q = torch.max(target_q, dim=1, keepdims=True)[0]
            target_q = batch.rewards + (1 - batch.dones) * self.gamma * target_q

        current_q = self(batch.observations)
        current_q = torch.gather(current_q, dim=1, index=batch.actions.long())

        loss = F.smooth_l1_loss(current_q, target_q)
        return loss

    @staticmethod
    def add_model_specific_args(parent_parser=None):
        if parent_parser:
            parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        else:
            parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--lr', type=float, default=3e-4)
        parser.add_argument('--hidden_size', type=int, default=64)
        parser.add_argument('--eps_init', type=float, default=1.0)
        parser.add_argument('--eps_decay', type=int, default=10000)
        parser.add_argument('--eps_final', type=float, default=0.05)
        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument('--buffer_length', type=int, default=int(1e6))
        parser.add_argument('--warmup_length', type=int, default=100)
        parser.add_argument('--train_freq', type=int, default=4)
        parser.add_argument('--episodes_per_rollout', type=int, default=-1)
        parser.add_argument('--num_rollouts', type=int, default=1024)
        parser.add_argument('--gradient_steps', type=int, default=1)
        parser.add_argument('--target_update_interval', type=int, default=512)
        parser.add_argument('--num_eval_episodes', type=int, default=10)
        parser.add_argument('--gamma', type=float, default=0.99)
        parser.add_argument('--verbose', action='store_true')
        parser.add_argument('--seed', type=int)
        return parser
    


if __name__ == '__main__':
    CHECKPOINT_PATH = "./saved_models"
    # Parse args separately so we don't have to abuse **kwargs
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--env', type=str, default='CartPole-v1')
    # If set to true, load model from model_fn and don't train
    parser.add_argument('--evaluate', action='store_true', default=False)
    parser.add_argument('--model_fn', type=str, default='saved_models/CartPole-v1_dqn_mlp/lightning_logs/version_1/checkpoints/mlp-epoch=02-val_reward_mean=144.10.pl.ckpt')
    parser.add_argument('--video_fn', type=str, default='ppo_mlp.mp4')
    args, ignored = parser.parse_known_args()

    if not args.evaluate:
        model_parser = DQN.add_model_specific_args()
        model_args, ignored = model_parser.parse_known_args()
        model_args = vars(model_args)

        trainer_parser = argparse.ArgumentParser(add_help=False)
        trainer_parser.add_argument('--gpus', type=int, default=1)
        trainer_parser.add_argument('--max_epochs', type=int, default=200)
        trainer_parser.add_argument('--gradient_clip_val', type=float, default=0.5)
        trainer_parser.add_argument('--default_root_dir', type=str, default=os.path.join(CHECKPOINT_PATH, args.env+'_dqn_mlp/'))
        # trainer_parser = pl.Trainer.add_argparse_args(trainer_parser)
        trainer_args, ignored = trainer_parser.parse_known_args()
        trainer_args = vars(trainer_args)

        env = gym.make(args.env)

        try:
            version_id = max([int(l.split('_')[1]) for l in os.listdir(trainer_args['default_root_dir'] + '/lightning_logs')]) + 1
        except:
            version_id = 0
            
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor='train/val_reward_mean',
            dirpath=trainer_args['default_root_dir'] + 'checkpoints',
            filename='dqn_mlp-version=' + str(version_id) + '-{epoch:02d}-{val_reward_mean:.2f}.pl',
            save_top_k=1,
            mode='max')

        model = DQN(
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
        model = DQN.load_from_checkpoint(
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