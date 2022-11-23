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
from common.distributions import SquashedMultivariateNormal
from common.utils import polyak_update

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


class SAC(OffPolicyModel):
    """
    Soft Actor-Critic (SAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup), from the softlearning repo
    (https://github.com/rail-berkeley/softlearning/)
    and from Stable Baselines (https://github.com/hill-a/stable-baselines)
    Paper: https://arxiv.org/abs/1801.01290
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

    Note: we use double q target and not value target as discussed
    in https://github.com/hill-a/stable-baselines/issues/270

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
    :param entropy_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param squashed_actions: Whether the actions are squashed between [-1, 1] and need to be unsquashed
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
        train_freq: int = 1,
        episodes_per_rollout: int = -1,
        num_rollouts: int = 1000,
        gradient_steps: int = 1,
        target_update_interval: int = 1,
        num_eval_episodes: int = 10,
        gamma: float = 0.99,
        entropy_coef: Union[str, float] = "auto",
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        squashed_actions: bool = True,
        verbose: int = 0,
        seed: Optional[int] = None,
        lr=3e-4,
        hidden_size=256,
        tau=0.005,
    ):
        self.entropy_coef = entropy_coef
        self.target_entropy = target_entropy
        self.target_update_interval = target_update_interval

        super(SAC, self).__init__(
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
            squashed_actions=squashed_actions,
            seed=seed,
            use_sde=use_sde,
            use_sde_at_warmup=use_sde_at_warmup,
            sde_sample_freq=sde_sample_freq
        )

        assert isinstance(self.action_space, gym.spaces.Box), "SAC only supports environments with Box action spaces"

        # We need manual optimization for this
        self.automatic_optimization = False

        self.lr = lr
        self.tau = tau

        # Note: The output layer of the actor must be Tanh activated
        self.actor = nn.Sequential(
            nn.Linear(self.observation_space.shape[0], hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, self.action_space.shape[0] * 2))

        in_dim = self.observation_space.shape[0] + self.action_space.shape[0]
        self.critic1 = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1))

        self.critic2 = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1))

        self.critic_target1 = copy.deepcopy(self.critic1)
        self.critic_target2 = copy.deepcopy(self.critic2)

        self.save_hyperparameters()

    def reset(self):
        """
        Resets the environment and automatic entropy
        """
        super(SAC, self).reset()
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        if isinstance(self.entropy_coef, str):
            if not hasattr(self, 'log_entropy_coef'):
                assert self.entropy_coef.startswith("auto")
                # Default initial value of entropy_coef when learned
                init_value = 1.0
                if "_" in self.entropy_coef:
                    init_value = float(self.entropy_coef.split("_")[1])
                    assert init_value > 0.0, "The initial value of entropy_coef must be greater than 0"

                # Note: we optimize the log of the entropy coeff which is slightly different from the paper
                # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
                self.log_entropy_coef = torch.log(torch.ones(1, device=self.device) * init_value)
                self.log_entropy_coef = nn.Parameter(self.log_entropy_coef.requires_grad_(True))
                self.entropy_coef_optimizer = torch.optim.Adam([self.log_entropy_coef], lr=3e-4)
        else:
            # I know this isn't very efficient but it makes the code cleaner
            # and it's only one extra operation
            self.log_entropy_coef = torch.log(float(self.entropy_coef))

    def forward_actor(self, x):
        out = list(torch.chunk(self.actor(x), 2, dim=1))
        out[1] = torch.diag_embed(
            torch.exp(torch.clamp(out[1], -5, 5)))
        dist = SquashedMultivariateNormal(
            loc=torch.tanh(out[0]), scale_tril=out[1])
        return dist


    def forward_critics(self, obs, action):
        out = [
            self.critic1(torch.cat([obs, action], dim=1)),
            self.critic2(torch.cat([obs, action], dim=1))]
        return out

    def forward_critic_targets(self, obs, action):
        out = [
            self.critic_target1(torch.cat([obs, action], dim=1)),
            self.critic_target1(torch.cat([obs, action], dim=1))]
        return out

    def update_targets(self) -> None:
        polyak_update(
            self.critic1.parameters(),
            self.critic_target1.parameters(),
            tau=self.tau)
        polyak_update(
            self.critic2.parameters(),
            self.critic_target2.parameters(),
            tau=self.tau)

    def predict(self, x, deterministic=True):
        out = self.actor(x)
        if deterministic:
            out = torch.chunk(out, 2, dim=1)[0]
        else:
            out = list(torch.chunk(out, 2, dim=1))
            out[1] = torch.diag_embed(
                torch.exp(torch.clamp(out[1], -5, 5)))
            out = SquashedMultivariateNormal(
                loc=torch.tanh(out[0]), scale_tril=out[1]).sample()
        return out.cpu().numpy()

    def configure_optimizers(self):
        opt_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        opt_critic = torch.optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=self.lr)
        return opt_critic, opt_actor

    def training_step(self, batch, batch_idx):
        """
        Specifies the update step for SAC. Override this if you wish to modify the SAC algorithm
        """
        # We need to sample because `log_std` may have changed between two gradient steps
        if self.num_timesteps < self.warmup_length:
            return

        opt_critic, opt_actor = self.optimizers(use_pl_optimizer=True)
        # Action by the current actor for the sampled state
        dist = self.forward_actor(batch.observations.to(torch.float32))
        actions = dist.rsample()
        log_probs = dist.log_prob(actions)

        log_entropy_coef = torch.clamp(self.log_entropy_coef, -10, 5)
        entropy_coef = torch.exp(log_entropy_coef)
        if hasattr(self, 'entropy_coef_optimizer'):
            # Important: detach the variable from the graph
            # so we don't change it with other losses
            # see https://github.com/rail-berkeley/softlearning/issues/60
            entropy_coef = entropy_coef.detach()
            entropy_coef_loss = -(log_entropy_coef * (log_probs + self.target_entropy).detach()).mean()
            self.log('train/entropy_coef_loss', entropy_coef_loss, on_step=True, prog_bar=True, logger=True)

        self.log('train/entropy_coef', entropy_coef, on_step=True, prog_bar=False, logger=True)

        # Optimize entropy coefficient, also called
        # entropy temperature or alpha in the paper
        if hasattr(self, 'entropy_coef_optimizer'):
            self.entropy_coef_optimizer.zero_grad()
            self.manual_backward(entropy_coef_loss, self.entropy_coef_optimizer)
            self.entropy_coef_optimizer.step()

        with torch.no_grad():
            # Select action according to policy
            next_dist = self.forward_actor(batch.next_observations.to(torch.float32))
            next_actions = next_dist.rsample()
            next_log_probs = next_dist.log_prob(next_actions)

            # Compute the target Q value: min over all critics targets
            targets = self.forward_critic_targets(batch.next_observations.to(torch.float32), next_actions)
            target_q = torch.minimum(*targets)
            # add entropy term
            target_q = target_q - entropy_coef * next_log_probs[..., None]
            # td error + entropy term
            target_q = batch.rewards + (1 - batch.dones) * self.gamma * target_q

        # Get current Q estimates for each critic network
        # using action from the replay buffer
        current_q_estimates = self.forward_critics(batch.observations.to(torch.float32), batch.actions)

        # Compute critic loss
        critic_loss = torch.stack([F.mse_loss(current_q, target_q) for current_q in current_q_estimates])
        critic_loss = torch.mean(critic_loss)
        self.log('train/critic_loss', critic_loss, on_step=True, prog_bar=True, logger=True)

        # Optimize the critic
        opt_critic.zero_grad()
        self.manual_backward(critic_loss, opt_critic)
        opt_critic.step()

        # Compute actor loss
        # Alternative: actor_loss = torch.mean(log_prob - qf1_pi)
        # Mean over all critic networks
        q_values_pi = self.forward_critics(batch.observations.to(torch.float32), actions)
        min_qf_pi = torch.minimum(*q_values_pi)
        actor_loss = (entropy_coef * log_probs[..., None] - min_qf_pi).mean()
        self.log('train/actor_loss', actor_loss, on_step=True, prog_bar=True, logger=True)

        # Optimize the actor
        opt_actor.zero_grad()
        self.manual_backward(actor_loss, opt_actor)
        opt_actor.step()

        # Update target networks
        if batch_idx % self.target_update_interval == 0:
            self.update_targets()



    @staticmethod
    def add_model_specific_args(parent_parser=None):
        if parent_parser:
            parser = argparse.ArgumentParser(
                parents=[parent_parser], add_help=False)
        else:
            parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument('--lr', type=float, default=3e-4)
        parser.add_argument('--hidden_size', type=int, default=256)
        parser.add_argument('--tau', type=float, default=0.005)
        parser.add_argument('--buffer_length', type=int, default=int(1e6))
        parser.add_argument('--warmup_length', type=int, default=10000)
        parser.add_argument('--batch_size', type=int, default=256)
        parser.add_argument('--train_freq', type=int, default=1)
        parser.add_argument('--num_rollouts', type=int, default=1000)
        parser.add_argument('--episodes_per_rollout', type=int, default=-1)
        parser.add_argument('--gradient_steps', type=int, default=1)
        parser.add_argument('--num_eval_episodes', type=int, default=10)
        parser.add_argument('--target_update_interval', type=int, default=1)
        parser.add_argument('--gamma', type=float, default=0.99)
        parser.add_argument('--entropy_coef', default='auto')
        parser.add_argument('--target_entropy', default='auto')
        parser.add_argument('--use_sde', action='store_true')
        parser.add_argument('--sde_sample_freq', type=int, default=-1)
        parser.add_argument('--use_sde_at_warmup', action='store_true')
        parser.add_argument('--squashed_actions', type=bool, default=False)
        parser.add_argument('--verbose', action='store_true')
        parser.add_argument('--seed', type=int, default=0)
        return parser




if __name__ == '__main__':
    CHECKPOINT_PATH = "./saved_models"
    # Parse args separately so we don't have to abuse **kwargs
    parser = argparse.ArgumentParser(add_help=False)
    # parser.add_argument('--env', type=str, default='LunarLanderContinuous-v2')
    parser.add_argument('--env', type=str, default='InvertedPendulum-v2')
    # If set to true, load model from model_fn and don't train
    parser.add_argument('--evaluate', action='store_true', default=False)
    parser.add_argument('--model_fn', type=str, default='saved_models/CartPole-v1_dqn_mlp/lightning_logs/version_1/checkpoints/mlp-epoch=02-val_reward_mean=144.10.pl.ckpt')
    parser.add_argument('--video_fn', type=str, default='ppo_mlp.mp4')
    args, ignored = parser.parse_known_args()

    if not args.evaluate:
        model_parser = SAC.add_model_specific_args()
        model_args, ignored = model_parser.parse_known_args()
        model_args = vars(model_args)

        trainer_parser = argparse.ArgumentParser(add_help=False)
        trainer_parser.add_argument('--gpus', type=int, default=1)
        trainer_parser.add_argument('--max_epochs', type=int, default=200)
        # trainer_parser.add_argument('--gradient_clip_val', type=float, default=0.5)
        trainer_parser.add_argument('--default_root_dir', type=str, default=os.path.join(CHECKPOINT_PATH, args.env+'_sac_mlp/'))
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
            filename='sac_mlp-version=' + str(version_id) + '-{epoch:02d}-{val_reward_mean:.2f}.pl',
            save_top_k=1,
            mode='max')

        model = SAC(
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
        model = SAC.load_from_checkpoint(
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