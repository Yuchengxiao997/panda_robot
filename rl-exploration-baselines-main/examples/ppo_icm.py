#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Project ：rl-exploration-baselines 
@File ：ppo_icm.py
@Author ：YUAN Mingqi
@Date ：2022/9/20 20:02 
'''
import os
import sys
curren_dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir_path = os.path.abspath(os.path.join(curren_dir_path, os.pardir))
sys.path.append(parent_dir_path)

import torch
from stable_baselines3 import PPO
from rlexplore.icm import ICM
from rlexplore.re3 import RE3
from rlexplore.utils import create_env

if __name__ == '__main__':
    device = torch.device('cuda:0')
    env_id = 'AntBulletEnv-v0'
    n_envs = 1
    n_steps = 128
    total_time_steps = 10000
    num_episodes = int(total_time_steps / n_steps / n_envs)
    # Create vectorized environments.
    envs = create_env(
        env_id=env_id,
        n_envs=n_envs,
        log_dir='./logs'
    )
    # Create ICM module.
    icm = ICM(
        envs=envs,
        device=device,
        lr=0.00025,
        batch_size=64,
        beta=1e-2,
        kappa=1e-5
        )
    # Create PPO agent.
    model = PPO(policy='MlpPolicy', env=envs, n_steps=n_steps)
    _, callback = model._setup_learn(total_timesteps=total_time_steps, eval_env=None)

    for i in range(num_episodes):
        model.collect_rollouts(
            env=envs,
            rollout_buffer=model.rollout_buffer,
            n_rollout_steps=n_steps,
            callback=callback
        )
        # Compute intrinsic rewards.
        intrinsic_rewards = icm.compute_irs(
            rollouts={'observations': model.rollout_buffer.observations,'actions': model.rollout_buffer.actions},
            time_steps=i * n_steps * n_envs)
        print('内在奖励',intrinsic_rewards)
        model.rollout_buffer.rewards += intrinsic_rewards[:, :, 0]
        # Update policy using the currently gathered rollout buffer.
        model.train()
