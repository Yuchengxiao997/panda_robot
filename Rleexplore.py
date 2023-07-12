import time
import gym
import torch
from stable_baselines3 import PPO
from rlexplore.rise import RISE
from rlexplore.icm import ICM
from rlexplore.utils import create_env
from kuka_reach_env import KukaReachEnv

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

    #envs = env=KukaReachEnv(is_render=True,is_good_view=True)
    # Create RE3 module.
    #re3 = RE3(envs=envs, device=device, latent_dim=64, beta=1e-2, kappa=1e-5)
    re3 = RISE(envs=envs, device=device, latent_dim=128, beta=1e-2, kappa=1e-5)
    # Create PPO agent.
    model = PPO(policy='MlpPolicy', env=envs, n_steps=n_steps)
    _, callback = model._setup_learn(total_timesteps=total_time_steps, eval_env=None)

    for i in range(num_episodes):
        #每次动n_step=128次
        model.collect_rollouts(
            env=envs,
            rollout_buffer=model.rollout_buffer,
            n_rollout_steps=n_steps,
            callback=callback
        )
        print(i)
        #time.sleep(5)
        # Compute intrinsic rewards.
        '''
        intrinsic_rewards = re3.compute_irs(
            buffer=model.rollout_buffer,
            time_steps=i * n_steps * n_envs,
            k=3)
        '''
        intrinsic_rewards = re3.compute_irs(
            buffer=model.rollout_buffer,
            time_steps=i * n_steps * n_envs)
        #print(i,"内在奖励是什么",intrinsic_rewards)
        model.rollout_buffer.rewards += intrinsic_rewards
        #print("原奖励",model.rollout_buffer.rewards)
        # Update policy using the currently gathered rollout buffer.
        model.train()