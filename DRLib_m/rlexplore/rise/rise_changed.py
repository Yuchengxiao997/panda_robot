#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Project ：rl-exploration-baselines 
@File ：rise.py
@Author ：YUAN Mingqi
@Date ：2022/9/20 13:38 
'''

from rlexplore.networks.random_encoder import CnnEncoder, MlpEncoder
import torch
import numpy as np

class RISE(object):
    def __init__(self,
                 obs_shape,
                 device,
                 latent_dim,
                 beta,
                 kappa
                 ):
        """
        Rényi State Entropy Maximization for Exploration Acceleration in Reinforcement Learning (RISE)
        Paper: https://ieeexplore.ieee.org/abstract/document/9802917/

        :param envs: The environment to learn from.
        :param device: Device (cpu, cuda, ...) on which the code should be run.
        :param latent_dim: The dimension of encoding vectors of the observations.
        :param beta: The initial weighting coefficient of the intrinsic rewards.
        :param kappa: The decay rate.
        """

        self.ob_shape = obs_shape
        self.device = device
        self.beta = beta
        self.kappa = kappa


        self.encoder = MlpEncoder(
            kwargs={'input_dim': self.ob_shape[0], 'latent_dim': latent_dim}
            )

        self.encoder.to(self.device)

        # freeze the network parameters
        for p in self.encoder.parameters():
            p.requires_grad = False

    def compute_irs(self, o,r, time_steps, alpha=0.5, k=3):
        """
        Compute the intrinsic rewards using the collected observations.
        :param buffer: The experiences buffer.
        :param time_steps: The current time steps.
        :param alpha: The order of Rényi divergence.
        :param k: The k value.
        :return: The intrinsic rewards
        """

        # compute the weighting coefficient of timestep t
        beta_t = self.beta * np.power(1. - self.kappa, time_steps)
        print("置信水平：", beta_t)
        intrinsic_rewards = np.zeros_like(r)

        # observations shape ((n_steps, n_envs) + obs_shape)
        obs_tensor = o

        encoded_obs = self.encoder(obs_tensor[:])
        dist = torch.norm(encoded_obs.unsqueeze(1) - encoded_obs, p=2, dim=2)
        intrinsic_rewards[:] = torch.pow(torch.kthvalue(dist, k + 1).values, 1. - alpha).cpu()

        return beta_t * intrinsic_rewards