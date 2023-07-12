#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Project ：rl-exploration-baselines
@File ：re3.py
@Author ：YUAN Mingqi
@Date ：2022/12/03 20:35
'''

from rlexplore.networks.random_encoder import CnnEncoder, MlpEncoder

import os
import torch
import numpy as np

class RE3(object):
    def __init__(self,
                 obs_shape,
                 action_shape,
                 device,
                 latent_dim,
                 beta,
                 kappa
                 ):
        """
        State Entropy Maximization with Random Encoders for Efficient Exploration (RE3)
        Paper: http://proceedings.mlr.press/v139/seo21a/seo21a.pdf

        :param obs_shape: The data shape of observations.
        :param action_shape: The data shape of actions.
        :param device: Device (cpu, cuda, ...) on which the code should be run.
        :param latent_dim: The dimension of encoding vectors of the observations.
        :param beta: The initial weighting coefficient of the intrinsic rewards.
        :param kappa: The decay rate.
        """

        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.device = device
        self.beta = beta
        self.kappa = kappa

        if len(self.obs_shape) == 3:
            self.encoder = CnnEncoder(obs_shape, latent_dim)
        else:
            self.encoder = MlpEncoder(obs_shape, latent_dim)

        self.encoder.to(self.device)

        # freeze the network parameters
        for p in self.encoder.parameters():
            p.requires_grad = False
    
    def compute_irs(self, rollouts, time_steps, k=3, average_entropy=True):
        """
        Compute the intrinsic rewards using the collected observations.
        :param rollouts: The collected experiences.
        :param time_steps: The current time steps.
        :param k: The k value.
        :param average_entropy: Use the average of entropy estimation.
        :return: The intrinsic rewards
        """
        # compute the weighting coefficient of timestep t
        beta_t = self.beta * np.power(1. - self.kappa, time_steps)
        n_steps = rollouts['observations'].shape[0]
        n_envs = rollouts['observations'].shape[1]
        intrinsic_rewards = np.zeros(shape=(n_steps, n_envs, 1))

        # observations shape ((n_steps, n_envs) + obs_shape)
        obs_tensor = torch.from_numpy(rollouts['observations'])
        obs_tensor = obs_tensor.to(self.device)

        with torch.no_grad():
            for idx in range(n_envs):
                src_feats = self.encoder(obs_tensor[:, idx])
                dist = torch.linalg.vector_norm(src_feats.unsqueeze(1) - src_feats, ord=2, dim=2)
                if average_entropy:
                    for sub_k in range(k):
                        intrinsic_rewards[:, idx, 0] += torch.log(
                            torch.kthvalue(dist, sub_k + 1, dim=1).values + 1.).cpu().numpy()
                    intrinsic_rewards[:, idx, 0] /= k
                else:
                    intrinsic_rewards[:, idx, 0] = torch.log(
                            torch.kthvalue(dist, k + 1, dim=1).values + 1.).cpu().numpy()
        
        return beta_t * intrinsic_rewards
