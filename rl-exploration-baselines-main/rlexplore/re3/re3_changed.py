#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Project ：rl-exploration-baselines
@File ：re3.py
@Author ：YUAN Mingqi
@Date ：2022/9/19 20:35
'''

from rlexplore.networks.random_encoder import CnnEncoder, MlpEncoder

import os
import torch
import numpy as np

class RE3(object):
    def __init__(self,
                 obs_shape,
                 device,
                 latent_dim,
                 beta,
                 kappa
                 ):
        """
        State Entropy Maximization with Random Encoders for Efficient Exploration (RE3)
        Paper: http://proceedings.mlr.press/v139/seo21a/seo21a.pdf

        :param envs: The environment to learn from.
        :param device: Device (cpu, cuda, ...) on which the code should be run.
        :param latent_dim: The dimension of encoding vectors of the observations.
        :param beta: The initial weighting coefficient of the intrinsic rewards.
        :param kappa: The decay rate.
        """


        self.device = device
        self.beta = beta
        self.kappa = kappa

        self.ob_shape = obs_shape
        self.encoder = MlpEncoder(
            kwargs={'input_dim': self.ob_shape, 'latent_dim': latent_dim}
        )
        self.encoder.to(self.device)

        # freeze the network parameters
        for p in self.encoder.parameters():
            p.requires_grad = False

    def compute_irs(self, o,r, time_steps,k=3):
        """
        Compute the intrinsic rewards using the collected observations.
        :param buffer: The experiences buffer.
        :param time_steps: The current time steps.
        :param k: The k value.
        :return: The intrinsic rewards
        """

        # compute the weighting coefficient of timestep t
        beta_t = self.beta * np.power(1. - self.kappa, time_steps)
        #print("置信水平：", beta_t)
        intrinsic_rewards = np.zeros_like(r.cpu())
        obs = o
        encoded_obs = obs[:]
        encoded_obs = self.encoder(encoded_obs)
        dist = torch.norm(encoded_obs.unsqueeze(1) - encoded_obs, p=2, dim=2)
        intrinsic_rewards[:] = torch.log(torch.kthvalue(dist, k + 1, dim=1).values + 1.).cpu().numpy()
        #print("re3内在奖励：",intrinsic_rewards)
        return beta_t * intrinsic_rewards
