#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Project ：rl-exploration-baselines
@File ：revd.py
@Author ：YUAN Mingqi
@Date ：2022/12/03 20:35
'''

from rlexplore.networks.random_encoder import CnnEncoder, MlpEncoder

import os
import torch
import numpy as np

class REVD(object):
    def __init__(self,
                 obs_shape,
                 device,
                 latent_dim,
                 beta,
                 kappa
                 ):
        """
        Rewarding Episodic Visitation Discrepancy for Exploration in Reinforcement Learning (REVD)
        Paper: https://openreview.net/pdf?id=V2pw1VYMrDo

        :param obs_shape: The data shape of observations.
        :param action_shape: The data shape of actions.
        :param device: Device (cpu, cuda, ...) on which the code should be run.
        :param latent_dim: The dimension of encoding vectors of the observations.
        :param beta: The initial weighting coefficient of the intrinsic rewards.
        :param kappa: The decay rate.
        """

        self.obs_shape = obs_shape
        self.device = device
        self.beta = beta
        self.kappa = kappa

        self.encoder = MlpEncoder(
            kwargs={'input_dim': self.obs_shape, 'latent_dim': latent_dim}
        )

        self.encoder.to(self.device)

        # freeze the network parameters
        for p in self.encoder.parameters():
            p.requires_grad = False
        
        self.num_updates = 0
        self.last_encoded_obs = list()
    
    def compute_irs(self, o,r, time_steps, alpha=0.5, k=3, average_divergence=False):
        """
        Compute the intrinsic rewards using the collected observations.
        :param rollouts: The collected experiences.
        :param time_steps: The current time steps.
        :param alpha: The order of Rényi divergence.
        :param k: The k value.
        :param average_divergence: Use the average of divergence estimation.
        :return: The intrinsic rewards
        """
        # compute the weighting coefficient of timestep t
        beta_t = self.beta * np.power(1. - self.kappa, time_steps)
        intrinsic_rewards = np.zeros_like(r.cpu())
        obs = o
        if self.num_updates == 0:

            src_feats = self.encoder(obs[:])
            self.last_encoded_obs=src_feats
            self.num_updates += 1
            return intrinsic_rewards

        with torch.no_grad():

            src_feats = self.encoder(obs[:])
            dist_intra = torch.linalg.vector_norm(src_feats.unsqueeze(1) - src_feats, ord=2, dim=2)
            dist_outer = torch.linalg.vector_norm(src_feats.unsqueeze(1) - self.last_encoded_obs, ord=2, dim=2)

            if average_divergence:
                pass
            else:
                D_step_intra = torch.kthvalue(dist_intra, k + 1, dim=1).values
                D_step_outer = torch.kthvalue(dist_outer, k + 1, dim=1).values
                L = torch.kthvalue(dist_intra, 2, dim=1).values.cpu().numpy().sum()
                intrinsic_rewards[:] = L * torch.pow(D_step_outer / (D_step_intra + 0.0001), 1. - alpha).cpu().numpy()

            self.last_encoded_obs = src_feats
        
        self.num_updates += 1
        #print(beta_t * intrinsic_rewards)
        return beta_t * intrinsic_rewards