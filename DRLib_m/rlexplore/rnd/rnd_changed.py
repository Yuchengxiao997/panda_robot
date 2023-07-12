#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Project ：rl-exploration-baselines 
@File ：rnd.py
@Author ：YUAN Mingqi
@Date ：2022/9/20 21:46 
'''

from rlexplore.networks.random_encoder import CnnEncoder, MlpEncoder
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np

class RND(object):
    def __init__(self,
                 obs_shape,
                 device,
                 latent_dim,
                 lr,
                 batch_size,
                 beta,
                 kappa
                 ):
        self.ob_shape = obs_shape
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.beta = beta
        self.kappa = kappa

        self.predictor_network = MlpEncoder(
            kwargs={'input_dim': self.ob_shape, 'latent_dim': latent_dim}
        )
        self.target_network = MlpEncoder(
            kwargs={'input_dim': self.ob_shape, 'latent_dim': latent_dim}
        )

        self.predictor_network.to(self.device)
        self.target_network.to(self.device)

        self.optimizer = optim.Adam(lr=self.lr, params=self.predictor_network.parameters())

        # freeze the network parameters
        for p in self.target_network.parameters():
            p.requires_grad = False

    def compute_irs(self, o,r, time_steps):
        """
        Compute the intrinsic rewards using the collected observations.
        :param buffer: The experiences buffer.
        :param time_steps: The current time steps.
        :return: The intrinsic rewards
        """

        # compute the weighting coefficient of timestep t
        beta_t = self.beta * np.power(1. - self.kappa, time_steps)
        intrinsic_rewards = np.zeros_like(r.cpu())
        obs = o

        with torch.no_grad():
            encoded_obs = self.predictor_network(obs[:])
            encoded_obs_target = self.target_network(obs[:])
            dist = torch.norm(encoded_obs - encoded_obs_target, p=2, dim=1)
            dist = (dist - dist.min()) / (dist.max() - dist.min() + 1e-6)

            intrinsic_rewards[:-1] = dist.cpu().numpy()[1:]

        self.update(o)

        return beta_t * intrinsic_rewards

    def update(self, o):
        obs = o



        dataset = TensorDataset(obs)
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, drop_last=True)

        for idx, batch_data in enumerate(loader):

            batch_obs = batch_data[0]
            encoded_obs = self.predictor_network(batch_obs)
            encoded_obs_target = self.target_network(batch_obs)

            loss = F.mse_loss(encoded_obs, encoded_obs_target)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()