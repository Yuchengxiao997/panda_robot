#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@Project ：rl-exploration-baselines 
@File ：icm.py
@Author ：YUAN Mingqi
@Date ：2022/9/20 14:43 
'''

from rlexplore.networks.inverse_forward_networks import InverseForwardDynamicsModel, CnnEncoder
# from rlexplore.utils.state_process import process

from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch
import numpy as np

class ICM(object):
    def __init__(self,
                 obs_shape,
                 act_shape,
                 device,
                 lr,
                 batch_size,
                 beta,
                 kappa
                 ):
        """
        Curiosity-Driven Exploration by Self-Supervised Prediction
        Paper: http://proceedings.mlr.press/v70/pathak17a/pathak17a.pdf

        :param envs: The environment to learn from.
        :param device: Device (cpu, cuda, ...) on which the code should be run.
        :param lr: The learning rate of inverse and forward dynamics model.
        :param batch_size: The batch size to train the dynamics model.
        :param beta: The initial weighting coefficient of the intrinsic rewards.
        :param kappa: The decay rate.
        """
        self.device = device
        self.beta = beta
        self.kappa = kappa
        self.lr = lr
        self.batch_size = batch_size


        self.ob_shape = obs_shape
        self.action_shape = act_shape
        self.action_type = 'cont'
        self.inverse_forward_model = InverseForwardDynamicsModel(
            kwargs={'latent_dim': self.ob_shape, 'action_dim': self.action_shape}
        ).to(device)
        self.im_loss = nn.MSELoss()


        self.fm_loss = nn.MSELoss()


        self.optimizer = optim.Adam(lr=self.lr, params=self.inverse_forward_model.parameters())

    def update(self, o,a):
        obs = o
        actions = a
        #做了那么多，只是为了转变一下格式而已
        encoded_obs = obs

        dataset = TensorDataset(encoded_obs[:-1], actions[:-1], encoded_obs[1:])
        loader = DataLoader(dataset=dataset, batch_size=self.batch_size, drop_last=True)

        for idx, batch_data in enumerate(loader):
            batch_obs = batch_data[0]
            batch_actions = batch_data[1]
            batch_next_obs = batch_data[2]

            pred_actions, pred_next_obs = self.inverse_forward_model(
                batch_obs, batch_actions, batch_next_obs
            )

            loss = self.im_loss(pred_actions, batch_actions) + \
                   self.fm_loss(pred_next_obs, batch_next_obs)

            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()

    def compute_irs(self, o,a,r, time_steps):
        # compute the weighting coefficient of timestep t

        beta_t = self.beta * np.power(1. - self.kappa, time_steps)
        #print("置信水平：",beta_t)
        #随着时间的推移，置信度逐渐降低，但还是不高啊
        intrinsic_rewards = np.zeros_like(r.cpu())
        obs = o
        actions = a
        #print("action",a)
        with torch.no_grad():
            encoded_obs = obs[:]
            pred_next_obs = self.inverse_forward_model(
                encoded_obs[:-1], actions[:-1], next_obs=None, training=False)
            #processed_next_obs = torch.clip(encoded_obs[1:], min=-1.0, max=1.0)
            #processed_pred_next_obs = torch.clip(pred_next_obs, min=-1.0, max=1.0)
            processed_next_obs = encoded_obs[1:]
            processed_pred_next_obs = pred_next_obs
            intrinsic_rewards[:-1] = F.mse_loss(processed_pred_next_obs, processed_next_obs, reduction='mean').cpu().numpy()
            # processed_next_obs = process(encoded_obs[1:n_steps], normalize=True, range=(-1, 1))
            # processed_pred_next_obs = process(pred_next_obs, normalize=True, range=(-1, 1))
        # train the icm
        self.update(o,a)
        #print("内在奖励",beta_t *intrinsic_rewards)
        return beta_t * intrinsic_rewards