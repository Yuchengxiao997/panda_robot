from rlexplore.networks.random_encoder import CnnEncoder, MlpEncoder
from torch import optim
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np

class NGU(object):
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
        :param rollouts: The collected experiences.
        :param time_steps: The current time steps.
        :return: The intrinsic rewards
        """

        # compute the weighting coefficient of timestep t
        beta_t = self.beta * np.power(1. - self.kappa, time_steps)
        intrinsic_rewards = np.zeros_like(r.cpu())
        obs = o

        with torch.no_grad():

            # compute the life-long intrinsic rewards
            rnd_encoded_obs = self.predictor_network(obs[:])
            rnd_encoded_obs_target = self.target_network(obs[:])
            dist = torch.norm(rnd_encoded_obs - rnd_encoded_obs_target, p=2, dim=1)
            dist = (dist - dist.min()) / (dist.max() - dist.min() + 1e-6)
            life_long_rewards = dist.cpu().numpy()[1:]
            life_long_rewards = np.where(life_long_rewards >= 1., life_long_rewards, 1.0)
            # L=5
            life_long_rewards = np.where(life_long_rewards <= 5., life_long_rewards, 1.0)

            encoded_obs = obs[:]

            episodic_rewards = self.pseudo_counts(encoded_obs)
            intrinsic_rewards[:-1] = episodic_rewards[:-1] * life_long_rewards

        # update the rnd module
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

    def pseudo_counts(self,
                     encoded_obs,
                     k=10,
                     kernel_cluster_distance=0.008,
                     kernel_epsilon=0.0001,
                     c=0.001,
                     sm=8,
                     ):
        counts = np.zeros(shape=(encoded_obs.size()[0], ))
        for step in range(encoded_obs.size(0)):
            ob_dist = torch.norm(encoded_obs[step] - encoded_obs, p=2, dim=1)
            ob_dist = torch.sort(ob_dist).values
            ob_dist = ob_dist[:k]
            dist = ob_dist.cpu().numpy()
            # TODO: moving average
            dist = dist / np.mean(dist)
            dist = np.max(dist - kernel_cluster_distance, 0)
            kernel = kernel_epsilon / (dist + kernel_epsilon)
            s = np.sqrt(np.sum(kernel)) + c

            if np.isnan(s) or s > sm:
                counts[step] = 0.
            else:
                counts[step] = 1 / s
        return counts