import torch
import numpy as np
from rlexplore.re3 import RE3

if __name__ == '__main__':
    ''' env setup '''
    device = torch.device('cuda:0')
    obs_shape = (4, 84, 84)
    action_shape = 1 # for discrete action space
    n_envs = 16
    n_steps = 256
    observations = np.random.randn(
       n_steps, n_envs, *obs_shape).astype('float32') # collected experiences

    ''' create RE3 instance '''
    re3 = RE3(obs_shape=obs_shape, action_shape=action_shape, device=device,
              latent_dim=128, beta=0.05, kappa=0.00001)

    ''' compute intrinsic rewards '''
    intrinsic_rewards = re3.compute_irs(rollouts={'observations': observations},
        time_steps=25600, k=3, average_entropy=False)

    print(intrinsic_rewards.shape, type(intrinsic_rewards))
    print(intrinsic_rewards)

# Output: (256, 16, 1) <class 'numpy.ndarray'>