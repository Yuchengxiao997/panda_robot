import panda_gym
import gym
from kuka_real_gym import KukaReachEnv
#env = gym.make('PandaReach-v2')
env=KukaReachEnv(is_render=True,is_good_view=True)
obs = env.reset()
print(obs)