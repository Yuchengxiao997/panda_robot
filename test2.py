import gym

from stable_baselines3 import PPO
from kuka_reach_env_last1 import KukaReachEnv

env = gym.make("CartPole-v1")
#env=KukaReachEnv(is_render='True',is_good_view='True')

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

obs = env.reset()
#print("返回观察空间：>>>>>>>>>>>>>>>>>>")
#print(obs)
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    print(obs)
    print(reward)
    env.render()
    if done:
      obs = env.reset()

env.close()