import time
import ArmEnv
import gym
import panda_gym
from kuka_reach_env import KukaReachEnv
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import DDPG
from stable_baselines3 import TD3
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
import os
log_path = os.path.join('Training','logs')
#env = gym.make('PandaReach-v2', render=True)
#obs = env.reset()
#print('Observation:',obs)
#env = gym.make('PointToPoint-v0',gui=True,mode='T')
env=KukaReachEnv(is_render=True,is_good_view=True)
#check_env(env)
#print("nice")
model = SAC("MultiInputPolicy", env,verbose=1,tensorboard_log=log_path)

path = os.path.join('train','ppo_with_gsde')

#model = PPO("MultiInputPolicy", env,verbose=1)
model = PPO("MlpPolicy", env,verbose=1)
start = time.time()
model.learn(total_timesteps=50000)
end = time.time()
print("花费了多长时间：",start -end)
model.save(path)

model = PPO.load(path,env = env)
obs = env.reset()
print('Observation:',obs)
done = False
temp = 0
flag = 0
fuck = 0
for i in range(1000):
    #action = env.action_space.sample() # random action
    temp = temp + 1
    action, _states = model.predict(obs,deterministic=True)
    #print("step----->",action)
    obs, reward, done, info = env.step(action)
    #print(info["is_success"])
    #print(reward,done)
    #obs, reward, done, info = env.step(action)
    #time.sleep(0.05)
    if reward ==3:
        flag = flag+1
    #print("第{}步".format(temp))
    if done:
        fuck = fuck+1
        print('共经历了几步：',temp)
        time.sleep(2)
        temp = 0
        obs = env.reset()
print(flag,fuck,flag/fuck)
env.close()