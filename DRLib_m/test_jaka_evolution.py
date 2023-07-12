#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
@File    :   kuka_reach_env.py
@Time    :   2021/03/20 14:33:24
@Author  :   Yan Wen
@Version :   1.0
@Contact :   z19040042@s.upc.edu.cn
@License :   (C)Copyright 2021-2022, Liugroup-NLPR-CASIA
@Desc    :   None
'''

# here put the import lib

import pybullet as p
import pybullet_data
import os
import sys
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from math import sqrt
import random
import time
from numpy import arange
import logging
import math
from termcolor import colored

#### 一些变量 ######
LOGGING_LEVEL = logging.INFO
# is_render=False
# is_good_view=False   #这个的作用是在step时加上time.sleep()，把机械比的动作放慢，看的更清，但是会降低训练速度
#########################

# logging.basicConfig(
#     level=LOGGING_LEVEL,
#     format='%(asctime)s - %(threadName)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
#     filename='../logs/reach_env.log'.format(time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())),
#     filemode='w')
# logger = logging.getLogger(__name__)
# env_logger=logging.getLogger('env.py')

# logging模块的使用
# 级别                何时使用
# DEBUG       细节信息，仅当诊断问题时适用。
# INFO        确认程序按预期运行
# WARNING     表明有已经或即将发生的意外（例如：磁盘空间不足）。程序仍按预期进行
# ERROR       由于严重的问题，程序的某些功能已经不能正常执行
# CRITICAL    严重的错误，表明程序已不能继续执行
def set_obstacle(position):
    scale = [1, 1, 1]
    visual_shape_id2 = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        rgbaColor=[0, 1, 1, 1],
        radius=0.01,
        specularColor=[0.4, 0.4, 0],
        visualFramePosition=[0, 0, 0],
        meshScale=scale)
    collision_shape_id2 = p.createCollisionShape(
        shapeType=p.GEOM_SPHERE,
        collisionFramePosition=[0, 0, 0],
        meshScale=scale)
    for i in range(len(position)):
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape_id2,
            basePosition=position[i],
            useMaximalCoordinates=True)

def produce_position(base_position):
    position = []
    for i in range(29):
        a = ring()
        b = ring()
        temp1 =[
            base_position[0]+a,
            base_position[1]+b,
            random.uniform(base_position[2]-0.05,
                           base_position[2]+0.05),
        ]
        position.append(temp1)
    return position
def ring():
    a = random.uniform(-0.13,0.13)
    while abs(a)<0.02:
        a = random.uniform(-0.13,0.13)
    return a

class KukaReachEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
    max_steps_one_episode = 4000

    def __init__(self, is_render=False, is_good_view=False):

        self.is_render = is_render
        self.is_good_view = is_good_view

        if self.is_render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        self.x_low_obs = 0.5
        self.x_high_obs = 1
        self.y_low_obs = -0.2
        self.y_high_obs = 0.2
        self.z_low_obs = 0
        self.z_high_obs = 0.55

        self.x_low_action = -0.8
        self.x_high_action = 0.8
        self.y_low_action = -0.8
        self.y_high_action = 0.8
        self.z_low_action = -0.8
        self.z_high_action = 0.8

        p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                     cameraYaw=0,
                                     cameraPitch=-40,
                                     cameraTargetPosition=[0.55, -0.35, 0.2])

        self.action_space = spaces.Box(low=np.array(
            [self.x_low_action, self.y_low_action, self.z_low_action]),
                                       high=np.array([
                                           self.x_high_action,
                                           self.y_high_action,
                                           self.z_high_action
                                       ]),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([self.x_low_obs, self.y_low_obs, self.z_low_obs]),
            high=np.array([self.x_high_obs, self.y_high_obs, self.z_high_obs]),
            dtype=np.float32)
        self.observation_space = spaces.Box(low=self.y_low_obs, high=self.x_high_obs, shape=(1, 3))
        self.step_counter = 0

        self.urdf_root_path = pybullet_data.getDataPath()
        print("路径：")
        print(self.urdf_root_path)
        # lower limits for null space
        self.lower_limits = [-2.88, -2.88, -2.88, -2.88, -2.88, -3.05]
        # upper limits for null space
        self.upper_limits = [2.88, 2.88, 2.88, 2.88, 2.88, 3.05]
        # joint ranges for null space
        self.joint_ranges = [ 5.76, 5.76, 5.76, 5.76, 5.76, 6.1]
        # restposes for null space
        self.rest_poses = [ 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
        # joint damping coefficents
        self.joint_damping = [
            0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001
        ]

        self.init_joint_positions = [
            0, 0, 0, -1.57, -1.57, 0
        ]
        #设置位姿一直朝下
        self.orientation = p.getQuaternionFromEuler([0, 0, 0])

        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        #p.connect(p.GUI)
        self.success_time = 0
        self.temp = 0
        self.her_flag = 0
        self.step_counter = 0

        p.resetSimulation()
        #p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        self.terminated = False
        #p.setGravity(0, 0, -10)

        #这些是周围那些白线，用来观察是否超过了obs的边界
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_low_obs, 0],
            lineToXYZ=[self.x_low_obs, self.y_low_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_high_obs, 0],
            lineToXYZ=[self.x_low_obs, self.y_high_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs, self.y_low_obs, 0],
            lineToXYZ=[self.x_high_obs, self.y_low_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs, self.y_high_obs, 0],
            lineToXYZ=[self.x_high_obs, self.y_high_obs, self.z_high_obs])

        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_low_obs, self.z_high_obs],
            lineToXYZ=[self.x_high_obs, self.y_low_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_high_obs, self.z_high_obs],
            lineToXYZ=[self.x_high_obs, self.y_high_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_low_obs, self.y_low_obs, self.z_high_obs],
            lineToXYZ=[self.x_low_obs, self.y_high_obs, self.z_high_obs])
        p.addUserDebugLine(
            lineFromXYZ=[self.x_high_obs, self.y_low_obs, self.z_high_obs],
            lineToXYZ=[self.x_high_obs, self.y_high_obs, self.z_high_obs])

        p.loadURDF(os.path.join(self.urdf_root_path, "plane.urdf"),
                   basePosition=[0, 0, -0.65])
        self.kuka_id = p.loadURDF(os.path.join(self.urdf_root_path,
                                               "Zu 5 URDF/urdf/Zu5.urdf"),
                                  useFixedBase=True)
        p.loadURDF(os.path.join(self.urdf_root_path, "table/table.urdf"),
                   basePosition=[0.5, 0, -0.65])
        # p.loadURDF(os.path.join(self.urdf_root_path, "tray/traybox.urdf"),basePosition=[0.55,0,0])
        #object_id=p.loadURDF(os.path.join(self.urdf_root_path, "random_urdfs/000/000.urdf"), basePosition=[0.53,0,0.02])
        #self.object_id = p.loadURDF(os.path.join(self.urdf_root_path,"random_urdfs/000/000.urdf"),basePosition=[0.8,0.1,0.4],useFixedBase=True)
        visual_shape_id1 = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName="/home/yuchengxiao/anaconda3/envs/spinningup/lib/python3.6/site-packages/pybullet_data/pear.obj",
            rgbaColor=[0, 1, 1, 1],
            specularColor=[0.4, 0.4, 0],
            visualFramePosition=[0, 0, 0],
            meshScale=[1, 1, 1])
        collision_shape_id1 = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName="/home/yuchengxiao/anaconda3/envs/spinningup/lib/python3.6/site-packages/pybullet_data/pear.obj",
            collisionFramePosition=[0, 0, 0],
            meshScale=[1, 1, 1])
        bp = [
            random.uniform(0.6,
                           0.8),
            random.uniform(-0.2,
                           0.2),
            random.uniform(0.3,
                           0.5),
        ]
        self.object_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape_id1,
            baseVisualShapeIndex=visual_shape_id1,
            basePosition=bp,
            useMaximalCoordinates=True)
        position = produce_position(bp)
        #print(position)

        set_obstacle(position)

        self.num_joints = p.getNumJoints(self.kuka_id)
        #print("关节：")

        #print(self.num_joints)
        #self.num_joints = 6
        for i in range(self.num_joints):
            p.resetJointState(
                bodyUniqueId=self.kuka_id,

                jointIndex=i,
                targetValue=self.init_joint_positions[i],
            )

        self.robot_pos_obs = p.getLinkState(self.kuka_id,
                                            self.num_joints - 1)[4]
        #print("初始位置")
        #print(self.robot_pos_obs)

        #logging.debug("init_pos={}\n".format(p.getLinkState(self.kuka_id,self.num_joints-1)))
        p.stepSimulation()
        self.object_pos = p.getBasePositionAndOrientation(self.object_id)[0]
        self.dd = []
        self.ff = []
        self.ff.append(self.object_pos)
        self.current_pos = p.getLinkState(self.kuka_id, self.num_joints - 1)[4]
        self.dd.append(self.current_pos[0])
        self.dd.append(self.current_pos[1])
        self.dd.append(self.current_pos[2])
        for i in range(len(position)):
            a = list(position[i])
            self.ff.append(a)
            self.dd.append(a[0]*0.5)
            self.dd.append(a[1]*0.5)
            self.dd.append(a[2]*0.5)

        #print("ff>>>>>",self.ff)
        #print("dd>>>>>",self.dd)


        rebuild = {'observation': np.array(self.current_pos), 'achieved_goal': np.array(self.current_pos),
                   'desired_goal': np.array(self.object_pos)}
        return rebuild
        #return np.array(self.dd).astype(np.float32)
        #return np.array(self.robot_pos_obs).astype(np.float32)

    def step(self, action):
        dv = 0.005
        #dv = 0.01
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv
        self.current_pos = p.getLinkState(self.kuka_id, self.num_joints - 1)[4]

        # logging.debug("self.current_pos={}\n".format(self.current_pos))
        self.new_robot_pos = [
            self.current_pos[0] + dx, self.current_pos[1] + dy,
            self.current_pos[2] + dz
        ]
        #print("现在所处的位置")
        #print(self.current_pos)
        #print("新的目标点")
        #print(self.new_robot_pos)
        #logging.debug("self.new_robot_pos={}\n".format(self.new_robot_pos))
        #开始求逆解
        self.robot_joint_positions = p.calculateInverseKinematics(
            bodyUniqueId=self.kuka_id,
            endEffectorLinkIndex=self.num_joints - 1,
            targetPosition=[
                self.new_robot_pos[0], self.new_robot_pos[1],
                self.new_robot_pos[2]
            ],
            targetOrientation=self.orientation,
            jointDamping=self.joint_damping,
            lowerLimits=self.lower_limits,
            upperLimits=self.upper_limits,
            jointRanges=self.joint_ranges
        )
        #print("关节位置")
        #print(self.robot_joint_positions)
        angle = []
        for i in range(6):
            angle.append(self.robot_joint_positions[i]/math.pi*180)
        #print("逆解求出的关节角度")
        #print(angle)
        #time.sleep(15)
        for i in range(self.num_joints):

            p.resetJointState(
                bodyUniqueId=self.kuka_id,
                jointIndex=i,
                targetValue=self.robot_joint_positions[i],
            )
        p.stepSimulation()

        #在代码开始部分，如果定义了is_good_view，那么机械臂的动作会变慢，方便观察
        if self.is_good_view:
            time.sleep(0.1)

        self.step_counter += 1
        return self._reward()

    def _reward(self):
        flag = 0
        #success_time = 0

        #一定注意是取第4个值，请参考pybullet手册的这个函数返回值的说明
        self.robot_state = p.getLinkState(self.kuka_id, self.num_joints - 1)[4]
        # self.object_state=p.getBasePositionAndOrientation(self.object_id)
        # self.object_state=np.array(self.object_state).astype(np.float32)
        #
        self.object_state = np.array(
            p.getBasePositionAndOrientation(self.object_id)[0]).astype(
                np.float32)
        #print("目标位置：>>>>>>>>>>>>>")
        #print(self.object_state)
        square_dx = (self.robot_state[0] - self.object_state[0])**2
        square_dy = (self.robot_state[1] - self.object_state[1])**2
        square_dz = (self.robot_state[2] - self.object_state[2])**2

        #用机械臂末端和物体的距离作为奖励函数的依据
        self.distance = sqrt(square_dx + square_dy + square_dz)
        #print(self.distance)
        ab = np.array(self.robot_state)
        tdd = []
        for i in range(len(self.ff)):
            #print("显示距离：")
            #print(self.ff[i])
            #print(ab)
            tdd.append(np.sqrt(np.sum((self.ff[i]-ab) ** 2)))
        #print("欧拉距离：")
        #print(tdd)
        tdd.pop(0)
        obstacle_distance = min(tdd)
        #print("与障碍物的最短距离")
        #print(obstacle_distance)


        x = self.robot_state[0]
        y = self.robot_state[1]
        z = self.robot_state[2]

        # 如果机械比末端超过了obs的空间，也视为done，而且会给予一定的惩罚
        terminated = bool(x < self.x_low_obs or x > self.x_high_obs
                          or y < self.y_low_obs or y > self.y_high_obs
                          or z < self.z_low_obs or z > self.z_high_obs)
        reward = -self.distance
        #reward = 0
        #if self.temp >0:
        #    reward = reward+(self.temp-self.distance)*100
        #if obstacle_distance < 0.05:
        #    reward = reward+((obstacle_distance-0.05)*100)
        if terminated:
            reward = -5
            self.temp = 0
            self.terminated = True
            #print("out")

        # 如果机械臂一直无所事事，在最大步数还不能接触到物体，也需要给一定的惩罚
        elif self.step_counter > self.max_steps_one_episode:
            reward = -5
            self.temp = 0
            self.terminated = True
            print("time up")
        elif self.distance < 0.05:
            reward = reward+(0.05-self.distance)
            flag = 1
            #print('good job,get it')
            #print(self.step_counter)
            self.temp = 0
            self.terminated = False
            self.success_time = self.success_time+1
            if self.success_time>5:
                self.terminated = True
        #if obstacle_distance<0.05:
        #    reward = reward+(obstacle_distance-0.05)*10
        if obstacle_distance<0.01:
            #time.sleep(2)
            #print(self.robot_state)
            #a, reward, b, c = self.step([0,0,-30])
            reward = -100
            self.temp = 0
            print("碰到障碍物，change position")
            #print(self.robot_state)
            #time.sleep(2)
            self.terminated = False
        if self.distance<0.015:
            reward=5
            #flag = 8
            print('good job,get it')
            self.terminated = True
        #elif obstacle_distance<0.05:
        #    reward = reward+((obstacle_distance-0.05)*100)

        #info = {'distance:', self.distance}
        info = dict(is_success=flag)
        self.temp = self.distance
        #self.observation=self.robot_state
        self.observation = self.dd
        #self.robot_state = p.getLinkState(self.kuka_id, self.num_joints - 1)[4]
        rebuild = {'observation': np.array(self.robot_state), 'achieved_goal': np.array(self.robot_state),
                   'desired_goal': np.array(self.object_state)}
        #print("reward",reward)
        return rebuild, reward, self.terminated, info
        #return np.array(self.observation).astype(np.float32), reward, self.terminated, info

    def close(self):
        p.disconnect()

    def compute_reward(self,achieved_goal, desired_goal,info) :
        square_dx = (achieved_goal[0] - desired_goal[0]) ** 2
        square_dy = (achieved_goal[1] - desired_goal[1]) ** 2
        square_dz = (achieved_goal[2] - desired_goal[2]) ** 2

        # 用机械臂末端和物体的距离作为奖励函数的依据
        #print('???>>>>',achieved_goal,desired_goal)
        d = sqrt(square_dx + square_dy + square_dz)
        #print("self.temp>>>",self.her_flag)
        #her_reward = 0
        #if self.her_flag>0:
        #    her_reward = her_reward+(self.her_flag-d)*100
        #self.her_flag = d
        #print(her_reward)

        return -d

if __name__ == '__main__':
    # 这一部分是做baseline，即让机械臂随机选择动作，看看能够得到的分数
    env = KukaReachEnv(is_render=True,is_good_view=True)
    print(env)
    print(env.observation_space.shape)
    # print(env.observation_space.sample())
    # print(env.action_space.sample())
    print(env.action_space.shape)
    obs = env.reset()
    print(obs)
    sum_reward=0
    for i in range(10):
        env.reset()

        for i in range(2000):
            action=env.action_space.sample()
            #print("执行工作：")
            #print(action)
            #action=np.array([0,0,0.47-i/1000])
            obs,reward,done,info=env.step(action)
            #print(obs)
          #  print("i={},\naction={},\nobs={},\ndone={},\n".format(i,action,obs,done,))
            #print(colored("reward={},info={}".format(reward,info),"cyan"))
           # print(colored("info={}".format(info),"cyan"))
            sum_reward+=reward
            if done:
                break
           # time.sleep(0.1)
    print()
    print(sum_reward)