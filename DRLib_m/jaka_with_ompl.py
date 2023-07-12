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
import torch
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
from algos.pytorch.td3_sp.td3_per_her import TD3Torch
from torch_arguments import get_args
from termcolor import colored
from statistics import mean
import utils
from itertools import product

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
def set_obstacle(position,rgb_color):
    scale = [1, 1, 1]
    visual_shape_id2 = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        rgbaColor=rgb_color,
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

def set_big_obstacle(position,rgb_color):
    scale = [0.01, 0.01, 0.01]
    pos_list = []
    visual_shape_id2 = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        rgbaColor=rgb_color,
        halfExtents =scale,
        specularColor=[0.4, 0.4, 0],
        visualFramePosition=[0, 0, 0])
    collision_shape_id2 = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=scale,
        collisionFramePosition=[0, 0, 0])
    for i in range(len(position)):
        temp = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape_id2,
            baseCollisionShapeIndex = collision_shape_id2,
            basePosition=position[i],
            useMaximalCoordinates=True)
        pos_list.append(temp)
    return pos_list

def produce_position(base_position):
    position = []
    for i in range(30):
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
def random_pos():
    bp = [
        random.uniform(0.6,
                       0.8),
        random.uniform(-0.2,
                       0.2),
        random.uniform(0.1,
                       0.4),
    ]
    return bp
def random_goal(r):
    a = random.uniform(-r,r)
    b = random.uniform(-r,r)
    c = random.uniform(-r,0)
    while a*a+b*b+c*c>r*r:
        a = random.uniform(-r, r)
        b = random.uniform(-r, r)
        c = random.uniform(-r, 0)
    return a,b,c
def cal_mean(position):
    position_x = 0
    position_y = 0
    position_z = 0
    for i in range(len(position)):
        position_x = position[i][0]+position_x
        position_y = position[i][1] + position_y
        position_z = position[i][2] + position_z
    return [position_x/len(position),position_y/len(position),position_z/len(position)]

def cal_distance_else(start,end,midle):
    start = np.array(start)
    end = np.array(end)
    midle = np.array(midle)
    x = start -end
    y = start - midle
    z = np.cross(x,y)
    if np.linalg.norm(y)>0.1:
        return 1
    else:
        return np.linalg.norm(z)/np.linalg.norm(x)
def cal_distance(start,end,midle):
    start = np.array(start)
    end = np.array(end)
    midle = np.array(midle)
    x = start -end
    y = start - midle
    z = np.cross(x,y)
    return np.linalg.norm(z)/np.linalg.norm(x)

class KukaReachEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
    max_steps_one_episode = 20

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

        self.x_low_action = -0.5
        self.x_high_action = 0.5
        self.y_low_action = -0.5
        self.y_high_action = 0.5
        self.z_low_action = -0.5
        self.z_high_action = 0.5

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
        self.observation_space = spaces.Box(low=self.y_low_action, high=self.x_high_action, shape=(1, 33))
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
            0.57, 0, 0, -1.57, -1.57, 0
        ]
        #设置位姿一直朝下
        self.orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.change1 = 1
        self.change2 = 1
        self.bp = [
            random.uniform(0.6,
                           0.8),
            random.uniform(-0.2,
                           0.2),
            random.uniform(0.3,
                           0.5),
        ]
        self.a = []
        for i in range(10):
            self.a.append(random_pos())
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        #p.connect(p.GUI)
        self.init_distance=1
        self.yeah = 0
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
        if self.change1==1:
            self.bp = [
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
            basePosition=self.bp,
            useMaximalCoordinates=True)
        if self.change2==1:
            self.a = []
            for i in range(10):
                self.a.append(random_pos())
        self.pos_list =set_big_obstacle(self.a,[4,1,1,1])
        #self.pos_list.append(self.object_id)
        position = self.a
        #print("ce ce ni de ma>>>>",self.goal)

        self.num_joints = p.getNumJoints(self.kuka_id)
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
        #self.ff.append(self.object_pos)
        self.current_pos = p.getLinkState(self.kuka_id, self.num_joints - 1)[4]
        self.current_pos_joint =[]
        for i in range(6):
            self.current_pos_joint.append(p.getJointState(self.kuka_id,i)[0])
        #print('yoyo',self.object_pos_joint,self.current_pos_joint)
        self.dd.append(self.current_pos[0])
        self.dd.append(self.current_pos[1])
        self.dd.append(self.current_pos[2])
        for i in range(len(position)):
            a = list(position[i])
            self.ff.append(a)
            self.dd.append(a[0])
            self.dd.append(a[1])
            self.dd.append(a[2])

        #print("ff>>>>>",self.ff)
        #print("dd>>>>>",self.dd)
        #self.object_best_pos = self.get_goal(self.dd, self.object_pos)
        #print("all i want ",self.object_best_pos)
        #set_obstacle([self.object_best_pos],[2,2,2,1])
        #self.object_pos_joint = self.return_joint_pos(self.object_best_pos)
        self.object_pos_joint = self.return_joint_pos(self.object_pos)


        rebuild = {'observation': np.array(self.dd), 'achieved_goal': np.array(self.current_pos),
                   'desired_goal': np.array(self.object_pos)}
        return rebuild
        #return np.array(self.dd).astype(np.float32)
        #return np.array(self.robot_pos_obs).astype(np.float32)

    def step(self, action):
        dv = 0.2
        #time.sleep(1)
        #dv = 0.01
        #print(action)
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
        #if self.is_good_view:
            #time.sleep(1)

        self.step_counter += 1
        return self._reward()
    def easy_step(self, joint_position):
        self.current_pos = p.getLinkState(self.kuka_id, self.num_joints - 1)[4]
        #time.sleep(1)
        self.current_pos_joint = []
        for i in range(6):
            self.current_pos_joint.append(p.getJointState(self.kuka_id,i)[0])
        angle = []
        for i in range(6):
            angle.append(joint_position[i]/math.pi*180)
        for i in range(self.num_joints):
            p.resetJointState(
                bodyUniqueId=self.kuka_id,
                jointIndex=i,
                targetValue=joint_position[i],
            )
        p.stepSimulation()

        #在代码开始部分，如果定义了is_good_view，那么机械臂的动作会变慢，方便观察
        #if self.is_good_view:
            #time.sleep(1)

        self.step_counter += 1
        return self._reward()
    def _reward(self):

        flag = 0
        #success_time = 0

        #一定注意是取第4个值，请参考pybullet手册的这个函数返回值的说明
        self.robot_state = p.getLinkState(self.kuka_id, self.num_joints - 1)[4]
        self.robot_state_joint = []
        for i in range(6):
            self.robot_state_joint.append(p.getJointState(self.kuka_id,i)[0])
        self.object_state = self.object_pos
        square_dx = (self.robot_state[0] - self.object_state[0])**2
        square_dy = (self.robot_state[1] - self.object_state[1])**2
        square_dz = (self.robot_state[2] - self.object_state[2])**2

        #用机械臂末端和物体的距离作为奖励函数的依据
        self.distance = sqrt(square_dx + square_dy + square_dz)
        moving_links = frozenset(
            [0, 1, 2, 3, 4, 5])
        moving_bodies = [(self.kuka_id, moving_links)]
        # print(moving_links)
        self.check_body_pairs = list(product(moving_bodies, self.pos_list))
        obstacle_distance = []
        for body1, body2 in self.check_body_pairs:
            if utils.pairwise_collision(body1, body2):
                obstacle_distance.append(1)
                # print("gg")
                #time.sleep(10)
                #print('body collision', body1, body2)
                #time.sleep(10)
                # print(get_body_name(body1), get_body_name(body2))
                # return False
            else:

                obstacle_distance.append(0)
        x = self.robot_state[0]
        y = self.robot_state[1]
        z = self.robot_state[2]
        terminated = False
        if self.step_counter==1:
            self.init_distance = self.distance
        reward = -self.distance
        #if z-self.object_state[2]>0.3:
        #    reward = -5
        #    self.terminated = True
        if z<0:
            reward = -5
            self.terminated = True
        if sum(obstacle_distance)>0:
            reward = -5
            flag = 0
            self.terminated = True
        if self.step_counter > self.max_steps_one_episode:
            reward = -5
            self.temp = 0
            self.terminated = True
        if (self.distance<0.1) & (reward==-self.distance):
            print("good job,get it")
            self.terminated = True
            flag = 1
            reward = 10

            # print("time up")

        #elif obstacle_distance<0.05:
        #    reward = reward+((obstacle_distance-0.05)*100)

        #info = {'distance:', self.distance}
        info = dict(is_success=flag)
        #self.temp = self.current_pos_joint
        #self.observation=self.robot_state
        self.dd[0]=self.robot_state[0]
        self.dd[1]=self.robot_state[1]
        self.dd[2]=self.robot_state[2]
        self.observation = self.dd
        #self.robot_state = p.getLinkState(self.kuka_id, self.num_joints - 1)[4]
        #print(">>>>>>>>>>",self.object_state)
        rebuild = {'observation': np.array(self.observation), 'achieved_goal': np.array(self.robot_state),
                   'desired_goal': np.array(self.object_pos)}
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
    def return_joint_pos(self,position):
        alpha = p.calculateInverseKinematics(
            bodyUniqueId=self.kuka_id,
            endEffectorLinkIndex=self.num_joints - 1,
            targetPosition=position,
            targetOrientation=self.orientation,
            jointDamping=self.joint_damping,
            lowerLimits=self.lower_limits,
            upperLimits=self.upper_limits,
            jointRanges=self.joint_ranges)
        return alpha
    def get_goal(self,observation,object):
        net = TD3Torch
        args = get_args()
        act_dim =3
        s_dim = 36
        a_bound = 0.5
        device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() and args.gpu_id != -1 else 'cpu')
        net = net(act_dim=act_dim,
                  obs_dim=s_dim,
                  a_bound=a_bound,
                  per_flag=args.per,
                  her_flag=args.her,
                  action_l2=args.action_l2,
                  state_norm=args.state_norm,
                  gamma=args.gamma,
                  sess_opt=args.sess_opt,
                  seed=args.seed,
                  clip_return=args.clip_return,
                  device=device,
                  )
        net.ac.load_state_dict(torch.load('goal_point/actor.pth'))
        net.load_norm("goal_point/norm.pkl")
        obs = list(observation)+list(object)
        a = net.get_action(obs)
        best_goal = []
        best_goal.append(self.object_pos[0]+a[0]*0.2)
        best_goal.append(self.object_pos[1]+a[1]*0.2)
        best_goal.append(self.object_pos[2]+a[2]*0.2)
        return best_goal



if __name__ == '__main__':
    # 这一部分是做baseline，即让机械臂随机选择动作，看看能够得到的分数
    env = KukaReachEnv(is_render=True,is_good_view=True)
    print(env)
    print(env.observation_space.shape)
    # print(env.observation_space.sample())
    # print(env.action_space.sample())
    print(env.action_space.shape)
    obs = env.reset()
    #print(obs)
    sum_reward=0
    for i in range(10):
        env.reset()

        for i in range(2000):
            action=env.action_space.sample()
            print("执行工作：")
            #print(action)

            time.sleep(1)
            #action=np.array([0,0,0.47-i/1000])
            obs,reward,done,info=env.step(action)
            print(info)
            #print(obs)
          #  print("i={},\naction={},\nobs={},\ndone={},\n".format(i,action,obs,done,))
            #print(colored("reward={},info={}".format(reward,info),"cyan"))
           # print(colored("info={}".format(info),"cyan"))
            sum_reward+=reward
            print(i)
            if done:
                #print("????")
                break
           # time.sleep(0.1)
    print()
    print(sum_reward)