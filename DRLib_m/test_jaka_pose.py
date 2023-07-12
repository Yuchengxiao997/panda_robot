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
import utils
from itertools import product
from termcolor import colored
from statistics import mean

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
    scale = [0.01, 0.01, 0.03]
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
            baseCollisionShapeIndex=collision_shape_id2,
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
def vector_to_euler(x,y,z):
    euler_x = math.atan2(z,y)
    euler_y = math.atan2(x,z)
    euler_z = math.atan2(x,y)
    euler_x = euler_x / math.pi
    euler_y = euler_y / math.pi
    euler_z = euler_z / math.pi
    return euler_x,euler_y,euler_z

class KukaReachEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
    max_steps_one_episode = 100

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

        self.x_low_action = -1
        self.x_high_action = 1
        self.y_low_action = -1
        self.y_high_action = 1
        self.z_low_action = -1
        self.z_high_action = 1

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
        self.observation_space = spaces.Box(low=self.y_low_obs, high=self.x_high_obs, shape=(1, 33))
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
            0.57, 0, 0, -1.57, 1.57, 0
        ]
        #设置位姿一直朝下
        self.orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.change1 = 1
        self.change2 = 0
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
        self.juila = 0
        self.fflag = 0
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
                           0.4),
            ]
        self.object_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_shape_id1,
            baseVisualShapeIndex=visual_shape_id1,
            basePosition=self.bp,
            useMaximalCoordinates=True)
        #position = produce_position(bp)
        #print(position)

        #set_obstacle(position,[4,1,1,1])
        #self.goal = cal_mean(position)
        #rint("???")
        #print(self.goal)
        #a = []
        #a.append(self.goal)
        #set_obstacle(a,[0,1,1,1])
        if self.change2==1:
            for i in range(10):
                self.a.append(random_pos())
        self.pos_list =set_big_obstacle(self.a,[4,1,1,1])
        self.pos_list.append(self.object_id)
        position = self.a
        #print("ce ce ni de ma>>>>",self.goal)

        self.num_joints = p.getNumJoints(self.kuka_id)
        #print("关节：")
        self.object_pos = p.getBasePositionAndOrientation(self.object_id)[0]
        #print(self.num_joints)
        #self.num_joints = 6
        while True:
            self.bp = [
                random.uniform(0.2,
                               0.8),
                random.uniform(-0.5,
                               0.5),
                random.uniform(0.1,
                               0.6),
            ]
            #self.init_joint_positions = self.return_joint_pos(self.bp)
            for i in range(self.num_joints):
                p.resetJointState(
                    bodyUniqueId=self.kuka_id,

                    jointIndex=i,
                    targetValue=self.init_joint_positions[i],
                )
            self.current_pos = p.getLinkState(self.kuka_id, self.num_joints - 1)[4]
            square_dx = (self.object_pos[0] - self.current_pos[0]) ** 2
            square_dy = (self.object_pos[1] - self.current_pos[1]) ** 2
            square_dz = (self.object_pos[2] - self.current_pos[2]) ** 2
            self.start_to_end = sqrt(square_dx + square_dy + square_dz)
            if (self.find_collision(self.pos_list)==0) & (self.start_to_end>0.2):
                #print('final',self.start_to_end)
                break
        self.temp_distance = self.start_to_end
        self.robot_pos_obs = p.getLinkState(self.kuka_id,
                                            self.num_joints - 1)[4]
        #print("初始位置")
        #print(self.robot_pos_obs)

        #logging.debug("init_pos={}\n".format(p.getLinkState(self.kuka_id,self.num_joints-1)))
        p.stepSimulation()
        self.object_pos = p.getBasePositionAndOrientation(self.object_id)[0]
        self.object_pos_joint = self.return_joint_pos(self.object_pos)
        self.dd = []
        self.ff = []
        #self.ff.append(self.object_pos)
        #self.current_pos = p.getLinkState(self.kuka_id, self.num_joints - 1)[4]
        self.current_pos_joint = self.return_joint_pos(self.current_pos)
        #print("the begin joint",self.current_pos_joint)
        #print("the final joint",self.object_pos_joint)
        self.dd.append(self.current_pos[0])
        self.dd.append(self.current_pos[1])
        self.dd.append(self.current_pos[2])
        for i in range(len(position)):
            a = list(position[i])
            self.ff.append(a)
            self.dd.append(a[0])
            self.dd.append(a[1])
            self.dd.append(a[2])



        #print("faffff",self.current_pos)
        rebuild = {'observation': np.array(self.dd), 'achieved_goal': np.array(self.current_pos),
                   'desired_goal': np.array(self.object_pos)}
        return rebuild
        #return np.array(self.dd).astype(np.float32)
        #return np.array(self.robot_pos_obs).astype(np.float32)

    def step(self, action):
        self.object_state = p.getBasePositionAndOrientation(self.object_id)[0]
        if self.step_counter == 1:
            self.init_joint_positions = [
                0.57, 0, 0, -1.57, -1.57, 0
            ]
            if self.fflag==0:
                for i in range(self.num_joints):
                    p.resetJointState(
                        bodyUniqueId=self.kuka_id,

                        jointIndex=i,
                        targetValue=self.init_joint_positions[i],
                    )
                self.fflag = self.fflag+1
            self.new_robot_pos = [
                self.object_state[0], self.object_state[1],
                self.object_state[2]-0.05
            ]
            self.orientation = p.getQuaternionFromEuler([0, 0, 0])
            #print("从下方往上方抓取")
        if self.step_counter == 2:
            self.new_robot_pos = [
                self.object_state[0] - 0.05, self.object_state[1],
                self.object_state[2]
            ]
            self.orientation = p.getQuaternionFromEuler([0, math.pi / 2, 0])
            #print("从前往后方抓取")
        if self.step_counter == 4:
            self.new_robot_pos = [
                self.object_state[0], self.object_state[1] + 0.05,
                self.object_state[2]
            ]
            self.orientation = p.getQuaternionFromEuler([math.pi / 2, 0, 0])
            #print("从左往右抓取")
        if self.step_counter == 3:
            self.new_robot_pos = [
                self.object_state[0], self.object_state[1] - 0.05,
                self.object_state[2]
            ]
            self.orientation = p.getQuaternionFromEuler([-math.pi / 2, 0, 0])
            #print("从右往左抓取")
        if self.step_counter == 0:
            self.init_joint_positions = [
                0.57, 0, 0, -1.57, 1.57, 0
            ]
            self.new_robot_pos = [
                self.object_state[0], self.object_state[1],
                self.object_state[2] + 0.1
            ]
            self.orientation = p.getQuaternionFromEuler([0,math.pi,0])
            #print("从上往下抓取")

        self.robot_joint_positions = p.calculateInverseKinematics(
            bodyUniqueId=self.kuka_id,
            endEffectorLinkIndex=self.num_joints - 1,
            targetPosition=[
                self.new_robot_pos[0], self.new_robot_pos[1],
                self.new_robot_pos[2]
            ],
            targetOrientation=self.orientation,
        )
        #print(self.robot_joint_positions)
        angle = []
        for i in range(6):
            angle.append(self.robot_joint_positions[i]/math.pi*180)
        for i in range(self.num_joints):

            p.resetJointState(
                bodyUniqueId=self.kuka_id,
                jointIndex=i,
                targetValue=self.robot_joint_positions[i],
            )
        p.stepSimulation()
        self.juila = self.juila+1
        return self._reward()
    def _reward(self):
        flag = 0
        #mask = 0
        #一定注意是取第4个值，请参考pybullet手册的这个函数返回值的说明
        self.robot_state = p.getLinkState(self.kuka_id, self.num_joints - 1)[4]
        #print("the goal position",self.new_robot_pos)
        #print("the end position",self.robot_state)
        self.object_state = self.object_pos
        self.robot_state = p.getLinkState(self.kuka_id, self.num_joints - 1)[4]
        self.object_state = self.object_pos
        square_dx = (self.robot_state[0] - self.new_robot_pos[0]) ** 2
        square_dy = (self.robot_state[1] - self.new_robot_pos[1]) ** 2
        square_dz = (self.robot_state[2] - self.new_robot_pos[2]) ** 2
        # 用机械臂末端和物体的距离作为奖励函数的依据
        self.distance = sqrt(square_dx + square_dy + square_dz)
        self.terminated = False
        reward = 0
        #avoid obstacle
        obstacle_distance=self.find_collision(self.pos_list)
        #if (obstacle_distance==0) &(self.start_to_end-self.distance2>0):

        if self.distance<0.01:
            flag = 1
            #print('best match')
            self.juila = 0
            #self.terminated = True
            print(self.step_counter,'used')
            if obstacle_distance==0:
                print('get it',self.step_counter)
                time.sleep(5)
            self.step_counter = self.step_counter + 1
        if self.step_counter>5:
            self.terminated = True
        if self.juila>50:
            self.step_counter = self.step_counter+1
            self.juila = 0
        info = dict(is_success=flag)

        self.dd[0] = self.robot_state[0]
        self.dd[1] = self.robot_state[1]
        self.dd[2] = self.robot_state[2]
        self.observation = self.dd
        rebuild = {'observation': np.array(self.observation), 'achieved_goal': np.array(self.robot_state),
                   'desired_goal': np.array(self.object_state)}
        #print("reward",reward)
        return rebuild, reward, self.terminated, info

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
    def find_collision(self,pos_list):
        moving_links = frozenset(
            [0, 1, 2, 3, 4, 5])
        moving_bodies = [(self.kuka_id, moving_links)]
        # print(moving_links)
        obstacle_distance = []
        self.check_body_pairs = list(product(moving_bodies, pos_list))
        for body1, body2 in self.check_body_pairs:
            if utils.pairwise_collision(body1, body2):
                obstacle_distance.append(1)
                # time.sleep(10)
                # print("gg")
                # print('body collision', body1, body2)
                # print(get_body_name(body1), get_body_name(body2))
                # return False
            else:
                obstacle_distance.append(0)
        return sum(obstacle_distance)



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
    for i in range(100):
        env.reset()
        time.sleep(5)

        for i in range(1000):
            action=env.action_space.sample()
            #print("执行工作：")
            #print(action)
            #action=np.array([0,0,0.47-i/1000])
            obs,reward,done,info=env.step(action)
            time.sleep(0.5)
            #print(obs)
          #  print("i={},\naction={},\nobs={},\ndone={},\n".format(i,action,obs,done,))
            #print(colored("reward={},info={}".format(reward,info),"cyan"))
           # print(colored("info={}".format(info),"cyan"))
            sum_reward+=reward
            #print(i)
            if done:
                #time.sleep(10)
                #print("????")
                break
           # time.sleep(0.1)
    print()
    print(sum_reward)