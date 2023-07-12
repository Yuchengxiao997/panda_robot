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
from algos.pytorch.td3_sp.td3_per_her import TD3Torch
from torch_arguments import get_args
import torch
#### 一些变量 ######
LOGGING_LEVEL = logging.INFO
def set_obstacle(position,rgb_color):
    pos_list = []
    a_list = []
    for i in range(len(position)):
        a = random.uniform(0.03, 0.1)
        a_list.append(a)
        visual_shape_id2 = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            rgbaColor=rgb_color,
            radius=a,
            specularColor=[0.4, 0.4, 0])
        collision_shape_id2 = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=a,)
        temp = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape_id2,
            baseCollisionShapeIndex=collision_shape_id2,
            basePosition=position[i])
        pos_list.append(temp)
    return pos_list,a_list
def set_big_obstacle(position,rgb_color):
    scale = [0.3, 0.01, 0.2]
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
        random.uniform(0.5,
                       0.7),
        random.uniform(-0.2,
                       0.2),
        random.uniform(0.1,
                       0.7),
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

        self.x_low_obs = 0.4
        self.x_high_obs = 0.8
        self.y_low_obs = -0.2
        self.y_high_obs = 0.2
        self.z_low_obs = 0
        self.z_high_obs = 0.55

        self.x_low_action = -3.14
        self.x_high_action = 3.14


        p.resetDebugVisualizerCamera(cameraDistance=1.5,
                                     cameraYaw=0,
                                     cameraPitch=-40,
                                     cameraTargetPosition=[0.55, -0.35, 0.2])

        self.action_space = spaces.Box(low=np.array(
            [self.x_low_action, self.x_low_action, self.x_low_action, self.x_low_action, self.x_low_action,
             self.x_low_action]),
            high=np.array([
                self.x_high_action,
                self.x_high_action,
                self.x_high_action,
                self.x_high_action,
                self.x_high_action,
                self.x_high_action
            ]),
            dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([self.x_low_obs, self.y_low_obs, self.z_low_obs]),
            high=np.array([self.x_high_obs, self.y_high_obs, self.z_high_obs]),
            dtype=np.float32)
        self.observation_space = spaces.Box(low=self.y_low_obs, high=self.x_high_obs, shape=(1, 40))
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
            0.77, 0.2, 0, -1.57, 1.57, 0
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
        self.init_distance=1
        self.yeah = 0
        self.success_time = 0
        self.temp = 0
        self.her_flag = 0
        self.step_counter = 0
        self.juila = 0
        self.fflag = 0
        self.step_counter1 = 0
        self.bad_flag = 0

        p.resetSimulation()
        #p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        self.terminated = False
        #p.setGravity(0, 0, -10)
        '''
        args = get_args()
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

        device = torch.device("cuda:" + str(args.gpu_id) if torch.cuda.is_available() and args.gpu_id != -1 else 'cpu')
        self.net = TD3Torch(act_dim=3,
                  obs_dim=6,
                  a_bound=0.8,
                  per_flag=args.per,
                  her_flag=False,
                  action_l2=args.action_l2,
                  state_norm=args.state_norm,
                  gamma=args.gamma,
                  sess_opt=args.sess_opt,
                  seed=args.seed,
                  clip_return=args.clip_return,
                  device=device,
                  )
        self.net.ac.load_state_dict(torch.load('low_policy/actor1.pth'))
        self.net.load_norm("low_policy/norm1.pkl")
        '''
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
            fileName="C:/Users/Mss/.conda/envs/baseline/Lib/site-packages/pybullet_data/pear.obj",
            rgbaColor=[0, 1, 1, 1],
            specularColor=[0.4, 0.4, 0],
            visualFramePosition=[0, 0, 0],
            meshScale=[1, 1, 1])
        collision_shape_id1 = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName="C:/Users/Mss/.conda/envs/baseline/Lib/site-packages/pybullet_data/pear.obj",
            collisionFramePosition=[0, 0, 0],
            meshScale=[1, 1, 1])

        self.bp = [
            0.68,
            0.25,
            0.3
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
        self.a = []
        for i in range(1):
            self.a.append(random_pos())
        #print("pos->>>>>",random_pos())
        self.a = [[0.7,0,0.15]]
        self.pos_list =set_big_obstacle(self.a,[4,1,1,1])



        self.pos_list.append(self.object_id)
        position = self.a
        self.num_joints = p.getNumJoints(self.kuka_id)
        self.object_pos = p.getBasePositionAndOrientation(self.object_id)[0]
        '''
        self.new_robot_pos = self.object_pos
        for _ in range(10):
            self.robot_joint_positions = p.calculateInverseKinematics(
                bodyUniqueId=self.kuka_id,
                endEffectorLinkIndex=self.num_joints - 1,
                targetPosition=[
                    self.new_robot_pos[0], self.new_robot_pos[1],
                    self.new_robot_pos[2]
                ],
                targetOrientation=self.orientation,
            )
            for i in range(self.num_joints):
                p.resetJointState(
                    bodyUniqueId=self.kuka_id,
                    jointIndex=i,
                    targetValue=self.robot_joint_positions[i],
                )
        '''
        for i in range(self.num_joints):
            p.resetJointState(
                bodyUniqueId=self.kuka_id,

                jointIndex=i,
                targetValue=self.init_joint_positions[i],
            )
        p.stepSimulation()
        self.current_pos = p.getLinkState(self.kuka_id, self.num_joints - 1)[4]
        #print("current pos",self.current_pos,"object pos",self.object_pos)
        square_dx = (self.object_pos[0] - self.current_pos[0]) ** 2
        square_dy = (self.object_pos[1] - self.current_pos[1]) ** 2
        square_dz = (self.object_pos[2] - self.current_pos[2]) ** 2
        self.start_to_end = sqrt(square_dx + square_dy + square_dz)
        self.temp_distance = self.start_to_end

        self.dd = []
        for i in range(6):
            self.dd.append(p.getJointState(self.kuka_id, i)[0])
        for i in range(6):
            temp_t = p.getLinkState(self.kuka_id, i)[4]
            self.dd.append(temp_t[0])
            self.dd.append(temp_t[1])
            self.dd.append(temp_t[2])
        for i in range(len(position)):
            a = list(position[i])
            self.dd.append(a[0])
            self.dd.append(a[1])
            self.dd.append(a[2])
        self.dd.append(0.3)
        self.dd.append(0.01)
        self.dd.append(0.2)
        #print(self.dd)
        #print('begin>>>>',self.current_pos)
        #time.sleep(5)

        self.rebuild = {'observation': np.array(self.dd), 'achieved_goal': np.array(self.current_pos),
                   'desired_goal': np.array(self.object_pos)}
        return self.rebuild
    def tree_step(self, obs,action):
        dv = 0.5
        #dv = 0.01
        #print(action)
        self.current_pos1 = obs[21:24]
        self.object_pos = obs[30:33]
        d0 = action[0] * dv
        d1 = action[1] * dv
        d2 = action[2] * dv
        d3 = action[3] * dv
        d4 = action[4] * dv
        d5 = action[5] * dv
        robot_joint_state = []
        for i in range(6):
            robot_joint_state.append(obs[i])
        self.robot_joint_positions = [
            robot_joint_state[0] + d0,
            robot_joint_state[1] + d1,
            robot_joint_state[2] + d2,
            robot_joint_state[3] + d3,
            robot_joint_state[4] + d4,
            robot_joint_state[5] + d5
        ]
        for i in range(self.num_joints):
            p.resetJointState(
                bodyUniqueId=self.kuka_id,
                jointIndex=i,
                targetValue=self.robot_joint_positions[i],
            )
        p.stepSimulation()
        #if self.is_good_view:
        #    time.sleep(1)
        self.step_counter += 1
        return self._reward()
    def step(self, action):
        self.bad_flag = 0
        self.bad_flag_collision = 0
        dv = 0.5
        # dv = 0.01
        # print(action)
        d0 = action[0] * dv
        d1 = action[1] * dv
        d2 = action[2] * dv
        d3 = action[3] * dv
        d4 = action[4] * dv
        d5 = action[5] * dv
        self.current_pos1 = p.getLinkState(self.kuka_id, self.num_joints - 1)[4]
        robot_joint_state = []
        for i in range(6):
            robot_joint_state.append(p.getJointState(self.kuka_id, i)[0])
        self.robot_joint_positions = [
            robot_joint_state[0] + d0,
            robot_joint_state[1] + d1,
            robot_joint_state[2] + d2,
            robot_joint_state[3] + d3,
            robot_joint_state[4] + d4,
            robot_joint_state[5] + d5
        ]
        # print("what we want to go",self.new_robot_pos)
        # demon = list(self.new_robot_pos)
        # set_obstacle([demon],[4,1,1,1])
        '''
        obs = list(self.current_pos1)+list(self.new_robot_pos)
        a = self.net.get_action(obs)
        done = True
        while done:
            obs_next,done= self.just_step(a)
            a = self.net.get_action(obs_next)
            #print(a)
            if self.bad_flag>200:
                break
        '''
        # print(self.robot_joint_positions)
        angle = []
        for i in range(6):
            angle.append(self.robot_joint_positions[i] / math.pi * 180)
        for i in range(self.num_joints):
            p.resetJointState(
                bodyUniqueId=self.kuka_id,
                jointIndex=i,
                targetValue=self.robot_joint_positions[i],
            )
        p.stepSimulation()
        # if self.is_good_view:
        #    time.sleep(1)
        self.step_counter += 1
        return self._reward()
    def _reward(self):
        reward = 0
        self.terminated = False
        flag = 0
        #mask = 0
        so_hard = 0
        #一定注意是取第4个值，请参考pybullet手册的这个函数返回值的说明
        self.robot_state = p.getLinkState(self.kuka_id, self.num_joints - 1)[4]
        self.object_state = self.object_pos
        self.distance = self.cal_dis(self.robot_state,self.object_state)
        #print('the distance',self.distance)
        self.distance2 = self.cal_dis(self.robot_state,self.current_pos1)
        midle = []
        midle.append((self.current_pos1[0] + self.object_state[0]) / 2)
        midle.append((self.current_pos1[1] + self.object_state[1]) / 2)
        midle.append((self.current_pos1[2] + self.object_state[2]) / 2)
        #print(self.current_pos1,self.object_state,midle)
        self.distance3 = self.cal_dis(self.robot_state,midle)
        self.distance4 = self.cal_dis(self.current_pos1,self.object_state)
        #print("mid->end",self.distance)
        #print("start->mid",self.distance2)
        #print("start->end",self.start_to_end)
        #ab = np.array(self.robot_state)
        #avoid obstacle

        reward = -abs(self.distance-self.distance2)*200
        #print(self.distance)
        #print('now we look',reward)
        #print('the reward number',reward)
        #reward = 0
        #print(reward)
        obstacle_distance=self.find_collision(self.pos_list)
        #obstacle_distance=0
        if self.robot_state[2]<0:
            reward = -100
            self.terminated = True
        if self.robot_state[0]<0:
            reward = -100
            self.terminated = True
        if obstacle_distance>0:
            reward = -100
            print("bad")
            #time.sleep(10)
            self.terminated = True
        #if (obstacle_distance==0) &(self.start_to_end-self.distance2>0):
        #print('wait',self.distance3/self.distance4)
        if (obstacle_distance == 0)& (self.terminated==False) :
            if abs(self.distance-self.distance2)<0.05:
                reward = 10-(self.distance3/self.distance4*10)
                #reward = 10 - self.distance3
                flag = 1
                print('right',reward)
                #time.sleep(2)

                if self.distance<0.1:
                    flag = 2
                    reward= 100
                    print('very well')
                    #print(self.step_counter)
                    #time.sleep(10)
                    self.terminated = True
        if self.step_counter>self.max_steps_one_episode:
            self.terminated = True
        #if self.robot_state[2]<0:
        #    reward = -10
        #    flag = 0
        #    self.terminated = True

        info = dict(is_success=flag)

        robot_joint_state = []
        for i in range(6):
            robot_joint_state.append(p.getJointState(self.kuka_id, i)[0])
        self.dd[0] = robot_joint_state[0]
        self.dd[1] = robot_joint_state[1]
        self.dd[2] = robot_joint_state[2]
        self.dd[3] = robot_joint_state[3]
        self.dd[4] = robot_joint_state[4]
        self.dd[5] = robot_joint_state[5]
        temp_flag = 6
        for i in range(6):
            temp_t = p.getLinkState(self.kuka_id, i)[4]
            self.dd[temp_flag] = temp_t[0]
            self.dd[temp_flag + 1] = temp_t[1]
            self.dd[temp_flag + 2] = temp_t[2]
            temp_flag = temp_flag + 3
        self.observation = self.dd
        rebuild = {'observation': np.array(self.observation), 'achieved_goal': np.array(self.robot_state),
                   'desired_goal': np.array(self.object_state)}
        #print("reward",reward)
        self.current_pos = p.getLinkState(self.kuka_id, self.num_joints - 1)[4]
        return rebuild, reward, self.terminated, info
        #return np.array(self.observation).astype(np.float32), reward, self.terminated, info
    def only_step(self,goal):
        #time.sleep(1)
        #print("adjusting")

        self.robot_joint_positions = p.calculateInverseKinematics(
            bodyUniqueId=self.kuka_id,
            endEffectorLinkIndex=self.num_joints - 1,
            targetPosition=[
                goal[0], goal[1],
                goal[2]
            ],
            targetOrientation=self.orientation,
        )
        # print(self.robot_joint_positions)
        for i in range(self.num_joints):
            p.resetJointState(
                bodyUniqueId=self.kuka_id,
                jointIndex=i,
                targetValue=self.robot_joint_positions[i],
            )
        #p.stepSimulation()
        self.current_pos = p.getLinkState(self.kuka_id, self.num_joints - 1)[4]
    def find_pose(self):
        #time.sleep(0.5)
        self.object_state = p.getBasePositionAndOrientation(self.object_id)[0]
        if self.step_counter1 == 1:
            self.init_joint_positions = [0.57, 2, -2, -1.57, -1.57, 0]
            if self.fflag == 0:
                for i in range(self.num_joints):
                    p.resetJointState(
                        bodyUniqueId=self.kuka_id,

                        jointIndex=i,
                        targetValue=self.init_joint_positions[i],
                    )
                self.fflag = self.fflag + 1
            self.new_robot_pos = [
                self.object_state[0], self.object_state[1],
                self.object_state[2]-0.05
            ]
            self.orientation = p.getQuaternionFromEuler([0, 0, 0])
            #print("从下方往上方抓取")
        if self.step_counter1 == 2:
            self.new_robot_pos = [
                self.object_state[0] - 0.05, self.object_state[1],
                self.object_state[2]
            ]
            self.orientation = p.getQuaternionFromEuler([0, math.pi / 2, 0])
            #print("从前往后方抓取")
        if self.step_counter1 == 4:
            self.new_robot_pos = [
                self.object_state[0], self.object_state[1] + 0.05,
                self.object_state[2]
            ]
            self.orientation = p.getQuaternionFromEuler([math.pi / 2, 0, 0])
            #print("从左往右抓取")
        if self.step_counter1 == 3:
            self.new_robot_pos = [
                self.object_state[0], self.object_state[1] - 0.05,
                self.object_state[2]
            ]
            self.orientation = p.getQuaternionFromEuler([-math.pi / 2, 0, 0])
            #print("从右往左抓取")
        if self.step_counter1 == 0:
            self.init_joint_positions = [0.57, 2, -2, -1.57, 1.57, 0]
            #self.init_joint_positions = [0.11, 0.2, 0.6, -0.745, -1.57, 0]
            self.new_robot_pos = [
                self.object_state[0], self.object_state[1],
                self.object_state[2] + 0.12
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
        for i in range(self.num_joints):
            p.resetJointState(
                bodyUniqueId=self.kuka_id,
                jointIndex=i,
                targetValue=self.robot_joint_positions[i],
            )
        p.stepSimulation()
        self.juila = self.juila+1
        self.robot_state = p.getLinkState(self.kuka_id, self.num_joints - 1)[4]
        square_dx = (self.robot_state[0] - self.new_robot_pos[0]) ** 2
        square_dy = (self.robot_state[1] - self.new_robot_pos[1]) ** 2
        square_dz = (self.robot_state[2] - self.new_robot_pos[2]) ** 2
        # 用机械臂末端和物体的距离作为奖励函数的依据
        self.distance = sqrt(square_dx + square_dy + square_dz)
        obstacle_distance = self.find_collision(self.pos_list)
        # if (obstacle_distance==0) &(self.start_to_end-self.distance2>0):

        if self.distance < 0.02:
            #print(self.step_counter1,"used",self.juila)
            #time.sleep(5)
            self.juila = 0
            if obstacle_distance == 0:
                #print('get it', self.step_counter1)
                return self.new_robot_pos
            self.step_counter1 = self.step_counter1 + 1
        if self.step_counter1 > 4:
            return 10
        if self.juila > 150:
            #print(self.step_counter,'used')
            #time.sleep(5)
            #print(self.distance)
            self.step_counter1 = self.step_counter1 + 1
            self.juila = 0
        return 0
    def close(self):
        p.disconnect()
    def compute_reward(self,achieved_goal, desired_goal,info) :
        square_dx = (achieved_goal[0] - desired_goal[0]) ** 2
        square_dy = (achieved_goal[1] - desired_goal[1]) ** 2
        square_dz = (achieved_goal[2] - desired_goal[2]) ** 2
        d = sqrt(square_dx + square_dy + square_dz)
        return -d
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
            else:
                obstacle_distance.append(0)
        return sum(obstacle_distance)

    def just_step(self, joint):
        for i in range(self.num_joints):
            p.resetJointState(
                bodyUniqueId=self.kuka_id,
                jointIndex=i,
                targetValue=joint[i],
            )
        p.stepSimulation()
    def cal_dis(self,begin,end):
        square_dx = (begin[0] - end[0]) ** 2
        square_dy = (begin[1] - end[1]) ** 2
        square_dz = (begin[2] - end[2]) ** 2
        # print('the gap is >>>>>>>',sqrt(square_dx),sqrt(square_dy),sqrt(square_dz))
        # 用机械臂末端和物体的距离作为奖励函数的依据
        distance = sqrt(square_dx + square_dy + square_dz)
        return distance


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
        time.sleep(1)

        for i in range(100):
            #print(i)
            action=env.action_space.sample()
            #print("执行工作：")
            #print(action)
            #action=np.array([0,0,0.47-i/1000])
            obs,reward,done,info=env.step(action)
            time.sleep(5)
            #print(obs)
          #  print("i={},\naction={},\nobs={},\ndone={},\n".format(i,action,obs,done,))
            #print(colored("reward={},info={}".format(reward,info),"cyan"))
           # print(colored("info={}".format(info),"cyan"))
            sum_reward+=reward
            #print(i)
            if done:
                time.sleep(10)
                #print("????")
                break
           # time.sleep(0.1)
    print()
    print(sum_reward)