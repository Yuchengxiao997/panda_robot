import os.path as osp
import pybullet as p
import math
import sys
import pybullet_data
import random
import time
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))

import pb_ompl

class BoxDemo():
    def __init__(self):
        self.obstacles = []

        p.connect(p.GUI)
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1./240.)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")

        # load robot
        robot_id = p.loadURDF("Zu 5 URDF/urdf/Zu5.urdf", (0,0,0), useFixedBase = 1)
        print(robot_id)
        robot = pb_ompl.PbOMPLRobot(robot_id)
        self.robot = robot

        # setup pb_ompl
        self.pb_ompl_interface = pb_ompl.PbOMPL(self.robot)
        self.pb_ompl_interface.set_planner("BITstar")

        # add obstacles
        for i in range(10):
            a= self.random_pos()
            self.add_obstacles(a)

    def clear_obstacles(self):
        for obstacle in self.obstacles:
            p.removeBody(obstacle)
    def random_pos(self):
    	bp = [
        random.uniform(0.6,
                       0.8),
        random.uniform(-0.2,
                       0.2),
        random.uniform(0.1,
                       0.4),
    	]
    	return bp

    def add_obstacles(self,pos):
        # add box
        self.add_box(pos, [0.01, 0.01, 0.03])

        # store obstacles
        self.pb_ompl_interface.set_obstacles(self.obstacles)

    def add_box(self, box_pos, half_box_size):
        colBoxId = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_box_size)
        box_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=colBoxId, basePosition=box_pos)

        self.obstacles.append(box_id)
        return box_id

    def demo(self):
        start = [0.57,0,0,-1.57,-1.57,0]
        goal = [0.384,1.05,-1.6,-2.12,-1.57,-1.18]
        #start = [0,1.5,0,-0.1,0,0.2]

        self.robot.set_state(start)
        #time.sleep(10)
        res, path = self.pb_ompl_interface.plan(goal)
        print('is what >>>>',res)
        if res:
            print(len(path))
            #for i in range(len(path)):
            #    print('the action is >>>',path[i])
            self.pb_ompl_interface.execute(path)
        return res, path

if __name__ == '__main__':
    env = BoxDemo()
    env.demo()
