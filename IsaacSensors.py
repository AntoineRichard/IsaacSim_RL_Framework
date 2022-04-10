from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix, NavSatStatus, Imu
from geometry_msgs.msg import Vector3Stamped

import PhysxUtils as utils
import numpy as np
from pxr import UsdPhysics
import math
import os

class GenericSensor:
    def __init__(self, stage, link):
        self.stage = stage
        self.path = link
    
    def get(self):
        raise Exception("Not implemented")

class PerfectPoseSensor(GenericSensor):
    def __init__(self, stage, PhysXIFace, link):
        super().__init__(stage, link)
        self.PhysXIFace = PhysXIFace
        self.rigidBodyAPI = UsdPhysics.RigidBodyAPI.Get(stage, self.path)
    
    def get(self):
        lin_vel = utils.getLinearVel(self.rigidBodyAPI)
        ang_vel = utils.getAngularVel(self.rigidBodyAPI)
        pose, quat = utils.getPose(self.PhysXIFace,self.path)
        return lin_vel, ang_vel, pose, quat