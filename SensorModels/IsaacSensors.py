from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix, NavSatStatus, Imu
from geometry_msgs.msg import Vector3Stamped

import BuoyancyPhysics.PhysxUtils as utils
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
    def __init__(self, stage, PhysXIFace, DCIFace, link):
        super().__init__(stage, link)
        self.DCIFace = DCIFace
        self.PhysxIFace = PhysXIFace
        self.rigidBodyAPI = UsdPhysics.RigidBodyAPI.Get(stage, self.path)
        self.rigid_body_handle = self.DCIFace.get_rigid_body(self.path)
    
    def get(self):
        self.rigid_body_handle = self.DCIFace.get_rigid_body(self.path) 
        lin_vel = np.array(self.DCIFace.get_rigid_body_local_linear_velocity(self.rigid_body_handle))/100
        ang_vel = utils.getAngularVel(self.rigidBodyAPI)
        pose, quat = utils.getPose(self.PhysxIFace,self.path)
        return lin_vel, ang_vel, pose, quat