"""
 Porting of the UUV Plugins to Isaac : https://github.com/uuvsimulator
 Some functions have been changed to better fit the use of Pythons.
 Some functions have been changed to be more accurate.
 2021 DreamLab  - Georgia Institute of Technology - IRL 2958 GT CNRS
 Antoine Richard: antoine.richard@gatech.edu
"""
from numpy.testing._private.utils import assert_
from scipy.spatial.transform import Rotation as SSTR
import numpy as np
import math
#from pxr import UsdGeom
#import carb

def Q2RPY(quat):
    # QX,QY,QZ,QW
    R = SSTR.from_quat(quat)
    return R.as_euler('xyz')

def Quaternion2RotationMatrix(quat):
    # QX,QY,QZ,QW
    R = SSTR.from_quat(quat)
    return R.as_matrix()

def Print(value):
    print(value)

def Assert(value, msg):
    assert value, msg

def getGravity(stage):
    raise NotImplementedError()

def getPrimAtLink(link):
    raise NotImplementedError()

def getRelativeLinearAccel(link):
    raise NotImplementedError()

def getRelativeAngularAccel(link):
    raise NotImplementedError()

def getPose(PhysXIFace,prim_path):
    transform = PhysXIFace.get_rigidbody_transformation(prim_path)
    return transform['position'], transform['quaternion']

def getRelativeLinearVel(prim):
    raise NotImplementedError()

def getRelativeAngularVel(prim):
    raise NotImplementedError()

def getName(stage, prim_path):
    return "None"
    #raise NotImplementedError()

def getBoundingBox(stage, prim_path):
    return np.array([1.0,1.0,1.0])
    prim = stage.GetPrimAtPath(prim_path)
    box = UsdGeom.Boundable(prim).ComputeLocalBound(0,"default").GetRange().GetSize()
    return box

def getMass(stage, prim_path):
    return 10

    
def AddForceAtRelativePosition(PhysXIFace, prim_path, force, position):
    # Projects force and position to world frame
    transform = PhysXIFace.get_rigidbody_transformation(prim_path)
    R = Quaternion2RotationMatrix(transform['quaternion'])
    force_wf = np.matmul(R,force)
    position_wf = np.matmul(R,position) + np.array(transform['position'])
    PhysXIFace.apply_force_at_pos(prim_path, numpy2carb3(force_wf), numpy2carb3(position_wf))

def AddRelativeForce(PhysXIFace, prim_path, force):
    # Projects force to world frame
    transform = PhysXIFace.get_rigidbody_transformation(prim_path)
    R = Quaternion2RotationMatrix(transform['quaternion'])
    force_wf = np.matmul(R,force)
    PhysXIFace.apply_force_at_pos(prim_path, numpy2carb3(force_wf), numpy2carb3(np.array(transform['position'])))

def AddRelativeTorque(PhysXIFace, prim_path, torque, dist=100):
    torque_norm = np.linalg.norm(torque)
    torque_normalized = torque/torque_norm
    # Projects torque to world frame
    transform = PhysXIFace.get_rigidbody_transformation(prim_path)
    R = Quaternion2RotationMatrix(transform['quaternion'])
    torque_normalized = np.matmul(R,torque_normalized)
    # Make virtual vector
    idx = np.argmin(torque)
    v = np.zeros([3])
    v[idx] = 1.0
    # Get a set two orthogonal vector to the torque
    w = np.cross(torque_normalized, v)
    w21 = np.cross(torque_normalized, w)
    w22 = -w21
    f1_dir = np.cross(torque_normalized,w21)
    f2_dir = -np.cross(torque_normalized,w22)
    p1 = w21 * dist
    p2 = w22 * dist
    # Apply forces at the position of the object
    PhysXIFace.apply_force_at_pos(prim_path,numpy2carb3(f1_dir*torque_norm/2),numpy2carb3(p1 + np.array(transform['position'])))
    PhysXIFace.apply_force_at_pos(prim_path,numpy2carb3(f2_dir*torque_norm/2),numpy2carb3(p2 + np.array(transform['position'])))

def numpy2carb3(array):
    return carb._carb.Float3([array[0],array[1],array[2]])