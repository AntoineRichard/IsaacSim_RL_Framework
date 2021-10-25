"""
 Porting of the UUV Plugins to Isaac : https://github.com/uuvsimulator
 Some functions have been changed to better fit the use of Pythons.
 Some functions have been changed to be more accurate.
 2021 DreamLab  - Georgia Institute of Technology - IRL 2958 GT CNRS
 Antoine Richard: antoine.richard@gatech.edu
"""

"""
When performing unit testing uncomment the returns with dummy values.
Comment the pxr and carb imports.

Don't forget to uncomment to run in IsaacSims

#TODO use Mock functions/classes during unit testing
"""

from scipy.spatial.transform import Rotation as SSTR
import numpy as np
import math
from pxr import UsdGeom, UsdPhysics
import carb

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

def getUsdPhysicsAPIs(stage, prim_path, physics_scene):
    #return None, None, None
    return UsdPhysics.RigidBodyAPI.Get(stage, prim_path), UsdPhysics.MassAPI.Get(stage, prim_path), UsdPhysics.Scene.Get(stage, physics_scene)

def getGravity(SceneAPI):
    #return 9.81
    return SceneAPI.GetGravityMagnitudeAttr().Get()

def getRigidBodyAtPath(stage, prim_path):
    #return True
    try:
        RB = UsdPhysics.RigidBodyAPI.Get(stage, prim_path)
        return True
    except:
        return False

#def getRelativeLinearAccel(link):
#    raise NotImplementedError()

#def getRelativeAngularAccel(link):
#    raise NotImplementedError()

def getPose(PhysXIFace, prim_path):
    #return np.array([1,1,1]), np.array([0,0,0,1])
    transform = PhysXIFace.get_rigidbody_transformation(prim_path)
    # Keeping centimeters to as the forces will be
    # applied in the world frame which is in cm
    return transform['position'], transform['rotation'] # cm, normalized.

def getRelativeLinearVel(rigidBodyAPI, rotWR):
    #return np.array([1,1,1])
    world_velocity = getLinearVel(rigidBodyAPI)
    robot_velocity = np.matmul(rotWR, world_velocity)
    return robot_velocity # m/s

def getRelativeAngularVel(rigidBodyAPI, rotWR):
    #return np.array([1,1,1])
    world_velocity = getAngularVel(rigidBodyAPI)
    robot_velocity = np.matmul(rotWR, world_velocity)
    return robot_velocity # rad/s

def getLinearVel(rigidBodyAPI):
    #return np.array([1,1,1])
    vec = rigidBodyAPI.GetVelocityAttr().Get() # cm/s
    return np.array([vec[0], vec[1], vec[2]]) * 0.01 # m/s

def getAngularVel(rigidBodyAPI):
    #return np.array([1,1,1])
    vec = rigidBodyAPI.GetAngularVelocityAttr().Get() # degrees/s
    return np.pi * np.array([vec[0],vec[1],vec[2]]) / 180 # rad/s

def getName(stage, prim_path):
    return "RobotName"

def getBoundingBox(stage, prim_path):
    #return np.array([1.0,1.0,1.0])
    prim = stage.GetPrimAtPath(prim_path)
    box = UsdGeom.Boundable(prim).ComputeLocalBound(0,"default").GetRange().GetSize() # cm
    return np.array(box[0],box[1],box[2]) * 0.01 # m

def getMass(massAPI):
    #return 10
    return massAPI.GetMassAttr().Get() # kg

    
def AddForceAtRelativePosition(PhysXIFace, prim_path, force, position):
    #return True
    # Projects force and position to world frame
    transform = PhysXIFace.get_rigidbody_transformation(prim_path)
    R = Quaternion2RotationMatrix(transform['rotation'])
    force_wf = np.matmul(R,force)
    position_wf = np.matmul(R,position) + np.array(transform['position'])
    PhysXIFace.apply_force_at_pos(prim_path, force_wf, position_wf)

def AddRelativeForce(PhysXIFace, prim_path, force):
    #return True
    # Projects force to world frame
    transform = PhysXIFace.get_rigidbody_transformation(prim_path)
    R = Quaternion2RotationMatrix(transform['rotation'])
    force_wf = np.matmul(R,force)
    PhysXIFace.apply_force_at_pos(prim_path, force_wf, np.array(transform['position']))

def AddRelativeTorque(PhysXIFace, prim_path, torque, dist=100):
    #return True
    torque_norm = np.linalg.norm(torque)
    torque_normalized = torque/torque_norm
    # Projects torque to world frame
    transform = PhysXIFace.get_rigidbody_transformation(prim_path)
    R = Quaternion2RotationMatrix(transform['rotation'])
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
    f2_dir = - f1_dir
    p1 = w21 * dist
    p2 = w22 * dist
    # Apply forces at the position of the object
    PhysXIFace.apply_force_at_pos(prim_path, f1_dir*torque_norm/2, p1 + np.array(transform['position']))
    PhysXIFace.apply_force_at_pos(prim_path, f2_dir*torque_norm/2, p2 + np.array(transform['position']))