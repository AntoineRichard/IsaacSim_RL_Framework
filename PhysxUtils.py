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

def getJointAxis(stage, joint_path):
    joint = UsdPhysics.RevoluteJoint.Get(stage, joint_path)
    axis = joint.GetAxisAttr().Get()
    if axis == 'X':
        ax = np.array([1,0,0])
    elif axis == 'Y':
        ax = np.array([0,1,0])
    elif axis == 'Z':
        ax = np.array([0,0,1])
    else:
        raise Exception("Failed to fetch thruster axis. Specified axis does not exist.")
    return ax

def getPose(PhysXIFace, prim_path):
    #return np.array([1,1,1]), np.array([0,0,0,1])
    transform = PhysXIFace.get_rigidbody_transformation(prim_path)
    # Keeping centimeters to as the forces will be
    # applied in the world frame which is in cm
    return np.array(transform['position']), np.array(transform['rotation']) # cm, normalized.

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

def getCenterOfMass(massAPI):
    #return 10
    return massAPI.GetCenterOfMassAttr().Get() # kg
    
def AddForceAtRelativePosition(PhysXIFace, prim_path, force, position):
    #return True
    # Projects force and position to world frame
    transform = PhysXIFace.get_rigidbody_transformation(prim_path)
    R = Quaternion2RotationMatrix(transform['rotation'])
    force_wf = np.matmul(R,force)
    position_wf = np.matmul(R,position) + np.array(transform['position'])
    PhysXIFace.apply_force_at_pos(prim_path, force_wf, position_wf)

def AddRelativeForceDC(DCIFace, rigid_body_handle, force):
    DCIFace.apply_body_force(rigid_body_handle, force, [0,0,0], False)

def AddForceDC(DCIFace, rigid_body_handle, PhysXIFace, prim_path, force):#, CoM):
    transform = PhysXIFace.get_rigidbody_transformation(prim_path)
    DCIFace.apply_body_force(rigid_body_handle, force, transform['position'], True)

def AddRelativeTorqueDC(DCIFace, rigid_body_handle, torque, dist=100):
    DCIFace.apply_body_torque(rigid_body_handle, torque*dist, False)