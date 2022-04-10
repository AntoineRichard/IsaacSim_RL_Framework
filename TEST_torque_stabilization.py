import os
import omni
import carb
from pxr import Gf, UsdGeom
from omni.isaac.python_app import OmniKitHelper
import time, math



CONFIG = {
    "experience": f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit',
    "renderer": "RayTracedLighting",
    "headless": False,
}

if __name__ == "__main__":

    omniverse_kit = OmniKitHelper(CONFIG)
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    omniverse_kit.set_setting("/app/window/drawMouse", True)
    omniverse_kit.set_setting("/app/livestream/proto", "ws")
    ext_manager.set_extension_enabled_immediate("omni.physx.bundle", True)
    ext_manager.set_extension_enabled_immediate("omni.syntheticdata", True)
    ext_manager.set_extension_enabled_immediate("omni.kit.livestream.core", True)
    ext_manager.set_extension_enabled_immediate("omni.kit.livestream.native", True)
    import utils
    import numpy as np
    #from omni.isaac.dynamic_control import _dynamic_control
    #from omni.physx import get_physx_scene_quety_interface
    from omni.physx import get_physx_interface
    from python_samples.Buoyancy.PhysxUtils import AddRelativeTorque, Q2RPY, Quaternion2RotationMatrix, getPose

    nucleus_server = utils.get_nucleus_server()
    asset_path = nucleus_server + "/Isaac/Props/Blocks/nvidia_cube.usd"

    stage = omni.usd.get_context().get_stage()
    utils.setup_cpu_physics(stage, "/World/physicsScene")
    #utils.create_ground_plane(stage, "/World/ground_plane", size=100)

    #dc = _dynamic_control.acquire_dynamic_control_interface()
    physxIFace = get_physx_interface()

    def physics_update(dt: float):
        pose, quat = getPose(physxIFace,prims[-1])
        roll, pitch, yaw = Q2RPY(quat)
        torque = np.array([
                -1 * math.sin(roll) * 100000,
                -1 * math.sin(pitch) * 100000,
                0])
        print(roll,pitch)
        AddRelativeTorque(physxIFace, prims[-1], torque, dist=100)

    physics_sub = omni.physx.acquire_physx_interface().subscribe_physics_step_events(physics_update)

    prims = []

    position = Gf.Vec3d(50, 50, 50)
    rotation = Gf.Quaternion(0.924, Gf.Vec3d(0.383,0,0))
    prims = utils.createObject('/World/nvidia_cube',stage,asset_path,False,position=position,rotation=utils.Euler2Quat(np.array([np.pi/4,np.pi/4,0])), group=prims, density=512, scale=Gf.Vec3d(3,3,3), allow_physics=True)
    prim = stage.GetPrimAtPath(prims[-1])
    print(UsdGeom.Boundable(prim).ComputeLocalBound(0,"default").GetRange().GetSize())

    time.sleep(10)
    while omniverse_kit.app.is_running():
        omniverse_kit.update(1.0/30, physics_dt=1/120.0, physics_substeps=4)
    
    omniverse_kit.stop()
    omni.usd.get_context().save_as_stage(nucleus_server + "/Users/test/saved.usd", None)
    time.sleep(10)
    omniverse_kit.shutdown()

""" TEST SCRIPT THRUST HERON
import omni
from pxr import UsdPhysics
from omni.physx import get_physx_interface
from scipy.spatial.transform import Rotation as SSTR
import numpy as np

stage = omni.usd.get_context().get_stage()
#massAPI = UsdPhysics.MassAPI.Get(stage,'/heron/dummy_link')
#print(massAPI.GetMassAttr().Get())
#print(massAPI.GetCenterOfMassAttr().Get())
#print(massAPI.GetDiagonalInertiaAttr().Get())
rigidBodyAPI = UsdPhysics.RigidBodyAPI.Get(stage, '/heron/dummy_link')
pose_base = PhysXIFace.get_rigidbody_transformation('/heron/dummy_link')
vlin_world = rigidBodyAPI.GetVelocityAttr().Get()
vang_world = rigidBodyAPI.GetAngularVelocityAttr().Get()
R_base = SSTR.from_quat(pose_base["rotation"])
rot_base = R_base.as_matrix()
vrob = np.matmul(np.linalg.inv(rot_base),np.array(vlin_world))
print(vrob)
pose0 = PhysXIFace.get_rigidbody_transformation('/heron/thruster_0')
pose1 = PhysXIFace.get_rigidbody_transformation('/heron/thruster_1')
p0 = pose0['position']
q0 = pose0['rotation']
p1 = pose1['position']
q1 = pose1['rotation']
R0 = SSTR.from_quat(q0)
R1 = SSTR.from_quat(q1)
rot0 = R0.as_matrix()
rot1 = R1.as_matrix()
F = np.array([10000000, 0, 0])
F0R = np.matmul(rot0, F)
F1R = np.matmul(rot1, F)

PhysXIFace = get_physx_interface()
PhysXIFace.apply_force_at_pos("/heron/thruster_1",F1R,p1)
PhysXIFace.apply_force_at_pos("/heron/thruster_0",F0R,p0)
"""