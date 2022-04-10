import os
import omni
import carb
from pxr import Gf, UsdGeom
import numpy as np
from omni.isaac.python_app import OmniKitHelper
from omni.isaac.kit import SimulationApp
import time
import sys

CONFIG = {
    "experience": f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit',
    "renderer": "RayTracedLighting",
    "headless": False,
}

if __name__ == "__main__":
    simulation_app = SimulationApp(CONFIG)
    #omniverse_kit = OmniKitHelper(CONFIG)
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    simulation_app.set_setting("/app/window/drawMouse", True)
    simulation_app.set_setting("/app/livestream/proto", "ws")
    ext_manager.set_extension_enabled_immediate("omni.physx.bundle", True)
    ext_manager.set_extension_enabled_immediate("omni.syntheticdata", True)
    ext_manager.set_extension_enabled_immediate("omni.kit.livestream.core", True)
    ext_manager.set_extension_enabled_immediate("omni.kit.livestream.native", True)
    ext_manager.set_extension_enabled_immediate("omni.kit.window.stage", True)
    ext_manager.set_extension_enabled_immediate("omni.kit.property.bundle", True)
    import utils
    from omni.physx import get_physx_interface, get_physx_authoring_interface
    #from omni.isaac.core.utils. import create_background
    from omni.isaac.dynamic_control import _dynamic_control
    from omni.isaac.core import SimulationContext


    from HeronSettings import HeronHydroSettings, HeronThrusters
    from UnderWaterObject import UnderwaterObject
    from Thruster import ThrusterPlugin


    nucleus_server = utils.get_nucleus_server()
    simulation_context = SimulationContext()

    asset_path = nucleus_server + "/LakeSimulation/heron3.usd"
    scene_path = nucleus_server + "/LakeSimulation/gen1_final.usd"

    stage = omni.usd.get_context().get_stage()
    utils.setup_cpu_physics(stage, "/physicsScene")
    PhysXIFace = get_physx_interface()
    PhysXAuthIFace = get_physx_authoring_interface()

    dc = _dynamic_control.acquire_dynamic_control_interface()

    def physics_update(dt: float):
        dt = PhysXAuthIFace.get_elapsed_time()
        #print('new_step')
        UWO.Update(dt)
        THR1.Update(dt)
        THR2.Update(dt)

    physics_sub = omni.physx.acquire_physx_interface().subscribe_physics_step_events(physics_update)

    prims = []

    #position = Gf.Vec3d(11210, 46000, 35)
    position = Gf.Vec3d(0, 0, 15)
    prims = utils.createObject('/heron', stage, asset_path, False, position=position, group=prims, allow_physics=False)
    #create_background(stage, scene_path, '/lake', offset=Gf.Vec3d(0, 0, 0))
    #prefix = "/heron"
    #prim_path = omni.usd.get_stage_next_free_path(stage, prefix, False)
    #robot_prim = stage.DefinePrim(prim_path, "Xform")
    #robot_prim.GetReferences().AddReference(asset_path)
    #prim = stage.GetPrimAtPath(prims[-1])
    #print(UsdGeom.Boundable(prim).ComputeLocalBound(0,"default").GetRange().GetSize())
    simulation_context.set_simulation_dt(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 30.0)
    simulation_app.update()

    UWO = UnderwaterObject(stage, PhysXIFace, dc)
    THR1 = ThrusterPlugin(stage, PhysXIFace, dc)
    THR2 = ThrusterPlugin(stage, PhysXIFace, dc)
    UWO.Load(HeronHydroSettings)
    THR1.Load(HeronThrusters[0])
    THR2.Load(HeronThrusters[1])
    THR1.UpdateCommand(0.2)
    THR2.UpdateCommand(0.2)
    UWO.UpdateFlowVelocity(np.array([0.0,0.0,0.0]))

    time.sleep(10)

    while simulation_app.is_running():
        simulation_app.update()
    
    omniverse_kit.stop()
    omni.usd.get_context().save_as_stage(nucleus_server + "/Users/test/heron_dynamic_test.usd", None)
    time.sleep(10)
    omniverse_kit.shutdown()

""" TEST SCRIPT THRUST HERON
import omni
from pxr import UsdPhysics
from omni.physx import get_physx_interface
from scipy.spatial.transform import Rotation as SSTR
import numpy as np
PhysXIFace = get_physx_interface()
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
F = np.array([2000, 0, 0])
F0R = np.matmul(rot0, F)
F1R = np.matmul(rot1, F)

PhysXIFace = get_physx_interface()
PhysXIFace.apply_force_at_pos("/heron/thruster_1",F1R,p1)
PhysXIFace.apply_force_at_pos("/heron/thruster_0",F0R/2,p0)
"""