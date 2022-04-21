import os
import omni
import carb
from pxr import Gf, Sdf, UsdGeom
from omni.isaac.python_app import OmniKitHelper
from omni.isaac.kit import SimulationApp
#from omni.isaac.utils.scripts.nucleus.utils import find_nucleus_server
from scipy.spatial.transform import Rotation as SSTR
import time
import numpy as np

CONFIG = {
    "experience": f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit',
    "renderer": "RayTracedLighting",
    "headless": False,
}

def setup_cpu_physics(stage, physics_name, gravity=9.81, gravity_direction=Gf.Vec3f(0.0, 0.0, -1.0)):
    from pxr import PhysicsSchemaTools, PhysxSchema
    # Add physics scene
    scene = UsdPhysics.Scene.Define(stage, Sdf.Path(physics_name))
    # Set gravity vector
    scene.CreateGravityDirectionAttr().Set(gravity_direction)
    scene.CreateGravityMagnitudeAttr().Set(gravity*100)
    # Set physics scene to use cpu physics
    PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath(physics_name))
    physxSceneAPI = PhysxSchema.PhysxSceneAPI.Get(stage,physics_name)
    physxSceneAPI.CreateEnableCCDAttr(True)
    physxSceneAPI.CreateEnableStabilizationAttr(True)
    physxSceneAPI.CreateEnableGPUDynamicsAttr(False)
    physxSceneAPI.CreateBroadphaseTypeAttr("MBP")
    physxSceneAPI.CreateSolverTypeAttr("TGS")

def get_nucleus_server():
    result, nucleus_server = find_nucleus_server()
    if result is False:
        carb.log_error(
            "Could not find nucleus server. Stopping."
        )
        exit(1)
    return nucleus_server

def Quaternion2RotationMatrix(quat):
    # QX,QY,QZ,QW
    R = SSTR.from_quat(quat)
    return R.as_matrix()

def getLinearVel(rigidBodyAPI):
    #return np.array([1,1,1])
    vec = rigidBodyAPI.GetVelocityAttr().Get() # cm/s
    return np.array([vec[0], vec[1], vec[2]]) * 0.01 # m/s

def getRelativeLinearVel(rigidBodyAPI, rotWR):
    #return np.array([1,1,1])
    world_velocity = getLinearVel(rigidBodyAPI)
    robot_velocity = np.matmul(rotWR, world_velocity)
    return robot_velocity # m/s


if __name__ == "__main__":

    #omniverse_kit = OmniKitHelper(CONFIG)
    simulation_app = SimulationApp(CONFIG)
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    simulation_app.set_setting("/app/window/drawMouse", True)
    simulation_app.set_setting("/app/livestream/proto", "ws")
    ext_manager.set_extension_enabled_immediate("omni.physx.bundle", True)
    ext_manager.set_extension_enabled_immediate("omni.syntheticdata", True)
    ext_manager.set_extension_enabled_immediate("omni.kit.livestream.core", True)
    ext_manager.set_extension_enabled_immediate("omni.kit.livestream.native", True)
    ext_manager.set_extension_enabled_immediate("omni.kit.window.stage", True)
    ext_manager.set_extension_enabled_immediate("omni.kit.property.bundle", True)
    from omni.isaac.dynamic_control import _dynamic_control
    from omni.physx import get_physx_interface
    from utils import setTransform
    from omni.physx.scripts import utils
    from pxr import UsdPhysics
    from omni.isaac.core.utils.nucleus import find_nucleus_server
    from omni.isaac.core import SimulationContext
    nucleus_server = get_nucleus_server()
    simulation_context = SimulationContext(stage_units_in_meters=1.0)
    UsdGeom.SetStageMetersPerUnit(omni.usd.get_context().get_stage(),1.0) 
    cube_asset_path = nucleus_server + "/Isaac/Props/Blocks/nvidia_cube.usd"
    stage = omni.usd.get_context().get_stage()
    setup_cpu_physics(stage, "/World/physicsScene")

    physxIFace = get_physx_interface()
    dc = _dynamic_control.acquire_dynamic_control_interface()

    global t
    global it
    t = 0
    it = 0

    def physics_update(dt: float):
        global t, it
        transform = physxIFace.get_rigidbody_transformation(cube_path)
        rot_mat = Quaternion2RotationMatrix(transform['rotation'])
        rot_mat_inv = np.linalg.inv(rot_mat)
        print('step:',it,'time:', it*1/120.0, 'speed:',getRelativeLinearVel(rigidbody,rot_mat_inv))
        t += dt
        it += 1
        rb = dc.get_rigid_body(cube_path)
        #dc.apply_body_force(rb, carb._carb.Float3([0,0,100]), np.array(transform['position']), True)
        #dc.apply_body_force(rb, carb._carb.Float3([0,0,100]), [100,100,0], False)
        #dc.apply_body_torque(rb, [100,0,0], True)
        physxIFace.apply_force_at_pos(cube_path, carb._carb.Float3([0,0,100]), np.array(transform['position']))

    physics_sub = omni.physx.acquire_physx_interface().subscribe_physics_step_events(physics_update)

    position = Gf.Vec3d(100, 100, 0)
    rotation=Gf.Quaternion(1, Gf.Vec3d(0,0,0))
    cube_path = omni.usd.get_stage_next_free_path(stage, '/World/nvidia_cube', False)
    obj_prim = stage.DefinePrim(cube_path, "Xform")
    obj_prim.GetReferences().AddReference(cube_asset_path)
    xform = UsdGeom.Xformable(obj_prim)
    xform = setTransform(xform, rotation, position)
    utils.setRigidBody(obj_prim, "convexHull", False)
    mass_api = UsdPhysics.MassAPI.Apply(obj_prim)
    mass_api.CreateMassAttr(1.0)
    rigidbody = UsdPhysics.RigidBodyAPI.Get(stage, cube_path)
    print(rigidbody.GetSchemaAttributeNames())
    omni.kit.commands.execute("ChangeProperty",prop_path="/World/nvidia_cube.physxRigidBody:disableGravity", value=True, prev=False)

    simulation_context.set_simulation_dt(physics_dt=1.0 / 120.0, rendering_dt=1.0 / 30.0)
    while simulation_app.is_running():
        simulation_app.update()
    
    omniverse_kit.stop()
    time.sleep(10)
    omniverse_kit.shutdown()