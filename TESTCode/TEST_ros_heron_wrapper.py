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
    ext_manager.set_extension_enabled_immediate("omni.isaac.ros_bridge", True)
    import utils
    from omni.physx import get_physx_interface, get_physx_authoring_interface
    from omni.isaac.imu_sensor import _imu_sensor
    #from omni.isaac.core.utils. import create_background
    from omni.isaac.dynamic_control import _dynamic_control
    from omni.isaac.core import SimulationContext


    from HeronSettings import HeronHydroSettings, HeronThrusters
    from RosUnderWaterObject import ROSHeronWrapper


    nucleus_server = utils.get_nucleus_server()
    simulation_context = SimulationContext()

    asset_path = nucleus_server + "/LakeSimulation/heron3.usd"
    scene_path = nucleus_server + "/LakeSimulation/gen1_final.usd"
    
    stage = omni.usd.get_context().get_stage()
    utils.setup_cpu_physics(stage, "/physicsScene")
    PhysXIFace = get_physx_interface()
    PhysXAuthIFace = get_physx_authoring_interface()
    IMUIFace = _imu_sensor.acquire_imu_sensor_interface()
    DCIFace = _dynamic_control.acquire_dynamic_control_interface()

    def physics_update(dt: float):
        dt = PhysXAuthIFace.get_elapsed_time()
        #print('new_step')
        RHW.Update(dt)

    physics_sub = omni.physx.().subscribe_physics_step_events(physics_update)
    result, check = omni.kit.commands.execute("RosBridgeRosMasterCheck")
    if not check:
        carb.log_error("Please run roscore before executing this script")
        simulation_app.close()
        exit()

    omni.kit.commands.execute("ROSBridgeCreacquire_physx_interfaceateClock", path="/ROS_Clock", enabled=False)

    prims = []
    lake = []
    #position = Gf.Vec3d(0, 0, 15)
    position = Gf.Vec3d(39000, 48000, 0)
    prims = utils.createObject('/heron', stage, asset_path, False, position=position, group=prims, allow_physics=False)
    lake = utils.createObject('/lake', stage, scene_path, False, position=Gf.Vec3d(0,0,0), group=lake, allow_physics=False)

    simulation_context.set_simulation_dt(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 30.0)
    simulation_app.update()
    import rospy
    RHW = ROSHeronWrapper(stage, PhysXIFace, DCIFace, IMUIFace, HeronHydroSettings, HeronThrusters)
    rospy.init_node("heron_isaac", anonymous=True, disable_signals=True, log_level=rospy.ERROR)
    while simulation_app.is_running():
        simulation_app.update()
    
    simulation_app.stop()
    omni.usd.get_context().save_as_stage(nucleus_server + "/Users/test/heron_dynamic_test.usd", None)
    time.sleep(10)
    simulation_app.shutdown()