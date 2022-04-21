import os
import omni
import carb
from pxr import Gf, UsdGeom
from omni.isaac.python_app import OmniKitHelper
import time
import sys

#import UnderWaterObject

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
    ext_manager.set_extension_enabled_immediate("omni.kit.window.stage", True)
    ext_manager.set_extension_enabled_immediate("omni.kit.property.bundle", True)
    import utils
    from omni.physx import get_physx_interface
    from omni.isaac.dynamic_control import _dynamic_control
    from python_samples.Buoyancy.HeronSettings import HeronHydroSettings
    from python_samples.Buoyancy.PhysxUtils import getPose
    from python_samples.Buoyancy.UnderWaterObject import UnderwaterObject

    nucleus_server = utils.get_nucleus_server()
    asset_path = nucleus_server + "/LakeSimulation/heron3.usd"
    scene_path = nucleus_server + "/LakeSimulation/gen1_final.usd"

    stage = omni.usd.get_context().get_stage()
    utils.setup_cpu_physics(stage, "/physicsScene")
    PhysXIFace = get_physx_interface()
    dc = _dynamic_control.acquire_dynamic_control_interface()

    def physics_update(dt: float):
        UWO.Update(dt)

    physics_sub = omni.physx.acquire_physx_interface().subscribe_physics_step_events(physics_update)

    prims = []

    position = Gf.Vec3d(11210, 46000, 35)
    prims = utils.createObject('/heron', stage, asset_path, False, position=position, group=prims, allow_physics=False)
    prims = utils.createObject('/World/lake', stage, scene_path, False, position=position, group=prims, allow_physics=False)
    UWO = UnderwaterObject(stage, PhysXIFace, dc)
    UWO.Load(HeronHydroSettings)

    time.sleep(10)
    while omniverse_kit.app.is_running():
        omniverse_kit.update(1.0/30, physics_dt=1/60.0, physics_substeps=4)
    
    omniverse_kit.stop()
    omni.usd.get_context().save_as_stage(nucleus_server + "/Users/test/heron_dynamic_test.usd", None)
    time.sleep(10)
    omniverse_kit.shutdown()