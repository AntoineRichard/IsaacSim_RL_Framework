import os
import omni
import carb
from pxr import Gf, UsdGeom
import numpy as np
from omni.isaac.python_app import OmniKitHelper
from omni.isaac.kit import SimulationApp
from matplotlib import pyplot as plt

import time
import datetime
import sys

CONFIG = {
    "experience": f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit',
    "renderer": "RayTracedLighting",
    "headless": True,
}

if __name__ == "__main__":
    simulation_app = SimulationApp(CONFIG)
    #omniverse_kit = OmniKitHelper(CONFIG)
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    #simulation_app.set_setting("/app/window/drawMouse", True)
    #simulation_app.set_setting("/app/livestream/proto", "ws")
    ext_manager.set_extension_enabled_immediate("omni.physx.bundle", True)
    #ext_manager.set_extension_enabled_immediate("omni.syntheticdata", True)
    #ext_manager.set_extension_enabled_immediate("omni.kit.livestream.core", True)
    #ext_manager.set_extension_enabled_immediate("omni.kit.livestream.native", True)
    #ext_manager.set_extension_enabled_immediate("omni.kit.window.stage", True)
    #ext_manager.set_extension_enabled_immediate("omni.kit.property.bundle", True)
    #ext_manager.set_extension_enabled_immediate("omni.isaac.ros_bridge", True)
    import utils
    #from omni.physx import get_physx_interface, get_physx_authoring_interface
    from omni.isaac.imu_sensor import _imu_sensor
    #from omni.isaac.core.utils. import create_background
    #from omni.isaac.dynamic_control import _dynamic_control
    #from omni.isaac.core import SimulationContext
    



    #from HeronSettings import HeronHydroSettings, HeronThrusters
    #from HeronWrapper import HeronWrapper
    #from lake_environment import Environment
    #from omni.isaac.core import World

    #simulation_context = SimulationContext()
    #simulation_context.set_simulation_dt(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 12.0)
    #world = World(stage_units_in_meters=0.01, physics_dt=1/60.0, rendering_dt=1.0/12.0)
    #world.step()
    #simulation_app.update()

    # Generate the environment
    # Make Z up
    #elf.omniverse_kit.set_up_axis(UsdGeom.Tokens.z)

    # Initialize and spawn the robot
    #stage = omni.usd.get_context().get_stage()
    #lake = Environment(stage, simulation_app)
    #heron = HeronWrapper(stage, HeronHydroSettings, HeronThrusters)
    #position, rotation = lake.getValidLocation(0)
    #heron.spawn(position)
    #simulation_app.update()
    #world.step()
    #def physics_update(dt: float):
    #    heron.update(dt)
    #world.add_physics_callback("heron_dyn", heron.update)
    #physics_sub = omni.physx.acquire_physx_interface().subscribe_physics_step_events(physics_update)
    #heron.activateLidar()
    #heron.loadPlugins()
    #world.reset()
    #HW.spawn()
    #world.play()
    #while simulation_app.is_running():
    from heron_environment import HeronEnvironment
    HE = HeronEnvironment()
    t = datetime.datetime.now()
    HE.reset()
    obs_laser = None
    obs_laser2= None
    for j in range(100):
        for i in range(500):
            obs, reward, _, _ = HE.step([0.25,0.25])
            #heron.updateCommands([0.5,0.5])
            #print(np.min(obs["laser"]))
            
            #print(obs["physics"])
            #plt.imshow(obs["image"])
            #plt.show()
            #plt.imsave("/tmp/"+str(i)+"_img.png",obs["image"])
        #print(heron.getObservation())
        HE.reset()
    t2 = datetime.datetime.now()
    print("ran 50.000 steps in :"+str((t2-t).seconds)+"s")
    simulation_app.close()
    #omni.usd.get_context().save_as_stage(nucleus_server + "/Users/test/heron_dynamic_test.usd", None)
    #time.sleep(10)
    #simulation_app.shutdown()