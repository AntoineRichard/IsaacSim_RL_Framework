import os
import omni
from pxr import Gf
import numpy as np
import pickle
from omni.isaac.python_app import OmniKitHelper
import argparse
CONFIG = {
    "experience": f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit',
    "renderer": "RayTracedLighting",
    "width": 128,
    "height": 72,
    "window_width": 192,
    "window_height": 108,
    "headless": True,
}

if __name__ == "__main__":

    omniverse_kit = OmniKitHelper(CONFIG)
    import utils
    parser = argparse.ArgumentParser()
    parser.add_argument("--lake_id", type=int, help="id of the lake to generate")
    args = parser.parse_args()

    nucleus_server = utils.get_nucleus_server()
    beds_path = nucleus_server + "/RLEnvironments/Assets/LakeBeds/parts_gen"+str(args.lake_id)

    stage = omni.usd.get_context().get_stage()
    material = utils.CreateBasicMaterial(stage)
    prims = []
    utils.setup_cpu_physics(stage, "/World/PhysicsScene")
    utils.createObject('/World/sky', stage, nucleus_server + '/NVIDIA/Assets/Skies/Dynamic/Overcast.usd', False)
    utils.createObject('/World/water_plane', stage, nucleus_server + '/RLEnvironments/Assets/LakeBeds/waterplane.usd', False, scale=Gf.Vec3d(100,100,100))
    utils.createObject('/World/lake', stage, nucleus_server + '/RLEnvironments/Worlds/Lakes/LakesAssembled/Lake_gen'+str(args.lake_id)+'.usd', False)
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    omni.usd.get_context().save_as_stage(nucleus_server + "/RLEnvironments/Worlds/Lakes/StandaloneLakesWater/Lake_"+str(args.lake_id)+".usd", None)
    
    omniverse_kit.stop()
    omniverse_kit.shutdown()
