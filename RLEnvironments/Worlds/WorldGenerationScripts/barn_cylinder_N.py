import os
import omni
from pxr import Gf, UsdGeom, Usd
import numpy as np
import pickle
from omni.isaac.python_app import OmniKitHelper
import argparse
CONFIG = {
    "experience": f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit',
    "renderer": "RayTracedLighting",
    "headless": True,
}

if __name__ == "__main__":

    omniverse_kit = OmniKitHelper(CONFIG)
    from RLEnvironments import IsaacUtils as utils
    from omni.physx.scripts.utils import setCollider
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_cylinders", type=int, help="the number of cylinders to generate")
    args = parser.parse_args()

    def add_cylinder(stage, path, radius: float = 7.5, height: float = 200, offset: Gf.Vec3d = Gf.Vec3d(0, 0, 0)):
        cylinderGeom = UsdGeom.Cylinder.Define(stage, path)
        cylinderGeom.CreateRadiusAttr(radius)
        cylinderGeom.CreateHeightAttr(height)
        cylinderGeom.AddTranslateOp().Set(offset)

    nucleus_server = utils.get_nucleus_server()

    stage = omni.usd.get_context().get_stage()

    count=0
    
    for i in range(args.num_cylinders):
        add_cylinder(stage, "/World/Barn/cylinder_"+str(i), offset=Gf.Vec3d(i*100,10000,0))

    print('Cylinder generation Done.')
    
    curr_prim = stage.GetPrimAtPath("/")

    for prim in Usd.PrimRange(curr_prim):
        # only process shapes
        if (prim.IsA(UsdGeom.Cylinder)
            or prim.IsA(UsdGeom.Cube)):
            setCollider(prim, approximationShape="convexHull")

    print('Collision Added.')

    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    omni.usd.get_context().save_as_stage(nucleus_server + "/RLEnvironments/Worlds/Barns/"+str(args.num_cylinders)+"Cylinders.usd", None)
    
    omniverse_kit.stop()
    omniverse_kit.shutdown()
