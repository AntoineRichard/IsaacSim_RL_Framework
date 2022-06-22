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
    "headless": True,
}

if __name__ == "__main__":

    omniverse_kit = OmniKitHelper(CONFIG)
    from RLEnvironments import IsaacUtils as utils
    parser = argparse.ArgumentParser()
    parser.add_argument("--lake_id", type=int, help="id of the lake to generate")
    args = parser.parse_args()

    nucleus_server = utils.get_nucleus_server()
    beds_path = nucleus_server + "/RLEnvironments/Assets/LakeBeds/parts_gen"+str(args.lake_id)

    stage = omni.usd.get_context().get_stage()
    material = utils.CreateBasicMaterial(stage)
    prims = []
    with open('/home/gpu_user/.local/share/ov/pkg/isaac_sim-2021.2.0/standalone_examples/python_samples/isaac-custom-code/raw_generation/gen'+str(args.lake_id)+'/objects.pkl','rb') as f:
        objects = pickle.load(f)
    count=0
    for y in range(0,700,50):
        for x in range(0,700,50):
            obj_name = "part_"+str(y)+"-"+str(x)+".usd"
            position = Gf.Vec3d(x,y,0)
            prims = utils.createObject('/World/lake_bed/tile',stage,beds_path+'/'+obj_name, material, position=position,group=prims,scale=Gf.Vec3d(100,100,100), allow_physics=False,collision=True, rotation=utils.Euler2Quat(np.array([np.pi/2,0,0])))
 
    print('Lake bed Done')
    obj_prims = []
    
    for i, obj in enumerate(objects):
        if i%1000 == 0:
            print(str(i) +' objects added')

        position = Gf.Vec3d(float(obj['position'][1]/(obj['scale'][0]*10)), float(obj['position'][0]/(obj['scale'][0]*10)), 0.0)
        scale = Gf.Vec3d(float(obj['scale'][0]*100), float(obj['scale'][1]*100), float(obj['scale'][2]*100))
        rotation = np.array(obj['orientation'])
        scene_path = '/World/vegetation/'+obj["type"]+'/obj'
        obj_path = nucleus_server + "/RLEnvironments/Assets/Vegetation/" + obj["asset"]
        obj_prims.append(utils.createObject(scene_path, stage, obj_path, False, position=position, scale=scale, rotation=utils.Euler2Quat(rotation), group=obj_prims, allow_physics=False, collision=False, is_instance=True))

    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    omni.usd.get_context().save_as_stage(nucleus_server + "/RLEnvironments/Worlds/Lakes/LakesAssembled/Lake_gen"+str(args.lake_id)+".usd", None)
    
    omniverse_kit.stop()
    omniverse_kit.shutdown()
