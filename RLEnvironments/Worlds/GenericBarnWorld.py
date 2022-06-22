import os
import numpy as np
from pxr import Gf, UsdGeom
import RLEnvironments.IsaacUtils as utils
from RLEnvironments.Worlds.BaseWorld import BaseWorld

class GenericBarnWorld(BaseWorld):
    def __init__(self, stage, world, **kwargs):
        super().__init__(stage, world, **kwargs)
    
    def loadWorld(self, scene_path="/RLEnvironments/Worlds/Barns/500Cylinders.usd", metadata_path="/RLEnvironments/Worlds/Barns/meta", randomness=0.0, **kwargs):
        scene_path = self.nucleus_server + scene_path
        self.metadata_path = metadata_path
        self.randomness = randomness
        self.lake = []
        self.lake = utils.addReference('/World', self.stage, scene_path)

    def loadMetaData(self,num):
        if num == -1:
            worlds = os.listdir(self.metadata_path)
            num = int(np.random.rand() * len(worlds))
        return np.load(os.path.join(self.metadata_path,"World_"+str(num)+".npy"))

    def moveWorld(self, num=-1):
        positions = self.loadMetaData(num)
        positions = positions + np.reshape(np.random.rand(positions.shape[0]*2)*self.randomness,(positions.shape[0],2))
        for i, xy in enumerate(positions):
            prim = self.stage.GetPrimAtPath("/World/Barn/cylinder_"+str(i))
            xform = UsdGeom.Xformable(prim)
            utils.setTranslate(xform, Gf.Vect3d(xy[0]*100,xy[1]*100,0))

    def reset(self):
        for i in range(500):
            prim = self.stage.GetPrimAtPath("/World/Barn/cylinder_"+str(i))
            xform = UsdGeom.Xformable(prim)
            utils.setTranslate(xform, Gf.Vect3d(i*100,10000,0))