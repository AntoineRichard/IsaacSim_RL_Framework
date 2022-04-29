from pxr import Gf
import RLEnvironments.IsaacUtils as utils
from RLEnvironments.Worlds.BaseWorld import BaseWorld

class SingleLakeWorld(BaseWorld):
    def __init__(self, stage, world, **kwargs):
        super.__init__(stage, world, **kwargs)
    
    def loadWorld(self, variation="0", scene_path="/LakeSimulation/", **kwargs):
        scene_path = scene_path+"gen"+str(variation)+"_lake.usd"
        self.lake = []
        self.lake = utils.createObject('/World', self.stage, scene_path, False, position=Gf.Vec3d(0,0,0), group=self.lake, allow_physics=False)
    
    def reset(self):
        pass