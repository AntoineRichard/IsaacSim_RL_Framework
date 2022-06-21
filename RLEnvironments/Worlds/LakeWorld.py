from pxr import Gf
import RLEnvironments.IsaacUtils as utils
from RLEnvironments.Worlds.BaseWorld import BaseWorld

class SingleLakeWorld(BaseWorld):
    def __init__(self, stage, world, **kwargs):
        super().__init__(stage, world, **kwargs)
    
    def loadWorld(self, variation="0", scene_path="/RLEnvironments/Worlds/Lakes/StandaloneLakesWater/", **kwargs):
        scene_path = self.nucleus_server + scene_path+"Lake_"+str(variation)+".usd"
        self.lake = []
        self.lake = utils.addReference('/World', self.stage, scene_path)
    
    def reset(self):
        pass