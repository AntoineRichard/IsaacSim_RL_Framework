from pxr import Gf
import RLEnvironments.IsaacUtils as utils

class SingleLakeEnvironment:
    def __init__(self, stage, world, variation="0", scene_path="/LakeSimulation/", **kwargs):
        #Load the scene (it's prebuilt)
        self.stage = stage
        self.world = world
        nucleus_server = utils.get_nucleus_server()
        scene_path = nucleus_server + scene_path
        self.setupPhysics()
        self.loadWorld(scene_path+"Lake_"+str(variation)+".usd")
        self.world.step()
        print("waiting for materials to load...")
    
    def loadWorld(self, scene_path):
        self.lake = []
        self.lake = utils.createObject('/World', self.stage, scene_path, False, position=Gf.Vec3d(0,0,0), group=self.lake, allow_physics=False)
    
    def reset(self):
        pass

    def setupPhysics(self):
        utils.setup_cpu_physics(self.stage, "/World/PhysicsScene")