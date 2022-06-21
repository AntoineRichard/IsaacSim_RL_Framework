from pxr import Gf
import RLEnvironments.IsaacUtils as utils

class BaseWorld:
    def __init__(self, stage, world, **kwargs):
        self.stage = stage
        self.world = world
        self.nucleus_server = utils.get_nucleus_server()
        self.setupPhysics()
        self.loadWorld(**kwargs)
        self.world.step()
        print("Waiting for materials to load...")
    
    def loadWorld(self, **kwargs):
        # Function to load the world.
        # Use named arguments in this function's header.
        raise NotImplementedError
    
    def reset(self):
        # Function to reset the world.
        # If not used replace the raise with a pass. 
        raise NotImplementedError

    def setupPhysics(self):
        utils.setup_cpu_physics(self.stage, "/World/PhysicsScene")
