import omni
from pxr import Gf, UsdGeom
from omni.isaac.range_sensor import _range_sensor
from SensorModels.IsaacSensors import PerfectPoseSensor
from USVConfigurations.HeronSettings import HeronHydroSettings, HeronThrustersSettings
import RLEnvironments.IsaacUtils as utils
import numpy as np

class BaseRobotWrapper:
    def __init__(self, stage):
        from omni.physx import get_physx_interface
        from omni.isaac.dynamic_control import _dynamic_control
        self.nucleus_server = utils.get_nucleus_server()
        self.stage = stage
        self.PhysXIFace = get_physx_interface()
        self._dynamic_control = _dynamic_control
        self.DCIFace = _dynamic_control.acquire_dynamic_control_interface()

    def loadPlugins(self):
        # Load plugins after kit is loaded
        # If no plugins, then replace the raise by a pass
        raise NotImplementedError

    def spawn(self, position):
        # The function used to spawn the robot
        raise NotImplementedError

    def teleport(self, position, rotation, settle=False):
        # The function to teleport the robot
        raise NotImplementedError

    def physicsCallback(self, dt):
        # The function to apply when updating the physics.
        # Use pass if not needed, keep the dt in the header.
        raise NotImplementedError
    
    def updateCommands(self, data):
        # The function to update the values of the command
        raise NotImplementedError

    def getObservation(self):
        # The function to get observations from the robot.
        raise NotImplementedError