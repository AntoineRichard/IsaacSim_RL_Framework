"""
 Porting of the UUV Plugins to Isaac : https://github.com/uuvsimulator
 Some functions have been changed to better fit the use of Pythons.
 Some functions have been changed to be more accurate.
 2021 DreamLab  - Georgia Institute of Technology - IRL 2958 GT CNRS
 Antoine Richard: antoine.richard@gatech.edu
"""

from BuoyantObject import BuoyantObject
import numpy as np
import PhysxUtils as utils

class HydrodynamicModel(BuoyantObject):
    def __init__(self, stage, prim_path, PhysxIFace, DCIFace, settings):
        super().__init__(stage, prim_path, PhysxIFace, DCIFace)
        # List of parameters needed
        self._params = []
        # Reynolds number (not used by all models)
        self._Re = 0.0
        # Temperature (not used by all models)
        self._temperature = 0.0
        self.buildHM(settings)
        self.Reset()
    
    def Reset(self):
        self._filtered_acc = np.zeros([6])
        self._last_time = - 10.0
        self._last_vel_rel = np.zeros([6])

    def buildHM(self, settings):
        if "volume" in settings.keys():
            self._volume = settings["volume"]
        if (("metacentric_width" in settings.keys()) and \
            ("metacentric_length" in settings.keys()) and \
            ("submerged_height" in settings.keys())):
            self._metacentric_width = settings["metacentric_width"]
            self._metacentric_length = settings["metacentric_length"]
            self._submerged_height = settings["submerged_height"]
            self._is_surface_vessel = True
            utils.Print("Surface vessel parameters")
            utils.Print("\tMetacentric width [m] = " + str(self._metacentric_width))
            utils.Print("\tMetacentric length [m] = " + str(self._metacentric_length))
            utils.Print("\tSubmerged height [m] = " + str(self._submerged_height))
        else:
            self.metacentric_width = 0.0
            self.metacentric_length = 0.0
            self.water_level_plane_area = 0.0
            self.is_surface_vessel = False
        if "center_of_buoyancy" in settings.keys():
            self.SetCoB(settings["center_of_buoyancy"])
        if "box" in settings.keys():
            self.SetBoundingBox(settings["box"])
        if "neutrally_buoyant" in settings.keys():
            if settings["neutrally_buoyant"]:
                self.SetNeutrallyBuoyant()

    def GetType(self):
        raise NotImplementedError()

    def ApplyHydrodynamicForces(self, time, flowVelWorld):
        raise NotImplementedError()

    def Print(paramName, message):
        raise NotImplementedError()

    def GetParam(tag):
        raise NotImplementedError()

    def SetParam(tag, input):
        raise NotImplementedError()

    def ComputeAcc(self, velRel, time, alpha):
        #Compute Fossen's nu-dot numerically. This is mandatory as Isaac does
        #not report accelerations
        if self._last_time < 0:
            self._last_time = time
            self._last_vel_rel = velRel
            return

        dt = time#time - self._last_time
        if dt <= 0.0:
            return

        acc = (velRel - self._last_vel_rel) / dt

        # TODO  We only have access to the acceleration of the previous simulation
        #       step. The added mass will induce a strong force/torque counteracting
        #       it in the current simulation step. This can lead to an oscillating
        #       system.
        #       The most accurate solution would probably be to first compute the
        #       latest acceleration without added mass and then use this to compute
        #       added mass effects. This is not how gazebo works, though.
        self._filtered_acc = (1.0 - alpha) * self._filtered_acc + alpha * acc
        self._last_time = time
        self._last_vel_rel = velRel.copy()

    @staticmethod
    def ToNED(vec):
        output = vec.copy()
        output[1] = -1 * vec[1]
        output[2] = -1 * vec[2]
        return output

    @staticmethod
    def FromNED(vec):
        output = vec.copy()
        output[1] = -1 * vec[1]
        output[2] = -1 * vec[2]
        return output

    def CheckParams(self, params):
        if not list(self._params.keys()):
            return True
        for tag in self._params.keys():
            if tag not in params:
                utils.Print("Hydrodynamic model: Expected element "+tag)
                return False
        return True

"""
class HydrodynamicModelFactory:
    def __init__(self):
        self._creators = {}

    def CreateHydrodynamicModel(self, params, prim):
        if "hydrodynamic_model" not in params.keys():
            raise ValueError("Hydrodynamic model is missing")
        else:
            hydroModel = params["hydrodynamic_model"]
        if "type" not in hydroModel.keys():
            utils.Print("Model has no type")
            return None
        else:
            identifier = hydroModel["type"]

        if identifier not in self._creators.keys():
            utils.Print("Cannot create HydrodynamicModel with unknown identifier: "+identifier)
            return None
        return self._creators[identifier](params, prim)

    @classmethod
    def GetInstance(cls):
        return cls

    def RegisterCreator(self, identifier, creator):
        if identifier in self._creators.keys():
            utils.Print("Warning: Registering HydrodynamicModel with identifier: twice")
        self.creators[identifier] = creator    
        utils.Print("Registered HydrodynamicModel type " + identifier)
        return True
"""