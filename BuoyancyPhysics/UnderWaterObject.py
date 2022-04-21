"""
 Porting of the UUV Plugins to Isaac : https://github.com/uuvsimulator
 Some functions have been changed to better fit the use of Pythons.
 2021 DreamLab  - Georgia Institute of Technology - IRL 2958 GT CNRS
 Antoine Richard: antoine.richard@gatech.edu
"""

from BuoyancyPhysics.HydrodynamicModel import *
from BuoyancyPhysics.HMFossenModels import *
import BuoyancyPhysics.PhysxUtils as utils

class UnderWaterObject:
    def __init__(self, stage, PhysXIFace, DCIFace):        
        # Pairs of links & corresponding hydrodynamic models
        self._models = {}
        self._stage = stage
        self._PhysXIFace = PhysXIFace
        self._DCIFace = DCIFace
        # Flow velocity vector read from topic
        self._flowVelocity = np.zeros([3])
        # Name of vehicle's base_link
        self._baseLinkName = None
        # Flag to use the global current velocity or the individually
        # assigned current velocity
        self._useGlobalCurrent = True

    def Load(self, settings):
        # Get the fluid density, if present
        fluidDensity = 1028.0
        if "fluid_density" in settings.keys():
            fluidDensity = settings["fluid_density"]
        if "use_global_current" in settings.keys():
            self._useGlobalCurrent = settings["use_global_current"]
        self.baseLinkName = ""
        if "link" in settings:
            for linkSettings in settings["link"]:
                linkName = ""
                if "name" in linkSettings.keys():
                    linkName = linkSettings["name"]
                    found = linkName.split("base_link")
                    if len(found) > 1:
                        self._baseLinkName = linkName
                        utils.Print("Name of the BASE_LINK: " + self._baseLinkName)
                    else:
                        utils.Print("Name of RigidBody: " + linkName)
                    rigidBody = utils.getRigidBodyAtPath(self._stage, linkName)
                    if not rigidBody:
                        utils.Print("Specified link [" + linkName + "] not found.")
                        continue
                else:
                    utils.Print("Attribute name missing from link [" + linkName + "]")
                    continue

                # Creating a new hydrodynamic model for this link
                hydro = HydroModelMap[linkSettings["hydrodynamic_model"]["type"]](self._stage, linkName, self._PhysXIFace, self._DCIFace, linkSettings)
                gAcc = utils.getGravity(hydro._SceneAPI)
                hydro.SetFluidDensity(fluidDensity)
                hydro.SetGravity(gAcc)

                self._models[linkName] = hydro
                self._models[linkName].Print("all")

    def Update(self, time):
        for hydro in self._models.values():
            hydro.ApplyHydrodynamicForces(time, self._flowVelocity)


    def UpdateFlowVelocity(self, value):
        if self._useGlobalCurrent:
            self._flowVelocity = value
