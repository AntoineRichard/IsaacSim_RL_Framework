"""
 Porting of the UUV Plugins to Isaac : https://github.com/uuvsimulator
 Some functions have been changed to better fit the use of Pythons.
 Some functions have been changed to be more accurate.
 2021 DreamLab  - Georgia Institute of Technology - IRL 2958 GT CNRS
 Antoine Richard: antoine.richard@gatech.edu
"""
import numpy as np
import math

import BuoyancyPhysics.ThrusterDynamics as ThrusterDynamics
import BuoyancyPhysics.ThrusterConversion as ThrusterConversion
import BuoyancyPhysics.PhysxUtils as utils

class ThrusterPlugin():
    def __init__(self, stage, PhysXIFace, DCIFace):
        self._inputCommand = 0
        self._clampMin = -1e16
        self._clampMax = 1e16
        self._thrustMin = 1e16
        self._thrustMax = 1e6
        self._gain = 1.0
        self._isOn = True
        self._thrustEfficiency = 1.0
        self._propellerEfficiency = 1.0
        self._thrusterID = -1
        self.stage = stage
        self._PhysXIFace = PhysXIFace
        self._DCIFace = DCIFace


    def Load(self, settings):
        # Link
        utils.Assert("linkName" in settings, "Could not find linkName.")
        self._thrusterLink = settings["linkName"]
        self._rigid_body_handle = self._DCIFace.get_rigid_body(self._thrusterLink)
        # Thruster dynamics
        utils.Assert("dynamics" in settings.keys(), "Could not find dynamics.")
        self._thrusterDynamics = ThrusterDynamics.ThrusterDynamicsMap[settings["dynamics"]["type"]](settings["dynamics"])
        # Thrust conversion function
        utils.Assert("conversion" in settings.keys(), "Could not find conversion.")
        self._conversionFunction = ThrusterConversion.ThrusterConversionMap[settings["conversion"]["type"]](settings["conversion"])
        # Clamping interval
        if "clampMin" in settings.keys():
            self._clampMin = settings["clampMin"]
        if "clampMax" in settings.keys():
            self._clampMax = settings["clampMax"]
        if self._clampMin >= self._clampMax:
            utils.print("clampMax must be greater than clampMin, returning to default values...")
            self._clampMin = -1e16
            self._clampMax = 1e16
        # Thrust force interval
        if "thrustMin" in settings.keys():
            self._thrustMin = settings["thrustMin"]
        if "thrustMax" in settings.keys():
            self._thrustMax = settings["thrustMax"]
        if self._thrustMin >= self._thrustMax:
            utils.print("thrustMax must be greater than thrustMin, returning to default values...")
            self._thrustMin = -1e16
            self._thrustMax = 1e16
        # Gain (1.0 by default)
        if "gain" in settings.keys():
          self._gain = settings["gain"]

        if "thrust_efficiency" in settings.keys():
            self._thrustEfficiency = settings["thrust_efficiency"]
            if (self._thrustEfficiency < 0.0) or (self._thrustEfficiency > 1.0):
                utils.print("Invalid thrust efficiency factor, setting it to 100%")
                self._thrustEfficiency = 1.0
        if "propeller_efficiency" in settings.keys():
            self._propellerEfficiency = settings["propeller_efficiency"]
            if (self._propellerEfficiency < 0.0) or (self._propellerEfficiency > 1.0):      
                utils.print("Invalid propeller dynamics efficiency factor, setting it to 100%")
                self.propellerEfficiency = 1.0

        if "jointName" in settings:
            self._thrusterJoint = settings["jointName"]
            self._thrusterAxis = utils.getJointAxis(self.stage, self._thrusterJoint)

    def Reset(self):
        self._thrusterDynamics.Reset()

    def Update(self, dt):
        utils.Assert(not math.isnan(self._inputCommand), "nan in this->inputCommand")
        # Test if the thruster has been turned off
        if self._isOn:
            dynamicsInput = np.clip(self._inputCommand, self._clampMin, self._clampMax)*self._gain
        else:
            # In case the thruster is turned off in runtime, the dynamic state
            # will converge to zero
            dynamicsInput = 0.0
        dynamicState = self._propellerEfficiency * self._thrusterDynamics.update(dynamicsInput, dt)
        utils.Assert(not math.isnan(dynamicState), "Invalid dynamic state")
        # Multiply the output force magnitude with the efficiency
        self._thrustForce = self._thrustEfficiency * self._conversionFunction.convert(dynamicState)
        utils.Assert(not math.isnan(self._thrustForce), "Invalid thrust force")
        # Use the thrust force limits
        self._thrustForce = np.clip(self._thrustForce, self._thrustMin, self._thrustMax)
        force = self._thrusterAxis * self._thrustForce
        self._rigid_body_handle = self._DCIFace.get_rigid_body(self._thrusterLink)
        utils.AddRelativeForceDC(self._DCIFace, self._rigid_body_handle, force*100)
        #self.dof_ptr = self._DCIFace.get_articulation(self._thrusterJoint)
        #self._DCIFace.set_dof_velocity_target(self.dof_ptr, dynamicState*3.14*2)
        #if self._thrusterJoint:
        # Let joint rotate with correct angular velocity.
        #this->joint->SetVelocity(0, dynamicState);

    def UpdateCommand(self, value):
        self._inputCommand = value