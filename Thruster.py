"""
 Porting of the UUV Plugins to Isaac : https://github.com/uuvsimulator
 Some functions have been changed to better fit the use of Pythons.
 Some functions have been changed to be more accurate.
 2021 DreamLab  - Georgia Institute of Technology - IRL 2958 GT CNRS
 Antoine Richard: antoine.richard@gatech.edu
"""
import numpy as np
import math

import ThrusterDynamics
import ThrusterConversion
import PhysxUtils as utils

class ThrusterPlugin():
    def __init__(self):
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

    def load(self, settings):
        # Link
        utils.Assert("linkName" in settings, "Could not find linkName.")
        self._thrusterLink = utils.GetLink(settings["linkName"])
        # Thruster dynamics
        utils.Assert("dynamics" in settings.keys(), "Could not find dynamics.")
        self._thrusterDynamics = ThrusterDynamics.ThrusterDynamicsMap[settings["dynamics"]](settings["dynamics"])
        # Thrust conversion function
        utils.Assert("conversion" in settings.keys(), "Could not find conversion.")
        self._conversionFunction = ThrusterConversion.ThrusterConversionMap[settings["conversion"]](settings["conversion"])
        # Optional paramters:
        # Rotor joint, used for visualization if available.
        #if (_sdf->HasElement("jointName"))
        #  this->joint = _model->GetJoint(_sdf->Get<std::string>("jointName"));
        # Clamping interval
        if "clampMin" in settings.keys():
            self._clampMin = settings["clampMin"]
        if "clampMax" in settings.keys():
            self._clampMax = settings["clampMxa"]
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

        self._thrusterAxis = utils.getOrientation(self._thrusterLink)

    def Reset(self):
        self._thrusterDynamics.Reset()

    def Update(self, info):
        utils.Assert(not math.isnan(self._inputCommand), "nan in this->inputCommand")
        #double dynamicsInput;
        #double dynamicState;
        # Test if the thruster has been turned off
        if self._isOn:
            dynamicsInput = np.clip(self._inputCommand, self._clampMin, self._clampMax)*self._gain
        else:
            # In case the thruster is turned off in runtime, the dynamic state
            # will converge to zero
            dynamicsInput = 0.0
        dynamicState = self._propellerEfficiency * self._thrusterDynamics.update(dynamicsInput, info)
        utils.Assert(not math.isnan(dynamicState), "Invalid dynamic state")
        # Multiply the output force magnitude with the efficiency
        self._thrustForce = self._thrustEfficiency * self._conversionFunction.convert(dynamicState)
        utils.Assert(not math.isnan(self._thrustForce), "Invalid thrust force")
        # Use the thrust force limits
        self._thrustForce = np.clip(self._thrustForce, self._thrustMin, self._thrustMax)
        self._thrustForceStamp = info
        force = np.matmul(self._thrusterAxis,self._thrustForce)
        utils.AddRelativeForce(force)

        #if (this->joint)
        #{
        #  // Let joint rotate with correct angular velocity.
        #  this->joint->SetVelocity(0, dynamicState);
        #}

    def UpdateCommand(value):
        self._inputCommand = value