"""
 Porting of the UUV Plugins to Isaac : https://github.com/uuvsimulator
 Some functions have been changed to better fit the use of Pythons.
 Some functions have been changed to be more accurate.
 2021 DreamLab  - Georgia Institute of Technology - IRL 2958 GT CNRS
 Antoine Richard: antoine.richard@gatech.edu
"""

import PhysxUtils as utils
import math


class Dynamics:
    def __init__(self):
        self._prevTime = None
        self._state = None
        self.Reset()

    def update(self, cmd, t):
      raise NotImplementedError()

    def Reset(self):
        self._prevTime = -10.
        self._state = 0.
      

class DynamicsZeroOrder(Dynamics):
    def __init__(self, settings):
        super().__init__()
    
    def update(self, cmd, t):
        return cmd


class DynamicsFirstOrder(Dynamics):
    def __init__(self, settings):
        super().__init__()
        if "timeConstant" not in settings.keys():
            utils.Print("DynamicsFirstOrder: expected element time_constant")
            exit(1)
        self.tau = settings["timeConstant"]

    def update(self, cmd, t):
        if self._prevTime < 0:
            self._prevTime = t
            return self._state

        dt = t - self._prevTime

        alpha = math.exp(-dt/self.tau)
        self._state = self._state*alpha + (1.0 - alpha)*cmd

        self._prevTime = t

        return self._state


class ThrusterDynamicsYoerger(Dynamics):
    def __init__(self, settings):
        super().__init__()
        if "alpha" not in settings.keys():
            utils.Print("ThrusterDynamicsYoerger: expected element alpha")
            raise ValueError()
        self._alpha = settings["alpha"]
        if "beta" not in settings.keys():
            utils.Print("ThrusterDynamicsYoerger: expected element beta")
            raise ValueError()
        self._beta = settings["beta"]

    def update(self, cmd, t):
        if self._prevTime < 0:
            self._prevTime = t
            return self._state

        dt = t - self._prevTime

        self._state += dt*(self._beta*cmd - self._alpha*self._state*abs(self._state))

        return self._state


class ThrusterDynamicsBessa(Dynamics):
    def __init__(self, settings):
        super().__init__()
        if "Jmsp" not in settings.keys():
            utils.Print("ThrusterDynamicsBessa: expected element Jmsp")
            raise ValueError()
        self._Jmsp = settings["Jmsp"]
        if "Kv1" not in settings.keys():
            utils.Print("ThrusterDynamicsBessa: expected element Kv1")
            raise ValueError()
        self._Kv1 = settings["Kv1"]
        if "Kv2" not in settings.keys():
            utils.Print("ThrusterDynamicsBessa: expected element Kv2")
            raise ValueError()
        self._Kv2 = settings["Kv2"]
        if "Kt" not in settings.keys():
            utils.Print("ThrusterDynamicsBessa: expected element Kt")
            raise ValueError()
        self._Kt = settings["Kt"]
        if "Rm" not in settings.keys():
            utils.Print("ThrusterDynamicsBessa: expected element Rm")
            raise ValueError()
        self._Rm = settings["Rm"]
    
    def update(self, cmd, t):
        if self._prevTime < 0:
            self._prevTime = t
            return self._state

        dt = t - self._prevTime
        
        self._state += dt*(cmd*self._Kt/self._Rm - self._Kv1*self._state
                      - self._Kv2*self._state*math.abs(self._state))/self._Jmsp

        return self._state


ThrusterDynamicsMap = {"ZeroOrder":DynamicsZeroOrder, "FirstOrder":DynamicsFirstOrder, "Yoerger":ThrusterDynamicsYoerger, "Bessa":ThrusterDynamicsBessa}