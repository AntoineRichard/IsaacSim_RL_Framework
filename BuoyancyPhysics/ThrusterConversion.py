"""
 Porting of the UUV Plugins to Isaac : https://github.com/uuvsimulator
 Some functions have been changed to better fit the use of Pythons.
 Some functions have been changed to be more accurate.
 2021 DreamLab  - Georgia Institute of Technology - IRL 2958 GT CNRS
 Antoine Richard: antoine.richard@gatech.edu
"""

from scipy.interpolate import interp1d
import BuoyancyPhysics.PhysxUtils as utils


class ConversionFunction:
    def __init__(self):
        self._type = None
    
    def GetParam(self, tag):
        if "_"+tag in self.__dict__:
            value = self.__getattribute__("_"+tag)
            utils.Print(self._type+"::GetParam < "+tag+" > = " + str(value))
            return value
        else:
            return False


class ConversionFunctionBasic(ConversionFunction):
    def __init__(self, settings):
        super().__init__()
        self._type = "ConversionFunctionBasic"
        if "rotorConstant" not in settings:
            utils.Print("ConversionFunctionBasic::Expected element rotorConstant")
            raise ValueError()
        self._rotorConstant = settings["rotorConstant"]
        utils.Print("ConversionFunctionBasic::Create conversion function")
        utils.Print("    rotorConstant: " + str(self._rotorConstant))

    def convert(self, cmd):
        return self._rotorConstant*abs(cmd)*cmd


class ConversionFunctionBessa(ConversionFunction):
    def __init__(self, settings):
        super().__init__()
        self._type = "ConversionFunctionBessa"
        if "rotorConstantL" not in settings.keys():
          utils.Print("ConversionFunctionBasic: expected element rotorConstantL")
        self._rotorConstantL = settings["rotorConstantL"]
        if "rotorConstantR" not in settings.keys():
          utils.Print("ConversionFunctionBasic::Expected element rotorConstantR")
        self._rotorConstantR = settings["rotorConstantR"]
        if "deltaL" not in settings.keys():
          utils.Print("ConversionFunctionBasic::Expected element deltaL")
        self._deltaL = settings["deltaL"]
        if "deltaR" not in settings.keys():
          utils.Print("ConversionFunctionBasic::Expected element deltaR")
        self._deltaR = settings["deltaR"]

        utils.Assert(self._rotorConstantL >= 0.0, "ConversionFunctionBessa: rotorConstantL should be >= 0")
        utils.Assert(self._rotorConstantR >= 0.0, "ConversionFunctionBessa: rotorConstantR should be >= 0")
        utils.Assert(self._deltaL <= 0.0, "ConversionFunctionBessa: deltaL should be <= 0")
        utils.Assert(self._deltaR >= 0.0, "ConversionFunctionBessa: deltaR should be >= 0")

        utils.Print("ConversionFunctionBessa:")
        utils.Print("    rotorConstantL: " + str(self._rotorConstantL))
        utils.Print("    rotorConstantR: " + str(self._rotorConstantR))
        utils.Print("    deltaL: " + str(self._deltaL))
        utils.Print("    deltaR: " + str(self._deltaR))

    def convert(self, cmd):
        basic = cmd*abs(cmd)
    
        if basic <= self._deltaL:
            return self._rotorConstantL*(basic - self._deltaL)
        elif (basic >= self._deltaR):
          return self._rotorConstantR*(basic - self._deltaR)
        else:
          return 0


class ConversionFunctionLinearInterp(ConversionFunction):
    def __init__(self, settings):
        super().__init__()
        self._type = "ConversionFunctionLinearInterp"

        if "inputValues" not in settings.keys():
            utils.Print("ConversionFunctionLinearInterp::Expected element inputValues")
            raise ValueError()

        if "outputValues" not in settings.keys():
            utils.Print("ConversionFunctionLinearInterp::Expected element outputValues")
            raise ValueError()

        x = settings["inputValues"]
        y = settings["outputValues"]

        self.BuildInterpolator(x,y)

    def convert(self, cmd):
        return self._interpolator(cmd)

    def BuildInterpolator(self, x, y):
        utils.Assert(x.shape[0] == y.shape[0], "input and output do not match")
        self._interpolator = interp1d(x,y, kind='linear', bounds_error=False, fill_value=(y[0],y[-1]))

    def GetTable(self):
        return self._interpolator

class ConversionFunctionCubicInterp(ConversionFunctionLinearInterp):
    def __init__(self, settings):
        super().__init__()
        self._type = "ConversionFunctionCubicInterp"

    def BuildInterpolator(self, x, y):
        utils.Assert(x.shape[0] == y.shape[1], "input and output do not match")
        self._interpolator = interp1d(x,y, kind='cubic', bounds_error=False, fill_value=(y[0],y[-1]))


ThrusterConversionMap = {"Basic":ConversionFunctionBasic, "Bessa":ConversionFunctionBessa, "LinearInterp":ConversionFunctionLinearInterp, "CubicInterp":ConversionFunctionCubicInterp}