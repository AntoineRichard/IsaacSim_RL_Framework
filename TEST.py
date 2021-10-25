"""
 Porting of the UUV Plugins to Isaac : https://github.com/uuvsimulator
 Some functions have been changed to better fit the use of Pythons.
 2021 DreamLab  - Georgia Institute of Technology - IRL 2958 GT CNRS
 Antoine Richard: antoine.richard@gatech.edu
"""

import numpy as np
import unittest
import math

import ThrusterConversion
import ThrusterDynamics
from BuoyantObject import BuoyantObject
import PhysxUtils as utils
from HydrodynamicModel import HydrodynamicModel
import HMFossenModels
import UnderWaterObject

basic={"conversion":{
    "type":"Basic",
    "rotorConstant":0.0049
    }
}

bessa={"conversion":{
    "type":"Bessa",
    "rotorConstantL":0.001,
    "rotorConstantR":0.002,
    "deltaL":-50,
    "deltaR":25
    }
}

interp_linear = {"conversion":{
    "type":"LinearInterp",
    "inputValues":np.array([-5.0,0,2.0,5.0]),
    "outputValues":np.array([-100,-10,20,120]),
    "alpha":np.array([0.1,0.5,0.9])
    }
}

zero_order = {"dynamics":{
    "type":"ZeroOrder"
    }
}

first_order = {"dynamics":{
    "type":"FirstOrder",
    "timeConstant":0.5
    }
}

yoerger_dyn = {"dynamics":{
    "type":"Yoerger",
    "alpha":0.5,
    "beta":0.5
    }
} 

bessa_dyn = {"dynamics":{
    "type":"Bessa",
    "Jmsp":0.5,
    "Kv1":0.5,
    "Kv2":0.5,
    "Kt":0.5,
    "Rm":0.5
    }
}

hm_init_usv = {
    "name": "base_link",
    "volume":0.10,
    "box":np.array([1.0, 0.5, 0.25]),
    "center_of_buoyancy":np.array([0,0,0]),
    "metacentric_width":0.1,
    "metacentric_length":0.1,
    "submerged_height":0.02,
    "hydrodynamic_model":{
        "type": "fossen",
        "added_mass": np.zeros([6,6]),
        "offset_added_mass":0.0,
        "linear_damping":np.eye(6)*np.array([-16.44998712, -15.79776044, -100,-13,-13, -6]),
        "quadratic_damping":np.eye(6)*np.array([-2.942, -2.7617212, -10, -5, -5, -5]),
        "linear_damping_forward_speed":np.eye(6)*np.zeros([6]),
        "offset_linear_damping":0.0,
        "offset_lin_forward_damping_speed":0.0,
        "offset_nonlin_damping":0.0,
        "scaling_damping":1.0
    }
}

hm_init_usv2 = {
    "name": "base_link",
    "volume":0.10,
    "box":np.array([1.0, 0.5, 0.25]),
    "center_of_buoyancy":np.array([0,0,0]),
    "metacentric_width":0.1,
    "metacentric_length":0.1,
    "submerged_height":0.02,
    "hydrodynamic_model":{
        "type": "fossen",
        "added_mass": np.ones([6,6])*0.5,
        "offset_added_mass":0.5,
        "linear_damping":np.eye(6)*np.array([-16.44998712, -15.79776044, -100,-13,-13, -6]),
        "quadratic_damping":np.eye(6)*np.array([-2.942, -2.7617212, -10, -5, -5, -5]),
        "linear_damping_forward_speed":np.eye(6)*np.zeros([6]),
        "offset_linear_damping":0.0,
        "offset_lin_forward_damping_speed":0.0,
        "offset_nonlin_damping":0.0,
        "scaling_damping":1.0
    }
}

HeronHydroSettings = {
    "fluid_density":1028,
    "link": [{
        "name": "dummy_link", #RigidBody path
        "volume":0.13,
        "box":np.array([1, 1, 1]),
        "center_of_buoyancy":np.array([0,0,0]),
        "metacentric_width":0.1,
        "metacentric_length":0.1,
        "metacentric_height":0.02,
        "hydrodynamic_model":{
            "type": "fossen",
            "added_mass": np.zeros([6,6]),
            "offset_added_mass":0.0,
            "linear_damping":np.eye(6)*np.array([-16.44998712, -15.79776044, -100,-13,-13, -6]),
            "quadratic_damping":np.eye(6)*np.array([-2.942, -2.7617212, -10, -5, -5, -5]),
            "linear_damping_forward_speed":np.eye(6)*np.zeros([6]),
            "offset_linear_damping":0.0,
            "offset_lin_forward_damping_speed":0.0,
            "offset_nonlin_damping":0.0,
            "scaling_damping":1.0
        }
    }]
}

hm_init_rov = {
}

class TestThrusterConversionBasic(unittest.TestCase):
    def test_ThrusterConversion_Basic_getParam(self):
        TC = ThrusterConversion.ThrusterConversionMap["Basic"](basic["conversion"])
        self.assertEqual(TC._rotorConstant, 0.0049)
        self.assertEqual(TC.GetParam("rotorConstant"), 0.0049)

    def test_ThrusterConversion_Basic_convert(self):
        TC = ThrusterConversion.ThrusterConversionMap["Basic"](basic["conversion"])
        self.assertEqual(TC.convert(0), 0)
        self.assertEqual(TC.convert(50), 50*50*0.0049)
        self.assertEqual(TC.convert(-50), -50*50*0.0049)

class TestThrusterConversionBessa(unittest.TestCase):
    def test_ThrusterConversion_Bessa_getParam(self):
        TC = ThrusterConversion.ThrusterConversionMap["Bessa"](bessa["conversion"])
        self.assertEqual(TC._rotorConstantL, 0.001)
        self.assertEqual(TC._rotorConstantR, 0.002)
        self.assertEqual(TC._deltaL, -50)
        self.assertEqual(TC._deltaR, 25)
        self.assertEqual(TC.GetParam("rotorConstantL"), 0.001)
        self.assertEqual(TC.GetParam("rotorConstantR"), 0.002)
        self.assertEqual(TC.GetParam("deltaL"), -50)
        self.assertEqual(TC.GetParam("deltaR"), 25)

    def test_ThrusterConversion_Bessa_convert_DeadZones(self):
        TC = ThrusterConversion.ThrusterConversionMap["Bessa"](bessa["conversion"])
        self.assertEqual(0.0, TC.convert(0.0))
        self.assertEqual(0.0, TC.convert(math.sqrt(25) - 1e-6))
        self.assertEqual(0.0, TC.convert(-math.sqrt(50) + 1e-6))
    
    def test_ThrusterConversion_Bessa_convert(self):
        TC = ThrusterConversion.ThrusterConversionMap["Bessa"](bessa["conversion"])
        cmdl = -50.0
        cmdr =  50.0
        self.assertEqual(0.001*(cmdl*abs(cmdl)+50), TC.convert(cmdl))
        self.assertEqual(0.002*(cmdr*abs(cmdr)-25), TC.convert(cmdr))

class TestThrusterConversionInterpLinear(unittest.TestCase):
    def test_ThrusterConversion_InterpLinear_sanityCheck(self):
        TC = ThrusterConversion.ThrusterConversionMap["LinearInterp"](interp_linear["conversion"])
        self.assertFalse(np.sum(TC.convert(interp_linear["conversion"]["inputValues"]) != interp_linear["conversion"]["outputValues"]))
    
    def test_ThrusterConversion_InterpLinear_checkOutside(self):
        TC = ThrusterConversion.ThrusterConversionMap["LinearInterp"](interp_linear["conversion"])
        self.assertEqual(interp_linear["conversion"]["outputValues"][0], TC.convert(interp_linear["conversion"]["inputValues"][0] - 0.5))
        self.assertEqual(interp_linear["conversion"]["outputValues"][-1], TC.convert(interp_linear["conversion"]["inputValues"][-1] + 0.5))

    def test_ThrusterConversion_InterpLinear_checkInterp(self):
        TC = ThrusterConversion.ThrusterConversionMap["LinearInterp"](interp_linear["conversion"])
        in_ = interp_linear["conversion"]["inputValues"][:-1]
        out_ = interp_linear["conversion"]["outputValues"][:-1]
        alpha = interp_linear["conversion"]["alpha"]
        for i in range(alpha.shape[0]-1):
            in1  = alpha[i]*in_[i] + (1-alpha[i])*in_[i+1] 
            out1 = alpha[i]*out_[i] + (1-alpha[i])*out_[i+1]
            self.assertAlmostEqual(out1, TC.convert(in1), 7)

class TestThrusterDynamicsZeroOrder(unittest.TestCase):
    def test_ThrusterDynamics_ZeroOrder_Update(self):
        TD = ThrusterDynamics.ThrusterDynamicsMap["ZeroOrder"](zero_order["dynamics"])
        self.assertEqual(TD.update(10.0,0.0), 10.0)
        self.assertEqual(TD.update(20.0,0.2), 20.0)

class TestThrusterDynamicsFirstOrder(unittest.TestCase):
    def test_ThrusterDynamics_FirstOrder_Update(self):
        TD = ThrusterDynamics.ThrusterDynamicsMap["FirstOrder"](first_order["dynamics"])
        self.assertEqual(TD.update(0.0,0.0), 0.0)
        self.assertAlmostEqual(TD.update(1.0,0.5),1-0.36787944, 5)

class TestThrusterDynamicsYoerger(unittest.TestCase):
    def test_ThrusterDynamics_YoergerDynamics_Update(self):
        TD = ThrusterDynamics.ThrusterDynamicsMap["Yoerger"](yoerger_dyn["dynamics"])
        self.assertEqual(TD.update(0.0,0.0), 0.0)

class TestThrusterDynamicsBessa(unittest.TestCase):
    def test_ThrusterDynamics_BessaDynamics_Update(self):
        TD = ThrusterDynamics.ThrusterDynamicsMap["Bessa"](bessa_dyn["dynamics"])
        self.assertEqual(TD.update(0.0,0.0), 0.0)

class TestUtilsQuat2Rot(unittest.TestCase):
    def test_Utils_Quaternion2RotationMatrix(self):
        q = np.array([0,0,0,1])
        R = utils.Quaternion2RotationMatrix(q)
        np.testing.assert_array_equal(R,np.eye(3))

    def test_Utils_Quat2RPY(self):
        q = np.array([0,0,0,1])
        r,p,y = utils.Q2RPY(q)
        self.assertEqual(r,0)
        self.assertEqual(p,0)
        self.assertEqual(y,0)

class TestBoyantObject(unittest.TestCase):
    def test_BuoyantObject_init(self):
        BO = BuoyantObject(None,None,None)
        self.assertTrue(True)

    def test_BuoyantObject_GetSetNeutrallyBuoyant(self):
        BO = BuoyantObject(None,None,None)
        BO.SetFluidDensity(1028.0)
        BO._mass = 20.0
        self.assertFalse(BO.IsNeutrallyBuoyant())
        BO.SetNeutrallyBuoyant()
        self.assertTrue(BO.IsNeutrallyBuoyant())
        self.assertEqual(BO.GetVolume(), 20.0 / 1028.0)
    
    def test_BuoyantObject_GetSetNeutrallyBuoyant(self):
        BO = BuoyantObject(None,None,None)
        BO.SetBoundingBox(np.array([2,2,2]))
        self.assertFalse(np.sum(BO._bounding_box != np.array([2,2,2])))

    def test_BuoyantObject_SetGetVolume(self):
        BO = BuoyantObject(None,None,None)
        BO.SetVolume(20.)
        self.assertEqual(BO.GetVolume(), 20.0)
        BO._offset_volume = 2.0
        BO._scaling_volume = 1.5
        self.assertEqual(BO.GetVolume(), 1.5*(20 + 2.0))

    def test_BuoyantObject_SetGetFluidDensity(self):
        BO = BuoyantObject(None,None,None)
        BO.SetFluidDensity(1028.0)
        self.assertEqual(1028.0, BO.GetFluidDensity())

    def test_BuoyantObject_SetGetCoB(self):
        BO = BuoyantObject(None,None,None)
        BO.SetCoB(np.array([2,1,3]))
        self.assertFalse(np.sum(np.array([2,1,3]) != BO.GetCoB()))

    def test_BuoyantObject_SetGetGravity(self):
        BO = BuoyantObject(None,None,None)
        BO.SetGravity(981)
        self.assertEqual(981, BO.GetGravity())

    def test_BuoyantObject_IsSubmerged(self):
        BO = BuoyantObject(None,None,None)
        BO.SetBoundingBox(np.array([2,2,2]))
        self.assertFalse(BO.IsSubmerged())
        BO.GetBuoyancyForce(np.array([0,0,0]),np.array([1,0,0,0]))
        self.assertFalse(BO.IsSubmerged())
        BO.GetBuoyancyForce(np.array([0,0,-2]),np.array([1,0,0,0]))
        self.assertTrue(BO.IsSubmerged())
        BO = BuoyantObject(None,None,None)
        BO.SetBoundingBox(np.array([2,2,2]))
        BO._is_surface_vessel = True
        self.assertFalse(BO.IsSubmerged())
        BO.GetBuoyancyForce(np.array([0,0,0]),np.array([1,0,0,0]))
        self.assertFalse(BO.IsSubmerged())
        BO.GetBuoyancyForce(np.array([0,0,-2]),np.array([1,0,0,0]))
        self.assertTrue(BO.IsSubmerged())

    def test_BuoyantObject_GetBuoyancyForce(self):
        BO = BuoyantObject(None,None,None)
        BO.SetBoundingBox(np.array([2,2,2]))
        BO._mass = 20.0
        BO.SetFluidDensity(1028.0)
        BO.SetVolume(0.3)
        BO.SetGravity(9.81)
        Force, Torque = BO.GetBuoyancyForce(np.array([0,0,1]),np.array([1,0,0,0]))
        np.testing.assert_array_equal(Force,np.array([0,0,0]))
        np.testing.assert_array_equal(Torque,np.array([0,0,0]))
        Force, Torque = BO.GetBuoyancyForce(np.array([0,0,-2]),np.array([1,0,0,0]))
        np.testing.assert_array_equal(Force,np.array([0,0,0.3 * 1028 * 9.81]))
        np.testing.assert_array_equal(Torque,np.array([0,0,0]))
        Force, Torque = BO.GetBuoyancyForce(np.array([0,0,0]),np.array([1,0,0,0]))
        np.testing.assert_array_equal(Force,np.array([0,0,0.5*0.3 * 1028 * 9.81]))
        np.testing.assert_array_equal(Torque,np.array([0,0,0]))
        BO._is_surface_vessel = True
        Force, Torque = BO.GetBuoyancyForce(np.array([0,0,1]),np.array([1,0,0,0]))
        np.testing.assert_array_equal(Force,np.array([0,0,0]))
        np.testing.assert_array_equal(Torque,np.array([0,0,0]))
        wLPA = 2*2
        Force, Torque = BO.GetBuoyancyForce(np.array([0,0,0]),np.array([1,0,0,0]))
        volume = 1 * wLPA
        np.testing.assert_array_equal(Force,np.array([0,0,volume*1028*9.81]))
        np.testing.assert_array_equal(Torque,np.array([0,0,0]))
        Force, Torque = BO.GetBuoyancyForce(np.array([0,0,-1]),np.array([1,0,0,0]))
        volume = 2 * wLPA
        np.testing.assert_array_equal(Force,np.array([0,0,volume*1028*9.81]))
        np.testing.assert_array_equal(Torque,np.array([0,0,0]))
        #TODO TEST TORQUE

class TestHydrodynamicModel(unittest.TestCase):
    def test_HydrodynamicModel_init(self):
        HM = HydrodynamicModel(None, None, None, hm_init_usv)
        self.assertTrue(True)

    def test_HydrodynamicModel_NED(self):
        HM = HydrodynamicModel(None, None, None, hm_init_usv)
        x = HM.ToNED(np.array([1,1,1]))
        x2 = HM.FromNED(x)
        np.testing.assert_array_equal(x,np.array([1,-1,-1]))
        np.testing.assert_array_equal(x2,np.array([1,1,1]))

    def test_HydrodynamicModel_ComputeAcc(self):
        HM = HydrodynamicModel(None, None, None, hm_init_usv)
        HM.Reset()
        HM.ComputeAcc(np.array([1,1,1,1,1,1]), 0, 0.3)
        np.testing.assert_array_equal(HM._filtered_acc, np.zeros(6))
        HM.ComputeAcc(np.array([2,2,2,2,2,2]), 1, 0.3)
        np.testing.assert_array_equal(HM._filtered_acc, np.ones(6)*0.3)
        HM.ComputeAcc(np.array([3,3,3,3,3,3]), 2, 0.3)
        np.testing.assert_array_equal(HM._filtered_acc, np.ones(6)*0.3*0.7 + np.ones(6)*0.3)

class TestHydrodynamicModelFossen(unittest.TestCase):
    def test_HydrodynamicModelFossen_init(self):
        HMF = HMFossenModels.HydroModelMap["fossen"](None, None, None, hm_init_usv)

    def test_HydrodynamicModelFossen_GetParam(self):
        HMF = HMFossenModels.HydroModelMap["fossen"](None, None, None, hm_init_usv)
        params = ["added_mass",
        "linear_damping",
        "linear_damping_forward_speed",
        "quadratic_damping",
        "center_of_buoyancy",
        "volume",
        "scaling_volume",
        "scaling_added_mass",
        "fluid_density",
        "offset_volume",
        "offset_added_mass",
        "offset_linear_damping",
        "offset_lin_forward_speed_damping",
        "offset_nonlin_damping"]
        for i in params:
            print(i)
            P = HMF.GetParam(i)
            if not(type(P) is np.ndarray):
                if (type(P) is bool) and (not P):
                    self.assertTrue(False)

    def test_HydrodynamicModelFossen_SetParam(self):
        HMF = HMFossenModels.HydroModelMap["fossen"](None, None, None, hm_init_usv)
        params = ["scaling_volume",
        "scaling_added_mass",
        "scaling_damping",
        "fluid_density",
        "offset_volume",
        "offset_added_mass",
        "offset_linear_damping",
        "offset_lin_forward_speed_damping",
        "offset_nonlin_damping"]
        for i in params:
            HMF.SetParam(i, 0)

    def test_HydrodynamicModelFossen_Print(self):
        HMF = HMFossenModels.HydroModelMap["fossen"](None, None, None, hm_init_usv)
        HMF.Print("all")
    
    def test_HydrodynamicModelFossen_GetAddedMass(self):
        HMF = HMFossenModels.HydroModelMap["fossen"](None, None, None, hm_init_usv)
        am = HMF.GetAddedMass()
        np.testing.assert_array_equal(am,np.zeros([6,6]))
        HMF = HMFossenModels.HydroModelMap["fossen"](None, None, None, hm_init_usv2)
        am = HMF.GetAddedMass()
        np.testing.assert_array_equal(am,np.ones([6,6])*0.5 + np.eye(6)*0.5)

    def test_HydrodynamicModelFossen_CrossProductOperator(self):
        HMF = HMFossenModels.HydroModelMap["fossen"](None, None, None, hm_init_usv)
        cp = HMF.CrossProductOperator([1,2,3])
        np.testing.assert_array_equal(cp, np.array([[0,-3,2],[3,0,-1],[-2,1,0]]))

    def test_HydrodynamicModelFossen_ComputeCoriolisMatrix(self):
        HMF = HMFossenModels.HydroModelMap["fossen"](None, None, None, hm_init_usv)
        HMF.ComputeAddedCoriolisMatrix([0,0,0,0,0,0])
        np.testing.assert_array_equal(HMF._Ca, np.zeros([6,6]))
        HMF = HMFossenModels.HydroModelMap["fossen"](None, None, None, hm_init_usv2)
        HMF.ComputeAddedCoriolisMatrix([0,0,0,0,0,0])
        np.testing.assert_array_equal(HMF._Ca, np.zeros([6,6]))
        HMF.ComputeAddedCoriolisMatrix([1.0,1.0,1.0,1.0,1.0,1.0])
        ca = np.array([
        [ 0. ,  0. ,  0. , -0. ,  3.5, -3.5],
        [ 0. ,  0. ,  0. , -3.5, -0. ,  3.5],
        [ 0. ,  0. ,  0. ,  3.5, -3.5, -0. ],
        [-0. ,  3.5, -3.5, -0. ,  3.5, -3.5],
        [-3.5, -0. ,  3.5, -3.5, -0. ,  3.5],
        [ 3.5, -3.5, -0. ,  3.5, -3.5, -0. ]])
        np.testing.assert_array_equal(HMF._Ca, ca)
        # TODO Double Check with their code
    
    def test_HydrodynamicModelFossen_ComputeDampingMatrix(self):
        HMF = HMFossenModels.HydroModelMap["fossen"](None, None, None, hm_init_usv)
        HMF.ComputeDampingMatrix(np.array([0,0,0,0,0,0]))
        np.testing.assert_array_equal(HMF._D,-hm_init_usv["hydrodynamic_model"]["linear_damping"])
        HMF.ComputeDampingMatrix(np.array([1,1,1,1,1,1]))
        np.testing.assert_array_equal(HMF._D,-hm_init_usv["hydrodynamic_model"]["linear_damping"]\
             - hm_init_usv["hydrodynamic_model"]["linear_damping_forward_speed"]\
             - hm_init_usv["hydrodynamic_model"]["quadratic_damping"])
    
    def test_HydrodynamicModelFossen_ComputeHydrodynamicForces(self):
        HMF = HMFossenModels.HydroModelMap["fossen"](None, None, None, hm_init_usv)
        #TODO do a unit test. Computing the output of this function is complicated.
        # This test makes sure that the function runs. Not that the output is correct.
        HMF.ComputeHydrodynamicForces(0, np.array([0,0,0]))
        HMF.ComputeHydrodynamicForces(1.0, np.array([0,0,0]))
        HMF.ComputeHydrodynamicForces(2.0, np.array([0,0,0]))
        HMF.ComputeHydrodynamicForces(3.0, np.array([0,0,0]))

class TestUnderWaterObject(unittest.TestCase):
    def test_UnderWaterObject_init(self):
        UWO = UnderWaterObject.UnderwaterObject(None,None)
    
    def test_UnderWaterObject_load(self):
        UWO = UnderWaterObject.UnderwaterObject(None,None)
        UWO.Load(HeronHydroSettings)

    def test_UnderWaterObject_updateFlowVel(self):
        UWO = UnderWaterObject.UnderwaterObject(None,None)
        UWO.Load(HeronHydroSettings)
        UWO.UpdateFlowVelocity(np.array([1,1,1]))

    def test_UnderWaterObject_update(self):
        UWO = UnderWaterObject.UnderwaterObject(None,None)
        UWO.Load(HeronHydroSettings)
        UWO.UpdateFlowVelocity(np.array([1,1,1]))
        UWO.Update(0)
        UWO.Update(1.0)
        UWO.Update(2.0)
        UWO.Update(3.0)

class TestPoweredUnderWaterObject(unittest.TestCase):
    def test_PoweredUnderWaterObject_init(self):
        UWO = UnderWaterObject.UnderwaterObject(None,None)

    def test_PoweredUnderWaterObject_updateFlowVel(self):
        UWO = UnderWaterObject.UnderwaterObject(None,None)
    
    def test_PoweredUnderWaterObject_updateThrustCMD(self):
        UWO = UnderWaterObject.UnderwaterObject(None,None)

    def test_PoweredUnderWaterObject_update(self):
        UWO = UnderWaterObject.UnderwaterObject(None,None)


if __name__ == '__main__':
    unittest.main()