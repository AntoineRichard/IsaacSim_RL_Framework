"""
 Porting of the UUV Plugins to Isaac : https://github.com/uuvsimulator
 Some functions have been changed to better fit the use of Pythons.
 2021 DreamLab  - Georgia Institute of Technology - IRL 2958 GT CNRS
 Antoine Richard: antoine.richard@gatech.edu
"""

import numpy as np

heron_length=1.35
heron_width=1.0
heron_height=0.32
heron_cob = np.array([0,0,0])

HeronHydroSettings = {
    "fluid_density":1028,
    "link": [{
        "name": "/heron/base_link", #RigidBody path
        "volume":0.13,
        "box":np.array([heron_length, heron_width, heron_height]),
        "center_of_buoyancy":heron_cob,
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
    }]
}

HeronThrustersSettings = [
    {
        "linkName":"/heron/thruster_0", #RigidBody path
        "jointName":"/heron/base_link/thruster_0_joint", #Joint path
        "thruster_id":"thruster_0",
        "gain":1,
        "clampMax":100,
        "clampMin":-100,
        "thrustMax":35,
        "thrustMin":-35,
        "thrust_efficiency":1,
        "propeller_efficiency":1,
        "dynamics":{
            "type":"FirstOrder",
            "timeConstant":0.05
        },
        "conversion":{
            "type":"LinearInterp",
            "inputValues": np.array([-1.0,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0]),
            "outputValues": np.array([-19.88,-16.52,-12.6,-5.6,-1.4,0.0,2.24,9.52,21.28,28.0,33.6])
        }
    },{
        "linkName":"/heron/thruster_1", #RigidBody path
        "jointName":"/heron/base_link/thruster_1_joint", #Joint path
        "thruster_id":"thruster_1",
        "gain":1,
        "clampMax":100,
        "clampMin":-100,
        "thrustMax":35,
        "thrustMin":-35,
        "thrust_efficiency":1,
        "propeller_efficiency":1,
        "dynamics":{
            "type":"FirstOrder",
            "timeConstant":0.05
        },
        "conversion":{
            "type":"LinearInterp",
            "inputValues": np.array([-1.0,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1.0]),
            "outputValues": np.array([-19.88,-16.52,-12.6,-5.6,-1.4,0.0,2.24,9.52,21.28,28.0,33.6])
        }
    }
]