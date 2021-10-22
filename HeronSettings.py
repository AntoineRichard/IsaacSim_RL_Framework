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
        "name": "base_link",
        "volume":0.13,
        "box":np.array([heron_length, heron_width, heron_height]),
        "center_of_buoyancy":heron_cob,
        "metacentric_width":0.1,
        "metacentric_length":0.1,
        "metacentric_height":0.02,
        "hydrodynamic_model":{
            "type": "fossen",
            "added_mass": np.zeros([6,6]),
            "offset_added_mass":0.0,
            "linear_damping":np.eye(6)*np.array([-16.44998712, -15.79776044, -100,-13,-13, -6]),
            "quadratic_damping":np.eye(6)*np.array([-2.942, -2.7617212, -10, -5, -5, -5]),
            "linear_damping_forward_speed":np.zeros([6]),
            "offset_linear_damping":0.0,
            "offset_lin_forward_damping_speed":0.0,
            "offset_nonlin_damping":0.0,
            "scaling_damping":1.0
        }
    }]
}