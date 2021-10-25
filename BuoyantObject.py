"""
 Porting of the UUV Plugins to Isaac : https://github.com/uuvsimulator
 Some functions have been changed to better fit the use of Pythons.
 2021 DreamLab  - Georgia Institute of Technology - IRL 2958 GT CNRS
 Antoine Richard: antoine.richard@gatech.edu
"""

import numpy as np
import math

import PhysxUtils as utils

class BuoyantObject:
    def __init__(self, stage, prim_path, PhysXIFace, physics_path="/physicsScene"):
        # Build APIs
        self._RigidBodyAPI, self._MassAPI, self._SceneAPI = utils.getUsdPhysicsAPIs(stage, prim_path, physics_path)
        # Volume of fluid displaced by the submerged object
        self._volume = 0.0
        # Scaling factor for the volume
        self._scaling_volume = 1.0
        # Offset for the volume
        self._offset_volume = 0.0
        # Fluid density
        self._fluid_density = 1028.0
        # Acceleration of gravity
        self._g = 981.0
        # Center of buoyancy in the body frame
        self._center_of_buoyancy = np.zeros(3)
        # Is submerged flag
        self._is_submerged = False
        # Pointer to the correspondent robot link
        self._prim_path = prim_path
        self._stage = stage
        self._PhysXIFace = PhysXIFace
        # If true, the restoring force will be equal to the gravitational force
        self._neutrally_buoyant = False
        # Metacentric width of the robot, used only for surface vessels and floating objects
        self._metacentric_width = 0.0
        # Metacentric length of the robot, used only for surface vessels and floating objects
        self._metacentric_length = 0.0
        # If the cross section area around water level of the surface vessel
        # is not given, it will be computed from the object's bounding box
        self._water_level_plane_area = 0.0
        # Height of the robot that is submerged (only for surface vessels)
        self._submerged_height = 1e-9
        # Flag set to true if the information about the metacentric width and height is available
        self._is_surface_vessel = False
        # Flag set to true if the vessel has reached its submerged height
        self._is_surface_vessel_floating = False
        # TMP for calculation of the buoyancy force close to the surface
        self._bounding_box = np.array([1,1,1])#utils.getBoundingBox(self._stage, self._prim_path)
        self._height = self._bounding_box[2]
        self._mass = utils.getMass(self._MassAPI)

    def SetNeutrallyBuoyant(self):
        self._neutrally_buoyant = True
        # Calculate the equivalent volume for the submerged body
        # so that it will be neutrally buoyant
        self._volume = self._mass / self._fluid_density
        utils.Print(utils.getName(self._prim_path)+" is neutrally buoyant")

    def GetBuoyancyForce(self, pose, quat):
        z = pose[2]
        roll,pitch,yaw = utils.Q2RPY(quat)
        volume = 0.0 
        buoyancyForce = np.array([0, 0, 0])
        buoyancyTorque = np.array([0, 0, 0])
     
        if not self._is_surface_vessel:
            if ((z + self._height/2.0) > 0) and ((z - self._height/2.0) < 0):
                self._is_submerged = False
                volume = self.GetVolume() * (-z + self._height/2.0) / self._height
            elif (z + self._height/2.0) < 0:
                self._is_submerged = True
                volume = self.GetVolume()
            if (not self._neutrally_buoyant) or (volume != self._volume):
                buoyancyForce = np.array([0, 0, volume * self._fluid_density * self._g])
            elif self._neutrally_buoyant:
                buoyancyForce = np.array([0, 0, self._mass * self._g])
        else:
            """
            Implementation of the linear (small angle) theory for boxed-shaped
            vessels. Further details can be seen at
            T. I. Fossen, "Handbook of Marine Craft Hydrodynamics and Motion Control," Apr. 2011.
            Page 65
            The original code behaved weirdly during unit testing.
            The waterLevelPlaneArea is now computed using the first expression.
            """
            if self._water_level_plane_area <= 0:
                self._water_level_plane_area = self._bounding_box[0] * self._bounding_box[1]
                utils.Print(utils.getName(self._stage, self._prim_path)+" :: waterLevelPlaneArea = "+str(self._water_level_plane_area))
            #self._waterLevelPlaneArea = self._mass / (self._fluidDensity * self._submergedHeight)
     
            if z > (self._height/2.0):
                # Vessel is completely out of the water
                self._is_submerged = False
                return buoyancyForce, buoyancyTorque
            elif z < (-self._height/2.0):
                curSubmergedHeight = self._bounding_box[2]
                self._is_submerged = True
            else:
                self._is_submerged = False
                curSubmergedHeight = self._height/2.0 - z
            print(curSubmergedHeight)
            volume = curSubmergedHeight * self._water_level_plane_area
            buoyancyForce = np.array([0, 0, volume * self._fluid_density * self._g])
            buoyancyTorque = np.array([
                -1 * self._metacentric_width * math.sin(roll) * buoyancyForce[2],
                -1 * self._metacentric_length * math.sin(pitch) * buoyancyForce[2],
                0])
     
        return buoyancyForce, buoyancyTorque

    def ApplyBuoyancyForce(self):
        # Prim's pose
        pose, quat = utils.getPose(self._PhysXIFace, self._prim_path)
        rotWR = utils.Quaternion2RotationMatrix(quat)
        # Get the buoyancy force in world coordinates
        buoyancyForce, buoyancyTorque = self.GetBuoyancyForce(pose, quat)
        cob_world = np.matmul(rotWR, self._center_of_buoyancy) + pose

        if not self._is_surface_vessel:
            utils.AddForceAtRelativePosition(self._PhysXIFace, self._prim_path, cob_world, buoyancyForce)
        else:
            utils.AddRelativeForce(self._PhysXIFace, self._prim_path, buoyancyForce)
            utils.AddRelativeTorque(self._PhysXIFace, self._prim_path, buoyancyTorque)

    def SetBoundingBox(self, bBox):
        self._bounding_box = bBox
        self._height = self._bounding_box[2]
        utils.Print("New bounding box for " + utils.getName(self._stage,
         self._prim_path) + " :: " + np.array2string(self._bounding_box,
          precision=2, separator=','))

    def SetVolume(self, volume):
        utils.Assert(volume > 0, "Invalid input volume")
        self._volume = volume

    def GetVolume(self):
        return self._scaling_volume * (self._volume + self._offset_volume)

    def SetFluidDensity(self, fluidDensity):
        utils.Assert(fluidDensity > 0, "Fluid density must be a positive value")
        self._fluid_density = fluidDensity

    def GetFluidDensity(self):
        return self._fluid_density
    
    #@property
    def GetCoB(self):
        return self._center_of_buoyancy

    #@_center_of_buoyancy.setter
    def SetCoB(self, centerOfBuoyancy):
        self._center_of_buoyancy = centerOfBuoyancy

    def SetGravity(self, g):
        utils.Assert(g > 0, "Acceleration of gravity must be positive")
        self._g = g

    def GetGravity(self):
        return self._g

    def IsSubmerged(self):
        return self._is_submerged

    def IsNeutrallyBuoyant(self):
        return self._neutrally_buoyant