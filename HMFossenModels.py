"""
 Porting of the UUV Plugins to Isaac : https://github.com/uuvsimulator
 Some functions have been changed to better fit the use of Pythons.
 Some functions have been changed to be more accurate.
 2021 DreamLab  - Georgia Institute of Technology - IRL 2958 GT CNRS
 Antoine Richard: antoine.richard@gatech.edu
"""

from HydrodynamicModel import *
import math


class HMFossen(HydrodynamicModel):
    def __init__(self, stage, prim_path, PhysXIFace, DCIFace, settings):
        super().__init__(stage, prim_path, PhysXIFace, DCIFace, settings)
        self._D = None
        self._Ca = None
        self.buildFossen(settings)

    def buildFossen(self, settings):
        # Initialize empty arrays
        addedMass = np.zeros([6,6])
        linDampCoef = np.zeros([6,6])
        linDampForward = np.zeros([6,6])
        quadDampCoef = np.zeros([6,6])

        if "hydrodynamic_model" not in settings.keys():
            raise ValueError("Hydrodynamic model is missing")
        hydroSettings = settings["hydrodynamic_model"]
        # Load added-mass coefficients, if provided. Otherwise, the added-mass
        # matrix is set to zero
        if "added_mass" in hydroSettings.keys():
            addedMass = hydroSettings["added_mass"]
            utils.Assert((addedMass.shape[0] == 6) and (addedMass.shape[1] == 6) and (len(addedMass.shape)==2),
              "Added-mass coefficients vector must be a 6x6 numpy matrix")
        else:
            utils.Print("HMFossen: Using added mass NULL")
        self._added_mass = addedMass        
        self._params.append("added_mass")
        if "scaling_added_mass" in hydroSettings.keys():
            self._scaling_added_mass = hydroSettings["scaling_added_mass"]
        else:
            self._scaling_added_mass = 1.0
        self._params.append("scaling_added_mass")
        if "offset_added_mass" in hydroSettings.keys():
            self._offset_added_mass = hydroSettings["offset_added_mass"]
        else:
            self._offset_added_mass = 0.0
        self._params.append("offset_added_mass")

        if "linear_damping_forward_speed" in hydroSettings:
            linDampForward = hydroSettings["linear_damping_forward_speed"]
            utils.Assert((linDampForward.shape[0] == 6) and (linDampForward.shape[1] == 6) and (len(linDampForward.shape)==2),
            "Linear damping coefficients proportional to the forward speed "
            "vector must be 6x6 numpy matrix")
        else:
            utils("HMFossen: Using linear damping for forward speed NULL")
        self._linear_damping_forward_speed = linDampForward
        self._params.append("linear_damping_forward_speed")

        if "linear_damping" in hydroSettings.keys():
            linDampCoef = hydroSettings["linear_damping"]
            utils.Assert((linDampCoef.shape[0] == 6) and (linDampCoef.shape[1] == 6) and (len(linDampCoef.shape)==2),
            "Linear damping coefficients vector must be a 6x6 numpy matrix")
        else:
            utils.Print("HMFossen: Using linear damping NULL")
        self._linear_damping = linDampCoef
        self._params.append("linear_damping")

        if "quadratic_damping" in hydroSettings.keys():
            quadDampCoef = hydroSettings["quadratic_damping"]
            utils.Assert((quadDampCoef.shape[0] == 6) and (quadDampCoef.shape[1] == 6) and (len(quadDampCoef.shape)==2),
            "Quadratic damping coefficients vector must be a 6x6 numpy matrix")
        else:
            utils.Print("HMFossen: Using quad damping NULL")
        self._quadratic_damping = quadDampCoef
        self._params.append("quadratic_damping")

        if "scaling_damping" in hydroSettings.keys():
            self._scaling_damping = hydroSettings["scaling_damping"]
        else:
            self._scaling_damping = 1.0
            self._params.append("scaling_damping")
        if "offset_linear_damping" in hydroSettings.keys():
            self._offset_linear_damping = hydroSettings["offset_linear_damping"]
        else:
            self._offset_linear_damping = 0.0
            self._params.append("offset_linear_damping")
        if "offset_lin_forward_speed_damping" in hydroSettings.keys():
            self._offset_lin_forward_speed_damping = hydroSettings["offset_lin_forward_speed_damping"]
        else:
            self._offset_lin_forward_speed_damping = 0.0
            self._params.append("offset_lin_forward_speed_damping")
        if "offset_nonlin_damping" in hydroSettings.keys():
            self._offset_nonlin_damping = hydroSettings["offset_nonlin_damping"]
        else:
            self._offset_nonlin_damping = 0.0
            self._params.append("offset_nonlin_damping")
            self._params.append("scaling_volume")
            self._params.append("volume")
  
    def ComputeHydrodynamicForces(self, time, _flowVelWorld):
        pose, quat = utils.getPose(self._PhysXIFace, self._prim_path)
        print(pose, quat)
        rot_mat = utils.Quaternion2RotationMatrix(quat)
        rot_mat_inv = np.linalg.inv(rot_mat)
        linVel = utils.getRelativeLinearVel(self._RigidBodyAPI, rot_mat_inv)
        angVel = utils.getRelativeAngularVel(self._RigidBodyAPI, rot_mat_inv)
        #print(linVel, angVel)

        # Transform the flow velocity to the body frame
        flowVel = np.matmul(rot_mat,_flowVelWorld)
        # Compute the relative velocity
        #velRel = np.hstack([self.ToNED(linVel - flowVel), self.ToNED(angVel)])
        velRel = np.hstack([linVel - flowVel, angVel])
        # Update added Coriolis matrix
        self.ComputeAddedCoriolisMatrix(velRel)
        # Update damping matrix
        self.ComputeDampingMatrix(velRel)
        # Filter acceleration (see issue explanation above)
        self.ComputeAcc(velRel, time, 0.3)
        # We can now compute the additional forces/torques due to thisdynamic
        # effects based on Eq. 8.136 on p.222 of Fossen: Handbook of Marine Craft ...
        # Damping forces and torques
        damping =  np.matmul(-self._D, velRel)
        # Added-mass forces and torques
        added = np.matmul(-self.GetAddedMass(), self._filtered_acc)
        # Added Coriolis term
        cor = np.matmul(-self._Ca, velRel)
        # All additional (compared to standard rigid body) Fossen terms combined.
        tau = damping + added + cor
        #print(tau)

        utils.Assert(not math.isnan(np.linalg.norm(tau)), "Hydrodynamic forces vector is nan")
        print("++++damping++++")
        print(angVel, tau[-3:])
        print(linVel, tau[:3])
        print("+++++++++++++++")
        return tau

    def ApplyHydrodynamicForces(self, time, _flowVelWorld):
        tau = self.ComputeHydrodynamicForces(time, _flowVelWorld)
        if not math.isnan(np.linalg.norm(tau)):
            # Convert the forces and moments back to Gazebo's reference frame
            #hydForce = self.FromNED(tau[:3])
            #hydTorque = self.FromNED(tau[-3:])
            hydForce = tau[:3]
            hydTorque = tau[-3:]
            # Forces and torques are also wrt link frame
            utils.AddRelativeForce(self._PhysXIFace, self._prim_path, hydForce*100)
            utils.AddRelativeTorque(self._PhysXIFace, self._prim_path, hydTorque*10)

        self.ApplyBuoyancyForce()

    @staticmethod
    def CrossProductOperator(A):
        B = np.zeros([3,3])
        B[0,1] = -A[2]
        B[1,0] = A[2]
        B[0,2] = A[1]
        B[2,0] = -A[1]
        B[2,1] = A[0]
        B[1,2] = -A[0]
        return B

    def ComputeAddedCoriolisMatrix(self, vel):
        """
        // This corresponds to eq. 6.43 on p. 120 in
        // Fossen, Thor, "Handbook of Marine Craft and Hydrodynamics and Motion
        // Control", 2011
        """
        self._Ca = np.zeros([6,6])
        ab = np.matmul(self.GetAddedMass(), vel)
        Sa = -1 * self.CrossProductOperator(ab[:3])
        self._Ca[-3:,:3] = Sa
        self._Ca[:3,-3:] = Sa
        self._Ca[-3:,-3:] = -1 * self.CrossProductOperator(ab[-3:])

    def ComputeDampingMatrix(self, vel):
        """
        // From Antonelli 2014: the viscosity of the fluid causes
        // the presence of dissipative drag and lift forces on the
        // body. A common simplification is to consider only linear
        // and quadratic damping terms and group these terms in a
        // matrix Drb
        """
        self._D = -1 * (self._linear_damping+ self._offset_linear_damping * np.eye(6))\
            - vel[0] * (self._linear_damping_forward_speed + self._offset_lin_forward_speed_damping * np.eye(6))
        # Nonlinear damping matrix is considered as a diagonal matrix
        self._D += -1 * (self._quadratic_damping + self._offset_nonlin_damping * np.eye(6))* np.abs(vel)
        self._D *= self._scaling_damping

    def GetAddedMass(self):
        return self._scaling_added_mass * (self._added_mass + self._offset_added_mass * np.eye(6))

    def GetParam(self, tag):
        if "_"+tag in self.__dict__.keys():
            value = self.__getattribute__("_"+tag)
            utils.Print("HydrodynamicModel::GetParam <" +tag+ "> is "+str(value))
            return value
        else:
            return False

    def SetParam(self, tag, value):
        if tag == "scaling_volume":
            if value < 0:
                return False
            self._scaling_volume = value
        elif tag == "scaling_added_mass":
            if value < 0:
                return False
            self._scaling_added_mass = value
        elif tag == "scaling_damping":
            if value < 0:
                return False
            self._scaling_damping = value
        elif tag == "fluid_density":
            if value < 0:
                return False
            self._fluid_density = value
        elif tag == "offset_volume":
            self._offset_volume = value
        elif tag == "offset_added_mass":
            self._offset_added_mass = value
        elif tag == "offset_linear_damping":
            self._offset_linear_damping = value
        elif tag == "offset_lin_forward_speed_damping":
            self._offset_lin_forward_speed_damping = value
        elif tag == "offset_nonlin_damping":
            self._offset_nonlin_damping = value
        else:
          return False
        utils.Print("HydrodynamicModel::SetParam <" +tag+ "> to "+str(value))
        return True

    def Print(self, paramName, message=""):
        if paramName == "all":
            for tag in self._params:
                self.Print(tag)
        if message:
            utils.Print(message)
        else:
            utils.Print(self.__class__.__name__ + " :: " +utils.getName(self._stage, self._prim_path) + " :: " + paramName)
        if paramName in self._params:
            utils.Print(self.__getattribute__("_"+paramName))

class HMSphere(HMFossen):
    def __init__(self, stage, prim_path, PhysxIFace, settings):
        super().__init__(stage, prim_path, PhysxIFace, settings)
        if "hydrodynamic_model" not in settings.keys():
            raise ValueError("Hydrodynamic model is missing")
        hydroSettings = settings["hydrodynamic_model"]
        if "radius" in hydroSettings.keys():
            self._radius = hydroSettings["radius"]
        else:
            utils.Print("HMSphere: Using the smallest length of bounding box as radius")
            self._radius = np.min(self._boundingBox)
        utils.Print("HMSphere::radius = "+str(self._radius))
        utils.Print("HMSphere: Computing added mass")
        self._params.append("radius")
        # Reynolds number for subcritical flow
        # Reference:
        #    - MIT Marine Hydrodynamic (Lecture Notes)
        # TODO Consider also critical flow
        self._Re = 3e5
        # Drag coefficient for a sphere in subcritical flow
        # Reference:
        #    - MIT Marine Hydrodynamic (Lecture Notes)
        self._Cd = 0.5
        # Area of the cross section
        self._areaSection = math.pi * self._radius**2
        # See derivation in MIT's Marine Hydrodynamics lecture notes
        # The sphere has the same projected area for X, Y and Z
        # TODO Interpolate temperatures in look-up table
        sphereMa = -2.0/3.0 * self._fluidDensity * math.pi * self._radius**3
        # At the moment, only pressure drag is calculated, no skin friction drag
        Dq = -0.5 * self._fluidDensity * self._Cd * self._areaSection
        self._Ma[:3,:3] = -sphereMa * np.eye(3)
        self._DNonLin[:3,:3] = Dq * np.eye(3)

class HMCylinder(HMFossen):
    def __init__(self, prim, physxIFace, settings):
        super.__init__(self, prim, physxIFace, settings)
        if "hydrodynamic_model" not in settings.keys():
            raise ValueError("Hydrodynamic model is missing")
        hydroSettings = settings["hydrodynamic_model"]
        if "radius" in hydroSettings.keys():
            self._radius = hydroSettings["radius"]
        else:
            utils.Print("HMSphere: Using the smallest length of bounding box as radius")
            self._radius = np.min(self._boundingBox)
        utils.Print("HMSphere::radius = "+str(self._radius))
        self._params.append("radius")
        if "length" in hydroSettings.keys():
            self._radius = hydroSettings["length"]
        else:
            utils.Print("HMSphere: Using the smallest length of bounding box as radius")
            self._length = np.max(self._boundingBox)
        utils.Print("HMSphere::length = "+str(self._length))
        self._params.append("length")
        self._dimRatio = self._length / (2* self._radius)

        utils.Print("HMCylinder::dimension_ratio = "+str(self._dimRatio))

        # Approximation of drag coefficients
        # Reference: http://www.mech.pk.edu.pl/~m52/pdf/fm/R_09.pdf
        # For the circular profile
        if (self._dimRatio <= 1):
            self._cdCirc = 0.91
        elif (self._dimRatio > 1) and (self._dimRatio <= 2):
            self._cdCirc = 0.85
        elif (self._dimRatio > 2) and (self._dimRatio <= 4):
            self._cdCirc = 0.87
        elif self._dimRatio > 4:
            self._cdCirc = 0.99
        # For the rectagular profile
        if (self._dimRatio <= 1):
            self._cdLength = 0.63
        elif (self._dimRatio > 1) and (self._dimRatio <= 2):
            self._cdLength = 0.68
        elif (self._dimRatio > 2) and (self._dimRatio <= 5):
            self._cdLength = 0.74
        elif (self._dimRatio > 5) and (self._dimRatio <= 10):
            self._cdLength = 0.82
        elif (self._dimRatio > 10) and (self.dimRatio <= 40):
            self._cdLength = 0.98
        elif self._dimRatio > 40:
            self._cdLength = 0.98

        if "axis" in hydroSettings:
            self._axis = hydroSettings["axis"]
            utils.Assert((self._axis == "i") or (self._axis == "j") or (self._axis == "k"), "Invalid axis of rotation")
        else:
            utils.Print("HMCylinder: Using the direction of biggest length as axis")
            ax = np.argmax(self._boundingBox)
            name = ["i","j","k"]
            self._axis = name[ax]
        utils.Print("HMCylinder::rotation_axis = "+self._axis)
        # Calculating the added mass and damping for the cylinder
        # Calculate added mass coefficients for the cylinder along its length
        MaLength = - self._fluidDensity * math.pi * (self._radius**2) * self._length
        # Calculate the added mass coefficients for the circular area
        MaCirc = - self._fluidDensity * math.pi * (self._radius**2)
        # Calculating added mass torque coefficients
        # Reference: Schjolberg, 1994 (Modelling and Control of Underwater-Vehicle
        # Manipulator System)
        MaLengthTorque = (-1.0/12.0) * self._fluidDensity * math.pi * (self._radius**2) * (self._length**3.0)
        # Calculate drag forces and moments
        # At the moment, only pressure drag is calculated, no skin friction drag
        DCirc = -0.5 * self._cdCirc * math.pi * (self._radius**2) * self._fluidDensity
        DLength = -0.5 * self._cdLength * self._radius * self._length * self._fluidDensity

        if self._axis == "i":
            self._Ma[0, 0] = -MaCirc
            self._Ma[1, 1] = -MaLength
            self._Ma[2, 2] = -MaLength
            self._Ma[4, 4] = -MaLengthTorque
            self._Ma[5, 5] = -MaLengthTorque
            self._DNonLin[0, 0] = DCirc
            self._DNonLin[1, 1] = DLength
            self._DNonLin[2, 2] = DLength
        elif self._axis == "j":
            self._Ma[0, 0] = -MaLength
            self._Ma[1, 1] = -MaCirc
            self._Ma[2, 2] = -MaLength
            self._Ma[3, 3] = -MaLengthTorque
            self._Ma[5, 5] = -MaLengthTorque
            self._DNonLin[0, 0] = DLength
            self._DNonLin[1, 1] = DCirc
            self._DNonLin[2, 2] = DLength
        else:
            self._Ma[0, 0] = -MaLength
            self._Ma[1, 1] = -MaLength
            self._Ma[2, 2] = -MaCirc
            self._Ma[3, 3] = -MaLengthTorque
            self._Ma[4, 4] = -MaLengthTorque
            self._DNonLin[0, 0] = DLength
            self._DNonLin[1, 1] = DLength
            self._DNonLin[2, 2] = DCirc


HydroModelMap = {"fossen":HMFossen, "sphere":HMSphere, "cylinder":HMCylinder}