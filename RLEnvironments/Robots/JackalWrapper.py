from distutils.spawn import spawn
import omni
#import carb
from pxr import Gf, UsdGeom
from RLEnvironments.Robots.BaseRobotWrapper import BaseRobotWrapper
from omni.isaac.range_sensor import _range_sensor
from SensorModels.IsaacSensors import PerfectPoseSensor
import RLEnvironments.IsaacUtils as utils
import numpy as np

class JackalWrapper(BaseRobotWrapper):
    def __init__(self, stage):
        super().__init__(stage)
        self.asset_path = self.nucleus_server + "/RLEnvironments/Robots/jackal_with_laser.usd"
        self.ar = None
        self.left_wheel_joints = ["front_left_wheel", "rear_left_wheel"]
        self.right_wheel_joints = ["front_right_wheel", "rear_right_wheel"]
        self.dist_chassis = 0.188
        self.wheel_radius = 0.10
        self.wheel_separation_multipier = 1.5
        self.wheel_radius_multiplier = 1.0
        self.vlin_max = 2.0
        self.vang_max = 4.0
        self.twist = [0,0]
        self.vw_L = 0
        self.vw_R = 0

    def loadPlugins(self):
        # Load plugins after kit is loaded
        self.base_link = "/jackal/base_link"
        # Add perfect pose sensor
        self.PPS = PerfectPoseSensor(self.stage, self.PhysXIFace, self.DCIFace, self.base_link)
        # Add lidar accessor
        self.lidar_path = "/jackal/front_laser/Lidar"
        self.activateLidar()

    def spawn(self, position):
        prims = []
        prims = utils.createObject('/jackal', self.stage, self.asset_path, False, position=position, group=prims, allow_physics=False)
        self.heron_prim = prims[0]

    def teleport(self, position, rotation, settle=False):
        if self.ar is None:
            self.ar = self.DCIFace.get_articulation("/jackal")
            self.base_link_id = self.DCIFace.get_articulation_root_body(self.ar)
        self.DCIFace.wake_up_articulation(self.ar)
        rot_quat = Gf.Rotation(Gf.Vec3d(0, 0, 1), 180*rotation[-1]/np.pi).GetQuaternion()
        tf = self._dynamic_control.Transform(position,
            (rot_quat.GetImaginary()[0],
            rot_quat.GetImaginary()[1],
            rot_quat.GetImaginary()[2],
            rot_quat.GetReal()))
        self.DCIFace.set_rigid_body_pose(self.base_link_id, tf)
        self.DCIFace.set_rigid_body_linear_velocity(self.base_link_id, [0,0,0])
        self.DCIFace.set_rigid_body_angular_velocity(self.base_link_id, [0,0,0])
        # Resets commands
        self.twist = [0,0]
        self.vw_L = 0
        self.vw_R = 0

        # Settle the robot
        #if settle:
        #    frame = 0
        #    velocity = 1
        #    while velocity > 0.1 and frame < 120:
        #        self.omni_kit.update(1.0 / 60.0)
        #        lin_vel = self.DCIFace.get_rigid_body_linear_velocity(self.base_link_id)
        #        velocity = np.linalg.norm([lin_vel.x, lin_vel.y, lin_vel.z])
        #        frame = frame + 1

    def smoothCommands(self, target, beta = 0.2):
        smooth = [0.0,0.0]
        if abs(self.twist[0] - target[0]) > beta:
            smooth[0] = self.twist[0] + np.sign(-self.twist[0] + target[0])*beta
        else:
            smooth[0] = target[0]
        if abs(self.twist[1] - target[1]) > beta:
            smooth[1] = self.twist[1] + np.sign(-self.twist[1] + target[1])*beta
        else:
            smooth[1] = target[1]
        return smooth

    def update(self, dt):
        self.applyCmd()
    
    def updateCommands(self, data):
        data[0] = data[0]*self.vlin_max
        data[1] = data[1]*self.vang_max
        data = self.smoothCommands(data)
        self.twist = data.copy()
        # Commands are sent as a twist (linear_vel,angular_vel).
        # It must be transformed to left right wheel velocities in rad/s.
        self.vw_L = (data[0] - self.wheel_separation_multipier*data[1]*self.dist_chassis)/(self.wheel_radius*self.wheel_radius_multiplier)
        self.vw_R = (data[0] + self.wheel_separation_multipier*data[1]*self.dist_chassis)/(self.wheel_radius*self.wheel_radius_multiplier)
        
    def applyCmd(self):
        # Only ran at start-up
        if self.ar is None:
            # Grabs the articulation at the robot path
            self.ar = self.DCIFace.get_articulation("/jackal")
            self.base_link_id = self.DCIFace.get_articulation_root_body(self.ar)
            # Gets the root of the articulation/robot
            self.chassis = self.DCIFace.get_articulation_root_body(self.ar)
            # Lists in which we store the ID associated to each wheel
            self.left_wheels = []
            self.right_wheels = []
            # Puts in the lists the ID of the wheels
            for joint in self.left_wheel_joints:
                self.left_wheels.append(self.DCIFace.find_articulation_dof(self.ar, joint))
            for joint in self.right_wheel_joints:
                self.right_wheels.append(self.DCIFace.find_articulation_dof(self.ar, joint))

        # First make sure the articulation is enabled
        self.DCIFace.wake_up_articulation(self.ar)
        # Apply the desired velocity to each wheel
        for wheel in self.left_wheels:
            self.DCIFace.set_dof_velocity_target(wheel, self.vw_L)
        for wheel in self.right_wheels:
            self.DCIFace.set_dof_velocity_target(wheel, self.vw_R)

    def activateLidar(self):
        self.lidar_interface = _range_sensor.acquire_lidar_sensor_interface()

    def getLaserData(self):
        depth = self.lidar_interface.get_linear_depth_data(self.lidar_path)
        depth[depth==20] = 100
        return depth

    def getState(self):
        return self.PPS.get()

    def getObservation(self):
        try: 
            laser = self.getLaserData()
            lin_vel, ang_vel, pose, quat = self.getState()
            return {"laser":laser,
                "linear_velocity":lin_vel,
                "angular_velocity":ang_vel,
                "position":pose,
                "quat":quat}
        except:
            return False