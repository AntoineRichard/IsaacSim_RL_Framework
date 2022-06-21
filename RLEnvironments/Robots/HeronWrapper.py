from distutils.spawn import spawn
import omni
#import carb
from pxr import Gf, UsdGeom
from RLEnvironments.Robots.BaseRobotWrapper import BaseRobotWrapper
from omni.isaac.range_sensor import _range_sensor
from SensorModels.IsaacSensors import PerfectPoseSensor
from USVConfigurations.HeronSettings import HeronHydroSettings, HeronThrustersSettings
import RLEnvironments.IsaacUtils as utils
import numpy as np

class HeronWrapper(BaseRobotWrapper):
    def __init__(self, stage):
        super().__init__(stage)
        self.asset_path = self.nucleus_server + "/LakeSimulation/heron_with_laser.usd"#"/RLEnvironments/Robots/heron_with_laser.usd"
        self.ar = None
        self.target = [0,0]

    def loadPlugins(self):
        # Load plugins after kit is loaded
        from BuoyancyPhysics.UnderWaterObject import UnderWaterObject
        from BuoyancyPhysics.Thruster import ThrusterPlugin
        # Hydrodynamics simulation with python plugin
        self.UWO = UnderWaterObject(self.stage, self.PhysXIFace, self.DCIFace)
        self.UWO.Load(HeronHydroSettings)
        self.base_link = HeronHydroSettings["link"][0]["name"]
        self.THR = []
        for i, settings in enumerate(HeronThrustersSettings):
            self.THR.append(ThrusterPlugin(self.stage, self.PhysXIFace, self.DCIFace))
            self.THR[i].Load(settings)
        # Add perfect pose sensor
        self.PPS = PerfectPoseSensor(self.stage, self.PhysXIFace, self.DCIFace, self.base_link)
        # Add lidar accessor
        self.lidar_path = "/heron/front_laser/Lidar"
        self.activateLidar()

    def spawn(self, position):
        prims = []
        prims = utils.createObject('/heron', self.stage, self.asset_path, False, position=position, group=prims, allow_physics=False)
        self.heron_prim = prims[0]

    def teleport(self, position, rotation, settle=False):
        if self.ar is None:
            self.ar = self.DCIFace.get_articulation("/heron")
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
        self.updateCommands([0.0,0.0])

        # Settle the robot
        #if settle:
        #    frame = 0
        #    velocity = 1
        #    while velocity > 0.1 and frame < 120:
        #        self.omni_kit.update(1.0 / 60.0)
        #        lin_vel = self.DCIFace.get_rigid_body_linear_velocity(self.base_link_id)
        #        velocity = np.linalg.norm([lin_vel.x, lin_vel.y, lin_vel.z])
        #        frame = frame + 1

    def smoothCommands(self, beta = 0.2):
        if abs(self.THR[0].getCmd() - self.target[0]) > beta:
            self.THR[0].UpdateCommand(self.THR[0].getCmd() + np.sign(-self.THR[0].getCmd() + self.target[0])*beta)
        else:
            self.THR[0].UpdateCommand(self.target[0])
        if abs(self.THR[1].getCmd() - self.target[1]) > beta:
            self.THR[1].UpdateCommand(self.THR[1].getCmd() + np.sign(-self.THR[1].getCmd() + self.target[1])*beta)
        else:
            self.THR[1].UpdateCommand(self.target[1])

    def update(self, dt):
        self.UWO.Update(dt)
        self.smoothCommands()
        for i in self.THR:
            i.Update(dt)
    
    def updateCommands(self, data):
        self.target[0] = data[1]
        self.target[1] = data[0]
        #self.THR[0].UpdateCommand(data[1])
        #self.THR[1].UpdateCommand(data[0])

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
    
    def flowVelCallback(self, data):
        self.UWO.UpdateFlowVelocity(data)
