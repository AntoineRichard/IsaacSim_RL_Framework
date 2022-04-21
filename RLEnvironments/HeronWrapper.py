from distutils.spawn import spawn
import omni
#import carb
from pxr import Gf, UsdGeom
from omni.isaac.range_sensor import _range_sensor
from SensorModels.IsaacSensors import PerfectPoseSensor
import RLEnvironments.IsaacUtils as utils
import numpy as np

class HeronWrapper():
    def __init__(self, stage, HydroSettings, ThrusterSettings):
        from omni.physx import get_physx_interface
        from omni.isaac.dynamic_control import _dynamic_control

        nucleus_server = utils.get_nucleus_server()
        self.asset_path = nucleus_server + "/LakeSimulation/heron_with_laser.usd"
        self.stage = stage
        self.HydroSettings = HydroSettings
        self.ThrusterSettings = ThrusterSettings
        self.PhysXIFace = get_physx_interface()
        self._dynamic_control = _dynamic_control
        self.DCIFace = _dynamic_control.acquire_dynamic_control_interface()
        self.ar = None

    def loadPlugins(self):
        # Load plugins after kit is loaded
        from BuoyancyPhysics.UnderWaterObject import UnderWaterObject
        from BuoyancyPhysics.Thruster import ThrusterPlugin
        # Hydrodynamics simulation with python plugin
        self.UWO = UnderWaterObject(self.stage, self.PhysXIFace, self.DCIFace)
        self.UWO.Load(self.HydroSettings)
        self.base_link = self.HydroSettings["link"][0]["name"]
        self.THR = []
        for i, settings in enumerate(self.ThrusterSettings):
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

    def update(self, dt):
        self.UWO.Update(dt)
        for i in self.THR:
            i.Update(dt)
    
    def updateCommands(self, data):
        self.THR[0].UpdateCommand(data[1])
        self.THR[1].UpdateCommand(data[0])

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
