from pxr import UsdGeom
from gym import spaces
import numpy as np
import omni

from RLEnvironments.LaserProjector import LaserProjector
from RLEnvironments.FollowShoreUtils import FollowingSampler
from omni.isaac.core import World

class FollowShoreVelocityGoal:
    def __init__(self, WorldLoader, RobotLoader, world_settings, task_settings, scene_settings):
        # Setup Isaac rates
        self.world = World(stage_units_in_meters=0.01, **world_settings)
        self.world.step()
        # Generate the environment
        self.stage = omni.usd.get_context().get_stage()
        self.lake = WorldLoader(self.stage, self.world, **scene_settings)
        self.FS = FollowingSampler(scene_settings["meta_data_path"]+"gen"+str(scene_settings["variation"]), **task_settings)
        # Initialize and spawn the robot
        self.robot = RobotLoader(self.stage)
        position, rotation = self.getValidLocation(0)
        self.robot.spawn(position)
        self.world.step()
        self.world.add_physics_callback("robot_dyn", self.robot.update)
        self.robot.activateLidar()
        self.robot.loadPlugins()
        # Start simulation
        self.world.play()
        # Initialize environment variables
        self.numsteps = 0
        self.numresets = 0
        # Projection parameters
        self.ideal_dist = task_settings["ideal_dist"]
        self.Proj = LaserProjector(task_settings["ideal_dist"])
        self.pose_coeff = task_settings["pose_coeff"]
        # Setup gym data
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = {"image":spaces.Box(low=0, high=255, shape=(64,64,3), dtype=np.uint8),
                                  "laser":spaces.Box(low=0, high=20.0, shape=(256,), dtype=np.float32),
                                  "physics":spaces.Box(low=-3.0, high=3.0, shape=(3,), dtype=np.float32)}
        for i in range(10):
            self.world.step()
        self.reset()
        self.world.step()

    def step(self, action):
        # Update robot commands
        self.robot.updateCommands(action)
        # Compute a render step (i.e. should be running all the physics substeps required)
        [self.world.step(render=False) for i in range(4)]
        self.world.step(render=True)
        # Compute reward
        obs = self.robot.getObservation()
        reward = self.computeReward(obs)
        obs = self.processObservations(obs)
        # Check if done (should add a death event like when the robot collides with stuff)
        done = False
        self.numsteps += 1
        return obs, reward, done, {}

    def computeReward(self, obs):
        min_dist = self.ideal_dist - np.min(obs["laser"])
        return max(-20.0, (1. - min_dist*min_dist*0.5)*self.pose_coeff)

    def processObservations(self, obs):
        obs["laser"] = np.min(np.reshape(obs["laser"],[-1,2]),axis=1)
        obs["image"] = self.Proj.projectLaser(obs["laser"])
        obs["laser"] = obs["laser"][52:-52]
        obs["physics"] = np.array([obs["linear_velocity"][0],obs["linear_velocity"][1],obs["angular_velocity"][2]])
        obs["physics_d"] = self.prev_phy
        self.prev_phy = obs["physics"].copy()
        return obs

    def reset(self, step=0):
        self.lake.reset()
        position, rotation = self.getValidLocation(step, mode="power")
        self.robot.teleport(position, rotation)
        self.numresets += 1
        self.prev_phy = [0,0,0]
        return self.step([0,0])
    
    def getValidLocation(self, step, reward=0.0, is_training=False, mode="random"):
        xyrz, _, _ = self.FS.sample(step, reward, is_training, mode)
        return [xyrz[1], xyrz[0], 0], [0, 0, xyrz[2]]


