from pxr import UsdGeom
from gym import spaces
import numpy as np
import omni

from omni.isaac.core import World

class BaseTask:
    def __init__(self, WorldLoader, RobotLoader, world_settings, task_settings, scene_settings):
        # Setup Isaac rates
        self.world = World(stage_units_in_meters=0.01, **world_settings)
        self.world.step()
        # Generate the environment
        self.stage = omni.usd.get_context().get_stage()
        self.lake = WorldLoader(self.stage, self.world, **scene_settings)
        # Initialize and spawn the robot
        self.robot = RobotLoader(self.stage)
        position, rotation = self.lake.getValidLocation(0)
        self.robot.spawn(position)
        self.world.step()
        self.world.add_physics_callback("robot_dyn", self.robot.update)
        self.robot.loadPlugins()
        # Start simulation
        self.world.play()
        # Initialize environment variables
        self.numsteps = 0
        self.numresets = 0
        self.initializeTask(scene_settings, task_settings)
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
        done = self.isDone(obs)
        self.numsteps += 1
        return obs, reward, done, {}

    def isDone(self, obs):
        raise NotImplementedError

    def computeReward(self, obs):
        raise NotImplementedError

    def processObservations(self, obs):
        raise NotImplementedError

    def reset(self, step=0):
        raise NotImplementedError

    def initializeTask(self, scene_settings, task_settings):
        raise NotImplementedError