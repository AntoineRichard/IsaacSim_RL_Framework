from pxr import UsdGeom
from gym import spaces
import numpy as np
import omni
import cv2


from USVConfigurations.HeronSettings import HeronHydroSettings, HeronThrusters
from RLEnvironments.HeronWrapper import HeronWrapper
from RLEnvironments.LaserProjector import LaserProjector
from RLEnvironments.LakeEnvironment import Environment
from omni.isaac.core import World

class HeronEnvironment:
    def __init__(self,physics_dt=1/60.0,render_dt=1/12.0,num_herons=1, ideal_dist=10.0, pose_coeff=2.5):
        # Setup Isaac rates
        self.world = World(stage_units_in_meters=0.01, physics_dt=physics_dt, rendering_dt=render_dt)
        self.world.step()
        # Make Z up
        #self.omniverse_kit.set_up_axis(UsdGeom.Tokens.z)
        # Generate the environment
        self.stage = omni.usd.get_context().get_stage()
        self.lake = Environment(self.stage, self.world)
        # Initialize and spawn the robot
        self.heron = HeronWrapper(self.stage, HeronHydroSettings, HeronThrusters)
        position, rotation = self.lake.getValidLocation(0)
        self.heron.spawn(position)
        self.world.step()
        self.world.add_physics_callback("heron_dyn", self.heron.update)
        self.heron.activateLidar()
        self.heron.loadPlugins()
        # Start simulation
        self.world.play()
        # Initialize environment variables
        self.numsteps = 0
        self.numresets = 0
        # Projection parameters
        self.ideal_dist = ideal_dist
        self.Proj = LaserProjector(ideal_dist)
        self.pose_coeff = pose_coeff
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
        # Update heron commands
        self.heron.updateCommands(action)
        # Compute a render step (i.e. should be running all the physics substeps required)
        [self.world.step(render=False) for i in range(4)]
        self.world.step(render=True)
        # Compute reward
        obs = self.heron.getObservation()
        reward = self.computeReward(obs)
        obs = self.processObservations(obs)
        # Check if done (should add a death event like when the boats collides with stuff)
        done = False
        self.numsteps += 1
        return obs, reward, done, {}

    def computeReward(self, obs):
        min_dist = self.ideal_dist - np.min(obs["laser"])
        #print(min_dist, max(-20.0, (1. - min_dist*min_dist*0.5)*self.pose_coeff))
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
        position, rotation = self.lake.getValidLocation(step, mode="power")
        self.heron.teleport(position, rotation)
        self.numresets += 1
        self.prev_phy = [0,0,0]
        return self.step([0,0])


