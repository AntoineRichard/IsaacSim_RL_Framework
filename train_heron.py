import os
import omni
import carb
from pxr import Gf, UsdGeom
import numpy as np
from omni.isaac.python_app import OmniKitHelper
from omni.isaac.kit import SimulationApp
from matplotlib import pyplot as plt
import argparse
import time
import datetime
import sys
import tensorflow as tf
import functools
import random

CONFIG = {
    "experience": f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit',
    "renderer": "RayTracedLighting",
    "headless": True,
}

if __name__ == "__main__":
    simulation_app = SimulationApp(CONFIG)
    #omniverse_kit = OmniKitHelper(CONFIG)
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    #simulation_app.set_setting("/app/window/drawMouse", True)
    #simulation_app.set_setting("/app/livestream/proto", "ws")
    ext_manager.set_extension_enabled_immediate("omni.physx.bundle", True)
    #ext_manager.set_extension_enabled_immediate("omni.syntheticdata", True)
    #ext_manager.set_extension_enabled_immediate("omni.kit.livestream.core", True)
    #ext_manager.set_extension_enabled_immediate("omni.kit.livestream.native", True)
    #ext_manager.set_extension_enabled_immediate("omni.kit.window.stage", True)
    #ext_manager.set_extension_enabled_immediate("omni.kit.property.bundle", True)
    from imgphy_flex2img_dreamer_large_newdyn import Dreamer, make_env, define_config, count_steps
    from tools import simulate, args_type

    def train(config):
        if config.gpu_growth:
            for gpu in tf.config.experimental.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(gpu, True)
        assert config.precision in (16, 32), config.precision
        #if config.precision == 16:
        #    prec.set_policy(prec.Policy('mixed_float16'))
        config.steps = int(config.steps)
        config.logdir.mkdir(parents=True, exist_ok=True)
        print('Logdir', config.logdir)

        # Create environments.
        datadir = config.logdir / 'episodes'
        writer = tf.summary.create_file_writer(str(config.logdir), max_queue=1000, flush_millis=20000)
        writer.set_as_default()
        train_envs = [make_env(config, writer, 'train', datadir, store=True)]
        #test_envs = [wrappers.Async(lambda: make_env(
        #    config, writer, 'test', datadir, store=False), config.parallel)
        #    for _ in range(config.envs)]
        actspace = train_envs[0].action_space
    
        # Prefill dataset with random episodes.
        step = count_steps(datadir, config)
        prefill = max(0, config.prefill - step)
        print(f'Prefill dataset with {prefill} steps.')
        random_agent = lambda o, d, _, __,___ : ([actspace.sample() for _ in d], None, None)
        simulate(random_agent, train_envs, prefill / config.action_repeat)
        writer.flush()

        # Build agent
        step = count_steps(datadir, config)
        print(f'Simulating agent for {config.steps-step} steps.')
        agent = Dreamer(config, datadir, actspace, writer)

        # Load weights
        if (config.logdir / 'variables.pkl').exists():
            print('Load checkpoint.')
            agent.load(config.logdir / 'variables.pkl')
        if (config.logdir / 'phy_dynamics_weights.pkl').exists():
            print('Loading physics_dynamics.')
            agent._phy_dynamics.load(config.logdir / 'phy_dynamics_weights.pkl')

        # Train and regularly evaluate the agent.
        state = None
        while step < config.steps:
            print('Start evaluation.')
            simulate(functools.partial(agent, training=False), train_envs, episodes=1, step=step, target_vel=[[random.random()+0.3]])
            writer.flush()
            print('Start collection.')
            steps = config.eval_every // config.action_repeat
            state = simulate(agent, train_envs, steps, state=state, step=step, target_vel=[[random.random()+0.3]])
            step = count_steps(datadir, config)
            agent.save(config.logdir / 'variables.pkl')

    parser = argparse.ArgumentParser()
    for key, value in define_config().items():
        parser.add_argument(f'--{key}', type=args_type(value), default=value)

    train(parser.parse_args())
    simulation_app.close()
