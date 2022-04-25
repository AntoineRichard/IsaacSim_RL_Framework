import os
import omni
from omni.isaac.kit import SimulationApp
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    from Dreamer.base_config import define_config
    from Dreamer.tools import args_type
    for key, value in define_config().items():
        parser.add_argument(f'--{key}', type=args_type(value), default=value)

    config = parser.parse_args()
    
    CONFIG = {
        "experience": f'{os.environ["EXP_PATH"]}/omni.isaac.sim.python.kit',
        "renderer": "RayTracedLighting",
        "headless": config.headless,
    }

    simulation_app = SimulationApp(CONFIG)
    ext_manager = omni.kit.app.get_app().get_extension_manager()
    ext_manager.set_extension_enabled_immediate("omni.physx.bundle", True)
    if not config.headless:
        simulation_app.set_setting("/app/window/drawMouse", True)
        simulation_app.set_setting("/app/livestream/proto", "ws")
        ext_manager.set_extension_enabled_immediate("omni.syntheticdata", True)
        ext_manager.set_extension_enabled_immediate("omni.kit.window.stage", True)
        ext_manager.set_extension_enabled_immediate("omni.kit.property.bundle", True)
    if config.livestream:
        ext_manager.set_extension_enabled_immediate("omni.kit.livestream.core", True)
        ext_manager.set_extension_enabled_immediate("omni.kit.livestream.native", True)

    from Dreamer.dreamer_img2img_goals_LinearRSSMPhysics import train

    train(config)
    simulation_app.close()
