def getEnvironment(environment_name, task_name, robot_name, world_specs={}, env_specs={}, task_specs={}):
    needed_world_specs = ["physics_dt","rendering_dt"]
    assert (set(world_specs) == set(needed_world_specs)), "Missing some world specs. Exepected fields are :"+str(needed_world_specs)+". Please edit the configuration."
    if robot_name == "heron":
        from RLEnvironments.Robots.HeronWrapper import HeronWrapper as RobotLoader
    elif robot_name == "husky":
        from RLEnvironments.Robots.HuskyWrapper import HuskyWrapper as RobotLoader
    elif robot_name == "jackal":
        from RLEnvironments.Robots.JackalWrapper import JackalWrapper as RobotLoader
    elif robot_name == "turtlebot2":
        from RLEnvironments.Robots.Turtlebot2Wrapper import Turtlebot2Wrapper as RobotLoader
    elif robot_name == "turtlebot3":
        from RLEnvironments.Robots.Turtlebot3Wrapper import Turtlebot3Wrapper as RobotLoader
    else:
        raise ValueError("unknown robot: "+str(robot_name))

    if environment_name=="lake_water":
        from RLEnvironments.Worlds.LakeWater import SingleLakeWorld as WorldLoader
        assert (robot_name == "heron"), "This environment is meant to be used with the heron. Please edit the config."
        needed_env_specs = ["scene_path", "meta_data_path", "variation"]
        assert (set(needed_env_specs) == set(env_specs)), "Missing some environment specs. Exepected fields are :"+str(needed_env_specs)+". Please edit the configuration."
        assert (type(env_specs["variation"]) == int), "Variation spec must be an int."
    elif environment_name=="lake_ground":
        from RLEnvironments.Worlds.LakeGround import SingleLakeWorld as WorldLoader
        assert (robot_name != "heron"), "This robot (the heron) is meant to be used in water. Please edit the config."
        needed_env_specs = ["scene_path", "meta_data_path", "variation"]
        assert (set(needed_env_specs) == set(env_specs)), "Missing some environment specs. Exepected fields are :"+str(needed_env_specs)+". Please edit the configuration."
        assert (type(env_specs["variation"]) == int), "Variation spec must be an int."
    else:
        raise ValueError("Unknown environment: "+str(environment_name))

    if task_name=="follow_shore_velocity_goal":
        from RLEnvironments.Tasks.FollowShoreTask_VelocityGoal import FollowShoreVelocityGoal as Task
        needed_task_specs = ["ideal_dist", "min_dist", "max_dist",
                             "warmup", "target_step", "alpha", "isaac_res",
                             "map_res","pose_coeff"]
        assert (set(task_specs)==set(needed_task_specs))
        return Task(WorldLoader, RobotLoader, world_specs, task_specs, env_specs)
    elif task_name=="follow_shore_fixed_velocity":
        from RLEnvironments.Tasks.FollowShoreTask_FixedVelocity import FollowShoreFixedVelocity as Task
        needed_task_specs = ["ideal_dist", "min_dist", "max_dist",
                             "warmup", "target_step", "alpha", "isaac_res", "target_vel",
                             "map_res","pose_coeff", "vel_coeff"]
        assert (set(task_specs)==set(needed_task_specs))
        return Task(WorldLoader, RobotLoader, world_specs, task_specs, env_specs)
    else:
        raise ValueError("Unknown task: "+str(task_name))

    
    