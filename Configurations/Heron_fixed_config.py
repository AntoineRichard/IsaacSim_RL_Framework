import pathlib 
import Dreamer.tools as tools

def define_config():
  config = tools.AttrDict()
  # General.
  config.logdir = pathlib.Path('.')
  config.seed = 0
  config.steps = 5e6
  config.eval_every = 5000
  config.log_every = 1e3
  config.log_scalars = True
  config.log_images = True
  config.gpu_growth = True
  config.precision = 32
  config.port = 8080
  # Environment.
  config.world_specs = {"physics_dt":1.0/60.0,"rendering_dt":1.0/12.0}
  config.env_name = 'lake_water'
  config.env_specs = {"scene_path":"/RLEnvironments/Worlds/Lakes/StandaloneLakesWater/",
                        "meta_data_path":"standalone_examples/python_samples/Buoyancy/raw_generation/",
                        "variation":0}
  config.robot_name = 'heron'
  config.task_name = 'follow_shore_fixed_velocity'
  config.task_specs = {"ideal_dist":10.5,
                        "min_dist":7.0,
                        "max_dist":13.0,
                        "warmup":2.5e5,
                        "target_step":1e6,
                        "alpha":1.0,
                        "target_vel":1.0,
                        "isaac_res":100,
                        "map_res":0.1,
                        "vel_coeff":1.0,
                        "pose_coeff":2.5}
  config.envs = 1
  config.parallel = 'none'
  config.action_repeat = 1
  config.time_limit = 500
  config.prefill = 2500
  config.eval_noise = 0.0
  config.clip_rewards = 'none'
  config.headless = True
  config.livestream = False
  # Model.
  config.deter_size = 300
  config.stoch_size = 30
  config.num_units = 400
  config.phy_deter_size = 50
  config.phy_stoch_size = 5
  config.phy_num_units = 60
  config.dense_act = 'elu'
  config.cnn_act = 'relu'
  config.cnn_depth = 32
  config.pcont = False
  config.free_nats = 3.0
  config.kl_scale = 1.0
  config.pcont_scale = 10.0
  config.weight_decay = 0.0
  config.weight_decay_pattern = r'.*'
  # Training.
  config.batch_size = 50
  config.batch_length = 50
  config.train_every = 500
  config.train_steps = 100
  config.pretrain = 100
  config.model_lr = 6e-4
  config.value_lr = 8e-5
  config.actor_lr = 8e-5
  config.grad_clip = 100.0
  config.dataset_balance = False
  # Behavior.
  config.discount = 0.97
  config.disclam = 0.95
  config.horizon = 15
  config.action_dist = 'tanh_normal'
  config.action_init_std = 5.0
  config.expl = 'additive_gaussian'
  config.expl_amount = 0.3
  config.expl_decay = 0.0
  config.expl_min = 0.0
  return config