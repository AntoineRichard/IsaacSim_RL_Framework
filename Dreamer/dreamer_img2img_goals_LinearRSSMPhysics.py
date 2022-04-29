import argparse
import collections
import functools
import json
import os
import glob
import pathlib
import sys
import time
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MUJOCO_GL'] = 'egl'

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.mixed_precision import experimental as prec

tf.get_logger().setLevel('ERROR')

from tensorflow_probability import distributions as tfd

sys.path.append(str(pathlib.Path(__file__).parent))

import Dreamer.models as models
import Dreamer.tools as tools
import Dreamer.wrappers as wrappers
from Dreamer.dreamer_img2img_goals_RSSMPhysics import DreamerImg2ImgGoalRSSMPhysics
from Dreamer.dreamer_img2img_goals_RSSMPhysics import count_steps, load_dataset, summarize_episode, get_last_episode_reward, make_env, preprocess
from Dreamer.base_config import define_config

class DreamerImg2ImgGoalLinearRSSMPhysics(DreamerImg2ImgGoalRSSMPhysics):
  def __init__(self, config, datadir, actspace, writer):
    self._c = config
    self._actspace = actspace
    self._actdim = actspace.n if hasattr(actspace, 'n') else actspace.shape[0]
    self._writer = writer
    self._random = np.random.RandomState(config.seed)
    with tf.device('cpu:0'):
      self._step = tf.Variable(count_steps(datadir, config), dtype=tf.int64)
    self._should_pretrain = tools.Once()
    self._should_train = tools.Every(config.train_every)
    self._should_log = tools.Every(config.log_every)
    self._last_log = None
    self._last_time = time.time()
    self._metrics = collections.defaultdict(tf.metrics.Mean)
    self._metrics['expl_amount']  # Create variable for checkpoint.
    self._float = prec.global_policy().compute_dtype
 
    self._dataset = iter(load_dataset(datadir, self._c))
    self._build_model()

  def _build_model(self):
    acts = dict(
        elu=tf.nn.elu, relu=tf.nn.relu, swish=tf.nn.swish,
        leaky_relu=tf.nn.leaky_relu)
    cnn_act = acts[self._c.cnn_act]
    act = acts[self._c.dense_act]
    self._encode = models.ConvEncoder(self._c.cnn_depth, cnn_act)
    self._env_dynamics = models.RSSMv2(self._c.stoch_size, self._c.deter_size, self._c.deter_size)
    self._phy_dynamics = models.RSSMv3(self._c.phy_stoch_size, self._c.phy_deter_size, self._c.phy_deter_size)
    self._decode = models.ConvDecoder(self._c.cnn_depth, cnn_act)
    self._reward = models.DenseDecoder((), 2, self._c.num_units, act=act)
    self._physics = models.DenseDecoder([3], 1, self._c.phy_num_units, act=act)
    if self._c.pcont:
      self._pcont = models.DenseDecoder(
          (), 3, self._c.num_units, 'binary', act=act)
    self._value = models.DenseDecoder((), 3, self._c.num_units, act=act)
    self._actor = models.ActionDecoder(
        self._actdim, 4, self._c.num_units, self._c.action_dist,
        init_std=self._c.action_init_std, act=act)
    env_modules = [self._encode, self._env_dynamics, self._decode, self._reward]
    phy_modules = [self._physics, self._phy_dynamics]
    if self._c.pcont:
      env_modules.append(self._pcont)
    Optimizer = functools.partial(
        tools.Adam, wd=self._c.weight_decay, clip=self._c.grad_clip,
        wdpattern=self._c.weight_decay_pattern)
    self._phy_opt = Optimizer('model', phy_modules, self._c.model_lr)
    self._env_opt = Optimizer('model', env_modules, self._c.model_lr)
    self._value_opt = Optimizer('value', [self._value], self._c.value_lr)
    self._actor_opt = Optimizer('actor', [self._actor], self._c.actor_lr)
    # Do a train step to initialize all variables, including optimizer
    # statistics. Ideally, we would use batch size zero, but that doesn't work
    # in multi-GPU mode.
    self.train(next(self._dataset))

  def _imagine_ahead(self, env_post, phy_post, targets):
    factor = tf.math.sigmoid((tf.cast(self._step,tf.float32)/1.0e6)*12 - 6)
    if self._c.pcont:  # Last step could be terminal.
      post = {k: v[:, :-1] for k, v in post.items()}
    # Get initial states
    flatten = lambda x: tf.reshape(x, [-1] + list(x.shape[2:]))
    env_start = {k: flatten(v) for k, v in env_post.items()}
    phy_start = {k: flatten(v) for k, v in phy_post.items()}
    # Define Randomization
    A_rand = tf.random.normal([targets.shape[0]]+tf.transpose(self._phy_dynamics.A).shape)*tf.transpose(self._phy_dynamics.A)*0.015*factor
    B_rand = tf.random.normal([targets.shape[0]]+tf.transpose(self._phy_dynamics.B).shape)*tf.transpose(self._phy_dynamics.B)*0.015*factor
    C_rand = tf.random.normal([targets.shape[0]]+tf.transpose(self._phy_dynamics.C).shape)*tf.transpose(self._phy_dynamics.C)*0.01*factor
    D_rand = tf.random.normal([targets.shape[0]]+self._phy_dynamics.D.shape)*tf.transpose(self._phy_dynamics.D)*0.01*factor
    E_rand = tf.transpose(tf.ones([2]+[targets.shape[0]])*(tf.random.normal([targets.shape[0]])*0.2)+1.0)*factor
    # Define Policy and Physics functions
    policy = lambda env_state, phy_state: self._actor(
        tf.concat([tf.stop_gradient(self._env_dynamics.get_feat(env_state)),
                         tf.stop_gradient(self._phy_dynamics.get_feat(phy_state)), targets],-1)).sample()
    physics = lambda state: self._physics(self._phy_dynamics.get_feat(state)).mode()
    # Run imagination
    env_states, phy_states = tools.forward_sync_dynamics_randomization(
        self._env_dynamics.img_step,
        self._phy_dynamics.img_step,
        env_start, phy_start,
        policy, physics,
        tf.range(self._c.horizon),
        randomize=True,
        rand=[A_rand, B_rand, C_rand, D_rand, E_rand])
    # Collect features
    img_env_feat = self._env_dynamics.get_feat(env_states)
    img_phy_feat = self._phy_dynamics.get_feat(phy_states)
    targets = tf.expand_dims(tf.transpose(tf.repeat(targets,img_env_feat.shape[0],1),[1,0]),-1)
    img_full_feat = tf.concat([img_env_feat, img_phy_feat, targets],-1)
    return img_env_feat, img_phy_feat, img_full_feat

def train(config):
  if config.gpu_growth:
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
      tf.config.experimental.set_memory_growth(gpu, True)
  assert config.precision in (16, 32), config.precision
  if config.precision == 16:
    prec.set_policy(prec.Policy('mixed_float16'))
  config.steps = int(config.steps)
  config.logdir.mkdir(parents=True, exist_ok=True)
  print('Logdir', config.logdir)

  # Create environments.
  datadir = config.logdir / 'episodes'
  writer = tf.summary.create_file_writer(str(config.logdir), max_queue=1000, flush_millis=20000)
  writer.set_as_default()
  train_envs = [make_env(config, writer, 'train', datadir, store=True)]
  #test_envs = [make_env(config, writer, 'test', datadir, store=False)]
  actspace = train_envs[0].action_space

  # Prefill dataset with random episodes.
  step = count_steps(datadir, config)
  prefill = max(0, config.prefill - step)
  print(f'Prefill dataset with {prefill} steps.')
  random_agent = lambda o, d, _, __,___ : ([actspace.sample() for _ in d], None, None)
  tools.simulate_2states_goal(random_agent, train_envs, prefill / config.action_repeat)
  writer.flush()

  # Build agent
  step = count_steps(datadir, config)
  print(f'Simulating agent for {config.steps-step} steps.')
  agent = DreamerImg2ImgGoalLinearRSSMPhysics(config, datadir, actspace, writer)

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
    tools.simulate_2states_goal(functools.partial(agent, training=False), train_envs, episodes=1, step=step, target_vel=[[random.random()+0.3]])
    writer.flush()
    print('Start collection.')
    steps = config.eval_every // config.action_repeat
    state = tools.simulate_2states_goal(agent, train_envs, steps, state=state, step=step, target_vel=[[random.random()+0.3]])
    step = count_steps(datadir, config)
    print('Saving.')
    agent.save(config.logdir / 'variables.pkl')
    agent.export4ROS()