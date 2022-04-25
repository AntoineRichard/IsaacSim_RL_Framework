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
from Dreamer.dreamer_img2img_NoPhysics import DreamerImg2ImgNoPhysics
from Dreamer.dreamer_img2img_NoPhysics import count_steps, summarize_episode, get_last_episode_reward, make_env  
from Dreamer.base_config import define_config

class DreamerImg2ImgRSSMPhysics(DreamerImg2ImgNoPhysics):
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

  def __call__(self, obs, reset, env_state=None, phy_state=None, training=True):
    step = self._step.numpy().item()
    tf.summary.experimental.set_step(step)
    if env_state is not None and reset.any():
      mask = tf.cast(1 - reset, self._float)[:, None]
      env_state = tf.nest.map_structure(lambda x: x * mask, env_state)
    if phy_state is not None and reset.any():
      mask = tf.cast(1 - reset, self._float)[:, None]
      phy_state = tf.nest.map_structure(lambda x: x * mask, phy_state)
    if self._should_train(step) and training:
      log = self._should_log(step)
      n = self._c.pretrain if self._should_pretrain() else self._c.train_steps
      print(f'Training for {n} steps.')
      for train_step in range(n):
        log_images = self._c.log_images and log and train_step == 0
        data = next(self._dataset)
        self.train(data, log_images)
        if log_images:
          with self._writer.as_default(): 
            tf.summary.experimental.set_step(step)
            tools.video_summary('agent/environment_reconstruction',np.array(self.image_summaries(data)),step = step)
            rec_phy, rec_phy_std, true_phy = self.plot_dynamics(data)
            tools.plot_summary('agent/dynamics_reconstruction', np.array(rec_phy), np.array(rec_phy_std), np.array(true_phy), step=step)
      if log:
        self._write_summaries()
    action, env_state, phy_state = self.policy(obs, env_state, phy_state, training)
    if training:
      self._step.assign_add(len(reset) * self._c.action_repeat)
    return action, env_state, phy_state

  def policy(self, obs, env_state, phy_state, training):
    if env_state is None:
      env_latent = self._env_dynamics.initial(len(obs['image']))
      action = tf.zeros((len(obs['image']), self._actdim), self._float)
    else:
      env_latent, action = env_state
    if phy_state is None:
      phy_latent = self._phy_dynamics.initial(len(obs['image']))
      action = tf.zeros((len(obs['image']), self._actdim), self._float)
    else:
      phy_latent, action = phy_state
    obs = preprocess(obs, self._c)
    embed = self._encode(obs)
    env_latent, _ = self._env_dynamics.obs_step(env_latent, obs['prev_phy'], embed, sample=False)
    phy_latent, _ = self._phy_dynamics.obs_step(phy_latent, action, obs['input_phy'], sample=False)
    feat = tf.concat([self._env_dynamics.get_feat(env_latent), self._phy_dynamics.get_feat(phy_latent)],-1)
    if training:
      action = self._actor(feat).sample()
    else:
      action = self._actor(feat).mode()
    action = self._exploration(action, training)
    env_state = (env_latent, action)
    phy_state = (phy_latent, action)
    return action, env_state, phy_state

  def _train(self, data, log_images):
    likes = tools.AttrDict()
    with tf.GradientTape() as phy_tape:
      # Observation
      phy_post, phy_prior = self._phy_dynamics.observe(data['input_phy'], data['action'])
      # Get features
      phy_feat = self._phy_dynamics.get_feat(phy_post)
      # Reconstruct
      physics_pred = self._physics(phy_feat)
      # Reconstruction errors
      likes.physics = tf.reduce_mean(physics_pred.log_prob(data['input_phy']))
      # World model loss
      phy_prior_dist = self._env_dynamics.get_dist(phy_prior)
      phy_post_dist = self._env_dynamics.get_dist(phy_post)
      phy_div = tf.reduce_mean(tfd.kl_divergence(phy_post_dist, phy_prior_dist))
      phy_div = tf.maximum(phy_div, 1.5)
      phy_loss = - likes.physics + self._c.kl_scale * phy_div
 
    with tf.GradientTape() as env_tape:
      # Observation
      embed = self._encode(data)
      env_post, env_prior = self._env_dynamics.observe(embed, data['prev_phy'])
      # Get features
      img_feat = self._env_dynamics.get_feat(env_post)
      full_feat = tf.concat([img_feat, tf.stop_gradient(phy_feat)],-1)
      # Reconstruct
      image_pred = self._decode(img_feat)
      reward_pred = self._reward(full_feat)
      # Reconstruction errors
      likes.image = tf.reduce_mean(image_pred.log_prob(data['image']))
      likes.reward = tf.reduce_mean(reward_pred.log_prob(data['reward']))
      # World model loss
      if self._c.pcont:
        pcont_pred = self._pcont(full_feat)
        pcont_target = self._c.discount * data['discount']
        likes.pcont = tf.reduce_mean(pcont_pred.log_prob(pcont_target))
        likes.pcont *= self._c.pcont_scale
      # Maximize the use of the internal state
      env_prior_dist = self._env_dynamics.get_dist(env_prior)
      env_post_dist = self._env_dynamics.get_dist(env_post)
      env_div = tf.reduce_mean(tfd.kl_divergence(env_post_dist, env_prior_dist))
      env_div = tf.maximum(env_div, self._c.free_nats)
      # World model loss
      env_loss = - likes.reward - likes.image + self._c.kl_scale * env_div
     
    # Actor
    with tf.GradientTape() as actor_tape:
      # Imagination
      img_env_feat, img_phy_feat, img_full_feat = self._imagine_ahead(env_post, phy_post)
      # Actor training
      reward = self._reward(img_full_feat).mode()
      if self._c.pcont:
        pcont = self._pcont(img_env_feat).mean()
      else:
        pcont = self._c.discount * tf.ones_like(reward)
      value = self._value(img_full_feat).mode()
      returns = tools.lambda_return(
          reward[:-1], value[:-1], pcont[:-1],
          bootstrap=value[-1], lambda_=self._c.disclam, axis=0)
      discount = tf.stop_gradient(tf.math.cumprod(tf.concat(
          [tf.ones_like(pcont[:1]), pcont[:-2]], 0), 0))
      actor_loss = -tf.reduce_mean(discount * returns)

    # Value
    with tf.GradientTape() as value_tape:
      value_pred = self._value(img_full_feat)[:-1]
      target = tf.stop_gradient(returns)
      value_loss = -tf.reduce_mean(discount * value_pred.log_prob(target))

    env_norm = self._env_opt(env_tape, env_loss)
    phy_norm = self._phy_opt(phy_tape, phy_loss)
    actor_norm = self._actor_opt(actor_tape, actor_loss)
    value_norm = self._value_opt(value_tape, value_loss)

    if self._c.log_scalars:
      self._scalar_summaries(
          data, full_feat, env_prior_dist, env_post_dist, likes, env_div,
          env_loss, value_loss, actor_loss, env_norm, value_norm,
          actor_norm)

  def _build_model(self):
    acts = dict(
        elu=tf.nn.elu, relu=tf.nn.relu, swish=tf.nn.swish,
        leaky_relu=tf.nn.leaky_relu)
    cnn_act = acts[self._c.cnn_act]
    act = acts[self._c.dense_act]
    self._encode = models.ConvEncoder(self._c.cnn_depth, cnn_act)
    self._env_dynamics = models.RSSMv2(self._c.stoch_size, self._c.deter_size, self._c.deter_size)
    self._phy_dynamics = models.RSSMv2(self._c.phy_stoch_size, self._c.phy_deter_size, self._c.phy_deter_size)
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

  def _imagine_ahead(self, env_post, phy_post):
    if self._c.pcont:  # Last step could be terminal.
      post = {k: v[:, :-1] for k, v in post.items()}
    # Get initial states
    flatten = lambda x: tf.reshape(x, [-1] + list(x.shape[2:]))
    env_start = {k: flatten(v) for k, v in env_post.items()}
    phy_start = {k: flatten(v) for k, v in phy_post.items()}

    # Define Policy and Physics functions
    policy = lambda env_state, phy_state: self._actor(
        tf.concat([tf.stop_gradient(self._env_dynamics.get_feat(env_state)),
                         tf.stop_gradient(self._phy_dynamics.get_feat(phy_state))],-1)).sample()
    physics = lambda state: self._physics(self._phy_dynamics.get_feat(state)).mode()
    # Run imagination
    env_states, phy_states = tools.forward_sync_RSSMv2(
        self._env_dynamics.img_step,
        self._phy_dynamics.img_step,
        env_start, phy_start,
        policy, physics,
        tf.range(self._c.horizon))
    # Collect features
    img_env_feat = self._env_dynamics.get_feat(env_states)
    img_phy_feat = self._phy_dynamics.get_feat(phy_states)
    img_full_feat = tf.concat([img_env_feat, img_phy_feat],-1)
    return img_env_feat, img_phy_feat, img_full_feat

  @tf.function
  def image_summaries(self, data):
    # Real 
    env_truth = data['image'][:6] + 0.5
    # Initial states (5 steps warmup)
    embed = self._encode(data)
    env_init, _ = self._env_dynamics.observe(embed[:6, :5], data['prev_phy'][:6, :5])
    env_init_feat = self._env_dynamics.get_feat(env_init)
    env_init = {k: v[:, -1] for k, v in env_init.items()}
    # Environment imagination
    env_prior = self._env_dynamics.imagine(data['prev_phy'][:6, 5:], env_init) 
    env_feat = self._env_dynamics.get_feat(env_prior)
    # Environment reconstruction
    env_obs = self._decode(env_init_feat).mode()
    openl = self._decode(env_feat).mode()
    env_model = tf.concat([env_obs[:, :5] + 0.5, openl + 0.5], 1)
    error = (env_model - env_truth + 1) / 2
    openl = tf.concat([env_truth, env_model, error], 2)
    return openl

  @tf.function
  def plot_dynamics(self, data):
    # Real 
    phy_truth = data['physics'][:3]
    # Initial states (5 steps warmup)
    phy_init, _ = self._phy_dynamics.observe(data['input_phy'][:3, :5], data['action'][:3, :5])
    phy_init_feat = self._phy_dynamics.get_feat(phy_init)
    phy_init = {k: v[:, -1] for k, v in phy_init.items()}
    # Physics imagination
    phy_prior = self._phy_dynamics.imagine(data['action'][:3, 5:], phy_init, sample=False)
    phy_feat = self._phy_dynamics.get_feat(phy_prior)
    # Physics reconstruction
    phy_obs = self._physics(phy_init_feat).mode()
    phy_pred = self._physics(phy_feat).mode()
    # Uncertainty
    phy_obs_std = self._physics(phy_init_feat).stddev()
    phy_pred_std = self._physics(phy_feat).stddev()
    # Concat and dump
    phy_model = tf.concat([phy_obs, phy_pred], 1)
    phy_model_std = tf.concat([phy_obs_std, phy_pred_std], 1)
    return phy_model, phy_model_std, phy_truth

  def export4ROS(self, path="latest"):
    os.makedirs(os.path.join(self._c.logdir,path), exist_ok=True)
    self._encode.save(os.path.join(self._c.logdir,path,'encoder_weights.pkl'))
    self._decode.save(os.path.join(self._c.logdir,path,'decoder_weights.pkl'))
    self._env_dynamics.save(os.path.join(self._c.logdir,path,'env_dynamics_weights.pkl'))
    self._phy_dynamics.save(os.path.join(self._c.logdir,path,'phy_dynamics_weights.pkl'))
    self._physics.save(os.path.join(self._c.logdir,path,'physics_weights.pkl'))
    self._actor.save(os.path.join(self._c.logdir,path,'actor_weights.pkl'))
    self._reward.save(os.path.join(self._c.logdir,path,'reward_weights.pkl'))
    self._value.save(os.path.join(self._c.logdir,path,'value_weights.pkl'))

def preprocess(obs, config):
  dtype = prec.global_policy().compute_dtype
  obs = obs.copy()
  with tf.device('cpu:0'):
    obs['input_phy'] = tf.cast(obs['physics'],dtype)
    obs['prev_phy'] = tf.cast(obs['physics_d'],dtype)
    obs['image'] = tf.cast(obs['image'], dtype) / 255.0 - 0.5
    clip_rewards = dict(none=lambda x: x, tanh=tf.tanh)[config.clip_rewards]
    obs['reward'] = clip_rewards(obs['reward'])
  return obs

def load_dataset(directory, config):
  episode = next(tools.load_episodes(directory, 1))
  types = {k: v.dtype for k, v in episode.items()}
  shapes = {k: (None,) + v.shape[1:] for k, v in episode.items()}
  generator = lambda: tools.load_episodes(
      directory, config.train_steps, config.batch_length,
      config.dataset_balance)
  dataset = tf.data.Dataset.from_generator(generator, types, shapes)
  dataset = dataset.batch(config.batch_size, drop_remainder=True)
  dataset = dataset.map(functools.partial(preprocess, config=config))
  dataset = dataset.prefetch(10)
  return dataset

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
  random_agent = lambda o, d, _, __ : ([actspace.sample() for _ in d], None, None)
  tools.simulate_2states(random_agent, train_envs, prefill / config.action_repeat)
  writer.flush()

  # Build agent
  step = count_steps(datadir, config)
  print(f'Simulating agent for {config.steps-step} steps.')
  agent = DreamerImg2ImgRSSMPhysics(config, datadir, actspace, writer)

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
    tools.simulate_2states(functools.partial(agent, training=False), train_envs, episodes=1, step=step)
    writer.flush()
    print('Start collection.')
    steps = config.eval_every // config.action_repeat
    state = tools.simulate_2states(agent, train_envs, steps, state=state, step=step)
    step = count_steps(datadir, config)
    print('Saving.')
    agent.save(config.logdir / 'variables.pkl')
    agent.export4ROS()