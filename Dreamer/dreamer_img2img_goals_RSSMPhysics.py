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
from Dreamer.dreamer_img2img_RSSMPhysics import DreamerImg2ImgRSSMPhysics
from Dreamer.dreamer_img2img_RSSMPhysics import count_steps, load_dataset, summarize_episode, get_last_episode_reward, make_env, preprocess
from Dreamer.base_config import define_config

class DreamerImg2ImgGoalRSSMPhysics(DreamerImg2ImgRSSMPhysics):
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

  def __call__(self, obs, reset, env_state=None, phy_state=None, target_vel=[[1.0]], training=True):
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
    action, env_state, phy_state = self.policy(obs, env_state, phy_state, training, target_vel)
    if training:
      self._step.assign_add(len(reset) * self._c.action_repeat)
    return action, env_state, phy_state
  
  def policy(self, obs, env_state, phy_state, training, target_vel):
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
    feat = tf.concat([self._env_dynamics.get_feat(env_latent), self._phy_dynamics.get_feat(phy_latent), target_vel],-1)
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
      reward_pred = self._reward(img_feat)
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
      targets = tf.random.uniform([img_feat.shape[0]*img_feat.shape[1],1], minval=0.3, maxval=1.3, dtype=tf.dtypes.float32)
      img_env_feat, img_phy_feat, img_full_feat = self._imagine_ahead(env_post, phy_post, targets)
      # Actor training
      # Reward: dist_reward + vel_reward.
      dist_reward = self._reward(img_env_feat).mode()
      linear_vel = tf.transpose(self._physics(img_phy_feat).mode(),[1,0,2])[:,:,0]
      vel_reward = 1 - tf.math.abs((targets - linear_vel)/targets)
      vel_reward = tf.transpose(vel_reward)
      reward = dist_reward + vel_reward

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

  def _imagine_ahead(self, env_post, phy_post, targets):
    if self._c.pcont:  # Last step could be terminal.
      post = {k: v[:, :-1] for k, v in post.items()}
    # Get initial states
    flatten = lambda x: tf.reshape(x, [-1] + list(x.shape[2:]))
    env_start = {k: flatten(v) for k, v in env_post.items()}
    phy_start = {k: flatten(v) for k, v in phy_post.items()}
    
    # Define Policy and Physics functions
    policy = lambda env_state, phy_state: self._actor(
        tf.concat([tf.stop_gradient(self._env_dynamics.get_feat(env_state)),
                         tf.stop_gradient(self._phy_dynamics.get_feat(phy_state)), targets],-1)).sample()
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
  tools.simulate(random_agent, train_envs, prefill / config.action_repeat)
  writer.flush()

  # Build agent
  step = count_steps(datadir, config)
  print(f'Simulating agent for {config.steps-step} steps.')
  agent = DreamerImg2ImgGoalRSSMPhysics(config, datadir, actspace, writer)

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
    tools.simulate(functools.partial(agent, training=False), train_envs, episodes=1, step=step, target_vel=[[random.random()+0.3]])
    writer.flush()
    print('Start collection.')
    steps = config.eval_every // config.action_repeat
    state = tools.simulate(agent, train_envs, steps, state=state, step=step, target_vel=[[random.random()+0.3]])
    step = count_steps(datadir, config)
    print('Saving.')
    agent.save(config.logdir / 'variables.pkl')
    agent.export4ROS()