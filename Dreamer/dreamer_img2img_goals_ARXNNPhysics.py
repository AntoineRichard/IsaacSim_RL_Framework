import argparse
import collections
import functools
import json
import os
import glob
import pathlib
import sys
import time

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

class DreamerImg2ImgGoalARXNNPhysics(DreamerImg2ImgGoalRSSMPhysics):
  def __init__(self, config, datadir, actspace, writer):
    super().__init__(config, datadir, actspace, writer)

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
    # Observation
    phy_post, phy_prior = self._phy_dynamics.observe(data['input_phy'], data['action'])
    # Get features
    phy_feat = self._phy_dynamics.get_feat(phy_post)

    with tf.GradientTape() as env_tape:
      # Observation
      embed = self._encode(data)
      env_post, env_prior = self._env_dynamics.observe(embed, data['prev_phy'])
      # Get features
      img_feat = self._env_dynamics.get_feat(env_post)
      #print(img_feat.shape)
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
      linear_vel = tf.transpose(img_phy_feat,[1,0,2])[...,0]
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
    self._phy_dynamics = models.Dynamics(64,activation="relu")
    self._decode = models.ConvDecoder(self._c.cnn_depth, cnn_act)
    self._reward = models.DenseDecoder((), 2, self._c.num_units, act=act)

    if self._c.pcont:
      self._pcont = models.DenseDecoder(
          (), 3, self._c.num_units, 'binary', act=act)
    self._value = models.DenseDecoder((), 3, self._c.num_units, act=act)
    self._actor = models.ActionDecoder(
        self._actdim, 4, self._c.num_units, self._c.action_dist,
        init_std=self._c.action_init_std, act=act)
    env_modules = [self._encode, self._env_dynamics, self._decode, self._reward]
    
    if self._c.pcont:
      env_modules.append(self._pcont)
    Optimizer = functools.partial(
        tools.Adam, wd=self._c.weight_decay, clip=self._c.grad_clip,
        wdpattern=self._c.weight_decay_pattern)
    
    self._env_opt = Optimizer('model', env_modules, self._c.model_lr)
    self._value_opt = Optimizer('value', [self._value], self._c.value_lr)
    self._actor_opt = Optimizer('actor', [self._actor], self._c.actor_lr)
    # Do a train step to initialize all variables, including optimizer
    # statistics. Ideally, we would use batch size zero, but that doesn't work
    # in multi-GPU mode.
    tmp = self._c.log_scalars
    self.train(next(self._dataset))

  def _imagine_ahead(self, env_post, phy_post, targets):
    if self._c.pcont:  # Last step could be terminal.
      post = {k: v[:, :-1] for k, v in post.items()}
    # Get initial states
    flatten = lambda x: tf.reshape(x, [-1] + list(x.shape[2:]))
    env_start = {k: flatten(v) for k, v in env_post.items()}
    phy_start = {k: flatten(v) for k, v in phy_post.items()}
    # Define Randomization
    #A_rand = tf.random.normal([targets.shape[0]]+tf.transpose(self._phy_dynamics.A).shape)*tf.transpose(self._phy_dynamics.A)*0.015
    #B_rand = tf.random.normal([targets.shape[0]]+tf.transpose(self._phy_dynamics.B).shape)*tf.transpose(self._phy_dynamics.B)*0.015
    # Define Policy and Physics functions
    policy = lambda env_state, phy_state: self._actor(
        tf.concat([tf.stop_gradient(self._env_dynamics.get_feat(env_state)),
                         tf.stop_gradient(self._phy_dynamics.get_feat(phy_state)), targets],-1)).sample()
    physics = lambda state: self._phy_dynamics.get_feat(state)
    # Run imagination
    env_states, phy_states = tools.forward_sync_dynamics_randomization(
        self._env_dynamics.img_step,
        self._phy_dynamics.img_step,
        env_start, phy_start,
        policy, physics,
        tf.range(self._c.horizon),
        randomize=False,
        rand=[])
    # Collect features
    img_env_feat = self._env_dynamics.get_feat(env_states)
    img_phy_feat = self._phy_dynamics.get_feat(phy_states)
    targets = tf.expand_dims(tf.transpose(tf.repeat(targets,img_env_feat.shape[0],1),[1,0]),-1)
    img_full_feat = tf.concat([img_env_feat, img_phy_feat, targets],-1)
    return img_env_feat, img_phy_feat, img_full_feat

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
    # Uncertainty
    phy_obs_std = tf.ones_like(phy_init_feat)*0.1
    phy_pred_std = tf.ones_like(phy_feat)*0.1
    # Concat and dump
    phy_model = tf.concat([phy_init_feat, phy_feat], 1)
    phy_model_std = tf.concat([phy_obs_std, phy_pred_std], 1)
    # Add actions
    phy_model_std = tf.concat([phy_model_std, tf.zeros_like(phy_model_std[:,:,:2])], -1)
    phy_model = tf.concat([phy_model, data['action'][:3]], -1)
    phy_truth = tf.concat([phy_truth, data['action'][:3]], -1)
    return phy_model, phy_model_std, phy_truth

  def export4ROS(self, path="latest"):
    os.makedirs(os.path.join(self._c.logdir,path), exist_ok=True)
    self._encode.save(os.path.join(self._c.logdir,path,'encoder_weights.pkl'))
    self._decode.save(os.path.join(self._c.logdir,path,'decoder_weights.pkl'))
    self._env_dynamics.save(os.path.join(self._c.logdir,path,'env_dynamics_weights.pkl'))
    self._phy_dynamics.save(os.path.join(self._c.logdir,path,'phy_dynamics_weights.pkl'))
    self._actor.save(os.path.join(self._c.logdir,path,'actor_weights.pkl'))
    self._reward.save(os.path.join(self._c.logdir,path,'reward_weights.pkl'))
    self._value.save(os.path.join(self._c.logdir,path,'value_weights.pkl'))