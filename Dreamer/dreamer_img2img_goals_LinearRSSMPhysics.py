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

class DreamerImg2ImgGoalLinearRSSMPhysics(DreamerImg2ImgGoalRSSMPhysics):
  def __init__(self, config, datadir, actspace, writer):
    super.__init__(self, config, datadir, actspace, writer)

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
    if self._c.pcont:  # Last step could be terminal.
      post = {k: v[:, :-1] for k, v in post.items()}
    # Get initial states
    flatten = lambda x: tf.reshape(x, [-1] + list(x.shape[2:]))
    env_start = {k: flatten(v) for k, v in env_post.items()}
    phy_start = {k: flatten(v) for k, v in phy_post.items()}
    # Define Randomization
    A_rand = tf.random.normal([targets.shape[0]]+tf.transpose(self._phy_dynamics.A).shape)*tf.transpose(self._phy_dynamics.A)*0.015
    B_rand = tf.random.normal([targets.shape[0]]+tf.transpose(self._phy_dynamics.B).shape)*tf.transpose(self._phy_dynamics.B)*0.015
    C_rand = tf.random.normal([targets.shape[0]]+tf.transpose(self._phy_dynamics.C).shape)*tf.transpose(self._phy_dynamics.C)*0.01
    D_rand = tf.random.normal([targets.shape[0]]+self._phy_dynamics.D.shape)*tf.transpose(self._phy_dynamics.D)*0.01
    E_rand = tf.transpose(tf.ones([2]+[targets.shape[0]])*(tf.random.normal([targets.shape[0]])*0.2)+1.0)
    # Define Policy and Physics functions
    policy = lambda env_state, phy_state: self._actor(
        tf.concat([tf.stop_gradient(self._env_dynamics.get_feat(env_state)),
                         tf.stop_gradient(self._phy_dynamics.get_feat(phy_state)), targets],-1)).sample()
    physics = lambda state: self._physics(self._phy_dynamics.get_feat(state)).mode()
    # Run imagination
    env_states, phy_states = tools.forward_sync_DR(
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