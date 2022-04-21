import argparse
import collections
import functools
import json
import os
import glob
import pathlib
import sys
import time
#import http.client
#from bottle import Bottle, request
from matplotlib import pyplot as plt
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
  config.task = 'dmc_walker_walk'
  config.envs = 1
  config.parallel = 'none'
  config.action_repeat = 1
  config.time_limit = 500
  config.prefill = 2500
  config.eval_noise = 0.0
  config.clip_rewards = 'none'
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
  config.train_every = 1000
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


class Dreamer(tools.Module):

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

  def load(self, filename):
    super().load(filename)
    self._should_pretrain()

  def __call__(self, obs, reset, env_state=None, phy_state=None, target_vel=[[1.0]], training=True):
    step = self._step.numpy().item()
    tf.summary.experimental.set_step(step)
    if env_state is not None and reset.any():
      mask = tf.cast(1 - reset, self._float)[:, None]
      env_state = tf.nest.map_structure(lambda x: x * mask, env_state)
    if phy_state is not None and reset.any():
      mask = tf.cast(1 - reset, self._float)[:, None]
      phy_state = tf.nest.map_structure(lambda x: x * mask, phy_state)
    if self._should_train(step):
      log = self._should_log(step)
      n = self._c.pretrain if self._should_pretrain() else self._c.train_steps
      print(f'Training for {n} steps.')
      #with self._strategy.scope():
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

  def _exploration(self, action, training):
    if training:
      amount = self._c.expl_amount
      if self._c.expl_decay:
        amount *= 0.5 ** (tf.cast(self._step, tf.float32) / self._c.expl_decay)
      if self._c.expl_min:
        amount = tf.maximum(self._c.expl_min, amount)
    elif self._c.eval_noise:
      amount = self._c.eval_noise
    else:
      return action
    if self._c.expl == 'additive_gaussian':
      return tf.clip_by_value(tfd.Normal(action, amount).sample(), -1, 1)
    if self._c.expl == 'completely_random':
      return tf.random.uniform(action.shape, -1, 1)
    if self._c.expl == 'epsilon_greedy':
      indices = tfd.Categorical(0 * action).sample()
      return tf.where(
          tf.random.uniform(action.shape[:1], 0, 1) < amount,
          tf.one_hot(indices, action.shape[-1], dtype=self._float),
          action)
    raise NotImplementedError(self._c.expl)

  @tf.function()
  def train(self, data, log_images=False):
    self._train(data, log_images)

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
      #targets_neg = tf.random.uniform([img_feat.shape[0]*img_feat.shape[1] - int(img_feat.shape[0]*img_feat.shape[1]*0.7),1], minval=-0.5, maxval=-0.2, dtype=tf.dtypes.float32)
      #targets = tf.concat([targets_pos, targets_neg],axis=0)
      #targets = tf.ones([img_feat.shape[0]*img_feat.shape[1],1], dtype=tf.dtypes.float32)
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
    #return data

  def _build_model(self):
    acts = dict(
        elu=tf.nn.elu, relu=tf.nn.relu, swish=tf.nn.swish,
        leaky_relu=tf.nn.leaky_relu)
    cnn_act = acts[self._c.cnn_act]
    act = acts[self._c.dense_act]
    self._encode = models.ConvEncoder(self._c.cnn_depth, cnn_act)
    self._env_dynamics = models.RSSMv2(self._c.stoch_size, self._c.deter_size, self._c.deter_size)
    self._phy_dynamics = models.Dynamics(64,activation="relu")#self._c.phy_stoch_size, self._c.phy_deter_size, self._c.phy_deter_size)
    self._decode = models.ConvDecoder(self._c.cnn_depth, cnn_act)
    self._reward = models.DenseDecoder((), 2, self._c.num_units, act=act)
    #self._physics = models.DenseDecoder([3], 1, self._c.phy_num_units, act=act)
    if self._c.pcont:
      self._pcont = models.DenseDecoder(
          (), 3, self._c.num_units, 'binary', act=act)
    self._value = models.DenseDecoder((), 3, self._c.num_units, act=act)
    self._actor = models.ActionDecoder(
        self._actdim, 4, self._c.num_units, self._c.action_dist,
        init_std=self._c.action_init_std, act=act)
    env_modules = [self._encode, self._env_dynamics, self._decode, self._reward]
    #phy_modules = [self._physics, self._phy_dynamics]
    if self._c.pcont:
      model_modules.append(self._pcont)
    Optimizer = functools.partial(
        tools.Adam, wd=self._c.weight_decay, clip=self._c.grad_clip,
        wdpattern=self._c.weight_decay_pattern)
    #self._phy_opt = Optimizer('model', phy_modules, self._c.model_lr)
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
    env_states, phy_states = tools.forward_sync_DR(
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

  def _scalar_summaries(
      self, data, feat, prior_dist, post_dist, likes, div,
      model_loss, value_loss, actor_loss, model_norm, value_norm,
      actor_norm):
    self._metrics['model_grad_norm'].update_state(model_norm)
    self._metrics['value_grad_norm'].update_state(value_norm)
    self._metrics['actor_grad_norm'].update_state(actor_norm)
    self._metrics['prior_ent'].update_state(prior_dist.entropy())
    self._metrics['post_ent'].update_state(post_dist.entropy())
    for name, logprob in likes.items():
      self._metrics[name + '_loss'].update_state(-logprob)
    self._metrics['div'].update_state(div)
    self._metrics['model_loss'].update_state(model_loss)
    self._metrics['value_loss'].update_state(value_loss)
    self._metrics['actor_loss'].update_state(actor_loss)
    #self._metrics['action_ent'].update_state(self._actor(feat).entropy())

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
    #phy_obs = phy_init_feat#self._physics(phy_init_feat)#.mode()
    #phy_pred = #self._physics(phy_feat)#.mode()
    # Uncertainty
    phy_obs_std = tf.ones_like(phy_init_feat)*0.1#self._physics(phy_init_feat)#.stddev()
    phy_pred_std = tf.ones_like(phy_feat)*0.1#self._physics(phy_feat)#.stddev()
    # Concat and dump
    phy_model = tf.concat([phy_init_feat, phy_feat], 1)
    phy_model_std = tf.concat([phy_obs_std, phy_pred_std], 1)
    # Add actions
    phy_model_std = tf.concat([phy_model_std, tf.zeros_like(phy_model_std[:,:,:2])], -1)
    phy_model = tf.concat([phy_model, data['action'][:3]], -1)
    phy_truth = tf.concat([phy_truth, data['action'][:3]], -1)
    return phy_model, phy_model_std, phy_truth

  def _write_summaries(self):
    step = int(self._step.numpy())
    metrics = [(k, float(v.result())) for k, v in self._metrics.items()]
    if self._last_log is not None:
      duration = time.time() - self._last_time
      self._last_time += duration
      metrics.append(('fps', (step - self._last_log) / duration))
    self._last_log = step
    [m.reset_states() for m in self._metrics.values()]
    with (self._c.logdir / 'metrics.jsonl').open('a') as f:
      f.write(json.dumps({'step': step, **dict(metrics)}) + '\n')
    [tf.summary.scalar('agent/' + k, m) for k, m in metrics]
    print(f'[{step}]', ' / '.join(f'{k} {v:.1f}' for k, v in metrics))
    self._writer.flush()

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

def count_steps(datadir, config):
  return tools.count_episodes(datadir)[1] * config.action_repeat

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

def summarize_episode(episode, config, datadir, writer, prefix):
  episodes, steps = tools.count_episodes(datadir)
  length = (len(episode['reward']) - 1) * config.action_repeat
  ret = episode['reward'].sum()
  print(f'{prefix.title()} episode of length {length} with return {ret:.1f}.')
  metrics = [
      (f'{prefix}/return', float(episode['reward'].sum())),
      (f'{prefix}/length', len(episode['reward']) - 1),
      (f'episodes', episodes)]
  step = count_steps(datadir, config)
  with (config.logdir / 'metrics.jsonl').open('a') as f:
    f.write(json.dumps(dict([('step', step)] + metrics)) + '\n')
  with writer.as_default():  # Env might run in a different thread.
    tf.summary.experimental.set_step(step)
    [tf.summary.scalar('sim/' + k, v) for k, v in metrics]
    #if prefix == 'test':
    tools.video_summary(f'sim/{prefix}/video', episode['image'][None], step=step)

def summarize_train(data, agent, step):
  with agent._writer.as_default(): 
    tf.summary.experimental.set_step(step)
    tools.video_summary('agent/environment_reconstruction',np.array(agent.image_summaries(data)),step = step)
    rec_phy, rec_phy_std, true_phy = agent.plot_dynamics(data)
    tools.plot_summary('agent/dynamics_reconstruction', np.array(rec_phy), np.array(rec_phy_std), np.array(true_phy), step=step)

def get_last_episode_reward(config, datadir, writer):
  list_of_files = glob.glob(str(datadir)+'/*.npz')
  latest_file = max(list_of_files, key=os.path.getctime)
  episode = np.load(latest_file)
  episode = {k: episode[k] for k in episode.keys()}
  ret = float(episode['reward'][-int(len(episode['reward'])/2):].sum())
  return ret

def make_env(config, writer, prefix, datadir, store):
  env = wrappers.IsaacSim()
  env = wrappers.TimeLimit(env, config.time_limit / config.action_repeat)
  callbacks = []
  if store:
    callbacks.append(lambda ep: tools.save_episodes(datadir, [ep]))
  callbacks.append(
      lambda ep: summarize_episode(ep, config, datadir, writer, prefix))
  env = wrappers.Collect(env, callbacks, config.precision)
  env = wrappers.RewardObs(env)
  return env

if __name__ == '__main__':
  try:
    import colored_traceback
    colored_traceback.add_hook()
  except ImportError:
    pass
  parser = argparse.ArgumentParser()
  for key, value in define_config().items():
    parser.add_argument(f'--{key}', type=tools.args_type(value), default=value)
  main(parser.parse_args())
