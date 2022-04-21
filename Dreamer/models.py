import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as tfkl
from tensorflow_probability import distributions as tfd
from tensorflow.keras.mixed_precision import experimental as prec

import Dreamer.tools as tools

def Conv1DTranspose(input_tensor, filters, kernel_size, strides=2, padding='same', activation='relu'):
    """
        input_tensor: tensor, with the shape (batch_size, time_steps, dims)
        filters: int, output dimension, i.e. the output tensor will have the shape of (batch_size, time_steps, filters)
        kernel_size: int, size of the convolution kernel
        strides: int, convolution step size
        padding: 'same' | 'valid'
    """
    x = tfkl.Lambda(lambda x: tf.expand_dims(x, axis=2))(input_tensor)
    x = tfkl.Conv2DTranspose(filters=filters, kernel_size=(kernel_size, 1), strides=(strides, 1), padding=padding, activation=activation)(x)
    x = tfkl.Lambda(lambda x: tf.squeeze(x, axis=2))(x)
    return x

class Dynamics(tools.Module):
    def __init__(self, hidden, action_size=2, state_size=3, history_length=2, activation="elu"):
        super().__init__()
        self._hidden_size = hidden
        self._action_size = action_size
        self._state_size = state_size
        self._history_length = history_length
        self._activation = activation
        self.A = tf.Variable(np.zeros([self._state_size*self._history_length ,self._state_size], dtype=np.float32),trainable=False, name="A")
        self.B = tf.Variable(np.zeros([self._action_size*self._history_length,self._state_size], dtype=np.float32),trainable=False,name="B")

    def initial(self, bs):
        dtype = tf.float32#prec.global_policy().compute_dtype
        return dict(
                physics=tf.zeros([bs, self._state_size*self._history_length],dtype),
                actions=tf.zeros([bs, self._action_size*self._history_length],dtype),
                prediction=tf.zeros([bs, self._state_size],dtype))

    @tf.function
    def observe(self, physics, action, state=None):
        if state is None: # No state --> set state to 0
            state = self.initial(physics.shape[0])
        physics = tf.transpose(physics, [1, 0, 2]) # BS,Length,Feat --> Length, BS, feat
        action = tf.transpose(action, [1, 0, 2]) # BS,Length,Feat --> Length, BS, feat
        post, prior = tools.static_scan(
            lambda prev, inputs: self.obs_step(prev[0], *inputs), # transforms the data to state, action, embed
            (action, physics), (state, state)) # Applies obs_step to each element of the sequence in a batch fashion
        post = {k: tf.transpose(v, [1, 0, 2]) for k, v in post.items()} # Undo previous transpose
        prior = {k: tf.transpose(v, [1, 0, 2]) for k, v in prior.items()} # Undo previous transpose
        return post, prior #post : (state,state), prior:

    @tf.function
    def imagine(self, action, state=None, sample=False, randomize=False, rand=[]):
        if state is None:
            state = self.initial(tf.shape(action)[0])
        assert isinstance(state, dict), state
        action = tf.transpose(action, [1, 0, 2])
        #prior = tools.static_scan(self.img_step, action, state, sample, rand)
        prior = tools.static_scan_dyn(self.img_step, action, state, sample=sample, randomize=randomize, rand=rand)
        prior = {k: tf.transpose(v, [1, 0, 2]) for k, v in prior.items()}
        return prior

    def get_feat(self, state):
        return state['physics'][...,-self._state_size:]

    def update_linear_model(self, A,B):
        self.A = A
        self.B = B

    @tf.function
    def update_history(self, state, phy, action):
        new_state = {}
        new_state['physics'] = tf.concat([state["physics"][:,-self._state_size*(self._history_length-1):], phy],axis=-1)
        new_state['actions'] = tf.concat([state["actions"][:,-self._action_size*(self._history_length-1):], action],axis=-1)
        new_state['prediction'] = phy
        return new_state
    
    @tf.function
    def update_physics(self, state, phy):
        new_state = {}
        new_state['physics'] = tf.concat([state["physics"][:,-self._state_size*(self._history_length-1):], phy],axis=-1)
        new_state['actions'] = state["actions"]
        return new_state
    
    @tf.function
    def update_actions(self, state, action):
        new_state = {}
        new_state['physics'] = state["physics"]
        new_state['actions'] = tf.concat([state["actions"][:,-self._action_size*(self._history_length-1):], action],axis=-1)
        return new_state

    @tf.function
    def apply_linear_model(self, state, randomize=False, rand=[]):
        if randomize:
            A_rand = rand[0]
            Ka = rand[1]
            B_rand = rand[2]
            Kb = rand[3]
            #tmp1 = tf.transpose(self.A) + A_rand
            #tf.print(tmp1.shape)
            #tmp2 = tf.transpose(self.B) + B_rand
            #tf.print(tmp2.shape)
            #tf.print(Kb.shape)
            linear = tf.squeeze(tf.matmul(a=(tf.transpose(self.A) + A_rand) * Ka, b=tf.expand_dims(state['physics'],axis=-1)))
            command = tf.squeeze(tf.matmul(a=(tf.transpose(self.B) + B_rand) * Kb, b=tf.expand_dims(state['actions'],axis=-1)))
        else:
            linear = tf.matmul(b=self.A, a=state['physics'])
            command = tf.matmul(b=self.B, a=state['actions'])
        linear_state = linear + command
        return linear_state

    @tf.function
    def nonlinear_residual(self, state, linear):
        x = tf.concat([state['physics'],state['actions'],linear],axis=-1)
        x = self.get('dyn1', tfkl.Dense, self._hidden_size, self._activation)(x)
        x = self.get('dyn2', tfkl.Dense, self._hidden_size, self._activation)(x)
        x = self.get('dyn3', tfkl.Dense, self._state_size)(x)
        return x

    @tf.function
    def obs_step(self, prev_state, prev_action, observation, sample=True):
        new_state = self.update_history(prev_state, observation, prev_action)
        linear_state = self.apply_linear_model(new_state)
        residual = self.nonlinear_residual(new_state, linear_state)
        new_state['prediction'] = linear_state+residual
        return new_state, prev_state

    @tf.function
    def img_step(self, prev_state, prev_action, sample=True, randomize=False, rand=[]):
        #new_state = self.update_actions(prev_state, prev_action)
        new_state = self.update_history(prev_state, prev_state['prediction'], prev_action)
        linear_state = self.apply_linear_model(new_state, randomize=randomize, rand=rand)
        residual = self.nonlinear_residual(new_state, linear_state)
        new_state['prediction'] = linear_state+residual
        #new_state = self.update_physics(new_state, linear_state + residual)
        return new_state

class RSSM(tools.Module):
  def __init__(self, stoch=30, deter=200, hidden=200, act=tf.nn.elu):
    super().__init__()
    self._activation = act
    self._stoch_size = stoch
    self._deter_size = deter
    self._hidden_size = hidden
    self._cell = tfkl.GRUCell(self._deter_size)

  def initial(self, batch_size):
    dtype = prec.global_policy().compute_dtype
    return dict(
        mean=tf.zeros([batch_size, self._stoch_size], dtype),
        std=tf.zeros([batch_size, self._stoch_size], dtype),
        stoch=tf.zeros([batch_size, self._stoch_size], dtype),
        deter=self._cell.get_initial_state(None, batch_size, dtype))

  @tf.function
  def observe(self, embed, action, state=None):
    if state is None: # No state --> set state to 0
      state = self.initial(tf.shape(action)[0])
    embed = tf.transpose(embed, [1, 0, 2]) # BS,Length,Feat --> Length, BS, feat
    action = tf.transpose(action, [1, 0, 2]) # BS,Length,Feat --> Length, BS, feat
    post, prior = tools.static_scan(
        lambda prev, inputs: self.obs_step(prev[0], *inputs), # transforms the data to state, action, embed
        (action, embed), (state, state)) # Applies obs_step to each element of the sequence in a batch fashion
    post = {k: tf.transpose(v, [1, 0, 2]) for k, v in post.items()} # Undo previous transpose
    prior = {k: tf.transpose(v, [1, 0, 2]) for k, v in prior.items()} # Undo previous transpose
    return post, prior #post : (state,state), prior:

  @tf.function
  def imagine(self, action, state=None):
    if state is None:
      state = self.initial(tf.shape(action)[0])
    assert isinstance(state, dict), state
    action = tf.transpose(action, [1, 0, 2])
    prior = tools.static_scan(self.img_step, action, state)
    prior = {k: tf.transpose(v, [1, 0, 2]) for k, v in prior.items()}
    return prior

  def get_feat(self, state):
    return tf.concat([state['stoch'], state['deter']], -1)

  def get_dist(self, state):
    return tfd.MultivariateNormalDiag(state['mean'], state['std'])

  @tf.function
  def obs_step(self, prev_state, prev_action, embed):
    prior = self.img_step(prev_state, prev_action)
    x = tf.concat([prior['deter'], embed], -1)
    x = self.get('obs1', tfkl.Dense, self._hidden_size, self._activation)(x)
    x = self.get('obs2', tfkl.Dense, 2 * self._stoch_size, None)(x)
    mean, std = tf.split(x, 2, -1)
    std = tf.nn.softplus(std) + 0.1
    stoch = self.get_dist({'mean': mean, 'std': std}).sample()
    post = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': prior['deter']}
    return post, prior

  @tf.function
  def img_step(self, prev_state, prev_action):
    x = tf.concat([prev_state['stoch'], prev_action], -1)
    x = self.get('img1', tfkl.Dense, self._hidden_size, self._activation)(x)
    x, deter = self._cell(x, [prev_state['deter']])
    deter = deter[0]  # Keras wraps the state in a list.
    x = self.get('img2', tfkl.Dense, self._hidden_size, self._activation)(x)
    x = self.get('img3', tfkl.Dense, 2 * self._stoch_size, None)(x)
    mean, std = tf.split(x, 2, -1)
    std = tf.nn.softplus(std) + 0.1
    stoch = self.get_dist({'mean': mean, 'std': std}).sample()
    prior = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': deter}
    return prior

#class RSSMv4(tools.Module):
#  def __init__(self, stoch=30, deter=200, hidden=200, action_size=2, state_size=3, history_length=2, act=tf.nn.elu):
#    super().__init__()
#    self._activation = act
#    self._stoch_size = stoch
#    self._deter_size = deter
#    self._hidden_size = hidden
#    self._action_size = action_size
#    self._state_size = state_size
#    self._history_length = history_length
#    self._cell = tfkl.GRUCell(self._deter_size)
#    self.A = tf.constant(np.ones([self._state_size, self._state_size*self._history_length], dtype=np.float32))
#    self.B = tf.constant(np.ones([self._action_size, self._state_size*self._history_length], dtype=np.float32))
#
#  def initial(self, batch_size, state, action):
#    dtype = prec.global_policy().compute_dtype
#    return dict(
#        mean=tf.zeros([batch_size, self._stoch_size], dtype),
#        std=tf.zeros([batch_size, self._stoch_size], dtype),
#        stoch=tf.zeros([batch_size, self._stoch_size], dtype),
#        deter=self._cell.get_initial_state(None, batch_size, dtype),
#        states=tf.cast(tf.repeat(state,self._history_length),dtype),
#        actions=tf.cast(tf.repeat(actions,self._history_length),dtype))
#
#  @tf.function
#  def update_history(state, phy, action):
#    state['states'] = tf.roll(state, -self._state_size, axis=-1)
#    state['actions'] = tf.roll(state, -self._action_size, axis=-1)
#    state['states'][:,-self._state_size:] = phy
#    state['actions'][:,-self._action_size:] = action
#    return state
#
#  @tf.function
#  def observe(self, embed, action, state=None):
#    if state is None: # No state --> set state to 0
#      state = self.initial(tf.shape(action)[0], state, action)
#
#    embed = tf.transpose(embed, [1, 0, 2]) # BS,Length,Feat --> Length, BS, feat
#    action = tf.transpose(action, [1, 0, 2]) # BS,Length,Feat --> Length, BS, feat
#    post, prior = tools.static_scan(
#        lambda prev, inputs: self.obs_step(prev[0], *inputs), # transforms the data to state, action, embed
#        (action, embed), (state, state)) # Applies obs_step to each element of the sequence in a batch fashion
#    post = {k: tf.transpose(v, [1, 0, 2]) for k, v in post.items()} # Undo previous transpose
#    prior = {k: tf.transpose(v, [1, 0, 2]) for k, v in prior.items()} # Undo previous transpose
#    return post, prior #post : (state,state), prior:
#
#  @tf.function
#  def imagine(self, action, state=None, sample=True, randomize=False, rand=[]):
#    if state is None:
#      state = self.initial(tf.shape(action)[0])
#    assert isinstance(state, dict), state
#    action = tf.transpose(action, [1, 0, 2])
#    prior = tools.static_scan_dyn(self.img_step, action, state, sample=sample, randomize=randomize, rand=rand)
#    prior = {k: tf.transpose(v, [1, 0, 2]) for k, v in prior.items()}
#    return prior
#
#  def get_feat(self, state):
#    return tf.concat([state['stoch'], state['deter']], -1)
#
#  def get_dist(self, state):
#    return tfd.MultivariateNormalDiag(state['mean'], state['std'])
#
#  def set_linear_matrices(self, A, B):
#      self.A = tf.constant(A)
#      self.B = tf.constant(B)
#
#  @tf.function
#  def obs_step(self, prev_state, prev_action, embed, sample=True):
#    prior = self.img_step(prev_state, prev_action)
#    x = tf.concat([prior['deter'], embed], -1)
#    x = self.get('obs1', tfkl.Dense, self._hidden_size, self._activation)(x)
#    x = self.get('obs2', tfkl.Dense, 2 * self._stoch_size, None)(x)
#    mean, std = tf.split(x, 2, -1)
#    std = tf.nn.softplus(std) + 0.1
#    dist = self.get_dist({'mean': mean, 'std': std})
#    stoch = dist.sample() if sample else dist.mode()
#    post = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': prior['deter']}
#    return post, prior
#
#  @tf.function
#  def apply_linear_model(self, prev_state, prev_action, randomize=False, rand=[]):
#    if randomize:
#      A_rand = rand[0]
#      B_rand = rand[1]
#      linear = tf.squeeze(tf.matmul(a=(tf.transpose(self.A) + A_rand), b=tf.expand_dims(prev_state['linear'],axis=-1)))
#      command = tf.squeeze(tf.matmul(a=(tf.transpose(self.B) + B_rand), b=tf.expand_dims(prev_action,axis=-1)))
#    else:
#      linear = tf.matmul(a=prev_state['linear'], b=self.A)
#      command = tf.matmul(a=prev_action, b=self.B)
#    linear_state = linear + command
#    return linear_state
#
#  @tf.function
#  def img_step(self, prev_state, prev_action, sample=True, randomize=False, rand = []):
#    linear_state = self.apply_linear_model(prev_state, prev_action, randomize=randomize, rand=rand)
#    x = tf.concat([prev_state['stoch'], linear_state, prev_action], -1)
#    x = self.get('img1', tfkl.Dense, self._hidden_size, self._activation)(x)
#    x, deter = self._cell(x, [prev_state['deter']])
#    deter = deter[0]  # Keras wraps the state in a list.
#    x = self.get('img2', tfkl.Dense, self._hidden_size, self._activation)(x)
#    x = self.get('img3', tfkl.Dense, 2 * self._stoch_size, None)(x)
#    mean, std = tf.split(x, 2, -1)
#    std = tf.nn.softplus(std) + 0.1
#    dist = self.get_dist({'mean': mean, 'std': std})
#    stoch = dist.sample() if sample else dist.mode()
#    prior = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': deter, 'linear':linear_state}
#    return prior

class RSSMv3(tools.Module):
  def __init__(self, stoch=30, deter=200, hidden=200, action_size=2, act=tf.nn.elu):
    super().__init__()
    self._activation = act
    self._stoch_size = stoch
    self._deter_size = deter
    self._hidden_size = hidden
    self._action_size = action_size
    self._cell = tfkl.GRUCell(self._deter_size)
    self.A = tf.Variable(np.ones([self._stoch_size, self._stoch_size + self._action_size], dtype=np.float32))
    self.B = tf.Variable(np.ones([self._action_size, self._stoch_size+self._action_size], dtype=np.float32))
    self.C = tf.Variable(np.ones([self._stoch_size+self._action_size,self._stoch_size+self._action_size], dtype=np.float32))
    self.D = tf.Variable(np.ones([self._action_size + self._stoch_size], dtype=np.float32))

  def initial(self, batch_size):
    dtype = prec.global_policy().compute_dtype
    return dict(
        mean=tf.zeros([batch_size, self._stoch_size], dtype),
        std=tf.zeros([batch_size, self._stoch_size], dtype),
        stoch=tf.zeros([batch_size, self._stoch_size], dtype),
        deter=self._cell.get_initial_state(None, batch_size, dtype))

  @tf.function
  def observe(self, embed, action, state=None):
    if state is None: # No state --> set state to 0
      state = self.initial(tf.shape(action)[0])
    embed = tf.transpose(embed, [1, 0, 2]) # BS,Length,Feat --> Length, BS, feat
    action = tf.transpose(action, [1, 0, 2]) # BS,Length,Feat --> Length, BS, feat
    post, prior = tools.static_scan(
        lambda prev, inputs: self.obs_step(prev[0], *inputs), # transforms the data to state, action, embed
        (action, embed), (state, state)) # Applies obs_step to each element of the sequence in a batch fashion
    post = {k: tf.transpose(v, [1, 0, 2]) for k, v in post.items()} # Undo previous transpose
    prior = {k: tf.transpose(v, [1, 0, 2]) for k, v in prior.items()} # Undo previous transpose
    return post, prior #post : (state,state), prior:

  @tf.function
  def imagine(self, action, state=None, sample=True, randomize=False, rand=[]):
    if state is None:
      state = self.initial(tf.shape(action)[0])
    assert isinstance(state, dict), state
    action = tf.transpose(action, [1, 0, 2])
    prior = tools.static_scan_dyn(self.img_step, action, state, sample=sample, randomize=randomize, rand=rand)
    prior = {k: tf.transpose(v, [1, 0, 2]) for k, v in prior.items()}
    return prior

  def get_feat(self, state):
    return tf.concat([state['stoch'], state['deter']], -1)

  def get_dist(self, state):
    return tfd.MultivariateNormalDiag(state['mean'], state['std'])

  @tf.function
  def obs_step(self, prev_state, prev_action, embed, sample=True):
    prior = self.img_step(prev_state, prev_action)
    x = tf.concat([prior['deter'], embed], -1)
    x = self.get('obs1', tfkl.Dense, self._hidden_size, self._activation)(x)
    x = self.get('obs2', tfkl.Dense, 2 * self._stoch_size, None)(x)
    mean, std = tf.split(x, 2, -1)
    std = tf.nn.softplus(std) + 0.1
    dist = self.get_dist({'mean': mean, 'std': std})
    stoch = dist.sample() if sample else dist.mode()
    post = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': prior['deter']}
    return post, prior

  @tf.function
  def img_step(self, prev_state, prev_action, sample=True, randomize=False, rand = []):
    x = tf.concat([prev_state['stoch'], prev_action], -1)
    # The randomization
    if randomize:
        # Add noise in transition function
        A_rand = rand[0]#tf.random.normal([prev_state['stoch'].shape[0]]+tf.transpose(self.A).shape)*0.1
        B_rand = rand[1]#tf.random.normal([prev_state['stoch'].shape[0]]+tf.transpose(self.B).shape)*0.1
        C_rand = rand[2]#tf.random.normal([prev_state['stoch'].shape[0]]+tf.transpose(self.C).shape)*0.1
        D_rand = rand[3]#tf.random.normal([prev_state['stoch'].shape[0]]+self.D.shape)*0.1 
        E_rand = rand[4]
        linear = tf.squeeze(tf.matmul(a=(tf.transpose(self.A) + A_rand), b=tf.expand_dims(prev_state['stoch'],axis=-1)))
        cmd_scaled = E_rand * prev_action
        command = tf.squeeze(tf.matmul(a=(tf.transpose(self.B) + B_rand), b=tf.expand_dims(cmd_scaled,axis=-1)))
        #command = tf.squeeze(tf.matmul(a=(tf.transpose(self.B) + B_rand), b=tf.expand_dims(prev_action,axis=-1)))
        non_linear = self._activation(tf.squeeze(tf.matmul(a=(tf.transpose(self.C) + C_rand), b=tf.expand_dims(x,axis=-1))))
        offset = self.D + D_rand
    else:
        # Changed the transition function from dense to control style equation
        linear = tf.matmul(a=prev_state['stoch'], b=self.A)
        command = tf.matmul(a=prev_action, b=self.B)
        non_linear = self._activation(tf.matmul(a=x, b=self.C))
        offset = self.D
    # update x
    x = linear + command + non_linear + offset

    x, deter = self._cell(x, [prev_state['deter']])
    deter = deter[0]  # Keras wraps the state in a list.
    x = self.get('img2', tfkl.Dense, self._hidden_size, self._activation)(x)
    x = self.get('img3', tfkl.Dense, 2 * self._stoch_size, None)(x)
    mean, std = tf.split(x, 2, -1)
    std = tf.nn.softplus(std) + 0.1
    dist = self.get_dist({'mean': mean, 'std': std})
    stoch = dist.sample() if sample else dist.mode()
    prior = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': deter}
    return prior

class RSSMv2(tools.Module):
  def __init__(self, stoch=30, deter=200, hidden=200, act=tf.nn.elu):
    super().__init__()
    self._activation = act
    self._stoch_size = stoch
    self._deter_size = deter
    self._hidden_size = hidden
    self._cell = tfkl.GRUCell(self._deter_size)

  def initial(self, batch_size):
    dtype = prec.global_policy().compute_dtype
    return dict(
        mean=tf.zeros([batch_size, self._stoch_size], dtype),
        std=tf.zeros([batch_size, self._stoch_size], dtype),
        stoch=tf.zeros([batch_size, self._stoch_size], dtype),
        deter=self._cell.get_initial_state(None, batch_size, dtype))

  @tf.function
  def observe(self, embed, action, state=None):
    if state is None: # No state --> set state to 0
      state = self.initial(tf.shape(action)[0])
    embed = tf.transpose(embed, [1, 0, 2]) # BS,Length,Feat --> Length, BS, feat
    action = tf.transpose(action, [1, 0, 2]) # BS,Length,Feat --> Length, BS, feat
    post, prior = tools.static_scan(
        lambda prev, inputs: self.obs_step(prev[0], *inputs), # transforms the data to state, action, embed
        (action, embed), (state, state)) # Applies obs_step to each element of the sequence in a batch fashion
    post = {k: tf.transpose(v, [1, 0, 2]) for k, v in post.items()} # Undo previous transpose
    prior = {k: tf.transpose(v, [1, 0, 2]) for k, v in prior.items()} # Undo previous transpose
    return post, prior #post : (state,state), prior:

  @tf.function
  def imagine(self, action, state=None, sample=True):
    if state is None:
      state = self.initial(tf.shape(action)[0])
    assert isinstance(state, dict), state
    action = tf.transpose(action, [1, 0, 2])
    prior = tools.static_scan(self.img_step, action, state) if sample else tools.static_scan_no_sampling(self.img_step, action, state)
    prior = {k: tf.transpose(v, [1, 0, 2]) for k, v in prior.items()}
    return prior

  def get_feat(self, state):
    return tf.concat([state['stoch'], state['deter']], -1)

  def get_dist(self, state):
    return tfd.MultivariateNormalDiag(state['mean'], state['std'])

  @tf.function
  def obs_step(self, prev_state, prev_action, embed, sample=True):
    prior = self.img_step(prev_state, prev_action)
    x = tf.concat([prior['deter'], embed], -1)
    x = self.get('obs1', tfkl.Dense, self._hidden_size, self._activation)(x)
    x = self.get('obs2', tfkl.Dense, 2 * self._stoch_size, None)(x)
    mean, std = tf.split(x, 2, -1)
    std = tf.nn.softplus(std) + 0.1
    dist = self.get_dist({'mean': mean, 'std': std})
    stoch = dist.sample() if sample else dist.mode()
    post = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': prior['deter']}
    return post, prior

  @tf.function
  def img_step(self, prev_state, prev_action, sample=True):
    x = tf.concat([prev_state['stoch'], prev_action], -1)
    x = self.get('img1', tfkl.Dense, self._hidden_size, self._activation)(x)
    x, deter = self._cell(x, [prev_state['deter']])
    deter = deter[0]  # Keras wraps the state in a list.
    x = self.get('img2', tfkl.Dense, self._hidden_size, self._activation)(x)
    x = self.get('img3', tfkl.Dense, 2 * self._stoch_size, None)(x)
    mean, std = tf.split(x, 2, -1)
    std = tf.nn.softplus(std) + 0.1
    dist = self.get_dist({'mean': mean, 'std': std})
    stoch = dist.sample() if sample else dist.mode()
    prior = {'mean': mean, 'std': std, 'stoch': stoch, 'deter': deter}
    return prior


class ConvEncoder(tools.Module):

  def __init__(self, depth=32, act=tf.nn.relu):
    self._act = act
    self._depth = depth

  def __call__(self, obs):
    kwargs = dict(strides=2, activation=self._act)
    x = tf.reshape(obs['image'], (-1,) + tuple(obs['image'].shape[-3:]))
    x = self.get('h1', tfkl.Conv2D, 1 * self._depth, 4, **kwargs)(x)
    x = self.get('h2', tfkl.Conv2D, 2 * self._depth, 4, **kwargs)(x)
    x = self.get('h3', tfkl.Conv2D, 4 * self._depth, 4, **kwargs)(x)
    x = self.get('h4', tfkl.Conv2D, 8 * self._depth, 4, **kwargs)(x)
    shape = tf.concat([tf.shape(obs['image'])[:-3], [32 * self._depth]], 0)
    return tf.reshape(x, shape)

class DenseEncoder(tools.Module):

  def __init__(self, depth=200, act=tf.nn.relu):
    self._act = act
    self._depth = depth

  def __call__(self, obs):
    kwargs = dict(activation=self._act)
    x = self.get('h1', tfkl.Dense, self._depth, **kwargs)(obs)
    x = self.get('h2', tfkl.Dense, self._depth, **kwargs)(x)
    x = self.get('h3', tfkl.Dense, self._depth, **kwargs)(x)
    return x

class LaserConvEncoder(tools.Module):

  def __init__(self, depth=32, act=tf.nn.relu):
    self._act = act
    self._depth = depth

  def __call__(self, obs):
    kwargs = dict(strides=2, activation=self._act)
    x = tf.reshape(obs['laser'], (-1,) + tuple(obs['laser'].shape[-2:]))
    x = self.get('h1', tfkl.Conv1D, 1 * self._depth, 6, **kwargs)(x)
    x = self.get('h2', tfkl.Conv1D, 2 * self._depth, 6, **kwargs)(x)
    x = self.get('h3', tfkl.Conv1D, 3 * self._depth, 6, **kwargs)(x)
    x = self.get('h4', tfkl.Conv1D, 4 * self._depth, 6, **kwargs)(x)
    x = self.get('h5', tfkl.Conv1D, 8 * self._depth, 6, **kwargs)(x)
    shape = tf.concat([tf.shape(obs['laser'])[:-2], [32*self._depth]], 0)
    return tf.reshape(x, shape)

class LaserConvEncoderWithBypass(tools.Module):

  def __init__(self, depth=32, act=tf.nn.relu):
    self._act = act
    self._depth = depth

  def __call__(self, obs):
    kwargs = dict(strides=2, activation=self._act)
    x = tf.reshape(obs['laser'], (-1,) + tuple(obs['laser'].shape[-2:]))
    x = self.get('h1', tfkl.Conv1D, 1 * self._depth, 6, **kwargs)(x)
    x = self.get('h2', tfkl.Conv1D, 2 * self._depth, 6, **kwargs)(x)
    x = self.get('h3', tfkl.Conv1D, 3 * self._depth, 6, **kwargs)(x)
    x = self.get('h4', tfkl.Conv1D, 4 * self._depth, 6, **kwargs)(x)
    x = self.get('h5', tfkl.Conv1D, 8 * self._depth, 6, **kwargs)(x)
    shape = tf.concat([tf.shape(obs['laser'])[:-2], [32*self._depth]], 0)
    #return tf.reshape(x, shape)
    x = tf.reshape(x, shape)
    x = tf.concat([x, obs['vel_as_laser']],axis=-1)
    return x


class ConvDecoder(tools.Module):

  def __init__(self, depth=32, act=tf.nn.relu, shape=(64, 64, 3)):
    self._act = act
    self._depth = depth
    self._shape = shape

  def __call__(self, features):
    kwargs = dict(strides=2, activation=self._act)
    x = self.get('h1', tfkl.Dense, 32 * self._depth, None)(features)
    x = tf.reshape(x, [-1, 1, 1, 32 * self._depth])
    x = self.get('h2', tfkl.Conv2DTranspose, 4 * self._depth, 5, **kwargs)(x)
    x = self.get('h3', tfkl.Conv2DTranspose, 2 * self._depth, 5, **kwargs)(x)
    x = self.get('h4', tfkl.Conv2DTranspose, 1 * self._depth, 6, **kwargs)(x)
    x = self.get('h5', tfkl.Conv2DTranspose, self._shape[-1], 6, strides=2)(x)

    mean = tf.reshape(x, tf.concat([tf.shape(features)[:-1], self._shape], 0))
    return tfd.Independent(tfd.Normal(mean, 1), len(self._shape))

class LaserConvDecoder(tools.Module):

  def __init__(self, depth=32, act=tf.nn.relu, shape=(256, 1)):
    self._act = act
    self._depth = depth
    self._shape = shape

  def __call__(self, features):
    kwargs = dict(strides=(2,1), activation=self._act, padding='same')
    kwargs2 = dict(strides=(4,1), activation=self._act, padding='same')
    x = self.get('h1', tfkl.Dense, 32 * self._depth, None)(features)
    x = tf.reshape(x, [-1, 1, 1, 32 * self._depth])
    x = self.get('h2', tfkl.Conv2DTranspose, 8 * self._depth, (5,1), **kwargs2)(x)
    x = self.get('h3', tfkl.Conv2DTranspose, 4 * self._depth, (5,1), **kwargs2)(x)
    x = self.get('h4', tfkl.Conv2DTranspose, 2 * self._depth, (5,1), **kwargs2)(x)
    x = self.get('h5', tfkl.Conv2DTranspose, 1 * self._depth, (5,1), **kwargs)(x)
    x = self.get('h6', tfkl.Conv2DTranspose, self._shape[-1], (5,1), strides=(2,1), padding='same')(x)
    mean = tf.reshape(x, tf.concat([tf.shape(features)[:-1], self._shape], 0))
    return tfd.Independent(tfd.Normal(mean, 1), len(self._shape))

class DenseDecoder(tools.Module):

  def __init__(self, shape, layers, units, dist='normal', act=tf.nn.elu):
    self._shape = shape
    self._layers = layers
    self._units = units
    self._dist = dist
    self._act = act

  def __call__(self, features):
    x = features
    for index in range(self._layers):
      x = self.get(f'h{index}', tfkl.Dense, self._units, self._act)(x)
    x = self.get(f'hout', tfkl.Dense, np.prod(self._shape))(x)
    x = tf.reshape(x, tf.concat([tf.shape(features)[:-1], self._shape], 0))
    if self._dist == 'normal':
      return tfd.Independent(tfd.Normal(x, 1), len(self._shape))
    if self._dist == 'binary':
      return tfd.Independent(tfd.Bernoulli(x), len(self._shape))
    raise NotImplementedError(self._dist)

class LinearPhysics(tools.Module):

  def __init__(self, shape, layers, units, dist='normal', act=tf.nn.elu):
    self._shape = shape
    self._layers = layers
    self._units = units
    self._dist = dist
    self._act = act

  def __call__(self, features):
    x = features
    for index in range(self._layers):
      x = self.get(f'h{index}', tfkl.Dense, self._units, self._act)(x)
    x = self.get(f'hout', tfkl.Dense, np.prod(self._shape))(x)
    x = tf.reshape(x, tf.concat([tf.shape(features)[:-1], self._shape], 0))
    return x

class ActionDecoder(tools.Module):

  def __init__(
      self, size, layers, units, dist='tanh_normal', act=tf.nn.elu,
      min_std=1e-4, init_std=5, mean_scale=5):
    self._size = size
    self._layers = layers
    self._units = units
    self._dist = dist
    self._act = act
    self._min_std = min_std
    self._init_std = init_std
    self._mean_scale = mean_scale

  def __call__(self, features):
    raw_init_std = np.log(np.exp(self._init_std) - 1)
    x = features
    for index in range(self._layers):
      x = self.get(f'h{index}', tfkl.Dense, self._units, self._act)(x)
    if self._dist == 'tanh_normal':
      # https://www.desmos.com/calculator/rcmcf5jwe7
      x = self.get(f'hout', tfkl.Dense, 2 * self._size)(x)
      mean, std = tf.split(x, 2, -1)
      mean = self._mean_scale * tf.tanh(mean / self._mean_scale)
      std = tf.nn.softplus(std + raw_init_std) + self._min_std
      dist = tfd.Normal(mean, std)
      dist = tfd.TransformedDistribution(dist, tools.TanhBijector())
      dist = tfd.Independent(dist, 1)
      dist = tools.SampleDist(dist)
    elif self._dist == 'onehot':
      x = self.get(f'hout', tfkl.Dense, self._size)(x)
      dist = tools.OneHotDist(x)
    else:
      raise NotImplementedError(dist)
    return dist
