import multiprocessing
import os
# import gym
import tensorflow as tf
from stable_baselines.a2c.utils import conv, linear, conv_to_fc
from stable_baselines.common.policies import ActorCriticPolicy, register_policy, nature_cnn
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack
from stable_baselines import PPO2, A2C, logger
from stable_baselines.bench import monitor
import gfootball.env as football_env
import numpy as np
# from keras_layer_normalization import LayerNormalization
import tensorflow as tf
import numpy as np


def residual_block(inputs, depth):
	out = tf.keras.layers.Conv2D(depth, 3, 1, padding='same')(inputs)
	out = tf.keras.layers.ReLU()(out)
	out = tf.keras.layers.Conv2D(depth, 3, 1, padding='same')(out)
	return out + inputs


def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
	q: query shape == (..., seq_len_q, depth)
	k: key shape == (..., seq_len_k, depth)
	v: value shape == (..., seq_len_v, depth_v)
	mask: Float tensor with shape broadcastable
		  to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
	output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
  print("q " + str(q.shape))
  print("k" + str(k.shape))
  print("qk" + str(matmul_qk.shape))
  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
  print("softmax" + str(scaled_attention_logits.shape))
  # add the mask to the scaled tensor.
  # if mask is not None:
  #   scaled_attention_logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.

  attention_weights = tf.nn.softmax(
      scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
  print("output" + str(output.shape))
  return output

# class MultiHeadAttention(tf.keras.layers.Layer):
#   def __init__(self, d_model, num_heads):
#     super(MultiHeadAttention, self).__init__()
#     self.num_heads = num_heads
#     self.d_model = d_model

#     assert d_model % self.num_heads == 0

#     self.depth = d_model // self.num_heads

#     self.wq = tf.keras.layers.Dense(d_model)
#     self.wk = tf.keras.layers.Dense(d_model)
#     self.wv = tf.keras.layers.Dense(d_model)

#     self.dense = tf.keras.layers.Dense(d_model)


def split_heads(x, num_heads):
  """Split the last dimension into (num_heads, depth).
  Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
  """
  c = tf.broadcast_to(x, [num_heads, tf.shape(x)[0], x.shape[1], x.shape[2]])
  return tf.transpose(c, perm=[1, 0, 2, 3])


def mlp(x):
	y = tf.keras.layers.Dense(384, activation='relu')(x)
	y = tf.keras.layers.Dense(384, activation='relu')(y)
	y = tf.keras.layers.Dense(54, activation='relu')(y)
	return y

# def layer_norm(x):
# 	return LayerNormalization()(x)

# def conv_lstm(x):
# 	tf.keras.layers.ConvLSTM2D(96,3,1,padding='same',)


def mhdpa(v, k, q, num_heads):
	#   batch_size = tf.shape(q)[0]

  # q = self.wq(q)  # (batch_size, seq_len, d_model)
  # k = self.wk(k)  # (batch_size, seq_len, d_model)
  # v = self.wv(v)  # (batch_size, seq_len, d_model)

  wq = tf.keras.layers.Dense(18)(q)
  wk = tf.keras.layers.Dense(18)(k)
  wv = tf.keras.layers.Dense(18)(v)
#   wq = LayerNormalization()(wq)
#   wk = LayerNormalization()(wk)
#   wv = LayerNormalization()(wv)
  print("wq" + str(wq.shape))
  wq = split_heads(wq, num_heads)  # (batch_size, num_heads, seq_len_q, depth)
  wk = split_heads(wk, num_heads)  # (batch_size, num_heads, seq_len_k, depth)
  wv = split_heads(wv, num_heads)  # (batch_size, num_heads, seq_len_v, depth)

  # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
  # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
  scaled_attention = scaled_dot_product_attention(
	  wq, wk, wv, num_heads)
  print("sca " + str(scaled_attention))
  # (batch_size, seq_len_q, num_heads, depth)
  scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
  print("sca_ " + str(scaled_attention))
  concat_attention = tf.reshape(scaled_attention,
                                (tf.shape(scaled_attention)[0], scaled_attention.shape[1], scaled_attention.shape[2]*scaled_attention.shape[3]))  # (batch_size, seq_len_q, d_model)
  print("concat" + str(concat_attention))
#   output = mlp(concat_attention)
#   output = output + q
  output = tf.keras.layers.Dense(scaled_attention.shape[2]*scaled_attention.shape[3])(concat_attention)
#   output = LayerNorma1lization()(output)
  return output

# def nature_cnn(scaled_images, **kwargs):
#     """
#     CNN from Nature paper.
#     :param scaled_images: (TensorFlow Tensor) Image input placeholder
#     :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
#     :return: (TensorFlow Tensor) The CNN output layer
#     """
#     activ = tf.nn.relu
#     layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs))
#     layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
#     layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
#     layer_3 = conv_to_fc(layer_3)
#     return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))
# def residual_block(inputs):
# 	depth = inputs.get_shape()[-1].value
# 	out = tf.nn.relu(inputs)
# 	out = tf.keras.layers.Conv2D(depth,3,1,padding='same')(out)
# 	out = tf.nn.relu(out)
# 	out = tf.keras.layers.Conv2D(depth,3,1,padding='same')(out)
# 	return out + inputs


def build_impala_cnn(unscaled_images, depths=[16, 32, 32], **conv_kwargs):
	"""
	Model used in the paper "IMPALA: Scalable Distributed Deep-RL with
	Importance Weighted Actor-Learner Architectures" https://arxiv.org/abs/1802.01561
	"""

	layer_num = 0

	def get_layer_num_str():
		nonlocal layer_num
		num_str = str(layer_num)
		layer_num += 1
		return num_str

	def conv_layer(out, depth):
		return tf.layers.conv2d(out, depth, 3, padding='same', name='layer_' + get_layer_num_str())

	def residual_block(inputs):
		depth = inputs.get_shape()[-1].value

		out = tf.nn.relu(inputs)

		out = conv_layer(out, depth)
		out = tf.nn.relu(out)
		out = conv_layer(out, depth)
		return out + inputs

	def conv_sequence(inputs, depth):
		out = conv_layer(inputs, depth)
		out = tf.layers.max_pooling2d(out, pool_size=3, strides=2, padding='same')
		out = residual_block(out)
		out = residual_block(out)
		return out

	out = tf.cast(unscaled_images, tf.float32) / 255.
	# y = tf.keras.layers.Conv2D(32, 4, 2, padding='same')(out)
	# y = tf.keras.layers.BatchNormalization()(y)
	# y = tf.keras.layers.ReLU()(y)
	# print(y)
	# y = residual_block(y, 32)
	# y = tf.keras.layers.BatchNormalization()(y)
	# y = tf.keras.layers.ReLU()(y)
	# y = tf.keras.layers.MaxPool2D()(y)
	# print(y)
	# y = tf.keras.layers.Conv2D(54, 4, 2, padding='same')(y)
	# y = tf.keras.layers.BatchNormalization()(y)
	# y = tf.keras.layers.ReLU()(y)
	# print(y)
	# y = residual_block(y, 54)
	# y = tf.keras.layers.BatchNormalization()(y)
	# out = tf.keras.layers.ReLU()(y)
	# print(out)
	for depth in depths:
		out = conv_sequence(out, depth)
	# out = tf.keras.layers.ConvLSTM2D(32,2,1,'same1')
	# out = tf.keras.layers.Conv2D(54,3,1,'same')(out)
	print(tf.shape(out)[0])
	print(tf.shape(out)[1])
	print(tf.shape(out)[2])
	print(tf.shape(out)[3])
	out = tf.reshape(
		out, (tf.shape(out)[0], out.shape[1]*out.shape[2], out.shape[3]))
	print(out.shape)
	out = mhdpa(out, out, out, 3)
	out = tf.keras.layers.Flatten()(out)
	out = tf.nn.relu(out)
	out = tf.layers.dense(out, 256, activation=tf.nn.relu)

	return out

# Custom MLP policy of three layers of size 128 each for the actor and 2 layers of 32 for the critic,
# with a nature_cnn feature extractor


class CustomPolicy(ActorCriticPolicy):
	def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
		super(CustomPolicy, self).__init__(sess, ob_space, ac_space,
                                     n_env, n_steps, n_batch, reuse=reuse, scale=True)

		with tf.variable_scope("model", reuse=reuse):
			activ = tf.nn.relu
			print("processed obs" + str(self.processed_obs.shape))
			extracted_features = build_impala_cnn(self.processed_obs, **kwargs)
			print("extracted_features"+str(extracted_features.shape))
			extracted_features = tf.layers.flatten(extracted_features)
			print("ex2 " + str(extracted_features.shape))

			pi_h = extracted_features
			for i, layer_size in enumerate([128, 128, 128]):
				pi_h = activ(tf.layers.dense(pi_h, layer_size, name='pi_fc' + str(i)))
			pi_latent = pi_h

			vf_h = extracted_features
			for i, layer_size in enumerate([32, 32]):
				vf_h = activ(tf.layers.dense(vf_h, layer_size, name='vf_fc' + str(i)))
			value_fn = tf.layers.dense(vf_h, 1, name='vf')
			vf_latent = vf_h

			self._proba_distribution, self._policy, self.q_value = \
				self.pdtype.proba_distribution_from_latent(
					pi_latent, vf_latent, init_scale=0.01)

		self._value_fn = value_fn
		self._setup_init()

	def step(self, obs, state=None, mask=None, deterministic=False):
		if deterministic:
			action, value, neglogp = self.sess.run([self.deterministic_action, self.value_flat, self.neglogp],
                                          {self.obs_ph: obs})
		else:
			action, value, neglogp = self.sess.run([self.action, self.value_flat, self.neglogp],
                                          {self.obs_ph: obs})
		return action, value, self.initial_state, neglogp

	def proba_step(self, obs, state=None, mask=None):
		return self.sess.run(self.policy_proba, {self.obs_ph: obs})

	def value(self, obs, state=None, mask=None):
		return self.sess.run(self.value_flat, {self.obs_ph: obs})


def create_single_football_env(seed):
  """Creates gfootball environment."""
#   env = football_env.create_environment(
#       env_name=FLAGS.level, stacked=('stacked' in FLAGS.state),
#       rewards=FLAGS.reward_experiment,
#       logdir=logger.get_dir() + str(seed),
#       enable_goal_videos=FLAGS.dump_scores and (seed == 0),
#       enable_full_episode_videos=FLAGS.dump_full_episodes and (seed == 0),
#       render= True and (seed == 0),
#       dump_frequency=50 if FLAGS.render and seed == 0 else 0)

  env = football_env.create_environment(env_name="academy_3_vs_1_with_keeper", stacked=True,
                                        representation='extracted', render=False and (seed == 0), channel_dimensions=(64, 64), rewards='scoring,checkpoints')
  env = monitor.Monitor(env, logger.get_dir()
                        and os.path.join(logger.get_dir(), str(seed)))
  return env



def callback(_, _t):
	model.save("Atari.pkl")
	print("model saved")
	return True

ncpu = multiprocessing.cpu_count()
config = tf.ConfigProto(allow_soft_placement=True,
                        intra_op_parallelism_threads=ncpu,
                        inter_op_parallelism_threads=ncpu)
config.gpu_options.allow_growth = True

tf.Session(config=config).__enter__()
logger.configure(folder="CNN")
vec_env = SubprocVecEnv([
	(lambda _i=i: create_single_football_env(_i))
	for i in range(8)])
# env = make_atari_env('SeaquestNoFrameskip-v0', num_env=4, seed=0)
# # Frame-stacking with 4 frames
# env = VecFrameStack(env, n_stack=4)

model = PPO2(CustomPolicy, vec_env, verbose=1,tensorboard_log="Atari",full_tensorboard_log=True,
				learning_rate=0.000343,n_steps=512,nminibatches=4,gamma=0.993,cliprange=0.08,
				ent_coef=0.03,lam=0.95,noptepochs=4,max_grad_norm=0.64)
# # model.setup_model()
# # model = PPO2.load("3_vs_1_Sep_3_full_mhdpa.pkl")
# # model.set_env(vec_env)
# # model.setup_model()
model.learn(total_timesteps=20000000, callback=callback,
            tb_log_name="Atari", reset_num_timesteps=False)

# model = PPO2.load("Hopefully_this_works_Continued_2.pkl")

# env = football_env.create_environment(env_name="academy_3_vs_1_with_keeper",stacked=False,render=True
# ,channel_dimensions=(96,96), logdir="",rewards='scoring,checkpoints')
# env.
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     # env.render()
#     if(dones):
#         obs = env.reset()