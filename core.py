import os
# import gym
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO2, logger
from stable_baselines3.common import monitor
import gfootball.env as football_env


def residual_block(inputs: torch.Tensor, depth: int) -> torch.Tensor:
	out = nn.Conv2D(depth, 3, 1, padding='same')(inputs)
	out = nn.ReLU()(out)
	out = nn.Conv2D(depth, 3, 1, padding='same')(out)
	return out + inputs

def split_heads(x, num_heads):
  """Split the last dimension into (num_heads, depth).
  Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
  """
  c = torch.broadcast_to(x, [num_heads, torch.shape(x)[0], x.shape[1], x.shape[2]])
  return torch.transpose(c, perm=[1, 0, 2, 3])


def mlp(x):
	y = nn.Dense(384, activation='relu')(x)
	y = nn.Dense(384, activation='relu')(y)
	y = nn.Dense(54, activation='relu')(y)
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

  wq = nn.Dense(18)(q)
  wk = nn.Dense(18)(k)
  wv = nn.Dense(18)(v)
#   wq = LayerNormalization()(wq)
#   wk = LayerNormalization()(wk)
#   wv = LayerNormalization()(wv)
  print("wq" + str(wq.shape))
  wq = split_heads(wq, num_heads)  # (batch_size, num_heads, seq_len_q, depth)
  wk = split_heads(wk, num_heads)  # (batch_size, num_heads, seq_len_k, depth)
  wv = split_heads(wv, num_heads)  # (batch_size, num_heads, seq_len_v, depth)

  # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
  # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
  scaled_attention = nn.functional.scaled_dot_product_attention(
	  wq, wk, wv)
  print("sca " + str(scaled_attention))
  # (batch_size, seq_len_q, num_heads, depth)
  scaled_attention = torch.transpose(scaled_attention, perm=[0, 2, 1, 3])
  print("sca_ " + str(scaled_attention))
  concat_attention = torch.reshape(scaled_attention,
                                (torch.shape(scaled_attention)[0], scaled_attention.shape[1], scaled_attention.shape[2]*scaled_attention.shape[3]))  # (batch_size, seq_len_q, d_model)
  print("concat" + str(concat_attention))
#   output = mlp(concat_attention)
#   output = output + q
  output = nn.Dense(scaled_attention.shape[2]*scaled_attention.shape[3])(concat_attention)
#   output = LayerNorma1lization()(output)
  return output

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
		return nn.functinal.conv2d(out, depth, 3, padding='same', name='layer_' + get_layer_num_str())

	def residual_block(inputs):
		depth = inputs.get_shape()[-1].value
		out = nn.relu(inputs)
		out = conv_layer(out, depth)
		out = nn.relu(out)
		out = conv_layer(out, depth)
		return out + inputs

	def conv_sequence(inputs, depth):
		out = conv_layer(inputs, depth)
		out = nn.functional.max_pool2d(out, pool_size=3, strides=2, padding='same')
		out = residual_block(out)
		out = residual_block(out)
		return out

	out = torch.cast(unscaled_images, torch.float32) / 255.
	
	for depth in depths:
		out = conv_sequence(out, depth)
	
	out = torch.reshape(
		out, (torch.shape(out)[0], out.shape[1]*out.shape[2], out.shape[3]))
	print(out.shape)
	out = mhdpa(out, out, out, 3)
	out = nn.Flatten()(out)
	out = nn.functional.relu(out)
	out = nn.functional.dense(out, 256, activation=nn.ReLU)

	return out

# Custom MLP policy of three layers of size 128 each for the actor and 2 layers of 32 for the critic,
# with a nature_cnn feature extractor


class CustomPolicy(ActorCriticPolicy):
	def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **kwargs):
		super(CustomPolicy, self).__init__(sess, ob_space, ac_space,
                                     n_env, n_steps, n_batch, reuse=reuse, scale=True)

		activation = nn.ReLU()
		print("processed obs" + str(self.processed_obs.shape))
		extracted_features = build_impala_cnn(self.processed_obs, **kwargs)
		print("extracted_features"+str(extracted_features.shape))
		extracted_features = nn.flatten(extracted_features)
		print("ex2 " + str(extracted_features.shape))

		pi_h = extracted_features
		for i, layer_size in enumerate([128, 128, 128]):
			pi_h = activation(nn.Dense(pi_h, layer_size, name='pi_fc' + str(i)))
		pi_latent = pi_h

		vf_h = extracted_features
		for i, layer_size in enumerate([32, 32]):
			vf_h = activation(nn.Dense(vf_h, layer_size, name='vf_fc' + str(i)))
		value_fn = nn.Dense(vf_h, 1, name='vf')
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

  env = football_env.create_environment(env_name="academy_3_vs_1_with_keeper", stacked=True,
                                        representation='extracted', render=False and (seed == 0), channel_dimensions=(64, 64), rewards='scoring,checkpoints')
  env = monitor.Monitor(env, logger.get_dir()
                        and os.path.join(logger.get_dir(), str(seed)))
  return env



def callback(_, _t):
	model.save("Atari.pkl")
	print("model saved")
	return True

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