"""Runs football_env on OpenAI's ppo2."""
from __future__ import absolute_import, division, print_function

import os

import gfootball.env as football_env
from stable_baselines3 import PPO, logger
from stable_baselines3.common import monitor
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# warnings.filterwarnings('ignore')
# TODO: where are these flags in pytorch
flags = {}
FLAGS = {}


flags.DEFINE_string('level', 'academy_5_vs_3_with_keeper',
                    'Defines type of problem being solved')
flags.DEFINE_enum('state', 'extracted_stacked', ['extracted',
                                                 'extracted_stacked'],
                  'Observation to be used for training.')
flags.DEFINE_enum('reward_experiment', 'scoring,checkpoints',
                  ['scoring', 'scoring,checkpoints'],
                  'Reward to be used for training.')
flags.DEFINE_enum('policy', 'gfootball_impala_cnn', ['cnn', 'lstm', 'mlp', 'impala_cnn',
                                    'gfootball_impala_cnn'],
                  'Policy architecture')
flags.DEFINE_integer('num_timesteps', int(2e8),
                     'Number of timesteps to run for.')
flags.DEFINE_integer('num_envs', 8,
                     'Number of environments to run in parallel.')
flags.DEFINE_integer('nsteps', 512, 'Number of environment steps per epoch; '
                     'batch size is nsteps * nenv')
flags.DEFINE_integer('noptepochs', 4, 'Number of updates per epoch.')
flags.DEFINE_integer('nminibatches', 8,
                     'Number of minibatches to split one epoch to.')
flags.DEFINE_integer('save_interval', 100,
                     'How frequently checkpoints are saved.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_float('lr', 0.000343, 'Learning rate')
flags.DEFINE_float('ent_coef', 0.003, 'Entropy coeficient')
flags.DEFINE_float('gamma', 0.993, 'Discount factor')
flags.DEFINE_float('cliprange', 0.08, 'Clip range')
flags.DEFINE_bool('render', False, 'If True, environment rendering is enabled.')
flags.DEFINE_bool('dump_full_episodes', False,
                  'If True, trace is dumped after every episode.')
flags.DEFINE_bool('dump_scores', False,
                  'If True, sampled traces after scoring are dumped.')
flags.DEFINE_string('load_path', "/home/aarav/5_vs_3_with_keeper_from_scratch/checkpoints/00600", 'Path to load initial checkpoint from.')


def create_single_football_env(seed):
	"""Creates gfootball environment."""
	env = football_env.create_environment(
		env_name=FLAGS.level, stacked=('stacked' in FLAGS.state),
		rewards=FLAGS.reward_experiment,
		logdir=logger.get_dir(),
		enable_goal_videos=FLAGS.dump_scores and (seed == 0),
		enable_full_episode_videos=FLAGS.dump_full_episodes and (seed == 0),
		render= True and (seed == 0),
		dump_frequency=50 if FLAGS.render and seed == 0 else 0)
	env = monitor.Monitor(env, logger.get_dir() and os.path.join(logger.get_dir(),
																str(seed)))
	return env


def train():
	"""Trains a PPO2 policy."""
	logger.configure("~/5_vs_3_with_keeper_from_scratch_2")
	vec_env = SubprocVecEnv([
		(lambda _i=i: create_single_football_env(_i))
		for i in range(1)
	], context=None)

	PPO.learn(network=FLAGS.policy,
				total_timesteps=FLAGS.num_timesteps,
				env=vec_env,
				seed=FLAGS.seed,
				nsteps=FLAGS.nsteps,
				nminibatches=FLAGS.nminibatches,
				noptepochs=FLAGS.noptepochs,
				gamma=FLAGS.gamma,
				ent_coef=FLAGS.ent_coef,
				lr=FLAGS.lr,
				log_interval=1,
				save_interval=FLAGS.save_interval,
				cliprange=FLAGS.cliprange,
				load_path=FLAGS.load_path,
				)

if __name__ == '__main__':
  	train()