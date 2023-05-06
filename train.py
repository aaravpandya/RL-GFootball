import wandb
wandb.login()
from tqdm import tqdm
import gymnasium
from gymnasium.spaces import Box, Discrete
import numpy as np
from collections import deque
from gfootball.env import create_environment
from gymnasium.wrappers import StepAPICompatibility
from gfootball.env import observation_preprocessing
import tempfile
import os
tempfile.tempdir = '/scratch/ap7641/hpmlproject/temp'
from gfootball.env.wrappers import Simple115StateWrapper
os.environ["DISPLAY"] = ""
import ray
from ray.runtime_env import RuntimeEnv
ray.init(runtime_env=RuntimeEnv(env_vars={"DISPLAY": ""}))

scenarios = {0: "academy_empty_goal_close",
             1: "academy_empty_goal",
             2: "academy_run_to_score",
             3: "academy_run_to_score_with_keeper",
             4: "academy_pass_and_shoot_with_keeper",
             5: "academy_run_pass_and_shoot_with_keeper",
             6: "academy_3_vs_1_with_keeper",
             7: "academy_corner",
             8: "academy_counterattack_easy",
             9: "academy_counterattack_hard",
             10: "academy_single_goal_versus_lazy",
             11: "11_vs_11_kaggle"}
scenario_name = scenarios[6]

class FootballGym(gymnasium.Env):
    spec = None
    metadata = None
    def debug(self):
        print(self.reset()[0].shape)
    def __init__(self, config=None):
        super(FootballGym, self).__init__()
        env_name = "academy_empty_goal_close"
        rewards = "scoring,checkpoints"
        if config is not None:
            env_name = config.get("env_name", env_name)
            rewards = config.get("rewards", rewards)
        print("Initializing env " + env_name)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(115,), dtype=np.float32)
        self.env = create_environment(
            env_name=env_name,
            stacked=False,
            representation="simple115v2",
            rewards = rewards,
            write_goal_dumps=False,
            write_full_episode_dumps=False,
            render=False,
            write_video=False,
            dump_frequency=1,
            logdir="/scratch/ap7641/hpmlproject/footballreplays",
            extra_players=None,
            number_of_left_players_agent_controls=1,
            number_of_right_players_agent_controls=0)  
        self.action_space = Discrete(19)
        self.reward_range = (-1, 1)
        self.obs_stack = deque([], maxlen=4)
        self.observation_wrapper = Simple115StateWrapper(self.env, True)
    
    def sample_action(self):
        return self.action_space.sample()
    def transform_obs(self, raw_obs):
        obs = raw_obs[0]
        obs = observation_preprocessing.generate_smm([obs])
        if not self.obs_stack:
            self.obs_stack.extend([obs] * 4)
        else:
            self.obs_stack.append(obs)
        obs = np.concatenate(list(self.obs_stack), axis=-1)
        obs = np.squeeze(obs)
        return obs

    def reset(self, *, seed=None, options=None):
        self.obs_stack.clear()
        obs = self.env.unwrapped.env.reset()
        obs = self.observation_wrapper.observation(obs)
        obs = np.squeeze(obs)
        return obs, {}
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step([action])
        return obs, float(reward), done, False, info
    def close(self):
        self.env.close()
    
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import pretty_print


algo = (
    PPOConfig()
    .rollouts(num_rollout_workers=8, num_envs_per_worker=1)
    .resources(num_gpus=1)
    .environment(env=FootballGym, env_config ={'env_name':scenario_name})
    .build()
)
run = wandb.init(
    # Set the project where this run will be logged
    project="HPML-Project",
    # Track hyperparameters and run metadata
    config={
        'n_envs':8,
        'n_steps':100,
        'env':scenario_name,
        'policy':'FCN'
        })
for i in tqdm(range(2000)):
    result = algo.train()
    learner_stats = result['info']['learner']['default_policy']['learner_stats']
    learner_stats['num_env_steps_sampled'] = result['num_env_steps_sampled']
    learner_stats['num_env_steps_trained'] = result['num_env_steps_trained']
    learner_stats['episode_reward_mean'] = result['episode_reward_mean']
    learner_stats['episodes_total'] = result['episodes_total']
    wandb.log(learner_stats)
    if i % 100 == 0:
        checkpoint_dir = algo.save(checkpoint_dir="\scratch\ap7641\hpmlproject\checkpoints")
        
checkpoint_dir = algo.save(checkpoint_dir="\scratch\ap7641\hpmlproject\checkpoints")
print(f"Checkpoint saved in directory {checkpoint_dir}")