import os
from collections import deque

import numpy as np
import torch as th
device = th.device("cuda" if th.cuda.is_available() else "cpu")

import gym
from gym.spaces import Box, Discrete

from gfootball.env import create_environment
from gfootball.env import observation_preprocessing

from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3.common import results_plotter
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.logger import configure
# import wandb
# wandb.login()
from tqdm import tqdm

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

class FootballGym(gym.Env):
    spec = None
    metadata = None
    
    def __init__(self, config=None):
        super(FootballGym, self).__init__()
        env_name = "academy_empty_goal_close"
        rewards = "scoring,checkpoints"
        if config is not None:
            env_name = config.get("env_name", env_name)
            rewards = config.get("rewards", rewards)
        self.env = create_environment(
            env_name=env_name,
            stacked=False,
            representation="raw",
            rewards = rewards,
            write_goal_dumps=False,
            write_full_episode_dumps=False,
            render=False,
            write_video=False,
            dump_frequency=1,
            logdir=".",
            extra_players=None,
            number_of_left_players_agent_controls=1,
            number_of_right_players_agent_controls=0)  
        self.action_space = Discrete(19)
        self.observation_space = Box(low=0, high=255, shape=(72, 96, 16), dtype=np.uint8)
        self.reward_range = (-1, 1)
        self.obs_stack = deque([], maxlen=4)
        
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

    def reset(self):
        self.obs_stack.clear()
        obs = self.env.reset()
        obs = self.transform_obs(obs)
        return obs
    
    def step(self, action):
        obs, reward, done, info = self.env.step([action])
        obs = self.transform_obs(obs)
        return obs, float(reward), done, info

from multiprocessing.connection import Pipe
import numpy as np
from stable_baselines3 import PPO
# from stable_baselines3.common.policies import CnnPolicy


# class ProgressBar(BaseCallback):
    
#     def __init__(self, verbose=0):
#         super(ProgressBar, self).__init__(verbose)
#         self.pbar = None

#     def _on_training_start(self):
#         factor = np.ceil(self.locals['total_timesteps'] / self.model.n_steps)
#         try:
#             n = len(self.training_env.remotes)
#         except AttributeError:
#             n = len(self.training_env.envs)
#         total = int(self.model.n_steps * factor / n)
#         self.pbar = tqdm(total=total)

#     def _on_rollout_start(self):
#         self.pbar.refresh()

#     def _on_step(self):
#         self.pbar.update(1)
#         return True

#     def _on_rollout_end(self):
#         self.pbar.refresh()

#     def _on_training_end(self):
#         self.pbar.close()
#         self.pbar = None

class SaveCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(SaveCallback, self).__init__(verbose)

    def _on_step(self):
        if self.n_calls % 1000 == 0:
            self.model.save('ppo_football')

        return True
def make_env(config=None, rank=0):
    def _init():
        env = FootballGym(config)
        log_file = os.path.join(".", str(rank))
        env = Monitor(env, log_file, allow_early_resets=True)
        return env
    return _init
   
if __name__ == "__main__":
    test_env = FootballGym({"env_name":scenario_name})
    check_env(env=test_env, warn=True)
    n_envs = 14
    n_steps = 500

    config={"env_name":scenario_name}
    # train_env = DummyVecEnv([make_env(config, rank=i) for i in range(n_envs)])
    train_env = SubprocVecEnv([make_env(config, rank=i) for i in range(n_envs)], start_method='fork')
    # train_env = CustomSubprocVecEnv([make_env(config, rank=i) for i in range(n_envs)])
#     run = wandb.init(
#         # Set the project where this run will be logged
#         project="HPML-Project",
#         # Track hyperparameters and run metadata
#         config={
#             'n_envs':n_envs,
#             'n_steps':n_steps,
#             'env':scenario_name,
#             'policy':'CNN'
#             })
#     model = PPO(CnnPolicy, train_env, n_steps=n_steps, verbose=1, tensorboard_log='./logs2',  device=device)
    model = PPO.load("/scratch/ap7641/hpmlproject/ppo_football", train_env, device=device)
    tmp_path = "./sb3_log2/"
# set up logger
    new_logger = configure(tmp_path, ["csv", "log", "tensorboard"])
    model.set_logger(new_logger)
    savecallback = SaveCallback()
   

#     progressbar = ProgressBar()
#     wandb_logging = WandbLoggingCallback()
    total_timesteps = n_steps * n_envs * 500
    model.learn(total_timesteps=total_timesteps, progress_bar=True, tb_log_name = '3v1', callback =[savecallback])
    model.save("ppo_gfootball")
