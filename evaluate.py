# import wandb
# wandb.login()
# from tqdm import tqdm
import gymnasium
from gymnasium.spaces import Box, Discrete
import numpy as np
from collections import deque
from gfootball.env import create_environment
from gymnasium.wrappers import StepAPICompatibility
from gfootball.env import observation_preprocessing
import tempfile
import os
import torch
import torch.nn as nn

from gfootball.env.wrappers import Simple115StateWrapper
import ray
from ray.runtime_env import RuntimeEnv
ray.init()

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
        render = True
        logdir = '.'
        if config is not None:
            env_name = config.get("env_name", env_name)
            rewards = config.get("rewards", rewards)
            render = config.get('render', render)
            logdir = config.get('logdir', logdir)
        print("Initializing env " + env_name)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(115,), dtype=np.float32)
        self.env = create_environment(
            env_name=env_name,
            stacked=False,
            representation="simple115v2",
            rewards = rewards,
            write_goal_dumps=False,
            write_full_episode_dumps=False,
            render=render,
            write_video=render,
            dump_frequency=1,
            logdir=logdir,
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
from tqdm.auto import tqdm
            
def main(args):
    tempfile.tempdir = args.tmpdir
    checkpoint = ray.rllib.algorithms.algorithm.Algorithm.from_checkpoint(args.checkpoint)
    policy = check.get_policy()
    env = FootballGym(config={'env_name':scenario_name, 'render':True, 'logdir': args.logdir})
    obs = env.reset()[0]
    with tqdm() as pbar:
        while(True):
            action = policy.compute_single_action(obs)
            obs, reward, done, truncated, info = env.step(action[0])
            if(done):
                break
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed RL Training")

    parser.add_argument("--scenario", type=int, choices=range(1, 12), default = 6,
                        help="Scenario (1-11)")
    parser.add_argument("--checkpoint", default=None, required=True,
                        help="Checkpoint directory (directory path or None)")
    parser.add_argument("--tmpdir", default=None, required=True,
                        help="Temporary directory to store your renders")
    parser.add_argument("--logdir", default=None, required=True,
                        help="Log dir to get your output renders")
    args = parser.parse_args()
    main(args)