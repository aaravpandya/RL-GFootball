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
import torch
import torch.nn as nn
import torch.nn.functional as F
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
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


class FootballGym(gymnasium.Env):
    spec = None
    metadata = None
    def debug(self):
        print(self.reset()[0].shape)
    def __init__(self, config=None):
        super(FootballGym, self).__init__()
        env_name = "academy_empty_goal_close"
        rewards = "scoring,checkpoints"
        render = False
        logdir = "."
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

class TorchCustomModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)
        self.input_size = 115
        self.output_size = 19
        self.d_model = 115
        self.fc0 = nn.Linear(115, 115)
        self.attention1 = nn.MultiheadAttention(embed_dim=115, num_heads=23)
        self.attention2 = nn.MultiheadAttention(embed_dim=115, num_heads=23)
        self.attention3 = nn.MultiheadAttention(embed_dim=115, num_heads=23)
        self.fc1 = nn.Linear(115, 115)
        self.fc2 = nn.Linear(115, 115)
        self.fc3 = nn.Linear(115, 115)  # Added an additional feedforward layer
        self.fc4 = nn.Linear(115, 19)
        self.value_branch = nn.Linear(115, 1)

    def forward(self, input_dict, state, seq_lens):
        input_dict["obs"] = input_dict["obs"].float()
        x = F.relu(self.fc0(input_dict["obs"]))
        x = F.relu(self.fc1(x))
        x = x.unsqueeze(0)
        x, _ = self.attention1(x, x, x)
        x = x.squeeze(0)
        x = F.relu(self.fc2(x))
        x = x.unsqueeze(0)
        x, _ = self.attention2(x, x, x)  # Added the second attention layer
        x = x.squeeze(0)
        x = F.relu(self.fc3(x))  # Added an additional feedforward layer
        x = x.unsqueeze(0)
        x, _ = self.attention3(x, x, x)
        x = x.squeeze(0)
        logits = F.relu(self.fc4(x))
        value_estimate = F.relu(self.value_branch(x))

        self._value_out = value_estimate.squeeze(1)
        return logits, []

    def value_function(self):
        return self._value_out
ModelCatalog.register_custom_model(
        "my_model", TorchCustomModel
    )


def main(args):
    tempfile.tempdir = args.tmpdir
    scenario_name = scenarios[args.scenario]
    if(args.algorithm == 'PPO'):
        algo = ( PPOConfig()
            .rollouts(num_rollout_workers=args.num_cpus, num_envs_per_worker=4)
            .resources(num_gpus=torch.cuda.device_count(), num_learner_workers = 16)
            .environment(env=FootballGym, env_config ={'env_name':scenario_name, 'logdir':logdir})
            .training(
                    model={
                        "custom_model": "my_model",
                        "vf_share_layers": True,
                    },
                    train_batch_size = 4000
                )
            .build()
        )
    elif (args.algorithm == 'IMPALA'):
        config = ImpalaConfig()
        config = config.resources(num_gpus = torch.cuda.device_count(), num_learner_workers = 4, num_gpus_per_learner_worker = args.num_gpus)
        config = config.rollouts(num_rollout_workers=args.num_cpus, num_envs_per_worker=1)
        config = config.environment(env=FootballGym, env_config ={'env_name':scenario_name, 'logdir':logdir})
        config = config.training(
                    model={
                        "custom_model": "my_model",
                        "vf_share_layers": True
                    }
                , train_batch_size=4000)
        algo = config.build()
    run = wandb.init(
    # Set the project where this run will be logged
    project="HPML-Project",
    # Track hyperparameters and run metadata
    config={
        'n_envs':args.num_cpus,
        'n_steps':100,
        'env':scenario_name,
        'policy':'Attention'
        })
    
    if(args.resume is not None):
        algo.restore(args.resume)

    for i in tqdm(range(args.num_iterations)):
        result = algo.train()
        learner_stats = result['info']['learner']['default_policy']['learner_stats']
        learner_stats['num_env_steps_sampled'] = result['num_env_steps_sampled']
        learner_stats['num_env_steps_trained'] = result['num_env_steps_trained']
        learner_stats['episode_reward_mean'] = result['episode_reward_mean']
        learner_stats['episodes_total'] = result['episodes_total']
        wandb.log(learner_stats)
        if i % 100 == 0:
            checkpoint_dir = algo.save(checkpoint_dir=args.checkpoint)
            
    checkpoint_dir = algo.save(checkpoint_dir=args.checkpoint)
    print(f"Checkpoint saved in directory {checkpoint_dir}")
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Distributed RL Training")

    parser.add_argument("--scenario", type=int, choices=range(1, 12), default = 6,
                        help="Scenario (1-11)")
    parser.add_argument("--num-cpus", type=int, required=True,
                        help="Number of CPUs")
    parser.add_argument("--num-gpus", type=int, required=True,
                        help="Number of GPUs per learner worker")
    parser.add_argument("--algorithm", choices=["PPO", "IMPALA"], required=True,
                        help="Algorithm (PPO or IMPALA)")
    parser.add_argument("--num-iterations", type=int, required=True,
                        help="Number of iterations")
    parser.add_argument("--resume", default=None,
                        help="Resume (directory path or None)")
    parser.add_argument("--checkpoint", default=None, required=True,
                        help="Checkpoint directory (directory path or None)")
    parser.add_argument("--logdir", default=None, required=True,
                        help="Log dir to get your output renders")
    parser.add_argument("--tmpdir", default=None, required=True,
                        help="Temporary directory to store your renders if rendering is enabled")
    args = parser.parse_args()
    main(args)