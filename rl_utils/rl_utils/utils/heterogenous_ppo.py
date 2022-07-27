import time
from collections import deque
from typing import Dict, Optional, Tuple

import gym
import numpy as np
import torch as th

from rl_utils.rl_utils.envs.pettingzoo_env import FlatlandPettingZooEnv
from rl_utils.rl_utils.utils.utils import call_service_takeSimStep
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common import logger
from stable_baselines3.ppo.ppo import PPO
from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper


class Heterogenous_PPO(object):
    def __init__(
        self,
        agent_ppo_dict: Dict[str, PPO],
        agent_env_dict: Dict[str, SB3VecEnvWrapper],
        n_envs: int,
        verbose: bool = True,
    ) -> None:
        self.agent_ppo_dict = agent_ppo_dict
        self.agent_env_dict = agent_env_dict

        self.n_envs, self.ns_prefix = n_envs, "sim_"
        self.num_timesteps = 0

    def _setup_learn(
        self,
        total_timesteps: int,
        eval_env: Optional[FlatlandPettingZooEnv],
        callback=None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
    ) -> Tuple[int, BaseCallback]:
        """
        Initialize different variables needed for training.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param eval_env: Environment to use for evaluation.
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param eval_freq: How many steps between evaluations
        :param n_eval_episodes: How many episodes to play per evaluation
        :param log_path: Path to a folder where the evaluations will be saved
        :param reset_num_timesteps: Whether to reset or not the ``num_timesteps`` attribute
        :param tb_log_name: the name of the run for tensorboard log
        :return:
        """
        self.start_time = time.time()

        for _, ppo in self.agent_ppo_dict.items():
            if ppo.ep_info_buffer is None or reset_num_timesteps:
                ppo.ep_info_buffer = deque(maxlen=100)
                ppo.ep_success_buffer = deque(maxlen=100)

            if ppo.action_noise is not None:
                ppo.action_noise.reset()

            if reset_num_timesteps:
                ppo.num_timesteps = 0
                ppo._episode_num = 0
            else:
                # Make sure training timesteps are ahead of the internal counter
                total_timesteps += ppo.num_timesteps

            self._total_timesteps = total_timesteps

            # Avoid resetting the environment when calling ``.learn()`` consecutive times
            if reset_num_timesteps or ppo._last_obs is None:
                ppo.env.reset()
                ppo._last_obs = ppo.env.reset()
                ppo._last_dones = np.zeros((ppo.env.num_envs,), dtype=bool)
                # Retrieve unnormalized observation for saving into the buffer
                if ppo._vec_normalize_env is not None:
                    ppo._last_original_obs = ppo._vec_normalize_env.get_original_obs()

            if eval_env is not None and ppo.seed is not None:
                eval_env.seed(ppo.seed)

            # Configure logger's outputs
            # utils.configure_logger(
            #     self.verbose, self.tensorboard_log, tb_log_name, reset_num_timesteps
            # )

            # Create eval callback if needed
            # callback = self._init_callback(
            #     callback, eval_env, eval_freq, n_eval_episodes, log_path
            # )

        return total_timesteps, None

    def collect_rollouts(
        self,
        callback: BaseCallback,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert all(
            list(map(lambda x: x._last_obs is not None, self.agent_ppo_dict.values()))
        ), "No previous observation was provided"

        n_steps = 0
        rollout_buffers = {
            agent: ppo.rollout_buffer for agent, ppo in self.agent_ppo_dict.items()
        }

        for agent, ppo in self.agent_ppo_dict.items():
            if ppo.use_sde:
                ppo.policy.reset_noise(self.agent_env_dict[agent].num_envs)

        # TODO: CALLBACK
        # callback.on_rollout_start()
        agent_actions_dict = {agent: None for agent in self.agent_ppo_dict.keys()}
        agent_dones_dict = {agent: None for agent in self.agent_ppo_dict.keys()}
        agent_values_dict = {agent: None for agent in self.agent_ppo_dict.keys()}
        agent_log_probs_dict = {agent: None for agent in self.agent_ppo_dict.keys()}
        agent_new_obs_dict = {agent: None for agent in self.agent_ppo_dict.keys()}

        while n_steps < n_rollout_steps:
            for agent, ppo in self.agent_ppo_dict.items():
                if (
                    ppo.use_sde
                    and ppo.sde_sample_freq > 0
                    and n_steps % ppo.sde_sample_freq == 0
                ):
                    ppo.policy.reset_noise(self.agent_env_dict[agent].num_envs)

                with th.no_grad():
                    # Convert to pytorch tensor
                    obs_tensor = th.as_tensor(ppo._last_obs).to(ppo.device)
                    (
                        actions,
                        agent_values_dict[agent],
                        agent_log_probs_dict[agent],
                    ) = ppo.policy.forward(obs_tensor)

                agent_actions_dict[agent] = actions.cpu().numpy()
                # Rescale and perform action
                clipped_actions = agent_actions_dict[agent]
                # Clip the actions to avoid out of bound error
                if isinstance(ppo.action_space, gym.spaces.Box):
                    clipped_actions = np.clip(
                        agent_actions_dict[agent],
                        ppo.action_space.low,
                        ppo.action_space.high,
                    )

                # Env step for all robots
                self.agent_env_dict[agent].step(clipped_actions)

            for i in range(1, self.n_envs):
                call_service_takeSimStep(ns=self.ns_prefix + str(i))

            for agent, ppo in self.agent_ppo_dict.items():
                (
                    agent_new_obs_dict[agent],
                    rewards,
                    agent_dones_dict[agent],
                    infos,
                ) = self.agent_env_dict[agent].step()

                # TODO: Training with different number of robots (num_envs)
                ppo.num_timesteps += self.agent_env_dict[agent].num_envs

                # TODO: CALLBACK
                # Give access to local variables
                # callback.update_locals(locals())
                # if callback.on_step() is False:
                #     return False

                ppo._update_info_buffer(infos)
                n_steps += 1

                if isinstance(ppo.action_space, gym.spaces.Discrete):
                    # Reshape in case of discrete action
                    actions = actions.reshape(-1, 1)

                rollout_buffers[agent].add(
                    ppo._last_obs,
                    agent_actions_dict[agent],
                    rewards,
                    ppo._last_dones,
                    agent_values_dict[agent],
                    agent_log_probs_dict[agent],
                )
                ppo._last_obs = agent_new_obs_dict[agent]
                ppo._last_dones = agent_dones_dict[agent]

        for agent, ppo in self.agent_ppo_dict.items():
            with th.no_grad():
                # Compute value for the last timestep
                obs_tensor = th.as_tensor(agent_new_obs_dict[agent]).to(ppo.device)
                _, agent_values_dict[agent], _ = ppo.policy.forward(obs_tensor)

        for agent in self.agent_ppo_dict.keys():
            rollout_buffers[agent].compute_returns_and_advantage(
                last_values=agent_values_dict[agent], dones=agent_dones_dict[agent]
            )

        # TODO: CALLBACK
        # callback.on_rollout_end()

        return True

    def train(self):
        for agent, ppo in self.agent_ppo_dict.items():
            print(f"[{agent}] Start Training Procedure")
            ppo.train()

    def learn(
        self,
        total_timesteps: int,
        callback=None,
        log_interval: int = 1,
        eval_env=None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "OnPolicyAlgorithm",
        eval_log_path=None,
        reset_num_timesteps: bool = True,
    ) -> "OnPolicyAlgorithm":
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name,
        )

        # callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(
                callback,
                n_rollout_steps=self.agent_ppo_dict["jackal"].n_steps,
            )

            if continue_training is False:
                break

            iteration += 1
            # self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            # if log_interval is not None and iteration % log_interval == 0:
            #     fps = int(self.num_timesteps / (time.time() - self.start_time))
            #     logger.record("time/iterations", iteration, exclude="tensorboard")
            #     if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            #         logger.record(
            #             "rollout/ep_rew_mean",
            #             safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]),
            #         )
            #         logger.record(
            #             "rollout/ep_len_mean",
            #             safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]),
            #         )
            #     logger.record("time/fps", fps)
            #     logger.record(
            #         "time/time_elapsed",
            #         int(time.time() - self.start_time),
            #         exclude="tensorboard",
            #     )
            #     logger.record(
            #         "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
            #     )
            #     logger.dump(step=self.num_timesteps)

            self.train()

        # callback.on_training_end()

        return self

    def _init_callback(self, callback):
        raise NotImplementedError()
