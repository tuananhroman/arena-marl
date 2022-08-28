import os
import time
from collections import deque
from typing import Dict, Optional, Tuple

import gym
import numpy as np
import rospy
import torch as th
from rl_utils.rl_utils.envs.pettingzoo_env import FlatlandPettingZooEnv
from rl_utils.rl_utils.utils.utils import call_service_takeSimStep
from rl_utils.rl_utils.utils.wandb_helper import WandbLogger
from stable_baselines3.common import utils

# from stable_baselines3.common import logger
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.ppo.ppo import PPO
from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper


class Heterogenous_PPO(object):
    def __init__(
        self,
        agent_ppo_dict: Dict[str, PPO],
        agent_env_dict: Dict[str, SB3VecEnvWrapper],
        n_envs: int,
        wandb_logger: WandbLogger,
        model_save_path_dict: Dict[str, str] = None,
        verbose: bool = True,
    ) -> None:
        self.agent_ppo_dict = agent_ppo_dict
        self.agent_env_dict = agent_env_dict

        self.model_save_path_dict = model_save_path_dict or {
            agent: None for agent in agent_ppo_dict
        }

        self.wandb_logger = wandb_logger

        self.n_envs, self.ns_prefix = n_envs, "sim_"
        self.num_timesteps = 0
        self.verbose = verbose

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

        ### First reset all environment states
        for agent, ppo in self.agent_ppo_dict.items():
            # ppo.device = "cpu"
            # ppo.policy.cpu()

            # reinitialize the rollout buffer when ppo was loaded and does not
            # have the appropriate shape
            if ppo.rollout_buffer.n_envs != self.agent_env_dict[agent].num_envs:
                self._update_rollout_buffer(agent, ppo)

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
                ### reset states
                ppo.env.reset()

        ### perform one step in each simulation to update the scene
        for i in range(1, self.n_envs + 1):
            call_service_takeSimStep(ns=self.ns_prefix + str(i))

        ### Now reset all environments to get respective last observations
        for _, ppo in self.agent_ppo_dict.items():
            ### get new observations
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

        # Create directories for best models and logs
        self._init_callback(callback)

        return total_timesteps, None

    def collect_rollouts(
        self,
        callback: BaseCallback,
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

        Heterogenous_PPO._reset_all_rollout_buffers(rollout_buffers)

        for agent, ppo in self.agent_ppo_dict.items():
            if ppo.use_sde:
                ppo.policy.reset_noise(self.agent_env_dict[agent].num_envs)

        if callback:
            callback.on_rollout_start()

        complete_collection_dict = {
            agent: False for agent in self.agent_ppo_dict.keys()
        }

        agent_actions_dict = {agent: None for agent in self.agent_ppo_dict.keys()}
        agent_dones_dict = {agent: None for agent in self.agent_ppo_dict.keys()}
        agent_values_dict = {agent: None for agent in self.agent_ppo_dict.keys()}
        agent_log_probs_dict = {agent: None for agent in self.agent_ppo_dict.keys()}
        agent_new_obs_dict = {agent: None for agent in self.agent_ppo_dict.keys()}

        # only end loop when all replay buffers are filled
        while not all(complete_collection_dict.values()):
            for agent, ppo in self.agent_ppo_dict.items():
                if (
                    ppo.use_sde
                    and ppo.sde_sample_freq > 0
                    and n_steps % ppo.sde_sample_freq == 0
                ):
                    ppo.policy.reset_noise(self.agent_env_dict[agent].num_envs)

                actions = Heterogenous_PPO.infer_action(
                    agent,
                    ppo,
                    agent_actions_dict,
                    agent_values_dict,
                    agent_log_probs_dict,
                )

                # Env step for all robots
                self.agent_env_dict[agent].step(actions)

            for i in range(1, self.n_envs + 1):
                call_service_takeSimStep(ns=self.ns_prefix + str(i))

            for agent, ppo in self.agent_ppo_dict.items():
                (
                    agent_new_obs_dict[agent],
                    rewards,
                    agent_dones_dict[agent],
                    infos,
                ) = self.agent_env_dict[agent].step(
                    actions
                )  # Apply dummy action

                ### Print size of rollout buffer
                #   For debugging purposes
                if n_steps % 100 == 0:
                    print(
                        "Size of rollout buffer for agent {}: {}".format(
                            agent, rollout_buffers[agent].pos
                        )
                    )

                # only continue memorizing experiences if buffer is not full
                # and if at least one robot is still alive
                if not complete_collection_dict[
                    agent
                ] and not Heterogenous_PPO.check_robot_model_done(
                    agent_dones_dict[agent]
                ):
                    ppo.num_timesteps += self.agent_env_dict[agent].num_envs
                    ppo._update_info_buffer(infos)

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

            # Give access to local variables
            if callback:
                callback.update_locals(locals())
                # Stop training if all model performances (mean rewards) are higher than threshold
                if callback.on_step() is False:
                    return False

            if Heterogenous_PPO.check_for_reset(agent_dones_dict, rollout_buffers):
                self.reset_all_envs()

            n_steps += 1

            self.check_for_complete_collection(complete_collection_dict, n_steps)

        ### Print size of rollout buffer
        #   For debugging purposes - print the last size of the rollout buffer that is skipped before
        # if n_steps % 100 == 0:
        #     print(
        #         "Size of rollout buffer for agent {}: {}".format(
        #             agent, rollout_buffers[agent].pos
        #         )
        #     )

        for agent, ppo in self.agent_ppo_dict.items():
            with th.no_grad():
                # Compute value for the last timestep
                obs_tensor = th.as_tensor(agent_new_obs_dict[agent]).to(ppo.device)
                _, agent_values_dict[agent], _ = ppo.policy.forward(obs_tensor)

        for agent in self.agent_ppo_dict.keys():
            rollout_buffers[agent].compute_returns_and_advantage(
                last_values=agent_values_dict[agent], dones=agent_dones_dict[agent]
            )

        ### Eval and save
        #   Perform evaluation phase and save best model for each robot, if it has improved
        if callback:
            return callback.on_rollout_end(), n_steps

        return True, n_steps

    def train(self):
        for agent, ppo in self.agent_ppo_dict.items():
            print(f"[{agent}] Start Training Procedure")
            ppo.train(agent)

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

        total_timesteps, _ = self._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name,
        )

        # if callback:
        # callback.on_training_start(locals(), globals())

        avg_n_robots = np.mean([envs.num_envs for envs in self.agent_env_dict.values()])

        while self.num_timesteps < total_timesteps:

            start_time = time.time()
            continue_training, n_steps = self.collect_rollouts(callback)

            if continue_training is False:
                break

            self.num_timesteps = n_steps * avg_n_robots

            iteration += 1

            # TODO: WandB LOGGING
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                duration = time.time() - start_time
                fps = int(self.num_timesteps / duration)
                if self.wandb_logger:
                    self.wandb_logger.log_single(
                        "time/fps", fps, step=self.num_timesteps
                    )
                    self.wandb_logger.log_single(
                        "time/total_timesteps",
                        self.num_timesteps,
                        step=self.num_timesteps,
                    )
                print("---------------------------------------")
                print(
                    "Iteration: {}\tTimesteps: {}\tFPS: {}".format(
                        iteration, self.num_timesteps, fps
                    )
                )
                print("---------------------------------------")
                # logger.record("time/iterations", iteration, exclude="tensorboard")
                # if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                #     logger.record(
                #         "rollout/ep_rew_mean",
                #         safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]),
                #     )
                #     logger.record(
                #         "rollout/ep_len_mean",
                #         safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]),
                #     )
                # logger.record("time/fps", fps)
                # logger.record(
                #     "time/time_elapsed",
                #     int(time.time() - self.start_time),
                #     exclude="tensorboard",
                # )
                # logger.record(
                #     "time/total_timesteps", self.num_timesteps, exclude="tensorboard"
                # )
                # logger.dump(step=self.num_timesteps)

            self.train()

        self.save_models()

        if callback:
            callback.on_training_end()

        return self

    def _init_callback(self, callback):
        callback._init_callback()

    @staticmethod
    def infer_action(
        agent: str,
        ppo: PPO,
        agent_actions_dict: dict,
        agent_values_dict: dict,
        agent_log_probs_dict: dict,
    ) -> np.ndarray:
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
        return clipped_actions

    @staticmethod
    def check_for_reset(
        dones_dict: dict, rollout_buffer_dict: Dict[str, RolloutBuffer]
    ) -> bool:
        """Check for each robot type if either the agents are done or the buffer is full. If one is true for each robot type prepare for reset.

        Args:
            dones_dict (dict): dictionary with all done values.
            rollout_buffer_dict (Dict[str, RolloutBuffer]): dictionary with the respective rolloutbuffers.

        Returns:
            bool: _description_
        """
        buffers_full = [
            rollout_buffer.full for rollout_buffer in rollout_buffer_dict.values()
        ]
        dones = list(map(lambda x: np.all(x == 1), dones_dict.values()))
        check = [_a or _b for _a, _b in zip(buffers_full, dones)]
        return all(check)

    @staticmethod
    def check_robot_model_done(dones_array: np.ndarray) -> bool:
        return all(list(map(lambda x: np.all(x == 1), dones_array)))

    def check_for_complete_collection(
        self, completion_dict: dict, curr_steps_count: int
    ) -> bool:
        # in hyperparameter file: n_steps = n_envs / batch_size
        # RolloutBuffer size = n_steps * (n_envs * n_robots)
        for agent, ppo in self.agent_ppo_dict.items():
            # if ppo.n_steps <= curr_steps_count:
            if ppo.rollout_buffer.full and not completion_dict[agent]:
                completion_dict[agent] = True

    @staticmethod
    def _reset_all_rollout_buffers(
        rollout_buffer_dict: Dict[str, RolloutBuffer]
    ) -> None:
        for buffer in rollout_buffer_dict.values():
            buffer.reset()

    def reset_all_envs(self) -> None:
        for agent, env in self.agent_env_dict.items():
            # reset states
            env.reset()
            # retrieve new simulation state
            self.agent_ppo_dict[agent]._last_obs = env.reset()
        # perform one step in each simulation to update the scene
        for i in range(1, self.n_envs + 1):
            call_service_takeSimStep(ns=self.ns_prefix + str(i))

    def _update_current_progress_remaining(
        self, num_timesteps: int, total_timesteps: int
    ) -> None:
        """
        Compute current progress remaining (starts from 1 and ends to 0)

        :param num_timesteps: current number of timesteps
        :param total_timesteps:
        """
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(
            total_timesteps
        )

    def save_models(self) -> None:
        if not rospy.get_param("debug_mode", False):
            for agent, ppo in self.agent_ppo_dict.items():
                if self.model_save_path_dict[agent]:
                    ppo.save(
                        os.path.join(self.model_save_path_dict[agent], "best_model")
                    )

    def _update_rollout_buffer(self, agent: str, ppo: PPO) -> None:
        ppo.rollout_buffer = RolloutBuffer(
            ppo.n_steps,
            ppo.observation_space,
            ppo.action_space,
            ppo.device,
            gamma=ppo.gamma,
            gae_lambda=ppo.gae_lambda,
            n_envs=self.agent_env_dict[agent].num_envs,
        )
