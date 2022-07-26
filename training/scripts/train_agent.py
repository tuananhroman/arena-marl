import os
import sys
import time
from datetime import time
from multiprocessing import cpu_count, set_start_method
from typing import Callable, List

import rospkg
import rospy
from rl_utils.rl_utils.envs.pettingzoo_env import env_fn
from rl_utils.rl_utils.training_agent_wrapper import TrainingDRLAgent
from rl_utils.rl_utils.utils.heterogenous_ppo import Heterogenous_PPO
from rl_utils.rl_utils.utils.supersuit_utils import vec_env_create
from rl_utils.rl_utils.utils.utils import instantiate_train_drl_agents
from rosnav.model.agent_factory import AgentFactory
from rosnav.model.base_agent import BaseAgent
from rosnav.model.custom_policy import *
from rosnav.model.custom_sb3_policy import *
from stable_baselines3.common.callbacks import (
    MarlEvalCallback,
    StopTrainingOnRewardThreshold,
)
from task_generator.robot_manager import init_robot_managers
from task_generator.tasks import get_MARL_task, get_task_manager, init_obstacle_manager
from training.tools import train_agent_utils
from training.tools.argsparser import parse_training_args
from training.tools.custom_mlp_utils import *

# from tools.argsparser import parse_training_args
from training.tools.staged_train_callback import InitiateNewTrainStage

# from tools.argsparser import parse_marl_training_args
from training.tools.train_agent_utils import *
from training.tools.train_agent_utils import (
    choose_agent_model,
    create_training_setup,
    get_MARL_agent_name_and_start_time,
    get_paths,
    initialize_hyperparameters,
    load_config,
)


def main(args):
    # load configuration
    config = load_config(args.config)
    robot_names = list(config["robots"].keys())

    # set debug_mode
    rospy.set_param("debug_mode", config["debug_mode"])

    # create dicts for all robot types with all necessary parameters,
    # and instances of the respective models and envs
    robots = create_training_setup(config)

    # set num of timesteps to be generated
    n_timesteps = 40000000 if config["n_timesteps"] is None else config["n_timesteps"]

    start = time.time()
    # Currently still only executes the model.learn() for one robot!!!
    # try:
    model = robots[robot_names[0]]["model"]
    model.learn(
        total_timesteps=n_timesteps,
        reset_num_timesteps=True,
        callback=get_evalcallback(
            robot_name=robot_names[0],
            train_env=robots[robot_names[0]]["env"],
            num_robots=robots[robot_names[0]]["robot_train_params"]["num_robots"],
            num_envs=config["n_envs"],
            agent_dict=robots[robot_names[0]]["agent_dict"],
            config=config,
            PATHS=robots[robot_names[0]]["paths"],
        ),
    )

    # agent_ppo_dict = {agent: robots[agent]["model"] for agent in robot_names}
    # agent_env_dict = {agent: robots[agent]["env"] for agent in robot_names}

    # het_ppo = Heterogenous_PPO(agent_ppo_dict, agent_env_dict)
    # het_ppo.learn(total_timesteps=n_timesteps)

    # except KeyboardInterrupt:
    #     print("KeyboardInterrupt..")
    # finally:
    # update the timesteps the model has trained in total
    # update_total_timesteps_json(n_timesteps, PATHS)

    # robots[robot_names[0]][model].env.close()
    print(f"Time passed: {time.time() - start}s")
    print("Training script will be terminated")
    sys.exit()


def get_evalcallback(
    robot_name: str,
    train_env,
    num_robots,
    num_envs,
    agent_dict: dict,
    config: dict,
    PATHS: dict,
) -> MarlEvalCallback:
    """Function which generates an evaluation callback with an evaluation environment.

    Args:
        train_env (VecEnv): Vectorized training environment
        num_robots (int): Number of robots in the environment
        num_envs (int): Number of parallel spawned environments
        task_mode (str): Task mode for the current experiment
        PATHS (dict): Dictionary which holds hyperparameters for the experiment

    Returns:
        MarlEvalCallback: [description]
    """

    # Setup whole task manager setup with respective robot managers
    task_manager = get_task_manager(
        ns="eval_sim",
        mode=config["task_mode"],
        curriculum_path=config["training_curriculum"]["training_curriculum_file"],
    )

    obstacle_manager = init_obstacle_manager(None, mode="eval")
    task_manager.set_obstacle_manager(obstacle_manager)

    robot_managers = init_robot_managers(
        n_envs=1, robot_type=robot_name, agent_dict=agent_dict, mode="eval"
    )
    task_manager.add_robot_manager(robot_type=robot_name, managers=robot_managers)

    eval_agent_list = instantiate_train_drl_agents(
        num_robots=num_robots,
        existing_robots=0,
        robot_model=robot_name,
        hyperparameter_path=PATHS["hyperparams"],
        ns="eval_sim",
    )

    # Setup evaluation environment
    eval_env = env_fn(
        agent_list=eval_agent_list,
        ns="eval_sim",
        task_manager_reset=task_manager.reset,
        max_num_moves_per_eps=config["periodic_eval"]["max_num_moves_per_eps"],
    )

    return MarlEvalCallback(
        train_env=train_env,
        eval_env=eval_env,
        num_robots=num_robots,
        n_eval_episodes=config["periodic_eval"]["n_eval_episodes"],
        eval_freq=config["periodic_eval"]["eval_freq"],
        deterministic=True,
        log_path=PATHS["eval"],
        best_model_save_path=PATHS["model"],
        callback_on_eval_end=InitiateNewTrainStage(
            n_envs=num_envs,
            treshhold_type=config["training_curriculum"]["threshold_type"],
            upper_threshold=config["training_curriculum"]["upper_threshold"],
            lower_threshold=config["training_curriculum"]["lower_threshold"],
            task_mode=config["training_curriculum"],
            verbose=1,
        ),
        callback_on_new_best=StopTrainingOnRewardThreshold(
            treshhold_type=config["stop_training"]["threshold_type"],
            threshold=config["stop_training"]["threshold"],
            verbose=1,
        ),
    )


if __name__ == "__main__":
    set_start_method("fork")
    # args, _ = parse_marl_training_args()
    args, _ = parse_training_args()
    # rospy.init_node("train_env", disable_signals=False, anonymous=True)
    main(args)
