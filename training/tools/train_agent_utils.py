from typing import Union, Type

import argparse
from datetime import datetime as dt
import gym
import json
import os

import yaml
import rosnode
import rospkg
import time
import warnings

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.utils import set_random_seed


# from rl_utils.envs.flatland_gym_env import (
#     FlatlandEnv,
# )
from rosnav.model.agent_factory import AgentFactory
from rosnav.model.base_agent import BaseAgent
from .custom_mlp_utils import get_act_fn
import rospy

""" 
Dict containing agent specific hyperparameter keys (for documentation and typing validation purposes)

:key agent_name: Precise agent name (as generated by get_agent_name())
:key robot: Robot name to load robot specific .yaml file containing settings
:key batch_size: Batch size (n_envs * n_steps)
:key gamma: Discount factor
:key n_steps: The number of steps to run for each environment per update
:key ent_coef: Entropy coefficient for the loss calculation
:key learning_rate: The learning rate, it can be a function
    of the current progress remaining (from 1 to 0)
    (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
:key vf_coef: Value function coefficient for the loss calculation
:key max_grad_norm: The maximum value for the gradient clipping
:key gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
:key m_batch_size: Minibatch size
:key n_epochs: Number of epoch when optimizing the surrogate loss
:key clip_range: Clipping parameter, it can be a function of the current progress
    remaining (from 1 to 0).
:key train_max_steps_per_episode: Max timesteps per training episode
:key eval_max_steps_per_episode: Max timesteps per evaluation episode
:key goal_radius: Radius of the goal
:key reward_fnc: Number of the reward function (defined in ../rl_agent/utils/reward.py)
:key discrete_action_space: If robot uses discrete action space
:key normalize: If observations are normalized before fed to the network
:key task_mode: Mode tasks will be generated in (custom, random, staged).
:key curr_stage: In case of staged training which stage to start with.
:param n_timesteps: The number of timesteps trained on in total.
"""
hyperparams = {
    key: None
    for key in [
        "agent_name",
        "robot",
        "actions_in_observationspace",
        "batch_size",
        "gamma",
        "n_steps",
        "ent_coef",
        "learning_rate",
        "vf_coef",
        "max_grad_norm",
        "gae_lambda",
        "m_batch_size",
        "n_epochs",
        "clip_range",
        "reward_fnc",
        "discrete_action_space",
        "normalize",
        "task_mode",
        "curr_stage",
        "train_max_steps_per_episode",
        "eval_max_steps_per_episode",
        "goal_radius",
    ]
}


def load_config(config_name: str) -> dict:
    """
    Load config parameters from config file
    """
    config_location = os.path.join(
        rospkg.RosPack().get_path("training"), "configs", config_name
    )
    with open(config_location, "r", encoding="utf-8") as target:
        config = yaml.load(target, Loader=yaml.FullLoader)

    return config


def initialize_hyperparameters(PATHS: dict, config: dict, n_envs: int) -> dict:
    """
    Write hyperparameters to json file in case agent is new otherwise load existing hyperparameters

    :param PATHS: dictionary containing model specific paths
    :param load_target: unique agent name (when calling --load)
    :param config_name: name of the hyperparameter file in /configs/hyperparameters
    :param n_envs: number of envs
    """
    # when building new agent
    if config["resume"] is None:
        hyperparams = load_hyperparameters_json(PATHS=PATHS, from_scratch=True)
        hyperparams["agent_name"] = PATHS["model"].split("/")[-1]
    else:
        hyperparams = load_hyperparameters_json(PATHS=PATHS)

    # dynamically adapt n_steps according to batch size and n envs
    # then update .json
    check_batch_size(n_envs, hyperparams["batch_size"], hyperparams["m_batch_size"])
    hyperparams["n_steps"] = int(hyperparams["batch_size"] / n_envs)
    if not rospy.get_param("debug_mode"):
        write_hyperparameters_json(hyperparams, PATHS)
    print_hyperparameters(hyperparams)
    return hyperparams


def write_hyperparameters_json(hyperparams: dict, PATHS: dict) -> None:
    """
    Write hyperparameters.json to agent directory

    :param hyperparams: dict containing model specific hyperparameters
    :param PATHS: dictionary containing model specific paths
    """
    doc_location = os.path.join(PATHS.get("model"), "hyperparameters.json")

    with open(doc_location, "w", encoding="utf-8") as target:
        json.dump(hyperparams, target, ensure_ascii=False, indent=4)


def load_hyperparameters_json(PATHS: dict, from_scratch: bool = False) -> dict:
    """
    Load hyperparameters from model directory when loading - when training from scratch
    load from ../configs/hyperparameters

    :param PATHS: dictionary containing model specific paths
    :param from_scatch: if training from scratch
    :param config_name: file name of json file when training from scratch
    """
    if from_scratch:
        doc_location = os.path.join(PATHS.get("hyperparams"))
    else:
        doc_location = os.path.join(PATHS.get("model"), "hyperparameters.json")

    if os.path.isfile(doc_location):
        with open(doc_location, "r") as file:
            hyperparams = json.load(file)
        check_hyperparam_format(loaded_hyperparams=hyperparams, PATHS=PATHS)
        return hyperparams
    else:
        if from_scratch:
            raise FileNotFoundError("Found no '%s'" % PATHS.get("hyperparams"))
        else:
            raise FileNotFoundError(
                "Found no 'hyperparameters.json' in %s" % PATHS.get("model")
            )


def update_total_timesteps_json(timesteps: int, PATHS: dict) -> None:
    """
    Update total number of timesteps in json file

    :param hyperparams_obj(object, agent_hyperparams): object containing containing model specific hyperparameters
    :param PATHS: dictionary containing model specific paths
    """
    doc_location = os.path.join(PATHS.get("model"), "hyperparameters.json")
    hyperparams = load_hyperparameters_json(PATHS=PATHS)

    try:
        curr_timesteps = int(hyperparams["n_timesteps"]) + timesteps
        hyperparams["n_timesteps"] = curr_timesteps
    except Exception:
        raise Warning(
            "Parameter 'total_timesteps' not found or not of type Integer in 'hyperparameter.json'!"
        )
    else:
        with open(doc_location, "w", encoding="utf-8") as target:
            json.dump(hyperparams, target, ensure_ascii=False, indent=4)


def print_hyperparameters(hyperparams: dict) -> None:
    print("\n--------------------------------")
    print("         HYPERPARAMETERS         \n")
    for param, param_val in hyperparams.items():
        print("{:30s}{:<10s}".format((param + ":"), str(param_val)))
    print("--------------------------------\n\n")


def check_hyperparam_format(loaded_hyperparams: dict, PATHS: dict) -> None:
    if set(hyperparams.keys()) != set(loaded_hyperparams.keys()):
        missing_keys = set(hyperparams.keys()).difference(
            set(loaded_hyperparams.keys())
        )
        redundant_keys = set(loaded_hyperparams.keys()).difference(
            set(hyperparams.keys())
        )
        raise AssertionError(
            f"unmatching keys, following keys missing: {missing_keys} \n"
            f"following keys unused: {redundant_keys}"
        )
    if not isinstance(loaded_hyperparams["discrete_action_space"], bool):
        raise TypeError("Parameter 'discrete_action_space' not of type bool")
    if loaded_hyperparams["task_mode"] not in ["custom", "random", "staged"]:
        raise TypeError("Parameter 'task_mode' has unknown value")


def update_hyperparam_model(
    model: PPO, PATHS: dict, params: dict, n_envs: int = 1
) -> None:
    """
    Updates parameter of loaded PPO agent when it was manually changed in the configs yaml.

    :param model(object, PPO): loaded PPO agent
    :param PATHS: program relevant paths
    :param params: dictionary containing loaded hyperparams
    :param n_envs: number of parallel environments
    """
    if model.batch_size != params["batch_size"]:
        model.batch_size = params["batch_size"]
    if model.gamma != params["gamma"]:
        model.gamma = params["gamma"]
    if model.n_steps != params["n_steps"]:
        model.n_steps = params["n_steps"]
    if model.ent_coef != params["ent_coef"]:
        model.ent_coef = params["ent_coef"]
    if model.learning_rate != params["learning_rate"]:
        model.learning_rate = params["learning_rate"]
    if model.vf_coef != params["vf_coef"]:
        model.vf_coef = params["vf_coef"]
    if model.max_grad_norm != params["max_grad_norm"]:
        model.max_grad_norm = params["max_grad_norm"]
    if model.gae_lambda != params["gae_lambda"]:
        model.gae_lambda = params["gae_lambda"]
    if model.n_epochs != params["n_epochs"]:
        model.n_epochs = params["n_epochs"]
    """
    if model.clip_range != params['clip_range']:
        model.clip_range = params['clip_range']
    """
    if model.n_envs != n_envs:
        model.update_n_envs()
    if model.rollout_buffer.buffer_size != params["n_steps"]:
        model.rollout_buffer.buffer_size = params["n_steps"]
    if model.tensorboard_log != PATHS["tb"]:
        model.tensorboard_log = PATHS["tb"]


def check_batch_size(n_envs: int, batch_size: int, mn_batch_size: int) -> None:
    assert (
        batch_size > mn_batch_size
    ), f"Mini batch size {mn_batch_size} is bigger than batch size {batch_size}"

    assert (
        batch_size % mn_batch_size == 0
    ), f"Batch size {batch_size} isn't divisible by mini batch size {mn_batch_size}"

    assert (
        batch_size % n_envs == 0
    ), f"Batch size {batch_size} isn't divisible by n_envs {n_envs}"

    assert (
        batch_size % mn_batch_size == 0
    ), f"Batch size {batch_size} isn't divisible by mini batch size {mn_batch_size}"


def get_agent_name(args: argparse.Namespace) -> str:
    """Function to get agent name to save to/load from file system

    Example names:
    "MLP_B_64-64_P_32-32_V_32-32_relu_2021_01_07__10_32"
    "DRL_LOCAL_PLANNER_2021_01_08__7_14"

    :param args (argparse.Namespace): Object containing the program arguments
    """
    START_TIME = dt.now().strftime("%Y_%m_%d__%H_%M")

    if args.custom_mlp:
        return (
            "MLP_B_"
            + args.body
            + "_P_"
            + args.pi
            + "_V_"
            + args.vf
            + "_"
            + args.act_fn
            + "_"
            + START_TIME
        )
    if args.load is None:
        return args.agent + "_" + START_TIME
    return args.load


def get_MARL_agent_name_and_start_time() -> str:
    """Function to get MARL agent parent dir, where seperate agents for different robots are saved"""
    START_TIME = dt.now().strftime("%Y_%m_%d__%H_%M")

    return "MARL_AGENTS_" + START_TIME, START_TIME


def get_paths(
    marl_dir: str,
    robot: str,
    agent_name: str,
    config_params: dict,
    curriculum: str,
    eval_log: bool,
    tb: bool,
) -> dict:
    """
    Function to generate agent specific paths

    :param agent_name: Precise agent name (as generated by get_agent_name())
    :param args (argparse.Namespace): Object containing the program arguments
    """
    dir = rospkg.RosPack().get_path("rosnav")
    training = rospkg.RosPack().get_path("training")
    simulation_setup = rospkg.RosPack().get_path("arena-simulation-setup")

    PATHS = {
        "model": os.path.join(dir, "agents", marl_dir, agent_name)
        if config_params["resume"] is None
        else config_params["resume"],
        "tb": os.path.join(training, "training_logs", "tensorboard", agent_name),
        "eval": os.path.join(
            training, "training_logs", "train_eval_log", agent_name
        ),
        "robot_setting": os.path.join(
            simulation_setup,
            "robot",
            robot,
            robot + ".model.yaml",
        ),
        "hyperparams": os.path.join(
            training, "configs", "hyperparameters", config_params["hyperparameter_file"]
        ),
        "robot_as": os.path.join(
            simulation_setup, "robot", robot, "default_settings.yaml"
        ),
        "curriculum": os.path.join(training, "configs", "training_curriculums", curriculum),
    }
    rospy.get_param("/debug_mode")

    # check for mode
    if not rospy.get_param("/debug_mode"):
        if config_params["resume"] is None:
            os.makedirs(PATHS["model"])
        elif not os.path.isfile(
            os.path.join(PATHS["model"], agent_name + ".zip")
        ) and not os.path.isfile(os.path.join(PATHS["model"], "best_model.zip")):
            raise FileNotFoundError(
                "Couldn't find model named %s.zip' or 'best_model.zip' in '%s'"
                % (agent_name, PATHS["model"])
            )

        # evaluation log enabled
        if eval_log:
            if not os.path.exists(PATHS["eval"]):
                os.makedirs(PATHS["eval"])
        else:
            PATHS["eval"] = None
        # tensorboard log enabled
        if tb:
            if not os.path.exists(PATHS["tb"]):
                os.makedirs(PATHS["tb"])
        else:
            PATHS["tb"] = None
    else:
        PATHS["eval"] = None
        PATHS["tb"] = None

    return PATHS


# def make_envs(
#     args: argparse.Namespace,
#     with_ns: bool,
#     rank: int,
#     params: dict,
#     seed: int = 0,
#     PATHS: dict = None,
#     train: bool = True,
# ):
#     """
#     Utility function for multiprocessed env

#     :param with_ns: (bool) if the system was initialized with namespaces
#     :param rank: (int) index of the subprocess
#     :param params: (dict) hyperparameters of agent to be trained
#     :param seed: (int) the inital seed for RNG
#     :param PATHS: (dict) script relevant paths
#     :param train: (bool) to differentiate between train and eval env
#     :param args: (Namespace) program arguments
#     :return: (Callable)
#     """

#     def _init() -> Union[gym.Env, gym.Wrapper]:
#         train_ns = f"sim_{rank+1}" if with_ns else ""
#         eval_ns = f"eval_sim" if with_ns else ""

#         if train:
#             # train env
#             env = FlatlandEnv(
#                 train_ns,
#                 params["reward_fnc"],
#                 params["discrete_action_space"],
#                 goal_radius=params["goal_radius"],
#                 max_steps_per_episode=params["train_max_steps_per_episode"],
#                 debug=args.debug,
#                 task_mode=params["task_mode"],
#                 curr_stage=params["curr_stage"],
#                 PATHS=PATHS,
#             )
#         else:
#             # eval env
#             env = Monitor(
#                 FlatlandEnv(
#                     eval_ns,
#                     params["reward_fnc"],
#                     params["discrete_action_space"],
#                     goal_radius=params["goal_radius"],
#                     max_steps_per_episode=params["eval_max_steps_per_episode"],
#                     train_mode=False,
#                     debug=args.debug,
#                     task_mode=params["task_mode"],
#                     curr_stage=params["curr_stage"],
#                     PATHS=PATHS,
#                 ),
#                 PATHS.get("eval"),
#                 info_keywords=("done_reason", "is_success"),
#             )
#         env.seed(seed + rank)
#         return env

#     set_random_seed(seed)
#     return _init


def wait_for_nodes(
    with_ns: bool, n_envs: int, timeout: int = 30, nodes_per_ns: int = 3
) -> None:
    """
    Checks for timeout seconds if all nodes to corresponding namespace are online.

    :param with_ns: (bool) if the system was initialized with namespaces
    :param n_envs: (int) number of virtual environments
    :param timeout: (int) seconds to wait for each ns
    :param nodes_per_ns: (int) usual number of nodes per ns
    """
    if with_ns:
        assert (
            with_ns and n_envs >= 1
        ), f"Illegal number of environments parsed: {n_envs}"
    else:
        assert (
            not with_ns and n_envs == 1
        ), f"Simulation setup isn't compatible with the given number of envs"

    for i in range(n_envs):
        for k in range(timeout):
            ns = "sim_" + str(i + 1) if with_ns else ""
            namespaces = rosnode.get_node_names(namespace=ns)

            if len(namespaces) >= nodes_per_ns:
                break

            warnings.warn(
                f"Check if all simulation parts of namespace '{ns}' are running properly"
            )
            warnings.warn(f"Trying to connect again..")
            assert (
                k < timeout - 1
            ), f"Timeout while trying to connect to nodes of '{ns}'"

            time.sleep(1)


from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.base_vec_env import VecEnv


def load_vec_normalize(params: dict, PATHS: dict, env: VecEnv, eval_env: VecEnv):
    if params["normalize"]:
        load_path = os.path.join(PATHS["model"], "vec_normalize.pkl")
        if os.path.isfile(load_path):
            env = VecNormalize.load(load_path=load_path, venv=env)
            eval_env = VecNormalize.load(load_path=load_path, venv=eval_env)
            print("Succesfully loaded VecNormalize object from pickle file..")
        else:
            env = VecNormalize(
                env,
                training=True,
                norm_obs=True,
                norm_reward=False,
                clip_reward=15,
            )
            eval_env = VecNormalize(
                eval_env,
                training=True,
                norm_obs=True,
                norm_reward=False,
                clip_reward=15,
            )
    return env, eval_env


def choose_agent_model(AGENT_NAME, PATHS, config, env, params, n_envs):
    if config["resume"] is None:
        agent: Union[
            Type[BaseAgent], Type[ActorCriticPolicy]
        ] = AgentFactory.instantiate(config["architecture_name"])
        if isinstance(agent, BaseAgent):
            model = PPO(
                agent.type.value,
                env,
                policy_kwargs=agent.get_kwargs(),
                gamma=params["gamma"],
                n_steps=params["n_steps"],
                ent_coef=params["ent_coef"],
                learning_rate=params["learning_rate"],
                vf_coef=params["vf_coef"],
                max_grad_norm=params["max_grad_norm"],
                gae_lambda=params["gae_lambda"],
                batch_size=params["m_batch_size"],
                n_epochs=params["n_epochs"],
                clip_range=params["clip_range"],
                tensorboard_log=PATHS.get("tb"),
                verbose=1,
            )
        elif issubclass(agent, ActorCriticPolicy):
            model = PPO(
                agent,
                env,
                gamma=params["gamma"],
                n_steps=params["n_steps"],
                ent_coef=params["ent_coef"],
                learning_rate=params["learning_rate"],
                vf_coef=params["vf_coef"],
                max_grad_norm=params["max_grad_norm"],
                gae_lambda=params["gae_lambda"],
                batch_size=params["m_batch_size"],
                n_epochs=params["n_epochs"],
                clip_range=params["clip_range"],
                tensorboard_log=PATHS.get("tb"),
                verbose=1,
            )
        else:
            architecture_name = config["architecture_name"]
            raise TypeError(
                f"Registered agent class {architecture_name} is neither of type"
                "'BaseAgent' or 'ActorCriticPolicy'!"
            )
    else:
        # load flag
        assert os.path.isfile(
            os.path.join(config["resume"], "best_model.zip")
        ), f"Couldn't find best model in {config['resume']}"
        model = PPO.load(os.path.join(config["resume"], "best_model.zip"), env)
        update_hyperparam_model(model, PATHS, params, n_envs)
    return model
