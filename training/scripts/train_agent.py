import sys

# import time
from datetime import time
from multiprocessing import set_start_method

import rospy
from rl_utils.rl_utils.utils.wandb_helper import WandbLogger


from rl_utils.rl_utils.utils.heterogenous_ppo import Heterogenous_PPO


from testing.scripts.evaluation import create_eval_callback

from training.tools.argsparser import parse_training_args
from training.tools.custom_mlp_utils import *

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
    ### load configuration
    config = load_config(args.config)
    robot_names = list(config["robots"].keys())
    if config["wandb"]:
        wandb_logger = WandbLogger("arena-marl")
    else:
        wandb_logger = None

    # set debug_mode
    rospy.set_param("debug_mode", config["debug_mode"])

    ### create dict for all robot types
    # those contain all necessary parameters, and instances of the respective models and envs
    robots = create_training_setup(config, wandb_logger)

    ### Create eval callback where also the trained models will be saved
    callback = create_eval_callback(config, robots, wandb_logger=wandb_logger)

    ### set num of timesteps to be generated
    n_timesteps = 40000000 if config["n_timesteps"] is None else config["n_timesteps"]

    agent_ppo_dict = {agent: robots[agent]["model"] for agent in robot_names}
    agent_env_dict = {agent: robots[agent]["env"] for agent in robot_names}

    het_ppo = Heterogenous_PPO(
        agent_ppo_dict, agent_env_dict, config["n_envs"], wandb_logger
    )

    start = time.time()
    try:
        het_ppo.learn(
            total_timesteps=n_timesteps,
            callback=callback,
            log_interval=config["log_interval"],
        )
    except KeyboardInterrupt:
        print("KeyboardInterrupt..")

    print(f"Time passed: {time.time() - start}s")
    print("Training script will be terminated")
    sys.exit()


if __name__ == "__main__":
    set_start_method("fork")
    args, _ = parse_training_args()
    main(args)
