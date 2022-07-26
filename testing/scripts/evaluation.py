from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import copy

from rl_utils.rl_utils.utils.utils import call_service_takeSimStep


def evaluate_policy(
    robots: Dict[str, Dict[str, Any]],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    This is made to work only with one env.

    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.
    """
    # is_monitor_wrapped = False
    # Avoid circular import
    # from stable_baselines3.common.env_util import is_wrapped
    # from stable_baselines3.common.monitor import Monitor

    obs = {robot: {} for robot in robots}

    not_reseted = True
    # Avoid double reset, as VecEnv are reset automatically.
    if not_reseted:
        # perform one step in the simulation to update the scene
        call_service_takeSimStep(ns="eval_sim")
        for robot in robots:
            # ('robot': {'agent1': obs, 'agent2': obs, ...})
            obs[robot] = robots[robot]["env"].reset()
            not_reseted = False

    # {'robot': [agents]} e.g. {'jackal': [agent1, agent2], 'burger': [agent1]}
    agents = {robot: robots[robot]["agent_dict"]["eval_sim"] for robot in robots}
    # {'robot': [robot_ns]} e.g. {'jackal': [robot1, robot2], 'burger': [robot1]}
    agent_names = {
        robot: [a_name._robot_sim_ns for a_name in agents[robot]] for robot in robots
    }

    # {
    #   'jackal':
    #       'robot1': [reward1, reward2, ...]
    #       'robot2': [reward1, reward2, ...]
    #   'burger':
    #       'robot1': [reward1, reward2, ...]
    #       'robot2': [reward1, reward2, ...]
    # }
    episode_rewards = {
        robot: {agent: [] for agent in agent_names[robot]} for robot in robots
    }
    episode_lengths = []

    # dones -> {'robot': {'robot_ns': False}}
    # e.g. {'jackal': {robot1: False, robot2: False}, 'burger': {'robot1': False}}
    default_dones = {robot: {a: False for a in agent_names[robot]} for robot in robots}
    default_actions = {robot: {a: [] for a in agent_names[robot]} for robot in robots}
    # states, actions, episode rewards -> {'robot': {'robot_ns': None}}
    # e.g. {'jackal': {robot1: None, robot2: None}, 'burger': {'robot1': None}}
    default_episode_reward = {
        robot: {a: 0 for a in agent_names[robot]} for robot in robots
    }
    while len(episode_lengths) < n_eval_episodes:
        # Number of loops here might differ from true episodes
        # played, if underlying wrappers modify episode lengths.

        # Avoid double reset, as VecEnv are reset automatically.
        if not_reseted:
            # perform one step in the simulation to update the scene
            call_service_takeSimStep(ns="eval_sim")
            for robot in robots:
                # ('robot': array[[obs1], [obs2], ...])
                obs[robot] = robots[robot]["env"].reset()
                not_reseted = False

        dones = copy.deepcopy(default_dones)
        actions = copy.deepcopy(default_actions)
        episode_reward = copy.deepcopy(default_episode_reward)
        episode_length = 0
        while not check_dones(dones):
            ### Get predicted actions and publish those
            for robot in robots:
                # Get predicted actions from each agent
                concat_obs = [obs[robot][agent] for agent in agent_names[robot]]
                pred_actions = robots[robot]["model"].predict(
                    concat_obs,
                    deterministic=deterministic,
                )[0]

                for i, agent in enumerate(agent_names[robot]):
                    actions[robot][agent] = pred_actions[i]

                # Publish actions in the environment
                env = robots[robot]["env"]
                env.apply_action(actions[robot])

            ### Make a step in the simulation
            # This moves all agents in the simulation and transfers them into the next state
            call_service_takeSimStep(ns="eval_sim")

            ### Get new obs, rewards, dones, and infos for all robots
            for robot in robots:
                obs[robot], rewards, dones[robot], _ = robots[robot]["env"].get_states()
                # Add up rewards for this episode
                for agent, reward in zip(agent_names[robot], rewards.values()):
                    episode_reward[robot][agent] += reward

                if callback is not None:
                    callback(locals(), globals())

                if render:
                    robots[robot]["env"].render()

            ### Document how long this episode was
            episode_length += 1

        ### Collect all episode rewards
        # For each robot append rewards for every of its agents
        # to the list of respective episode rewards
        for robot in robots:
            for agent, reward in episode_reward[robot].items():
                episode_rewards[robot][agent].append(reward)

        ### Document all episode lengths
        episode_lengths.append(episode_length)

        ### Set to reset the environment after each episode
        not_reseted = True

    ### Calculate average rewards
    mean_rewards = {
        robot: {
            agent: np.mean(episode_rewards[robot][agent])
            for agent in agent_names[robot]
        }
        for robot in robots
    }

    ### Calculate standard deviation of rewards
    std_rewards = {
        robot: {
            agent: np.std(episode_rewards[robot][agent]) for agent in agent_names[robot]
        }
        for robot in robots
    }

    if reward_threshold is not None:
        for robot in robots:
            for agent in agent_names[robot]:
                if mean_rewards[robot][agent] < reward_threshold:
                    raise ValueError(
                        "Mean reward for agent {} of robot {} is below threshold! {} < {}".format(
                            agent, robot, mean_rewards[robot][agent], reward_threshold
                        )
                    )

    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_rewards, std_rewards


def check_dones(dones):
    # Check if all agents for every robot are done
    for robot in dones:
        for agent in dones[robot]:
            if dones[robot][agent]:
                continue
            else:
                return False
    return True
