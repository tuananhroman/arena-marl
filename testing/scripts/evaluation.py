from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import supersuit.vector.sb3_vector_wrapper as sb3vw
from stable_baselines3.common import base_class


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
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.env_util import is_wrapped
    from stable_baselines3.common.monitor import Monitor

    obs = {robot: {} for robot in robots}

    not_reseted = True
    # Avoid double reset, as VecEnv are reset automatically.
    if not_reseted:
        for robot in robots:
            # ('robot': {'agent1': obs, 'agent2': obs, ...})
            obs[robot] = robots[robot]["env"].reset()
            not_reseted = False

    # {'robot': [agents]} e.g. {'jackal': [agent1, agent2], 'burger': [agent1]}
    agents = {robot: robots[robot]["agent_dict"] for robot in robots}
    # {'robot': [robot_ns]} e.g. {'jackal': [robot1, robot2], 'burger': [robot1]}
    agent_names = {
        robot: [a_name._robot_sim_ns for a_name in agents[robot]["sim_1"]]
        for robot in robots
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
    default_dones = {robot: {a: None for a in agent_names[robot]} for robot in robots}
    default_states = {robot: None for robot in robots}
    default_actions = {robot: None for robot in robots}
    # states, actions, episode rewards -> {'robot': {'robot_ns': None}}
    # e.g. {'jackal': {robot1: None, robot2: None}, 'burger': {'robot1': None}}
    default_episode_reward = {
        robot: {a: None for a in agent_names[robot]} for robot in robots
    }
    while len(episode_lengths) < n_eval_episodes:
        # Number of loops here might differ from true episodes
        # played, if underlying wrappers modify episode lengths.
        # Avoid double reset, as VecEnv are reset automatically.
        if not_reseted:
            for robot in robots:
                # ('robot': array[[obs1], [obs2], ...])
                obs[robot] = robots[robot]["env"].reset()
                not_reseted = False

        dones = default_dones.copy()
        states = default_states.copy()
        actions = default_actions.copy()
        episode_reward = default_episode_reward.copy()
        episode_length = 0
        while not check_dones(dones):
            # Go over all robots
            for robot in robots:
                # Get predicted actions and new states from each agent
                actions[robot], states[robot] = robots[robot]["model"].predict(
                    obs[robot], states[robot], deterministic=deterministic
                )

                # Publish actions in the environment
                env = robots[robot]["env"]
                env.apply_action(actions[robot])

                # shouldn't we here make the takeSimStep() call?

                # Take one step in the simulation (applies to all robots and all agents, respectively!)
                # todo: outsource call_service_takeSimStep from env, since it is not env-specific
                # robots[robot]["env"].call_service_takeSimStep()

                # and then continue with a new for robots in robots loop to collect the obs and rewards?
                # for robot in robots:
                # And get new obs, rewards, dones, and infos
                obs[robot], rewards, dones[robot], _ = robots[robot]["env"].get_states()
                # Add up rewards for this episode
                for agent, reward in zip(agent_names[robot], rewards.values()):
                    episode_reward[robot][agent] += reward

                if callback is not None:
                    callback(locals(), globals())

                if render:
                    robots[robot]["env"].render()

            # Take one step in the simulation (applies to all robots and all agents, respectively!)
            # todo: outsource call_service_takeSimStep from env, since it is not env-specific
            robots[robot]["env"].call_service_takeSimStep()

            episode_length += 1

        # if is_monitor_wrapped:
        #     # Do not trust "done" with episode endings.
        #     # Remove vecenv stacking (if any)
        #     # if isinstance(env, VecEnv):
        #     #     info = info[0]
        #     if "episode" in info.keys():
        #         # Monitor wrapper includes "episode" key in info if environment
        #         # has been wrapped with it. Use those rewards instead.
        #         episode_rewards.append(info["episode"]["r"])
        #         episode_lengths.append(info["episode"]["l"])
        # else:

        # For each robot append rewards for every of its agents
        # to the list of respective episode rewards
        for robot in robots:
            for agent, reward in episode_reward[robot].items():
                episode_rewards[robot][agent].append(reward)

        episode_lengths.append(episode_length)

    mean_rewards = {
        robot: {
            agent: np.mean(episode_rewards[robot][agent])
            for agent in agent_names[robot]
        }
        for robot in robots
    }

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

    # if reward_threshold is not None:
    #     assert min(mean_rewards.values()) > reward_threshold, (
    #         "Atleast one mean reward below threshold: "
    #         f"{min(mean_rewards.values()):.2f} < {reward_threshold:.2f}"
    #     )
    # if return_episode_rewards:
    #     return episode_rewards, episode_lengths
    # return mean_rewards, std_rewards


def check_dones(dones):
    # Check if all agents for every robot are done
    for robot in dones:
        for agent in dones[robot]:
            if dones[robot][agent]:
                continue
            else:
                return False
    return True
