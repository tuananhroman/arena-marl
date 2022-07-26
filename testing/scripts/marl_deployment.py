import rospy
from testing.scripts.evaluation import evaluate_policy
from training.tools.argsparser import parse_training_args
from training.tools.train_agent_utils import load_config, create_deployment_setup


def main(args):
    rospy.init_node("marl_deployment")

    # load configuration
    config = load_config(args.config)

    # set debug_mode - this mode hinders the creation of several training directories and models.
    rospy.set_param("debug_mode", config["debug_mode"])

    # load robots
    # assert len(config["robots"]) == 1, "Only one robot type is currently supported"

    robots = create_deployment_setup(config)

    # Evaluate the policy for one robot type!
    # Integration of multiple robot types is not yet implemented
    mean_rewards, std_rewards = evaluate_policy(
        robots=robots,
        n_eval_episodes=config["periodic_eval"]["n_eval_episodes"],
        return_episode_rewards=False,
    )

    print("Mean rewards: {}".format(mean_rewards))
    print("Std rewards: {}".format(std_rewards))


if __name__ == "__main__":
    args, _ = parse_training_args()
    main(args)
