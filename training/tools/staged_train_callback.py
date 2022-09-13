import json
import os
import warnings
import rospy
import numpy as np
import time
from filelock import FileLock

from typing import List
from std_msgs.msg import Bool
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    MarlEvalCallback,
)


class InitiateNewTrainStage(BaseCallback):
    """
    Introduces new training stage when threshhold reached.
    It must be used with "EvalCallback".

    :param treshhold_type (str): checks threshhold for either percentage of successful episodes (succ) or mean reward (rew)
    :param rew_threshold (int): mean reward threshold to trigger new stage
    :param succ_rate_threshold (float): threshold percentage of succesful episodes to trigger new stage
    :param task_mode (str): training task mode, if not 'staged' callback won't be called
    :param verbose:
    """

    def __init__(
        self,
        n_envs: int = 1,
        treshhold_type: str = "succ",
        upper_threshold: float = 0,
        lower_threshold: float = 0,
        task_mode: str = "staged",
        model_paths: List[str] = None,
        verbose=0,
    ):

        super(InitiateNewTrainStage, self).__init__(verbose=verbose)
        self.n_envs = n_envs
        self.threshhold_type = treshhold_type
        self.model_paths = model_paths

        # hyperparamters.json location
        self.json_files = {
            robot: os.path.join(path, "hyperparameters.json")
            for robot, path in self.model_paths.items()
        }
        if not rospy.get_param("debug_mode"):
            for _, path in self.json_files.items():
                assert os.path.isfile(
                    path
                ), f"Could not find hyperparameters.json file at {path}"

        self._lock_json = {
            robot: FileLock(f"{json_file}.lock")
            for robot, json_file in self.json_files.items()
        }

        self._trigger = Bool()
        self._trigger.data = True

        self.activated = False
        if task_mode == "staged":
            self.activated = True
            self._instantiate_publishers()

        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold

        assert self.threshhold_type in {
            "rew",
            "succ",
        }, "given theshhold type neither 'rew' or 'succ'"

        # default values
        if self.threshhold_type == "rew" and upper_threshold == 0:
            self.upper_threshold = 13
            self.lower_threshold = 7
        elif self.threshhold_type == "succ" and upper_threshold == 0:
            self.upper_threshold = 0.85
            self.lower_threshold = 0.6
        else:
            self.upper_threshold = upper_threshold
            self.lower_threshold = lower_threshold

        assert (
            self.upper_threshold > self.lower_threshold
        ), "upper threshold has to be bigger than lower threshold"
        assert (
            self.upper_threshold >= 0 and self.lower_threshold >= 0
        ), "upper/lower threshold have to be positive numbers"
        if self.threshhold_type == "succ":
            assert (
                self.upper_threshold <= 1 and self.lower_threshold >= 0
            ), "succ thresholds have to be between [1.0, 0.0]"

        self.verbose = verbose
        self.activated = bool(task_mode == "staged")

        if self.activated:
            rospy.set_param("/last_stage_reached", False)
            self._instantiate_publishers()

            self._trigger = Bool()
            self._trigger.data = True

    def _instantiate_publishers(self):
        self._publishers_next = []
        self._publishers_previous = []

        self._publishers_next.append(
            rospy.Publisher(f"/eval_sim/next_stage", Bool, queue_size=1)
        )
        self._publishers_previous.append(
            rospy.Publisher(f"/eval_sim/previous_stage", Bool, queue_size=1)
        )

        for env_num in range(self.n_envs):
            self._publishers_next.append(
                rospy.Publisher(f"/sim_{env_num+1}/next_stage", Bool, queue_size=1)
            )
            self._publishers_previous.append(
                rospy.Publisher(f"/sim_{env_num+1}/previous_stage", Bool, queue_size=1)
            )

    def _on_step(self, EvalObject: EvalCallback) -> bool:
        assert isinstance(EvalObject, EvalCallback) or isinstance(
            EvalObject, MarlEvalCallback
        ), f"InitiateNewTrainStage must be called within EvalCallback"

        if self.activated:
            if EvalObject.n_eval_episodes < 20:
                warnings.warn(
                    "Only %d evaluation episodes considered for threshold monitoring,"
                    "results might not represent agent performance well"
                    % EvalObject.n_eval_episodes
                )

            if self._check_lower_threshold(EvalObject):
                if self.verbose:
                    print("Dropped below lower threshold, going to previous stage.")
                for i, pub in enumerate(self._publishers_previous):
                    pub.publish(self._trigger)
                    if i == 0:
                        self.log_curr_stage(EvalObject.logger)
                        if not rospy.get_param("debug_mode"):
                            for robot, lock in self._lock_json.items():
                                with lock:
                                    self._update_curr_stage_json(robot)

            if self._check_upper_threshold(EvalObject):
                if not rospy.get_param("/last_stage_reached"):
                    EvalObject.best_mean_rewards = {
                        robot: -np.inf for robot in EvalObject.robots
                    }
                    EvalObject.last_success_rates = {
                        robot: -np.inf for robot in EvalObject.robots
                    }
                    if self.verbose:
                        print(
                            "Performance above upper threshold! Entering next training stage."
                        )
                    for i, pub in enumerate(self._publishers_next):
                        pub.publish(self._trigger)
                        if i == 0:
                            self.log_curr_stage(EvalObject.logger)
                            if not rospy.get_param("debug_mode"):
                                for robot, lock in self._lock_json.items():
                                    with lock:
                                        self._update_curr_stage_json(robot)

    def _check_lower_threshold(self, EvalObject: EvalCallback) -> bool:
        ### if all values (all robots) are smaller than lower threshold return true
        if (
            self.threshhold_type == "rew"
            and all(
                np.asarray([rew for rew in EvalObject.best_mean_rewards.values()])
                <= self.lower_threshold
            )
        ) or (
            self.threshhold_type == "succ"
            and all(
                np.asarray([sr for sr in EvalObject.last_success_rates.values()])
                <= self.lower_threshold
            )
        ):
            return True
        else:
            return False

    def _check_upper_threshold(self, EvalObject: EvalCallback) -> bool:
        ### if all values (all robots) are higher than upper threshold return true
        if (
            self.threshhold_type == "rew"
            and all(
                np.asarray([rew for rew in EvalObject.best_mean_rewards.values()])
                >= self.upper_threshold
            )
        ) or (
            self.threshhold_type == "succ"
            and all(
                np.asarray([sr for sr in EvalObject.last_success_rates.values()])
                >= self.upper_threshold
            )
        ):
            return True
        else:
            return False

    def log_curr_stage(self, logger):
        time.sleep(1)
        curr_stage = rospy.get_param("/curr_stage", -1)
        if logger is not None:
            logger.record("train_stage/stage_idx", curr_stage)

    def _update_curr_stage_json(self, robot):
        curr_stage = rospy.get_param("/curr_stage", -1)
        with open(self.json_files[robot], "r") as file:
            hyperparams = json.load(file)
        try:
            hyperparams["curr_stage"] = curr_stage
        except Exception as e:
            raise Warning(
                f" {e} \n Parameter 'curr_stage' not found in 'hyperparameters.json' of {robot}!"
            )
        else:
            with open(self.json_files[robot], "w", encoding="utf-8") as target:
                json.dump(hyperparams, target, ensure_ascii=False, indent=4)
