"""PPO flavour of the RLlib inner optimizer config."""

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.ppo import PPO

from core.adaptors.ray.optimizer_config import RayOptimizerConfig


class PPOptimizerConfig(RayOptimizerConfig):
    """``RayOptimizerConfig`` whose default RLlib config is ``PPO``'s.

    When to use: a synchronous inner learner, simpler to reason about than
    ``APPOptimizerConfig`` when debugging sample flow at the cost of idle
    environments during the learner update.
    """

    algo_class: Algorithm = PPO
