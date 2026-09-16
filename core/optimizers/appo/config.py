"""APPO (asynchronous PPO) flavour of the RLlib inner optimizer config."""

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.appo import APPO

from core.adaptors.ray.optimizer_config import RayOptimizerConfig


class APPOptimizerConfig(RayOptimizerConfig):
    """``RayOptimizerConfig`` whose default RLlib config is ``APPO``'s.

    When to use: the default inner learner of the bilevel fishery example;
    its asynchronous sampling keeps every regulated environment busy while
    the learner updates. Use ``PPOptimizerConfig`` for synchronous PPO.
    """

    algo_class: Algorithm = APPO
