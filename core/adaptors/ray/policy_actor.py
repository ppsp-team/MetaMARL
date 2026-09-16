"""Ray actor that owns the RLlib ``Algorithm`` of the inner optimizer.

``RayOptimizer`` never holds an ``Algorithm`` itself; it drives this actor
through ``.remote()`` calls. Building the algorithm inside the actor keeps the
learner, its env runners and their evaluation counterparts in one process tree
and lets the outer loop rebuild the policy between ES generations without
touching the driver.
"""

from __future__ import annotations

import logging

from core.metrics.schemas import MetricSchema

import numpy as np
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.utils.typing import ResultDict

from core.adaptors.ray.utils import hash_weights

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=1)
class PolicyActor:
    """Own the RLlib ``Algorithm`` of one inner optimizer.

    The ``Algorithm`` is built from ``algo_config`` inside the actor and never
    leaves it. The weights right after construction are kept in
    ``_init_weights`` so that ``reset`` can restore the same starting point
    at each outer iteration.

    Parameters
    ----------
    algo_config : AlgorithmConfig
        Fully resolved RLlib configuration (environment registered, RLModule
        specs and policy mapping applied). Kept so that ``reset`` can rebuild.
    """

    def __init__(self, algo_config: AlgorithmConfig):
        # Store config for reset capability
        self.algo_config = algo_config

        # Build Algorithm INSIDE actor (critical)
        self.algo: Algorithm = algo_config.build_algo()

        # Store initial weights
        self._init_weights = self.algo.get_weights()

    def train(self) -> ResultDict:
        """Run one ``Algorithm.train()`` iteration.

        Returns
        -------
        ResultDict
            The raw RLlib result dictionary of the iteration.
        """

        # TODO config ability to debug remote actors
        # import debugpy, os
        # debugpy.listen(("127.0.0.1", 5678))
        # print(f"[debugpy] worker pid={os.getpid()} listening on 5678")
        # debugpy.wait_for_client()
        # debugpy.breakpoint()
        # TODO mapping result
        return self.algo.train()

    def evaluate(self) -> ResultDict:
        """Run one ``Algorithm.evaluate()`` pass on the evaluation env runners.

        With the configuration produced by ``RayOptimizerConfig.evaluation``,
        this executes ``_evaluate_with_fixed_duration_once``: exactly one
        episode per ``(mechanism, policy seed, eval seed)`` environment.

        Returns
        -------
        ResultDict
            RLlib evaluation results.
        """

        return self.algo.evaluate()

    def compute_actions(
        self,
        policy_id: str,
        obs_batch: np.ndarray,
    ) -> np.ndarray:
        """Sample actions for a batch of observations from one RLModule.

        Parameters
        ----------
        policy_id : str
            RLModule ID (``<policy>_m<mechanism_idx>_s<seed>``).
        obs_batch : numpy.ndarray
            Observations, shape ``(B, obs_dim)``.

        Returns
        -------
        numpy.ndarray
            Actions, shape ``(B, act_dim)``. The new API stack path samples
            from the inference distribution; if it fails, the old-stack
            ``Policy.compute_single_action`` fallback is used without
            exploration.
        """

        try:
            module = self.algo.get_module(policy_id)
            out = module.forward_inference({"obs": obs_batch})
            dist_cls = module.get_inference_action_dist_cls()
            dist = dist_cls.from_logits(out["action_dist_inputs"])

            return dist.sample().cpu().numpy()
        except Exception:
            policy = self.algo.get_policy(policy_id)
            actions = []

            for obs in obs_batch:
                a, _, _ = policy.compute_single_action(obs, explore=False)

                actions.append(a)

            return np.asarray(actions)

    def reset(self) -> None:
        """Rebuild the ``Algorithm`` and restore the initial weights.

        A brand-new ``Algorithm`` is built from the stored config, then the
        weights captured at actor construction are loaded into it, so every
        outer iteration starts the inner policy from the same parameters. The
        hash of the restored weights is logged for cross-run comparison.

        Notes
        -----
        The previous ``Algorithm`` is replaced without calling ``stop()`` on
        it, so its env runners and learner threads are left to garbage
        collection.
        """

        # TODO verify reset is using the same seed
        # self.algo.set_weights(self._init_weights)
        self.algo = self.algo_config.build_algo()

        # TODO tie the init_weights with seeding
        self.algo.set_weights(self._init_weights)

        weights = self.algo.get_weights()

        logger.info(
            "[PPO] Initial policy weight hash: %s",
            hash_weights(weights),
        )

    def stop(self) -> None:
        """Stop the owned ``Algorithm`` and release its workers."""

        self.algo.stop()
