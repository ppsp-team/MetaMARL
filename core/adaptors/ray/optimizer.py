"""Inner-loop optimizer backed by an RLlib ``Algorithm`` running in Ray.

``RayOptimizer`` is the ``Optimizer`` node the regulator drives: ``run`` is one
RLlib training iteration, ``evaluate`` one fixed-duration evaluation pass and
``reset`` a rebuild of the policy from its initial weights. The algorithm itself
lives in a ``PolicyActor``; this class only forwards calls and keeps light
bookkeeping (inner iteration counter, per-iteration return and loss) for logs.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np
import ray
from ray.rllib.utils.typing import AgentID, ResultDict
from ray.train._internal.checkpoint_manager import _TrainingResult

from core.adaptors.ray.schema import EvalSchema, RaySchema, TrainSchema

# TODO temporary
from core.adaptors.ray.utils import (
    build_learner,
    build_performance,
    build_rollout,
    get_env_steps,
    get_episode_return_mean,
    get_policy_loss_if_present,
)
from core.annotations import override
from core.metrics.logger import MetricLogger
from core.optimizers.base import Optimizer

# Deprecated
from core.utils import to_float

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from core.adaptors.ray.optimizer_config import RayOptimizerConfig


# TODO perhaps we would first want an adaptor for core ray algorithm and then the PPO inherits it


class RayOptimizer(Optimizer):
    """Optimizer wrapping an RLlib algorithm through a ``PolicyActor``.

    Parameters
    ----------
    config : RayOptimizerConfig
        Frozen configuration whose ``rllib_cfg`` is fully resolved (this is
        what ``RayOptimizerConfig.build_optimizer`` passes). A ``PolicyActor``
        is spawned from it immediately.

    Attributes
    ----------
    policy_actor : ActorHandle[PolicyActor]
        Remote owner of the ``Algorithm``.
    eval_episodes : int
        ``evaluation_duration // evaluation_config["rollout_fragment_length"]``,
        an estimate of episodes per evaluation. Raises ``TypeError`` at
        construction when the evaluation config has no
        ``rollout_fragment_length``.
    _inner_iter : int
        Training iterations since the last ``reset``.
    _es_round : int
        Number of ``reset`` calls so far, i.e. the outer (ES) generation.
    _training_rewards, _training_losses : list of float
        Per-iteration mean return and policy loss since the last ``reset``.
    """

    def __init__(
        self,
        # algo: Algorithm,
        config: RayOptimizerConfig,
    ):
        super().__init__(config)

        # self.algo = algo

        # TODO maybe this either needs to be an actor. or atleast have method to serialize data
        self.logger = MetricLogger.from_schema(RaySchema)

        # self.eval_episodes = config.eval_episodes
        # TODO fallback if rollout_fragment_length not in eval_cfg
        self.eval_episodes = (
            config.rllib_cfg.evaluation_duration
            // config.rllib_cfg.evaluation_config.get("rollout_fragment_length")
        )

        # self.rollout_fragment_length = config.rollout_fragment_length

        from core.adaptors.ray.policy_actor import PolicyActor

        self.policy_actor = PolicyActor.remote(config.rllib_cfg)

        # Track training metrics for plotting
        self._training_rewards: list[float] = []
        self._training_losses: list[float] = []
        self._inner_iter: int = 0
        self._es_round: int = 0

    @property
    @override(Optimizer)
    def batch_capacity(self) -> int:
        """Number of mechanism candidates evaluated in parallel per iteration.

        ``RayOptimizerConfig.debugging`` multiplies
        ``num_envs_per_env_runner`` by the number of training seeds, so the
        original count of mechanisms is recovered as
        ``num_envs_per_env_runner // len(seeds)``. The outer ES optimizer
        reads this to size its population.

        Returns
        -------
        int
            Mechanisms per env runner.

        Raises
        ------
        ZeroDivisionError
            If the config has no training seeds (``debugging`` not called or
            called without a seed).
        """

        num_envs = self.config.rllib_cfg.num_envs_per_env_runner
        num_seeds = len(self.config.seeds)
        num_mechanisms = num_envs // num_seeds

        return num_mechanisms

    # TODO move to utils
    def _build_agent_policy_map(self) -> dict[AgentID, str]:
        """Map ``"<agent_type>:<i>"`` agent IDs to their base policy name.

        Currently unused. The mapping ignores the per-mechanism, per-seed
        suffix (``_m<idx>_s<seed>``) that ``_apply_agents_to_rllib`` appends
        to the real RLModule IDs.

        Returns
        -------
        dict[AgentID, str]
            One entry per agent instance declared in ``config.agent_specs``.
        """

        agent_to_policy: dict[AgentID, str] = {}

        for agent_type, spec in self.config.agent_specs.items():
            policy_id = spec["policy"]
            count = spec["count"]

            for i in range(count):
                agent_id = f"{agent_type}:{i}"
                agent_to_policy[agent_id] = policy_id

        return agent_to_policy

    def _get_policy_handle(self, policy_id: str):
        """Return the RLModule (new stack) or Policy (old stack) for an ID.

        Dead code in the current design: it references ``self.algo``, which
        ``RayOptimizer`` never defines because the ``Algorithm`` lives inside
        ``PolicyActor``. Calling it raises ``AttributeError`` from the fallback
        branch.
        """

        # RLModule API (newer)
        try:
            return self.algo.get_module(policy_id)
        except Exception:
            # Policy API (older / classic)
            return self.algo.get_policy(policy_id)

    # TODO (nadine) : in the future this could be separated into a different class if justified
    def _to_logger_payload(
        self, result: ResultDict, is_eval: bool = False
    ) -> RaySchema:
        if is_eval:
            evaluation = EvalSchema(
                rollout=build_rollout(result),
                performance=build_performance(result),
            )

            return RaySchema(train=None, eval=evaluation)

        train = TrainSchema(
            rollout=build_rollout(result),
            learner=build_learner(result),
            performance=build_performance(result),
        )
        eval_result = result.get("evaluation")
        evaluation = None

        if isinstance(eval_result, dict):
            evaluation = EvalSchema(
                rollout=build_rollout(eval_result),
                performance=build_performance(eval_result),
            )

        return RaySchema(train=train, eval=evaluation)

    @override(Optimizer)
    def run(self) -> None:
        """Run one RLlib training iteration on the policy actor.

        Increments the inner iteration counter, extracts the mean episode
        return, the env step counters and (when present) the policy loss from
        the result, appends them to the tracking lists and logs one summary
        line. RLlib's own lifetime ``training_iteration`` is logged for
        reference only; the per-generation counter is ``_inner_iter``.
        Reporting to W&B is currently disabled (commented out).
        """

        logger.info("[PPO] Training step started")

        result = ray.get(self.policy_actor.train.remote())

        # step = int(to_float(result.get("training_iteration")) or 0)

        # Local inner-loop iteration. This resets for each outer ES round.
        self._inner_iter += 1
        step = self._inner_iter

        self.logger.push(key=("iter",), value=step)

        # RLlib's own lifetime training counter, retained only for debugging.
        rllib_training_iteration = int(to_float(result.get("training_iteration")) or 0)
        metrics = self._to_logger_payload(result)

        self.logger.push_data(metrics)

        # TODO temporary to be moved to a logger Extract metrics
        ep_return = get_episode_return_mean(result)
        steps_iter, steps_life = get_env_steps(result)

        # Track metrics
        self._training_rewards.append(ep_return)

        policy_loss = get_policy_loss_if_present(result)

        self._training_losses.append(policy_loss)
        logger.info(
            "[PPO] Training step completed | "
            "outer_iter=%d | inner_iter=%d | rllib_iter_lifetime=%d | "
            "ep_return=%.4f | env_steps_iter=%d | "
            "env_steps_lifetime=%d | policy_loss=%s",
            self._es_round,
            step,
            rllib_training_iteration,
            ep_return,
            steps_iter,
            steps_life,
            f"{policy_loss:.6f}" if np.isfinite(policy_loss) else "NA",
        )

    @override(Optimizer)
    def evaluate(self) -> None:
        """Run one evaluation pass on the policy actor and log start/end.

        The evaluation results themselves are not returned; the environments
        publish their ``EnvStepContext`` records to the World, which is where
        the regulator reads the outcome.
        """

        logger.info("[PPO] Evaluation started")

        result = ray.get(self.policy_actor.evaluate.remote())
        metrics = self._to_logger_payload(result, is_eval=True)

        self.logger.push_data(metrics)
        logger.info("[PPO] Evaluation completed")

    @override(Optimizer)
    def reset(self) -> None:
        """Reset policy weights to random initialization."""

        logger.info("[PPO] Resetting policy weights")

        self._training_rewards = []
        self._training_losses = []
        self._inner_iter = 0
        self._es_round += 1

        ray.get(self.policy_actor.reset.remote())
        self.logger.reset()

    @override(Optimizer)
    def stop(self) -> None:
        """Stop the RLlib ``Algorithm`` held by the policy actor."""

        ray.get(self.policy_actor.stop.remote())

        return self.logger.reduce(complie=True)

    @override(Optimizer)
    def save(self, checkpoint_dir: Optional[str] = None) -> _TrainingResult:
        """Checkpointing stub: does nothing and returns ``None``.

        The ``_TrainingResult`` return annotation describes the intended
        contract, not the current behaviour.
        """

        # TODO
        pass
