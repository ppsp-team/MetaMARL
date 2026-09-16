"""Regulator environment of the fishery example: scores mechanism candidates.

``FisheryRegulatorEnv`` is the environment the ES outer optimizer steps. A
step publishes the population of candidate mechanisms, runs the inner
optimizer (handled by the ``RegulatorEnv`` base class) and, in
:meth:`FisheryRegulatorEnv.aggregate_rewards`, turns the inner rollouts into
one fitness per candidate through a ``FitnessContext``.
"""

import logging
from collections import defaultdict
from typing import Any

import numpy as np
from gymnasium.core import ObsType

from core.annotations import override
from core.envs.regulator import RegulatorEnv
from core.metrics.schemas import MetricSchema
from core.world.context import (
    MechanismContext,
    MechanismStatus,
)
from examples.bilevel_fishery.contexts import FitnessContext

logger = logging.getLogger(__name__)


class FisheryRegulatorEnv(RegulatorEnv):
    """Outer-loop environment that scores fishery mechanism candidates.

    The fitness of a candidate is computed on the last ``fitness_tail_steps``
    steps of each inner episode (tail averaging), on the ``train`` or ``eval``
    split selected by ``aggregation_status``.

    Parameters
    ----------
    ecology_cfg : dict
        ``sustainability_weight`` (weight of the mean normalized biomass in
        the objective, default 5.0), ``sustainability_threshold`` (normalized
        biomass below which a step counts as collapsed, default 0.1), ``K``
        (carrying capacity, biomass units, used to denormalize the threshold
        for plots), ``aggregation_status`` (``"train"`` or ``"eval"``, default
        ``"eval"``) and ``fitness_tail_steps`` (default 50).
    """

    def __init__(
        self,
        *,
        ecology_cfg: dict[str, Any],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.sustainability_weight = ecology_cfg.get("sustainability_weight", 5.0)
        self.sustainability_threshold = ecology_cfg.get("sustainability_threshold", 0.1)
        self.K = ecology_cfg.get("K")

        # Denormalized threshold for visualization
        self.raw_sustainability_threshold = self.sustainability_threshold * self.K
        self.trajectories: dict[int, list[dict[str, Any]]] = {}
        self.last_metrics: list[dict[str, float]] = []
        target_status = ecology_cfg.get("aggregation_status", "eval")
        self.aggregation_status = MechanismStatus(target_status)

        # tail averaging
        self.fitness_tail_steps = int(ecology_cfg.get("fitness_tail_steps", 50))

    @override(RegulatorEnv)
    def observation(self, obs: ObsType) -> ObsType:
        """Return a constant ``0.0``: the ES outer loop is stateless and ignores observations."""

        return 0.0

    @override(RegulatorEnv)
    def aggregate_rewards(self, metrics: MetricSchema) -> list[float]:
        """Compute one fitness per candidate from the inner optimizer's metrics.

        ``metrics`` is the inner ``RaySchema`` peeked after training; the
        ``aggregation_status`` split (``train`` or ``eval``) is read, then the
        rollouts are walked by mechanism, seed and episode. Each episode
        contributes its tail-averaged reward, biomass and harvest statistics;
        seeds are averaged per mechanism and folded into a ``FitnessContext``.
        """

        # TODO move num_steps here
        # num_steps = getattr(metrics, "iter")
        metrics = getattr(metrics, self.aggregation_status.value)

        per_mech_metrics: list[dict[str, float]] = []

        # TODO when running parallel eval, async may duplicate runs ! should not statistically change the result
        metrics_by_mechanism: dict[int, list[dict[str, Any]]] = defaultdict(list)

        self.trajectories = {}

        # TODO ensure aggregation by policy seed
        for mechanism_id, mechanism_metrics in metrics.rollout.by_mechanism.items():
            idx = int(mechanism_id)

            for seed_id, seed_metrics in mechanism_metrics.by_seed.items():
                seed = int(seed_id)

                for episode_metrics in seed_metrics.by_episode.values():
                    # TODO this is the mean however this is not good representation for late learning mechanisms
                    rewards = np.atleast_1d(
                        np.asarray(episode_metrics.reward_mean, dtype=np.float32)
                    )
                    fish = np.atleast_1d(
                        np.asarray(
                            episode_metrics.fish_norm_next_mean, dtype=np.float32
                        )
                    )
                    realized_harvest = np.atleast_1d(
                        np.asarray(episode_metrics.H_realized, dtype=np.float32)
                    )
                    msy = np.atleast_1d(
                        np.asarray(episode_metrics.MSY, dtype=np.float32)
                    )
                    harvest_scores = realized_harvest / np.maximum(msy, 1e-6)
                    sustainability_penalties = np.maximum(
                        0.0,
                        (self.sustainability_threshold - fish)
                        / max(1e-6, self.sustainability_threshold),
                    )
                    num_steps = len(fish)
                    tail_steps = min(self.fitness_tail_steps, num_steps)
                    tail_start = num_steps - tail_steps
                    tail_rewards = rewards[tail_start:]
                    tail_fish = fish[tail_start:]
                    tail_realized_harvest = realized_harvest[tail_start:]
                    tail_harvest_scores = harvest_scores[tail_start:]
                    sustainability_penalties = np.maximum(
                        0.0,
                        (self.sustainability_threshold - tail_fish)
                        / max(1e-6, self.sustainability_threshold),
                    )

                    metrics_by_mechanism[idx].append(
                        {
                            "seed": seed,
                            "mean_reward": float(tail_rewards.mean()),
                            "reward_std": float(tail_rewards.std()),
                            "mean_realized_harvest": float(
                                tail_realized_harvest.mean()
                            ),
                            "harvest_score": float(tail_harvest_scores.mean()),
                            "collapse_rate": float(
                                (tail_fish < self.sustainability_threshold).mean()
                            ),
                            "sustainability_penalty": float(
                                sustainability_penalties.mean()
                            ),
                            "min_fish": float(tail_fish.min()),
                            "mean_fish": float(tail_fish.mean()),
                            "mean_fines": float(tail_fish.mean()),
                        }
                    )

        max_idx = max(metrics_by_mechanism)
        fitness = np.full(max_idx + 1, -np.inf, dtype=np.float32)

        for idx, seed_metrics in metrics_by_mechanism.items():
            mean_reward = float(np.mean([m["mean_reward"] for m in seed_metrics]))
            reward_std = float(np.mean([m["reward_std"] for m in seed_metrics]))
            mean_realized_harvest = float(
                np.mean([m["mean_realized_harvest"] for m in seed_metrics])
            )
            harvest_score = float(np.mean([m["harvest_score"] for m in seed_metrics]))
            collapse_rate = float(np.mean([m["collapse_rate"] for m in seed_metrics]))
            sustainability_penalty = float(
                np.mean([m["sustainability_penalty"] for m in seed_metrics])
            )
            min_fish = float(np.mean([m["min_fish"] for m in seed_metrics]))
            mean_fish = float(np.mean([m["mean_fish"] for m in seed_metrics]))
            mean_fines = float(np.mean([m["mean_fines"] for m in seed_metrics]))
            fitness_ctx = FitnessContext.from_metrics(
                mean_reward=mean_reward,
                collapse_rate=collapse_rate,
                sustainability_penalty=sustainability_penalty,
                sustainability_weight=self.sustainability_weight,
                total_fines=mean_fines,
                mean_fish=mean_fish,
                min_fish=min_fish,
                mean_realized_harvest=mean_realized_harvest,
                harvest_score=harvest_score,
            )
            objective = float(fitness_ctx.objective_score)
            fitness[idx] = objective

            self._publish(
                MechanismContext(
                    index=idx,
                    seed=None,
                    env_id=self.env_id,
                    status=MechanismStatus.done,
                    mechanism=None,
                    metrics=fitness_ctx,
                )
            )
            per_mech_metrics.append(
                {
                    "idx": idx,
                    "objective": objective,
                    "mean_reward": mean_reward,
                    "reward_std": reward_std,
                    "mean_realized_harvest": mean_realized_harvest,
                    "harvest_score": harvest_score,
                    "collapse_rate": collapse_rate,
                    "min_fish": min_fish,
                    "mean_fish": mean_fish,
                    "total_fines": mean_fines,
                    "num_seeds": float(len(seed_metrics)),
                }
            )

        objectives = np.asarray(
            [m["objective"] for m in per_mech_metrics],
            dtype=np.float32,
        )
        collapse_rates = np.asarray(
            [m["collapse_rate"] for m in per_mech_metrics],
            dtype=np.float32,
        )
        best_position = int(np.argmax(objectives))
        worst_position = int(np.argmin(objectives))
        best = per_mech_metrics[best_position]
        worst = per_mech_metrics[worst_position]

        logger.info(
            "[Regulator][summary] "
            "mean_obj=%.4f | best_obj=%.4f (θ=%d) | "
            "worst_obj=%.4f (θ=%d) | "
            "collapse(mean=%.3f max=%.3f)",
            float(objectives.mean()),
            best["objective"],
            int(best["idx"]),
            worst["objective"],
            int(worst["idx"]),
            float(collapse_rates.mean()),
            float(collapse_rates.max()),
        )

        self.last_metrics = per_mech_metrics

        return fitness.tolist()
