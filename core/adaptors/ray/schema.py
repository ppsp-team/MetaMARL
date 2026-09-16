"""Typed view of an RLlib training/evaluation ``ResultDict``.

``RaySchema`` has an optional ``train`` and ``eval`` branch, each with the
rollout statistics (``aggregate`` plus the ``by_mechanism -> by_seed ->
by_episode`` grouping of the environment episode schemas), the per-policy
learner statistics and performance timers. Instances are produced by the
builders in ``core.adaptors.ray.utils``.
"""

from typing import Optional, TypeAlias

from pydantic import Field

from core.envs.schema import EpisodeRolloutSchema
from core.metrics.enums import ReduceProtocol
from core.metrics.schemas import MetricSchema

PolicyID: TypeAlias = str
EpisodeID: TypeAlias = str
MechanismID: TypeAlias = str
SeedID: TypeAlias = str


class PolicyLearnerSchema(MetricSchema):
    """Learner statistics of one RLModule (policy) for one training iteration.

    Every field is optional and dimensionless unless stated otherwise;
    ``json_schema_extra["reduce"]`` gives the reduction applied across the
    iteration's mini-batches. ``batch_size`` is the number of samples per
    update; ``total_loss``, ``policy_loss`` and ``value_loss`` are the
    optimized losses; ``residual_variance`` is the unexplained fraction of
    the value target; ``sample_staleness`` is the age of the samples in
    learner updates; ``policy_entropy`` and ``policy_entropy_coeff`` are the
    action-distribution entropy (nats) and its coefficient;
    ``policy_relative_entropy`` and ``entropy_pressure`` track entropy
    against its initial level; ``policy_kl`` and ``policy_kl_coeff`` are the
    KL divergence to the behaviour policy (nats) and its coefficient;
    ``value_mean`` and ``value_target`` are the mean predicted and target
    returns (reward units); ``gradient_norm`` and ``gradient_noise`` describe
    the update direction.
    """

    batch_size: Optional[int] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )

    # Value, Q, advantage debugging
    total_loss: Optional[float] = Field(
        default=None, json_schema_extra={"source": "total_loss"}
    )
    residual_variance: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )

    # TODO this is a callable
    sample_staleness: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )

    # Policy (π) debugging
    policy_loss: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    policy_entropy: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    policy_entropy_coeff: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    policy_relative_entropy: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    entropy_pressure: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    policy_kl: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    policy_kl_coeff: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )

    # TODO what is total loss ?
    # TODO kl vs kl loss
    # TODO curr_kl_coeff
    # TODO entropy vs entropy coeff

    # Value (V) debgging
    value_loss: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    value_mean: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    value_target: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )

    # TODO Q-statistics
    # TODO Advantage statistics
    # TODO advantage statistics

    # TODO Reward (R) debugging
    # Reward metrics. N.B. episode == trajectory

    # Gradient debugging
    gradient_norm: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    gradient_noise: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )


class PerformanceSchema(MetricSchema):
    """Throughput and timing counters of one RLlib iteration.

    ``env_steps_this_iter`` and ``agent_steps_this_iter_sum`` count the
    environment and agent steps sampled in the iteration; the ``_lifetime``
    variants are cumulative since the algorithm was built.
    ``env_steps_throughput`` is in environment steps per second. The
    ``*_s`` fields are wall-clock durations in seconds of the whole
    iteration, of one training step, of sampling and of the learner update.
    ``weights_seq_no`` is RLlib's counter of weight broadcasts to the env
    runners.
    """

    env_steps_this_iter: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.LAST},
    )
    env_steps_lifetime: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.LAST},
    )
    agent_steps_this_iter_sum: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.LAST},
    )
    agent_steps_lifetime_sum: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.LAST},
    )
    env_steps_throughput: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    training_iteration_s: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    training_step_s: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    sample_s: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    learner_update_s: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    weights_seq_no: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.LAST},
    )


class SeedRolloutSchema(MetricSchema):
    """Episodes collected under one training seed, keyed by episode ID."""

    by_episode: dict[EpisodeID, EpisodeRolloutSchema] = Field(default_factory=dict)


class MechanismRolloutSchema(MetricSchema):
    """Rollouts of one mechanism candidate, keyed by training seed."""

    by_seed: dict[SeedID, SeedRolloutSchema] = Field(default_factory=dict)


class RolloutSchema(MetricSchema):
    """Rollout statistics of one iteration.

    ``aggregate`` is the episode schema reduced over every episode of the
    iteration; ``by_mechanism`` keeps the per-mechanism, per-seed,
    per-episode breakdown that the regulator uses to compute one fitness
    per candidate.
    """

    aggregate: EpisodeRolloutSchema
    by_mechanism: dict[MechanismID, MechanismRolloutSchema] = Field(
        default_factory=dict
    )


class SeedLearnerSchema(MetricSchema):
    by_policy: dict[PolicyID, PolicyLearnerSchema] = Field(default_factory=dict)


class MechanismLearnerSchema(MetricSchema):
    """Training/policy initialization seed, NOT evaluation environment seed."""

    by_seed: dict[SeedID, SeedLearnerSchema] = Field(default_factory=dict)


class LearnerSchema(MetricSchema):
    by_mechanism: dict[MechanismID, MechanismLearnerSchema] = Field(
        default_factory=dict
    )


class TrainSchema(MetricSchema):
    """Training branch of a result: rollouts, learner statistics and timers."""

    rollout: RolloutSchema
    learner: LearnerSchema
    performance: PerformanceSchema


class EvalSchema(MetricSchema):
    """Evaluation branch of a result: rollouts and timers, no learner statistics."""

    rollout: RolloutSchema
    performance: PerformanceSchema


class RaySchema(MetricSchema):
    """Typed root of an RLlib ``ResultDict`` with optional ``train`` and ``eval`` branches.

    When to use: pass it as the ``schema`` of ``RayOptimizerConfig.reporting``
    so the inner optimizer logs typed metrics that the regulator environment
    can aggregate into a fitness per mechanism.
    """

    train: Optional[TrainSchema] = Field(
        default=None,
        json_schema_extra={"subtree_reduce": ReduceProtocol.LAST},
    )
    eval: Optional[EvalSchema] = Field(
        default=None,
        json_schema_extra={"subtree_reduce": ReduceProtocol.LAST},
    )


# TODO
# num_env_steps_sampled_lifetime_throughput
# timers
# throughput_since_last_restore
# num_agent_steps_sampled
# num_agent_steps_sampled_lifetime
# prevent non terminal leaves to have reduce objects
