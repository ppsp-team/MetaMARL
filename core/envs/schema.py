"""Environment-level metric schemas.

``AgentEnvStepSchema`` holds what is logged per agent at each step;
``EpisodeRolloutSchema`` holds episode-level statistics plus the ``by_agent``
dynamic node. Benchmarks subclass both to add their own fields (see
``examples/bilevel_fishery/metric_schema.py``).
"""

from typing import Optional, TypeAlias

from pydantic import Field

from core.metrics.enums import ReduceProtocol
from core.metrics.schemas import MetricSchema

AgentID: TypeAlias = str


class AgentEnvStepSchema(MetricSchema):
    """Generic metrics produced for one agent during environment steps."""

    # statistics collected at env-step level
    action: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    observation: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    reward: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    terminated: Optional[bool] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.LAST},  # TODO mean to support bool
    )
    truncated: Optional[bool] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.LAST},  # TODO mean to supprot bool
    )

    # calculated stats
    logp: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    value_pred: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    value_target: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    advantage: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    td_error: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    q_pred: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )

    # TODO is this necessary ?
    intrinsic_utility: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    violation_signal: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )


# TODO add recducer metadata attachment.
class EpisodeRolloutSchema(
    MetricSchema
):  # attention this is aggregate by env not by env step
    """Generic environment-step metrics."""

    env_id: Optional[int] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.LAST},
    )
    mechanism_id: Optional[int] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.LAST},
    )
    seed: Optional[int] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.LAST},
    )
    policy_seed: Optional[int] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.LAST},
    )

    # Reward (R) statistics
    reward_total: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.SUM}
    )
    reward_mean: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    reward_min: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MIN}
    )
    reward_max: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MAX}
    )
    reward_terminal: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.LAST}
    )

    # Value (V) statistics
    value_terminal: Optional[float] = None
    value_penultimate: Optional[float] = None
    episode_len_mean: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    episode_len_max: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    episode_len_min: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    num_episodes: Optional[int] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.LAST},
    )
    num_episodes_lifetime: Optional[int] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.LAST},
    )

    # TODO models such as PILCO, Dyna, Qyna-Q
    by_agent: dict[AgentID, AgentEnvStepSchema] = Field(
        default_factory=dict,
    )
