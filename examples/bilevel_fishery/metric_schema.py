"""Metric schema logged by ``FisheryRegulatedEnv`` at every step.

``FisheryMetricSchema`` extends the generic episode schema with the stock
dynamics (biomass, growth, harvests, reference points, quota allowance) and
``FisheryAgentMetricSchema`` with the per-fisher harvest requests. The
regulated environment pushes one value per step into these fields, and the
regulator reads the ``fish_norm_next``, ``H_realized`` and ``MSY`` series
back to compute a candidate's fitness.
"""

from typing import Optional, TypeAlias

from pydantic import Field

from core.envs.schema import AgentEnvStepSchema, EpisodeRolloutSchema
from core.metrics.enums import ReduceProtocol

AgentID: TypeAlias = str


class FisheryAgentMetricSchema(AgentEnvStepSchema):
    """Per-fisher harvest metrics, one value per step.

    ``requested_harvest`` and ``delivered_harvest`` are in biomass units;
    ``requested_frac`` is the harvest fraction of the agent's maximal request
    in ``[0, 1]``; ``quota_violation``, ``quota_penalty`` and ``risk_penalty``
    are the mechanism's dimensionless violation measure and the penalties it
    subtracts from the reward.
    """

    requested_harvest: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    delivered_harvest: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    requested_frac: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    quota_violation: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    quota_penalty: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    risk_penalty: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )


class FisheryMetricSchema(EpisodeRolloutSchema):
    """Stock-level fishery metrics, one value per step.

    ``fish_stock`` and ``fish_stock_next`` are the biomass before and after
    the transition, ``fish_norm`` and ``fish_norm_next`` the same divided by
    the carrying capacity ``K``. ``growth`` is the biomass change before
    harvest (surplus production plus noise plus restoration) and
    ``growth_noise`` its stochastic part. ``H_attempted`` and ``H_realized``
    are the requested and realized total harvests, ``allowed_harvest`` the
    total the quota permits (all in biomass units), ``total_usage_norm`` the
    realized harvest over ``K`` and ``quota_stress`` the allowed harvest
    fraction of the quota in force. ``B_msy`` (biomass), ``MSY`` (biomass
    per step) and ``F_msy`` (per step) are the maximum-sustainable-yield
    reference points of the surplus-production model.
    """

    # TODO move this into logging for mechanism
    # max_demand_frac: float = Field(
    #     json_schema_extra={"reduce": ReduceProtocol.MEAN},
    # )
    quota_stress: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    allowed_harvest: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    fish_stock: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    growth: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    growth_noise: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    H_attempted: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    H_realized: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    total_usage_norm: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    B_msy: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    MSY: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    F_msy: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    fish_stock_next: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    fish_norm: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    fish_norm_next_mean: Optional[float] = Field(
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
    fish_norm_next_min: Optional[float] = Field(
        json_schema_extra={"reduce": ReduceProtocol.MIN},
    )
    fish_norm_next_max: Optional[float] = Field(
        json_schema_extra={"reduce": ReduceProtocol.MAX},
    )
    fish_norm_next_last: Optional[float] = Field(
        json_schema_extra={"reduce": ReduceProtocol.MAX},
    )

    # full_required_harvest: float = Field(
    #     json_schema_extra={"reduce": ReduceProtocol.MEAN},
    # )

    # realized_harvest: float = Field(
    #     json_schema_extra={"reduce": ReduceProtocol.MEAN},
    # )

    # harvest_to_msy: float = Field(
    #     json_schema_extra={"reduce": ReduceProtocol.MEAN},
    # )

    by_agent: dict[AgentID, FisheryAgentMetricSchema] = Field(
        default_factory=dict,
    )
