"""Context records exchanged between the fishery levels through the World actor.
``FitnessContext`` is what the regulator environment publishes for each
mechanism candidate once the inner rollouts have been aggregated: the scalar
objective the ES maximizes plus the summary statistics it was computed from.
"""

from typing import SupportsFloat

from core.world.context import ContextSchema


class FitnessContext(ContextSchema):
    """Fitness of one mechanism candidate, the sole scalar feedback of the bilevel loop.

    Built by ``FisheryRegulatorEnv.aggregate_rewards`` from the inner
    optimizer's evaluation rollouts and consumed by the ES outer loop.
    ``objective_score`` is ``harvest_score + sustainability_weight *
    mean_fish``. ``mean_reward`` is the tail-averaged per-step reward
    (delivered harvest fraction, dimensionless); ``collapse_rate`` the
    fraction of tail steps with normalized biomass below the sustainability
    threshold; ``sustainability_penalty`` the mean relative shortfall below
    that threshold; ``total_fines`` the fines paid (reward units);
    ``mean_fish`` and ``min_fish`` the mean and minimum normalized biomass
    (fraction of the carrying capacity ``K``); ``mean_realized_harvest`` the
    mean realized total harvest (biomass units); ``harvest_score`` the mean
    realized harvest divided by the maximum sustainable yield.
    """

    objective_score: float
    mean_reward: float
    collapse_rate: float
    sustainability_penalty: float
    total_fines: float
    mean_fish: float
    min_fish: float
    mean_realized_harvest: float
    harvest_score: float

    @classmethod
    def from_metrics(
        cls,
        *,
        mean_reward: SupportsFloat,
        collapse_rate: SupportsFloat,
        sustainability_penalty: SupportsFloat,
        sustainability_weight: SupportsFloat,
        total_fines: SupportsFloat = 0.0,
        mean_fish: SupportsFloat = 0.0,
        min_fish: SupportsFloat = 0.0,
        mean_realized_harvest: SupportsFloat = 0.0,
        harvest_score: SupportsFloat = 0.0,
    ) -> "FitnessContext":
        """Build the context and compute ``objective_score`` from the statistics.

        The objective is ``harvest_score + sustainability_weight * mean_fish``;
        ``mean_reward``, ``collapse_rate`` and ``sustainability_penalty`` are
        stored for reporting but do not enter the objective.
        """

        # TODO
        # objective = mean_reward - sustainability_weight * (1.0 - mean_fish)
        # objective = harvest_score
        objective = harvest_score + sustainability_weight * mean_fish

        # objective = mean_fish

        # reward = float(mean_reward)
        # fish = float(mean_fish)
        # alpha = float(sustainability_weight)

        # if not 0.0 <= alpha <= 1.0:
        #     raise ValueError(
        #         "sustainability_weight must be between 0 and 1"
        #     )

        # eps = 1e-8

        # objective = (
        #     (1.0 - alpha) * np.log(max(reward, eps))
        #     + alpha * np.log(max(fish, eps))
        # )

        return cls(
            objective_score=objective,
            mean_reward=float(mean_reward),
            collapse_rate=float(collapse_rate),
            sustainability_penalty=float(sustainability_penalty),
            total_fines=float(total_fines),
            mean_fish=float(mean_fish),
            min_fish=float(min_fish),
            mean_realized_harvest=float(mean_realized_harvest),
            harvest_score=float(harvest_score),
        )
