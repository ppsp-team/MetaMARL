"""Metric schema of one Evolution Strategies generation.

Series fields (``generation``, ``sigma``, ``fitness_mean``, ...) grow by one
value per generation; ``by_mechanism`` holds each candidate's fitness and
parameter values, ``search_mean``/``global_best``/``generation_best`` are keyed
by parameter name, and ``inner`` is the reduced metric schema of the inner
optimizer, specialized at runtime (``RaySchema`` for RLlib).
"""

from typing import Optional, TypeAlias

from pydantic import Field

from core.metrics.enums import ReduceProtocol
from core.metrics.schemas import MetricSchema

MechanismID: TypeAlias = str
ParameterName: TypeAlias = str


class ESParameterSchema(MetricSchema):
    """One normalized mechanism parameter tracked as a series over generations.

    ``value`` lies in ``[0, 1]`` (the ES search space, before the mechanism
    decodes it to its own units).
    """

    value: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.SERIES},
    )


class ESCandidateSchema(MetricSchema):
    """One candidate of the population: its fitness series and its parameters.

    ``fitness`` is the scalar returned by the regulator environment for the
    candidate (objective units defined by the example); ``by_parameter`` is
    keyed by the mechanism's parameter names.
    """

    fitness: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.SERIES},
    )

    # Parameter names are runtime-defined by MechanismSpace.
    by_parameter: dict[str, ESParameterSchema] = Field(default_factory=dict)


class ESSchema(MetricSchema):
    """Metrics of one ES generation (see module docstring).

    ``generation`` is the generation index, ``sigma`` the search standard
    deviation in logit space, ``population_size`` the number of candidates,
    ``fitness_mean`` and ``fitness_best`` the population statistics of the
    generation, ``best_mechanism_idx`` the index of the generation's best
    candidate and ``best_fitness_global`` the best fitness seen so far.
    """

    generation: Optional[int] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.SERIES},
    )
    sigma: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.SERIES},
    )
    population_size: Optional[int] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.SERIES},
    )
    fitness_mean: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.SERIES},
    )
    fitness_best: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.SERIES},
    )
    best_mechanism_idx: Optional[int] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.SERIES},
    )
    best_fitness_global: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.SERIES},
    )
    by_mechanism: dict[MechanismID, ESCandidateSchema] = Field(default_factory=dict)
    search_mean: dict[MechanismID, ESParameterSchema] = Field(default_factory=dict)
    global_best: dict[MechanismID, ESParameterSchema] = Field(default_factory=dict)
    generation_best: dict[ParameterName, ESParameterSchema] = Field(
        default_factory=dict
    )
    inner: Optional[MetricSchema] = None
