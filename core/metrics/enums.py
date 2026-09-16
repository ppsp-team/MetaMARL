"""Reduction protocols attachable to metric schema fields."""

from enum import Enum


class ReduceProtocol(Enum):
    """How a metric leaf collapses its pushed values when the logger reduces.

    ``MEAN``, ``SUM``, ``MAX`` and ``MIN`` reduce numeric values to one
    scalar; ``LAST`` keeps the most recent value of any primitive type;
    ``SERIES`` keeps the whole history as a list. ``EMA`` is declared but has
    no implementation yet, so :class:`~core.metrics.metric.factory.MetricFactory`
    raises ``NotImplementedError`` for it. The protocol is attached to a
    schema field through ``Field(json_schema_extra={"reduce": ...})``.
    """

    MEAN = "mean"
    SUM = "sum"
    MAX = "max"
    MIN = "min"
    LAST = "last"
    EMA = "ema"
    SERIES = "series"
