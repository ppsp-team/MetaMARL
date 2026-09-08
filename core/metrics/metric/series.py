"""Metric keeping the full history of pushed values (``SERIES``)."""

from __future__ import annotations

from core.metrics.metric.base import Metric, PrimitiveType


class SeriesMetric(Metric):
    """Metric keeping every pushed value in order.

    ``peek`` and ``reduce`` both return the whole history as a list whatever
    ``compile`` is; ``reduce`` additionally clears it. It is also the base
    class of the scalar metrics, which reuse its ``values`` list and only
    override the reduction.

    When to use: for values you want to plot against an x axis (a curve per
    iteration, the fitness of every candidate); reporters expect series
    leaves. Use :class:`~core.metrics.metric.last.LastMetric` for a counter
    and :class:`~core.metrics.metric.mean.MeanMetric` for a value averaged
    over an iteration.
    """

    def __init__(self) -> None:
        self.values: list[PrimitiveType] = []

    def __len__(self) -> int:
        return len(self.values)

    def __repr__(self) -> str:
        return f"SeriesMetric(len={len(self)})"

    def push(self, value: PrimitiveType) -> None:
        """Append ``value`` to the history without type checking."""

        self.values.append(value)

    def peek(
        self,
        compile: bool = True,
    ) -> list[PrimitiveType]:
        """Return a copy of the history; ``compile`` is ignored."""

        return list(self.values)

    def reduce(
        self,
        compile: bool = True,
    ) -> list[PrimitiveType] | SeriesMetric:
        """Return the history and clear it.

        With ``compile`` false the history is returned inside a new
        ``SeriesMetric`` instead of a plain list.
        """

        values = list(self.values)

        self.flush()

        if compile:
            return values

        metric = SeriesMetric()
        metric.values = values

        return metric

    def flush(self) -> None:
        """Clear the history in place."""

        self.values.clear()
