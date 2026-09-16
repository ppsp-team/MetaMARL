"""Metric reducing to the sum of numeric values (``SUM``, empty -> 0)."""

from __future__ import annotations

from core.annotations import override
from core.metrics.metric.base import PrimitiveType
from core.metrics.metric.series import SeriesMetric


class SumMetric(SeriesMetric):
    """Metric reducing to the sum of the pushed numbers.

    Only ``int`` and ``float`` are accepted (``bool`` is rejected). Unlike the
    other scalar metrics an empty sum compiles to ``0``, not ``None``.

    When to use: for quantities accumulated over an iteration (total catch,
    number of episodes, environment steps).
    """

    @override(SeriesMetric)
    def push(self, value: PrimitiveType) -> PrimitiveType:
        """Append a number, rejecting booleans and non-numeric values with ``TypeError``."""

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"SumMetric only accepts int or float, got {type(value).__name__}."
            )

        self.values.append(value)

    def peek(
        self,
        compile: bool = True,
    ) -> PrimitiveType | list[PrimitiveType]:
        """Return the sum (``0`` when empty), or the history when ``compile`` is false."""

        if not compile:
            return list(self.values)

        return sum(self.values)

    # TODO move to base cls
    def reduce(
        self,
        compile: bool = True,
    ) -> PrimitiveType | SumMetric:
        """Return the sum and clear the history.

        With ``compile`` false a new ``SumMetric`` holding only the sum is
        returned instead.
        """

        sum = self.peek(compile=True)

        self.flush()

        if compile:
            return sum

        metric = SumMetric()
        metric.values = [sum]

        return metric

    def __repr__(self) -> str:
        return f"SumMetric({self.peek()}; len={len(self)})"
