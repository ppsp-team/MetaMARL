"""Metric reducing to the maximum of numeric values (``MAX``)."""

from __future__ import annotations

from core.annotations import override
from core.metrics.metric.base import PrimitiveType
from core.metrics.metric.series import SeriesMetric


class MaxMetric(SeriesMetric):
    """Metric reducing to the maximum of the pushed numbers.

    Only ``int`` and ``float`` are accepted (``bool`` is rejected); ``peek``
    returns ``None`` while empty.

    When to use: for peak values over an iteration, such as the best reward of a batch of episodes.
    """

    @override(SeriesMetric)
    def push(self, value: PrimitiveType) -> PrimitiveType:
        """Append a number, rejecting booleans and non-numeric values with ``TypeError``."""

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"MaxMetric only accepts int or float, got {type(value).__name__}."
            )

        self.values.append(value)

    def peek(
        self,
        compile: bool = True,
    ) -> PrimitiveType | list[PrimitiveType] | None:
        """Return the maximum (``None`` when empty), or the history when ``compile`` is false."""

        if not compile:
            return list(self.values)

        if not self.values:
            return None

        return max(self.values)

    def reduce(
        self,
        compile: bool = True,
    ) -> PrimitiveType | MaxMetric | None:
        """Return the maximum and clear the history.

        With ``compile`` false a new ``MaxMetric`` holding only that value is
        returned instead; an empty metric reduces to ``None``.
        """

        if not self.values:
            return None if compile else MaxMetric()

        max = self.peek(compile=True)

        self.flush()

        if compile:
            return max

        metric = MaxMetric()
        metric.values = [max]

        return metric

    def __repr__(self) -> str:
        return f"MaxMetric({self.peek()}; len={len(self)})"
