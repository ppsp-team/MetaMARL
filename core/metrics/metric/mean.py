"""Metric reducing to the arithmetic mean of numeric values (``MEAN``)."""

from __future__ import annotations

from core.annotations import override
from core.metrics.metric.base import PrimitiveType
from core.metrics.metric.series import SeriesMetric


class MeanMetric(SeriesMetric):
    """Metric reducing to the arithmetic mean of the pushed numbers.

    Only ``int`` and ``float`` are accepted (``bool`` is rejected); ``peek``
    returns ``None`` while empty.

    When to use: the default protocol, for per-step quantities averaged over an iteration (rewards, catches, losses).
    """

    @override(SeriesMetric)
    def push(self, value: PrimitiveType) -> None:
        """Append a number, rejecting booleans and non-numeric values with ``TypeError``."""

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"MeanMetric only accepts int or float, got {type(value).__name__}."
            )

        self.values.append(value)

    def peek(
        self,
        compile: bool = True,
    ) -> float | list[float]:
        """Return the arithmetic mean (``None`` when empty), or the history when ``compile`` is false."""

        if not compile:
            return list(self.values)

        if not self.values:
            return None

        return sum(self.values) / len(self.values)

    def reduce(
        self,
        compile: bool = True,
    ) -> float | MeanMetric:
        """Return the arithmetic mean and clear the history.

        With ``compile`` false a new ``MeanMetric`` holding only that value is
        returned instead; an empty metric reduces to ``None``.
        """

        if not self.values:
            return None if compile else MeanMetric()

        mean = self.peek(compile=True)

        self.flush()

        if compile:
            return mean

        metric = MeanMetric()
        metric.values = [mean]

        return metric

    def __repr__(self) -> str:
        return f"MeanMetric({self.peek()}; len={len(self)})"
