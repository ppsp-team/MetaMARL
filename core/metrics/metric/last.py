"""Metric reducing to the last pushed value (``LAST``)."""

from __future__ import annotations

from core.metrics.metric.base import PrimitiveType
from core.metrics.metric.series import SeriesMetric


class LastMetric(SeriesMetric):
    """Metric reducing to the most recently pushed value.

    Any primitive type is accepted; ``peek`` returns ``None`` while empty.

    When to use: for counters and state where only the latest value matters
    (``iter``, a generation index, the current best fitness).
    """

    def peek(
        self,
        compile: bool = True,
    ) -> PrimitiveType | list[PrimitiveType]:
        """Return the last value (``None`` when empty), or the history when ``compile`` is false."""

        if not compile:
            return list(self.values)

        if not self.values:
            return None

        return self.values[-1]

    # TODO move to base cls
    def reduce(
        self,
        compile: bool = True,
    ) -> float | LastMetric:
        """Return the last value and clear the history.

        With ``compile`` false a new ``LastMetric`` holding only that value is
        returned instead; an empty metric reduces to ``None``.
        """

        if not self.values:
            return None if compile else LastMetric()

        last = self.peek(compile=True)

        self.flush()

        if compile:
            return last

        metric = LastMetric()
        metric.values = [last]

        return metric

    def __repr__(self) -> str:
        return f"LastMetric({self.peek()}; len={len(self)})"
