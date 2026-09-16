"""Abstract accumulator behind every metric schema leaf.

A ``Metric`` receives raw values through ``push``, exposes them without
side effects through ``peek`` and collapses them through ``reduce`` according
to its protocol (mean, sum, ...). ``compile=False`` returns the raw history or a
copy of the metric instead of the compiled scalar.
"""

from __future__ import annotations

# TODO what is an ABCMeta
from abc import ABC, abstractmethod
from typing import Self, TypeAlias, Union

# NOTE this is restrictive can be relaxed in the future
PrimitiveType: TypeAlias = Union[int, float, bool, str]


class Metric(ABC):
    """Accumulator of primitive values behind one schema field.

    Each subclass implements one reduction protocol (see
    :class:`~core.metrics.enums.ReduceProtocol`). ``float()`` and ``int()``
    on a metric return its compiled value and raise ``ValueError`` when that
    value is a list.
    """

    @abstractmethod
    def __len__(self) -> int:
        """Returns the length of the internal values list."""

        ...

    def __float__(self):
        value = self.peek(compile=True)

        if isinstance(value, (list)):  # , tuple, deque
            raise ValueError(f"Can not convert {self} to float.")

        return float(value)

    def __int__(self):
        value = self.peek(compile=True)

        if isinstance(value, (list)):
            raise ValueError(f"Can not convert {self} to int.")

        return int(value)

    def empty_copy(self) -> Self:
        """Return a fresh, empty metric of the same class."""

        return type(self)()

    @abstractmethod
    def peek(
        self,
        compile: bool = True,
    ) -> Union[PrimitiveType, list[PrimitiveType]]:
        """Returns the result of reducing the internal values list.

        Note that this method does NOT alter the internal values list in this process.
        Thus, users can call this method to get an accurate look at the reduced value(s)
        given the current internal values list.

        Args:
            compile: If True, the result is compiled into a single value if possible.
        Returns:
            The result of reducing the internal values list on CPU memory.
        """

    @abstractmethod
    def reduce(
        self,
        compile: bool = True,
    ) -> Union[PrimitiveType, list[PrimitiveType], Metric]:
        """Reduces the internal values.

        This method should NOT be called directly by users.
        It can be used as a hook to prepare the stats object for sending it to the root metrics logger and starting a new 'reduce cycle'.

        The reduction logic depends on the implementation of the subclass.
        Meaning that some classes may reduce to a single value, while others do not or don't even contain values.

        Args:
            compile: If True, the result is compiled into a single value if possible.
                If False, the result is a Stats object similar to itself, but with the internal values reduced.
        Returns:
            The reduced value or a Stats object similar to itself, but with the internal values reduced.
        """

    @abstractmethod
    def push(self, value: PrimitiveType) -> None:
        """Append one value to the internal history."""

        ...

    @abstractmethod
    def flush(self) -> None:
        """Discard every accumulated value."""

    ...
