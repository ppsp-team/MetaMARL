"""Reporter configurations: serializable factories for reporter instances.

A ``ReporterConfig`` travels through the optimizer configs and is copied into
every environment; ``build(label=...)`` creates the backend-specific reporter
for one owner (an optimizer or an environment instance).
"""

import copy
from abc import ABC, abstractmethod
from typing import Self, Union

from core.reporting.base import Reporter


class ReporterConfig(ABC):
    """Serializable description of a reporter, built into one instance per owner.

    ``project`` names the run group of the backend. ``world`` and
    ``outer_iters`` are filled in later by the optimizer that owns the config,
    before :meth:`build` is called; :meth:`copy` gives each environment its
    own instance.
    """

    def __init__(
        self,
        project: str,
    ):
        self.project_name: str = project
        self._world_name: Union[str | None] = None
        self._outer_iters: Union[int | None] = None

    @property
    def world(self) -> Union[str, None]:
        """Name of the world reported on (``None`` until the optimizer sets it)."""

        return self._world_name

    @world.setter
    def world(self, world: str) -> None:
        """Set the world name used in run names and output directories."""

        self._world_name = world

    @property
    def outer_iters(self) -> Union[int, None]:
        """Number of outer-loop iterations of the run (``None`` until set)."""

        return self._outer_iters

    @outer_iters.setter
    def outer_iters(self, outer_iters: int) -> None:
        """Set the number of outer-loop iterations forwarded to the backend."""

        self._outer_iters = outer_iters

    def copy(self) -> Self:
        """Return a deep copy of this reporter configuration."""

        return copy.deepcopy(self)

    @abstractmethod
    def build(self) -> Reporter:
        """
        Compiles config based on the reporter type and initiates the reporter
        """

        ...
