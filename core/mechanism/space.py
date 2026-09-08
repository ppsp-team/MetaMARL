"""Geometry of the mechanism manifold.

``MechanismSpace`` is the protocol a benchmark implements to connect its
semantic :class:`~core.mechanism.base.Mechanism` to the outer optimizer:
``encode`` and ``decode`` map between a mechanism and a normalized vector of
fixed ``dimension``, ``clip`` enforces the parameter ranges, ``sample`` draws a
random mechanism and ``default`` provides the one in force before any candidate
is published. Environments receive a space (class or instance) through the
``mechanism_space`` argument of :class:`core.envs.base.BaseEnv`.
"""

from abc import abstractmethod
from typing import Protocol

import numpy as np

from core.mechanism.base import Mechanism


# TODO why not have these methods be abstract ?


class MechanismSpace(Protocol):
    """Geometry + constraints over a mechanism manifold."""

    dimension: int

    def _validate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)

        if x.shape != (self.dimension,):
            raise ValueError(f"Expected shape ({self.dimension},), got {x.shape}")

        if not np.isfinite(x).all():
            raise ValueError(f"Non-finite values in vector: {x}")

        return x

    @classmethod
    def default(cls) -> "Mechanism":
        """Return the mechanism in force before the optimizer publishes a candidate."""

        raise NotImplementedError

    @abstractmethod
    def encode(self, m: Mechanism) -> np.ndarray:
        """Map ``m`` to its optimizer vector, shape ``(dimension,)`` in ``[0, 1]``."""

        ...

    @abstractmethod
    def decode(self, x: np.ndarray) -> Mechanism:
        """Build the mechanism parameterized by ``x``, shape ``(dimension,)``."""

        ...

    def clip(self, m: Mechanism) -> Mechanism:
        """Return a copy of ``m`` whose parameters are clipped to their valid ranges."""

        ...

    def sample(self) -> Mechanism:
        """Draw a random mechanism from the space."""

        ...
