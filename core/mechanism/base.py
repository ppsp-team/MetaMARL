"""Mechanism interface and its generic vector implementation.

A mechanism is the regulatory intervention the outer optimizer searches over
and the inner environment applies to its agents. This module defines the
:class:`Mechanism` protocol every mechanism satisfies (a normalized vector
representation exposed to the agents, parameter names and a default instance)
and :class:`VectorMechanism`, a parameter-free wrapper around a raw array used
when no semantic mechanism class is available. The geometry of the mechanism
manifold (encoding, decoding, clipping, sampling) lives in
:mod:`core.mechanism.space`.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Mechanism(Protocol):
    """Semantic representation of a regulatory mechanism."""

    def to_vector(self) -> list[float]:
        """Convert semantic mechanism to normalized vector in [0,1]^d."""

        ...

    def param_names(self) -> list[str]:
        """Names corresponding exactly to the entries of :meth:`to_vector`."""

        ...

    @classmethod
    def default(cls) -> "Mechanism":
        """Return the mechanism in force before the optimizer publishes a candidate."""

        ...


@dataclass(frozen=True)
class VectorMechanism(Mechanism):
    """Mechanism whose only content is a raw parameter vector.

    When to use: benchmarks that have no semantic mechanism class and let the
    optimizer act directly on ``[0, 1]^d`` vectors;
    :class:`core.envs.regulator.RegulatorEnv` wraps optimizer outputs in it
    when no mechanism space is configured.

    Parameters
    ----------
    x : np.ndarray
        Parameter vector, shape ``(d,)``, normalized floats.
    """

    x: np.ndarray

    def to_vector(self) -> list[float]:
        """Return ``x`` flattened as a list of ``float32`` values."""

        return np.asarray(self.x, dtype=np.float32).ravel().tolist()

    @classmethod
    def from_vector(cls, v: list[float]) -> "VectorMechanism":
        """Build a ``VectorMechanism`` from any array-like of floats."""

        return cls(np.asarray(v, dtype=np.float32))
