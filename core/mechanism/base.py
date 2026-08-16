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
from abc import ABC, abstractmethod
from typing import Any, Callable, Self

from core.types import MultiAgentDict

import numpy as np

class Mechanism(ABC):
    """Semantic representation of a regulatory mechanism."""

    bindings: dict[str, Callable[[Any], Any]]

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimension of this mechanism in optimizer space."""
        ...

    @abstractmethod
    def encode(self) -> np.ndarray:
        """Encode this mechanism into its normalized optimizer representation.""" 
        ...

    @abstractmethod
    def decode(self, x: np.ndarray) -> Self: 
        """
        Return the same mechanism structure parameterized by x.

        For composite mechanisms, decoding propagates recursively to children.
        """
        ...

    @abstractmethod
    def clip(self) -> Self: 
        ...

    @abstractmethod
    def param_names(self) -> list[str]: 
        """Names corresponding exactly to encode()."""
        ...

    @abstractmethod
    def to_vector(self) -> np.ndarray:
        """Full semantic representation exposed to agents."""
        ...

    def _validate(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)

        if x.shape != (self.dimension,):
            raise ValueError(f"Expected shape ({self.dimension},), got {x.shape}")

        if not np.isfinite(x).all():
            raise ValueError(f"Non-finite values in vector: {x}")

        return x


    def resolve(
        self,
        env: Any, #TODO env should not be any
    ) -> dict[str, Any]:
        bindings = getattr(self, "bindings", {})
        return {name: binding(env) for name, binding in bindings.items()}


    def action(
            self,
            action_dict: MultiAgentDict,
            **kwargs,
    ) -> MultiAgentDict:
        """Transform agent actions."""
        return action_dict

    def observation(
            self,
            observation_dict: MultiAgentDict,
            **kwargs,
    ) -> MultiAgentDict:
        """Transform agent observations."""
        return observation_dict

    def reward(
            self,
            reward_dict: MultiAgentDict,
            **kwargs,
    ) -> MultiAgentDict:
        """Transform agent rewards."""
        return reward_dict
