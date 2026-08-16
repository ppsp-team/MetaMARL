from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import numpy as np

from core.annotations import override
from core.mechanism.base import Mechanism
from core.types import MultiAgentDict


@dataclass(frozen=True)
class ChainedMechanism(Mechanism):
    """
    Functional composition of mechanisms.

    For children [f, h, g]:

        action(x) = g(h(f(x)))

    The same ordering is used for reward and observation transformations.
    """

    children: tuple[Mechanism, ...]

    bindings: dict[str, Callable[[Any], Any]] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not self.children:
            raise ValueError(
                "ChainedMechanism requires at least one child."
            )

    @property
    def dimension(self) -> int:
        return sum(child.dimension for child in self.children)

    # TODO replace with __str__
    def param_names(self) -> list[str]:
        names: list[str] = []
        for index, child in enumerate(self.children):
            prefix = (f"{index}:{type(child).__name__}")
            names.extend(f"{prefix}.{name}" for name in child.param_names())
        return names

    def encode(self) -> np.ndarray:
        if self.dimension == 0:
            return np.empty(0, dtype=np.float32)

        return np.concatenate(
            [child.encode() for child in self.children], 
            axis=0,
        ).astype(np.float32, copy=False,)

    def decode(
        self,
        x: np.ndarray,
    ) -> "ChainedMechanism":
        x = self._validate(x)
        decoded_children: list[Mechanism] = []
        start = 0
        for child in self.children:
            stop = (start + child.dimension)
            decoded_children.append(child.decode(x[start:stop]))
            start = stop
        return replace(self, children=tuple(decoded_children))

    @override(Mechanism)
    def reward(
        self,
        reward_dict: MultiAgentDict,
        *,
        env: Any,
        **kwargs,
    ) -> MultiAgentDict:
        for child in self.children:
            context = child.resolve(env)
            child_kwargs = {**kwargs, **context}
            reward_dict = child.reward(reward_dict, **child_kwargs)
        return reward_dict

    @override(Mechanism)
    def action(
        self,
        action_dict: MultiAgentDict,
        *,
        env: Any,
        **kwargs,
    ) -> MultiAgentDict:
        for child in self.children:
            context = child.resolve(env)
            child_kwargs = {**kwargs, **context}
            action_dict = child.action(action_dict, **child_kwargs)
        return action_dict

    @override(Mechanism)
    def observation(
        self,
        observation_dict: MultiAgentDict,
        *,
        env: Any,
        **kwargs,
    ) -> MultiAgentDict:
        for child in self.children:
            context = child.resolve(env)
            child_kwargs = {**kwargs, **context}
            observation_dict = child.observation(observation_dict, **child_kwargs)
        return observation_dict