from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import numpy as np

from core.mechanism.base import Mechanism
from core.types import MultiAgentDict


MergeFn = Callable[
    [
        MultiAgentDict,
        tuple[MultiAgentDict, ...],
    ],
    MultiAgentDict,
]


@dataclass(frozen=True)
class ParallelMechanism(Mechanism):
    """
    Parallel composition of mechanisms.

    Given children [f, h, g]:

                    -> f(x) --
                   /           |
        x --------> h(x) ------+--> merge
                   \\           |
                    -> g(x) --

    Each child receives the same original input.
    """

    children: tuple[Mechanism, ...]

    action_merge: MergeFn
    reward_merge: MergeFn
    observation_merge: MergeFn

    bindings: dict[str, Callable[[Any], Any]] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not self.children:
            raise ValueError(
                "ParallelMechanism requires at least one child."
            )

    @property
    def dimension(self) -> int:
        return sum(
            child.dimension
            for child in self.children
        )

    def param_names(self) -> list[str]:
        names: list[str] = []

        for index, child in enumerate(self.children):
            prefix = f"{index}:{type(child).__name__}"

            names.extend(
                f"{prefix}.{name}"
                for name in child.param_names()
            )

        return names

    def encode(self) -> np.ndarray:
        if self.dimension == 0:
            return np.empty(
                0,
                dtype=np.float32,
            )

        return np.concatenate(
            [
                child.encode()
                for child in self.children
            ],
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        )

    def decode(
        self,
        x: np.ndarray,
    ) -> "ParallelMechanism":
        x = self._validate(x)

        children: list[Mechanism] = []
        start = 0

        for child in self.children:
            stop = start + child.dimension

            children.append(
                child.decode(
                    x[start:stop]
                )
            )

            start = stop

        return replace(
            self,
            children=tuple(children),
        )

    def apply_action(
        self,
        action_dict: MultiAgentDict,
        *,
        env: Any,
    ) -> MultiAgentDict:
        outputs = tuple(
            child.apply_action(
                deepcopy(action_dict),
                env=env,
            )
            for child in self.children
        )

        return self.action_merge(
            action_dict,
            outputs,
        )

    def apply_reward(
        self,
        reward_dict: MultiAgentDict,
        *,
        env: Any,
    ) -> MultiAgentDict:
        outputs = tuple(
            child.apply_reward(
                deepcopy(reward_dict),
                env=env,
            )
            for child in self.children
        )

        return self.reward_merge(
            reward_dict,
            outputs,
        )

    def apply_observation(
        self,
        observation_dict: MultiAgentDict,
        *,
        env: Any,
    ) -> MultiAgentDict:
        outputs = tuple(
            child.apply_observation(
                deepcopy(observation_dict),
                env=env,
            )
            for child in self.children
        )

        return self.observation_merge(
            observation_dict,
            outputs,
        )