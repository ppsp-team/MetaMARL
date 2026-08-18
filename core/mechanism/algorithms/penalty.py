from dataclasses import dataclass, field
from typing import Any, Callable, Self

import numpy as np

from core.mechanism.base import Mechanism
from core.types import MultiAgentDict


@dataclass(frozen=True)
class ThresholdPenaltyMechanism(Mechanism):
    """
    Smoothly penalize rewards when a normalized signal falls below
    a specified threshold.
    """

    threshold: float = 0.20
    penalty_amount: float = 0.10
    transition_width: float = 0.03

    bindings: dict[str, Callable[[Any], Any]] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if "resource_level" not in self.bindings:
            raise ValueError(
                "ThresholdPenaltyMechanism requires a "
                "'resource_level' binding."
            )

        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(
                "threshold must be in [0, 1]."
            )

        if self.penalty_amount < 0.0:
            raise ValueError(
                "penalty_amount must be non-negative."
            )

        if self.transition_width <= 0.0:
            raise ValueError(
                "transition_width must be positive."
            )

    @property
    def dimension(self) -> int:
        # Fixed regulatory algorithm for now.
        return 0

    def encode(self) -> np.ndarray:
        return np.empty(
            0,
            dtype=np.float32,
        )

    def decode(
        self,
        x: np.ndarray,
    ) -> Self:
        self._validate(x)
        return self

    def param_names(self) -> list[str]:
        return []

    def reward(
        self,
        reward_dict: MultiAgentDict,
        **kwargs,
    ) -> MultiAgentDict:
        resource_level = float(
            kwargs["resource_level"]
        )

        penalty = (
            self.penalty_amount
            / (
                1.0
                + np.exp(
                    np.clip(
                        (
                            resource_level
                            - self.threshold
                        )
                        / self.transition_width,
                        -60.0,
                        60.0,
                    )
                )
            )
        )

        return {
            agent_id: float(
                reward - penalty
            )
            for agent_id, reward
            in reward_dict.items()
        }