# core/mechanism/quota.py

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from core.annotations import override
from core.mechanism.base import Mechanism
from core.types import MultiAgentDict
from core.utils import (
    sigmoid,
    smooth_positive_zero_at_origin,
)


EPS = 1e-8


# TODO what if two mechanisms interfere by requiring context from each other ? 
# for example a penalty based on how much quota is violated ?
@dataclass(frozen=True)
class QuotaMechanism(Mechanism):
    fixed_quota: float

    # NOTE why reinstantiate in subclass ?
    bindings: dict[str, Callable[[Any], Any]] = field(
        repr=False,
        compare=False,
    )
    action_component: int = 0

    # Fixed algorithmic parameters.
    quota_transition_width: float = 0.03
    usage_transition_width: float = 0.005
    violation_transition_width: float = 0.03

    _context: dict[str, Any] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        required = {"resource_level"}

        missing = required - self.bindings.keys()
        if missing: raise ValueError(f"Missing quota bindings: {missing}")

        assert 0.0 <= self.fixed_quota <= 1.0
        assert self.quota_transition_width > 0.0
        assert self.usage_transition_width > 0.0
        assert self.violation_transition_width > 0.0

    def to_vector(self) -> np.ndarray:
        return np.array([self.fixed_quota], dtype=np.float32)

    def param_names(self) -> list[str]:
        return ["fixed_quota"]

    def observation_names(self) -> list[str]:
        return ["effective_quota"]

    # TODO action needs to know about #1 current resource level and #2 full required
    # TODO we assume prior normalizaiton of action space 
    # TODO we assume prior selection of action component.
    @override(Mechanism)
    def action(
        self,
        action_dict: MultiAgentDict,
        **kwargs,
    ) -> MultiAgentDict:
        resource_level = kwargs["resource_level"]

        width = max(self.quota_transition_width, EPS)
        lower = sigmoid((0.0 - self.fixed_quota) / width)
        upper = sigmoid((1.0 - self.fixed_quota) / width)
        current = sigmoid((resource_level - self.fixed_quota) / width)

        allowed_frac = (current - lower) / max(upper - lower, EPS)
        self._context["allowed_frac"] = allowed_frac

        requested = {}
        regulated = {}

        for agent_id, action in action_dict.items():
            action = np.asarray(action, dtype=np.float32)
            requested[agent_id] = action.copy()
            regulated_action = action.copy()
            requested_frac = float(action[self.action_component])
            regulated_action[self.action_component] = (
                requested_frac - smooth_positive_zero_at_origin(
                    requested_frac - allowed_frac,
                    self.usage_transition_width,
                )
            )
            regulated[agent_id] = regulated_action

        self._context["requested_action_dict"] = requested
        self._context["delivered_action_dict"] = regulated
        return regulated


    # TODO this can be potentially removed
    @override(Mechanism)
    def observation(
        self,
        observation_dict: MultiAgentDict,
        **kwargs,
    ) -> MultiAgentDict:
        if "allowed_frac" not in self._context:
            return observation_dict
        return {
            agent_id: np.concatenate(
                [observation, np.array([self._context["allowed_frac"]], dtype=np.float32)]
            )
            for agent_id, observation in observation_dict.items()
        }