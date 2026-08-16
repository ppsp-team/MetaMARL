from dataclasses import dataclass

import numpy as np

from core.mechanism.base import Mechanism
from core.types import MultiAgentDict


@dataclass(frozen=True)
class SubsidyMechanism(Mechanism):
    subsidy: float
    cost: float
    action_component: int = 1

    # default_cost = 

    def __post_init__(self) -> None:
        assert 0.0 <= self.subsidy <= 0.5
        assert 0.0 <= self.cost <= 1.0 # TODO

    def to_vector(self) -> np.ndarray:
        return np.array([self.subsidy / 0.5], dtype=np.float32)

    def param_names(self) -> list[str]:
        return ["restoration_subsidy"]

    def reward(
        self,
        rewards: MultiAgentDict,
        **kwargs,
    ) -> MultiAgentDict:
        actions = kwargs["action_after"] # TODO fix this, passing action after and before

        return {
            agent_id: 
            reward 
            + self.subsidy 
            * actions[agent_id][self.action_component]
            - self.cost 
            * actions[agent_id[self.action_component]] ** 2
            for agent_id, reward in rewards.items()
        }