from dataclasses import dataclass

import numpy as np

from core.mechanism.base import Mechanism
from core.types import MultiAgentDict


@dataclass(frozen=True)
class SocialInfluenceMechanism(Mechanism):
    influence_weight: float

    def observation(
        self,
        observation_dict: MultiAgentDict,
        **kwargs,
    ) -> MultiAgentDict:
        previous_actions = kwargs["previous_actions"]
        agent_ids = kwargs["agent_ids"]

        regulated = {}
        for agent_id, observation in observation_dict.items():
            peer_actions = [
                np.asarray(previous_actions[other_id], dtype=np.float32).reshape(-1)
                for other_id in agent_ids if other_id != agent_id
            ]
            regulated[agent_id] = np.concatenate(
                [observation, *peer_actions]
            ).astype(np.float32,copy=False,)

        return regulated