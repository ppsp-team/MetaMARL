import logging
from typing import SupportsFloat

import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import AgentID, MultiAgentDict

from core.annotations import override
from core.envs.marl_regulated import MultiAgentRegulatedEnv
from core.world.context import EnvStepContext

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)


# Numerical stability constant
EPS = 1e-8

# TODO add multiagent state in types
# TODO ban proportional to violation severity


# TODO number of agents spawned dynamically as a byproduct of config stating number of agents


class CartpoleRegulatedEnv(MultiAgentRegulatedEnv):
    def __init__(
        self,
        *,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Ensure number of agents == 1
        if len(self.agents) != 1:
            raise ValueError(
                "CartPoleRegulatedEnv is a single-agent QC environment. "
                f"Got agents={self.agents}."
            )

        self.agent_id = self.agents[0]

        # Initialize Cartpole env
        self.env = gymnasium.make("CartPole-v1", render_mode=render_mode)

        # override observationa and action spaces
        self.action_spaces = {self.agent_id: self.env.action_space}
        self.observation_spaces = {self.agent_id: self.env.observation_space}
        self.action_space = spaces.Dict(self.action_space)
        self.observation_space = spaces.Dict(self.observation_space)
        self.S_t: np.ndarray | None = None
        self._last_reset_seed: int | None = None

    @override(MultiAgentRegulatedEnv)
    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[MultiAgentDict, MultiAgentDict]:
        self._base_reset(seed=seed)

        self._last_reset_seed = seed
        obs, info = self.env.reset(seed=seed, options=options)
        self.S_t = np.asarray(obs, dtype=np.float32)
        observations = {self.agent_id: self.S_t}
        infos = {self.agent_id: info}

        return observations, infos

    def _reset(self) -> MultiAgentDict:
        obs, _ = self.env.reset(seed=self._last_reset_seed)
        self.S_t = np.asarray(obs, dtype=np.float32)

        return {self.agent_id: self.S_t}

    @override(MultiAgentRegulatedEnv)
    def step(self, action_dict: MultiAgentDict):
        action = int(action_dict[self.agent_id])
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.S_t = np.asarray(obs, dtype=np.float32)
        observations = {self.agent_id: self.S_t}
        rewards = {self.agent_id: float(reward)}
        terminateds = {
            self.agent_id: bool(terminated),
            "__all__": bool(terminated),
        }
        truncateds = {
            self.agent_id: bool(truncated),
            "__all__": bool(truncated),
        }
        infos = {self.agent_id: info}

        self._publish(
            EnvStepContext(
                mechanism=self.m_ctx.index if self.m_ctx else None,
                observation=observations,
                observation_map=self.obs_map,
                reward=rewards,
                action={self.agent_id: action},
                info=infos,
            )
        )

        self._t += 1

        return observations, rewards, terminateds, truncateds, infos

    def _is_truncated(self) -> bool:
        # Gym Cartpole truncation is handled by wrapped env
        return False

    def intrinsic_utility(
        self, agent_id: AgentID, action: ActType, S_t: dict[str, MultiAgentDict]
    ) -> SupportsFloat:
        del agent_id, action, S_t

        return 0.0

    def violation_signal(
        self, agent_id: AgentID, u_i: SupportsFloat, S_t: dict[str, MultiAgentDict]
    ) -> SupportsFloat:
        del agent_id, u_i, S_t

        return 0.0

    def penalty(self) -> SupportsFloat:
        return np.array(0.0, dtype=np.float32)

    def transition_kernel(
        self, A_t: MultiAgentEnv, S_t: dict[str, float]
    ) -> dict[str, float]:
        pass  # TODO

    @override(MultiAgentRegulatedEnv)
    def aggregate_rewards(self, rewards: MultiAgentDict) -> MultiAgentDict:
        """Aggregate rewards across agents without scaling or clipping."""

        return rewards

    # TODO canonical observation in base multiagent env
    def _observation(self, agent_id: AgentID, S_t: dict[str, MultiAgentDict]):
        """We assume complete transparency. Observations normalized to [0, 1]."""

        del agent_id

        return np.asarray(S_t, dtype=np.float32)

    def close(self) -> None:
        self.env.close()
