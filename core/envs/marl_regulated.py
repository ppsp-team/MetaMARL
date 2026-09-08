"""Multi-agent environment regulated by a published mechanism.

``MultiAgentRegulatedEnv`` is the RLlib-facing environment of the inner
optimization level. It combines :class:`core.envs.regulated.RegulatedEnv`
(mechanism lifecycle and regulated reward) with RLlib's ``MultiAgentEnv``. A
concrete benchmark implements the abstract pieces of the step:
``transition_kernel`` (``S_{t+1} = T(S_t, A_t)``), ``intrinsic_utility``
(``u_i = U(a_i, S_t)``), ``violation_signal`` and ``penalty`` (the regulated
reward ``u_i - lambda(M) * v_i``), ``_observation`` (``o_i = O_i(S_t)``) and
``_is_truncated``. The base class owns the step lifecycle: it computes the
intrinsic utilities, shapes and aggregates the rewards, advances the state,
appends the mechanism vector ``theta`` to every observation and publishes an
``EnvStepContext`` to the ``World``.

The mechanism in force is fetched from the ``World`` at ``reset`` by
``mechanism_id``. Until one is published the environment returns zero rewards
and does not advance its dynamics, so RLlib's environment checks can run
before training starts.

Metrics are optional: with a ``schema`` the env owns a ``MetricLogger`` fed by
``_log`` (episode identity at ``reset``, rewards and ``iter`` at every step),
and with a ``reporter_cfg`` a ``Reporter`` renders the configured ``queries``
against it (see ``core.callbacks``).
"""

import logging
from abc import abstractmethod
from typing import Any, Optional, SupportsFloat

import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, ObsType
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import AgentID, MultiAgentDict

from core.annotations import override
from core.envs.base import BaseEnv
from core.envs.regulated import RegulatedEnv
from core.world.context import EnvStepContext, MechanismStatus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

# TODO create a reward type

# TODO remove inheritance from regulatedEnv completley
# class SingleAgentBaseEnv(gym.Env): ...
# class MultiAgentBaseEnv(MultiAgentEnv): ...

# class RegulatedEnv(SingleAgentBaseEnv): ...
# class MultiAgentRegulatedEnv(MultiAgentBaseEnv): ...


class MultiAgentRegulatedEnv(RegulatedEnv, MultiAgentEnv):
    """Base class for mechanism-regulated multi-agent benchmarks.

    Parameters
    ----------
    agents : list[AgentID]
        Agent identifiers; ``possible_agents`` is a copy of this list.
    action_spaces, observation_spaces : dict, optional
        Per-agent gymnasium spaces read from ``kwargs`` (forwarded by the
        RLlib env creator); they also build the ``Dict`` spaces of the env.
    **kwargs
        Forwarded to :class:`RegulatedEnv` and :class:`BaseEnv`
        (``mechanism_id``, ``world``, ``mechanism_space``, ``horizon``,
        ``seed``, ``policy_seed``, ``mode``, ``reporter_cfg``, ``queries``,
        ``schema``, ...).
    """

    def __init__(
        self,
        *,
        agents: list[AgentID],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.agents = agents
        self.possible_agents = list(self.agents)
        self.action_spaces = kwargs.get("action_spaces", {})
        self.observation_spaces = kwargs.get("observation_spaces", {})

        # TODO move this to baseenv later
        self.observation_space = spaces.Dict(self.observation_spaces)
        self.action_space = spaces.Dict(self.action_spaces)
        self._infos: MultiAgentDict = {agent_id: {} for agent_id in self.agents}

    def _update_infos(self, key: str, values: MultiAgentDict | SupportsFloat):
        values = (
            values
            if isinstance(values, dict)
            else {agent_id: values for agent_id in self._infos}
        )

        for agent_id, value in values.items():
            self._infos[agent_id][key] = value

    @abstractmethod
    def _reset(self) -> MultiAgentDict:
        raise NotImplementedError

    # TODO options doesnt get consumed
    @override(MultiAgentEnv)
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[MultiAgentDict, MultiAgentDict]:
        """Start a new episode under the mechanism currently in force.

        The step counter restarts (the ``iter`` metric is flushed) and the
        episode identity (``env_id``, ``mechanism_id``, ``seed``,
        ``policy_seed``) is logged. If no published candidate has been assigned
        yet, one is fetched from the ``World`` by ``mechanism_id``; otherwise
        the current mechanism is kept for the whole run. The abstract
        ``_reset`` then produces the initial per-agent observations and the
        per-agent infos are cleared. The seed fixed at construction takes
        precedence over the per-call ``seed``; ``options`` is ignored.

        Returns
        -------
        tuple[MultiAgentDict, MultiAgentDict]
            Per-agent initial observations and per-agent (empty) infos.
        """

        if seed is not None and self.seed is not None and seed != self.seed:
            pass  # do not mutate seed after construction

        # if seed is not None and seed != self.seed:
        #     self.seed = seed
        #     self.rng = np.random.default_rng(seed)
        self._t = 0

        self.logger.flush(key=("iter",))
        self.logger.push(key=("env_id",), value=self.env_id)
        self.logger.push(key=("mechanism_id",), value=self.mechanism_id)
        self.logger.push(key=("seed",), value=self.seed)
        self.logger.push(key=("policy_seed",), value=self.policy_seed)

        # TODO only log iter when a metric is counted, meaning pushing a value with its attached
        # throughput makes logic less fragile
        # self.logger.push(key=("iter",), value=self._t)

        effective_seed = self.seed if self.seed is not None else seed

        self._pre_reset(seed=effective_seed)

        obs = self._reset()
        self._infos = {agent_id: {} for agent_id in self.agents}

        return obs, self._infos

    @override(MultiAgentEnv)
    def step(
        self, action_dict: MultiAgentDict
    ) -> tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        """Run one regulated step of the benchmark.

        Actions go through :meth:`action`; ``_step`` then computes the
        intrinsic utilities with :meth:`intrinsic_utility`, shapes and
        aggregates them with :meth:`reward`, advances ``S_t`` with
        :meth:`transition_kernel` and builds the next observations with
        :meth:`observation`. Rewards are logged per agent and as
        ``reward_mean``, the intrinsic utilities are recorded in the infos as
        ``intrinsic_utility``, and an ``EnvStepContext`` is published to the
        ``World``. Episodes never terminate; ``truncated`` is raised for every
        agent and for ``"__all__"`` when ``_is_truncated`` reports the horizon.

        Before any candidate mechanism has been published (RLlib's environment
        checks), the step is inert: observations of the unchanged state, zero
        rewards, nothing published.

        Returns
        -------
        tuple
            ``(obs, rewards, terminated, truncated, infos)``, each keyed by
            agent ID; ``terminated`` and ``truncated`` also carry ``"__all__"``.
        """

        actions = self.action(action_dict)

        if not self.published_mechanism_assigned:
            obs = {
                agent_id: self.observation(agent_id, self.S_t)
                for agent_id in self.agents
            }
            rewards = {agent_id: 0.0 for agent_id in self.agents}
            terminated = {agent_id: False for agent_id in self.agents}
            terminated["__all__"] = False
            truncated = {agent_id: False for agent_id in self.agents}
            truncated["__all__"] = False
            self._t += 1

            self.logger.push(key=("iter",), value=self._t)

            return obs, rewards, terminated, truncated, self._infos

        obs, rewards, terminated, truncated, self._infos = self._step(actions)
        [
            self.logger.push(key=("by_agent", aid, "reward"), value=r)
            for aid, r in rewards.items()
        ]

        self._aggregate_rewards(rewards)
        self._publish(
            EnvStepContext(
                env_id=self.env_id,
                seed=self.seed,
                policy_seed=self.policy_seed,
                status=MechanismStatus(self.mode),
                mechanism=self.mechanism_id,
                observation=obs,
                observation_map=self.obs_map,
                reward=rewards,
                action=actions,
                info=self._infos,
            )
        )

        self._t += 1

        self.logger.push(key=("iter",), value=self._t)

        return obs, rewards, terminated, truncated, self._infos

    def _step(
        self, action_dict: dict[AgentID, ActType]
    ) -> tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        intrinsic_rewards: MultiAgentDict = self.intrinsic_utility(A_t=action_dict)

        rewards = self.reward(rewards=intrinsic_rewards, A_t=action_dict)
        self.S_t = self.transition_kernel(A_t=action_dict, S_t=self.S_t.copy())
        obs = {
            agent_id: self.observation(agent_id, self.S_t) for agent_id in self.agents
        }
        time_limit = self._is_truncated()
        terminated = {aid: False for aid in self.agents}
        terminated["__all__"] = False
        truncated = {aid: time_limit for aid in self.agents}
        truncated["__all__"] = time_limit

        self._update_infos(key="intrinsic_utility", values=intrinsic_rewards)

        return obs, rewards, terminated, truncated, self._infos

    @abstractmethod
    def transition_kernel(
        self,
        *,
        A_t: MultiAgentDict,
        S_t: dict[str, MultiAgentDict],
        **kwargs: Any,
    ) -> dict[str, float]:
        """S_{t+1} = T(S_t, A_t)"""

        raise NotImplementedError

    @abstractmethod
    def intrinsic_utility(
        self,
        *,
        A_t: MultiAgentDict,
        **kwargs: Any,
    ) -> SupportsFloat:
        """u_i = U(a_i, S_t)"""

        raise NotImplementedError

    @abstractmethod
    def violation_signal(self, u_i: SupportsFloat, **kwargs: Any) -> SupportsFloat:
        """v_i = V(a_i, S_t, M)"""

        raise NotImplementedError

    @abstractmethod
    def penalty(self, u_i: SupportsFloat, **kwargs: Any) -> SupportsFloat:
        """λ = λ(M)"""

        raise NotImplementedError

    @abstractmethod
    def _observation(
        self, agent_id: AgentID, S_t: dict[str, MultiAgentDict]
    ) -> ObsType:
        """o_i = O_i(S_t)"""

        raise NotImplementedError

    @abstractmethod
    def _is_truncated(self) -> bool:
        raise NotImplementedError

    # TODO Restrict Any Type
    @override(BaseEnv)
    def observation(self, agent_id: AgentID, S_t: dict[str, MultiAgentDict]) -> Any:
        """o_i = O_i(S_t, theta)"""

        # TODO may wanna normalize base_obs later
        base_obs = self._observation(agent_id=agent_id, S_t=S_t)
        theta = self.mechanism.to_vector()

        return np.concatenate([base_obs, theta], axis=0)

    def _aggregate_rewards(self, rewards: MultiAgentDict) -> MultiAgentDict:
        mean_reward = float(np.mean(list(rewards.values())))

        return {agent_id: mean_reward for agent_id in self.agents}

    @override(RegulatedEnv)
    def reward(self, rewards: MultiAgentDict, **kwargs) -> SupportsFloat:
        """Regulate each agent's utility as ``u_i - lambda(M) * v_i``, then average.

        ``rewards`` maps agent ID to intrinsic utility; ``kwargs`` (``A_t``)
        are forwarded to :meth:`penalty` and :meth:`violation_signal`. The
        aggregated result gives every agent the mean regulated reward.
        """

        reward_by_agent: MultiAgentDict = {}

        for aid, u_i in rewards.items():
            r = -self.penalty(u_i, **kwargs) * self.violation_signal(u_i, aid, **kwargs)

            self.logger.push(key=("by_agent", aid, "reward"), value=r)

            reward_by_agent[aid] = r

        return self._aggregate_rewards(rewards=reward_by_agent)
