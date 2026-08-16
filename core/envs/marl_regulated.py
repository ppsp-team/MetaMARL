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
from typing import Any, ClassVar, Optional, SupportsFloat

import numpy as np

# TODO remove ray and gymnasium dependency
from gymnasium.core import ActType, ObsType
from gymnasium import spaces
import ray
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from core.annotations import override
from core.mechanism.base import Mechanism
from core.types import AgentID, MultiAgentDict
from core.utils import sigmoid
from core.types import OptimizerID
from core.world.base import World
from core.world.context import (
    Context, 
    ContextSchema, 
    EnvStepContext, 
    MechanismStatus,
    MechanismContext
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

# TODO create a reward type

class MultiAgentRegulatedEnv(MultiAgentEnv):
    """Base class for mechanism-regulated multi-agent benchmarks.

    Parameters
    ----------
    world : World
        Ray actor handle of the shared blackboard.
    opt_id : OptimizerID, optional
        Identifier of the optimizer owning this env (set on published contexts).
    horizon : int, optional
        Episode length in steps.
    mechanism_id : str
        Identifier of the candidate mechanism this env instance trains
        against; used to fetch its ``MechanismContext`` from the ``World``.
    seed, policy_seed : int, optional
        Environment RNG seed and seed of the associated policy.
    mode : {"train", "eval"}
        Lifecycle status stamped on published contexts.
    agents : list[AgentID]
        Agent identifiers; ``possible_agents`` is a copy of this list.
    action_spaces, observation_spaces : dict, optional
        Per-agent gymnasium spaces read from ``kwargs`` (forwarded by the
        RLlib env creator); they also build the ``Dict`` spaces of the env.
    **kwargs
        Forwarded to :class:`MultiAgentEnv`.
    """

    _reset: ClassVar[str | None] = None
    _action: ClassVar[str | None] = None
    _reward: ClassVar[str | None] = None
    _observation: ClassVar[str | None] = None
    _transition: ClassVar[str | None] = None

    def __init__(
        self,
        *,
        world: World,
        opt_id: Optional[OptimizerID] = None,
        horizon: Optional[int] = None,
        mechanism_id: str,
        seed: Optional[int] = None,
        policy_seed: Optional[int] = None,
        mode: Optional[str] = "train",
        agents: list[AgentID],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.world = world
        self._opt_id = opt_id

        # training
        self.horizon = horizon
        self._t = 0
        self.env_id = None

        # seeding
        self.seed = seed
        self.policy_seed = policy_seed
        self.rng = np.random.default_rng(seed)
        self.mode = MechanismStatus(mode) # TODO change name to just Status

        # Mechanism
        self.mechanism_id = mechanism_id
        self.m_ctx: MechanismContext = None
        self.m: Mechanism = None
        self._using_default_mechanism = True

        # observation map
        self.obs_map: Optional[dict[int, str]] = None
        self.observation_spaces = kwargs.get("observation_spaces", {})
        self.observation_space = spaces.Dict(self.observation_spaces)

        # Multi-agent environment
        self.agents = agents
        self.possible_agents = list(self.agents)
        self.action_spaces = kwargs.get("action_spaces", {})
        self.action_space = spaces.Dict(self.action_spaces)
        self._infos : MultiAgentDict = {agent_id: {} for agent_id in self.agents}

    def __init_subclass__(
            cls,
            **kwargs,
        ):
        super().__init_subclass__(**kwargs)

        for name, func in cls.__dict__.items():
            if getattr(func, "reset", False): cls._reset = name
            if getattr(func, "action", False): cls._action = name
            if getattr(func, "reward", False): cls._reward = name
            if getattr(func, "observation", False): cls._observation = name  # o_i = O_i(S_t)
            if getattr(func, "transition", False): cls._transition = name  # S_{t+1} = T(S_t, A_t)

    @property
    def mechanism(self) -> Mechanism:
        if self.m is not None:
            return self.m
        return self.m_space.default()
    
    @property
    def published_mechanism_assigned(self) -> bool:
        return self.m is not None and not self._using_default_mechanism

    # Setter
    def set_opt_id(self, opt_id: OptimizerID) -> None:
        self._opt_id = opt_id

    # private methods
    def _publish(self, payload: ContextSchema):
        ctx = Context(
            id=None,
            opt_id=self._opt_id,
            step=self._t,
            env=self.__class__.__name__,
            payload=payload,
        )
        ray.get(self.world.append_context.remote(ctx))


    def _update_infos(self, key: str, values: MultiAgentDict | SupportsFloat):
        values = (
            values
            if isinstance(values, dict)
            else {agent_id: values for agent_id in self._infos}
        )

        for agent_id, value in values.items():
            self._infos[agent_id][key] = value

    def _normalize_action(
        self,
        action: ActType,
    ) -> np.ndarray:
        z = np.asarray(action, dtype=np.float32).reshape(-1)
        temperature = 4.0
        return np.asarray([sigmoid(float(value) / temperature) for value in z], dtype=np.float32)
    

    # TODO options doesnt get consumed
    @override(MultiAgentEnv)
    def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[dict[str, Any]] = None,
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
            pass # do not mutate seed after construction
        self._t = 0

        self.logger.flush(key=("iter",))
        self.logger.push(key=("env_id",), value=self.env_id)
        self.logger.push(key=("mechanism_id",), value=self.mechanism_id)
        self.logger.push(key=("seed",), value=self.seed)
        self.logger.push(key=("policy_seed",), value=self.policy_seed)

        # Try to fetch a new mechanism if one is available (published)
        # Otherwise keep the current mechanism for subsequent episodes
        if self.mechanism_id is None:
            raise RuntimeError(
                "RegulatedEnv has no mechanism_id. "
                "mechanism_id must be injected at env creation."
            )

        if not self.published_mechanism_assigned: 
            try:
                new_ctx = ray.get(
                    self.world.get_mechanism_by_id.remote(
                        mechanism_id = self.mechanism_id, 
                        seed=self.policy_seed,
                        mode=self.mode
                    )
                )
            except Exception as e:
                self._debug_remote(
                    "pre_reset_fetch_failed",
                    {
                        "error_type": type(e).__name__,
                        "error_repr": repr(e),
                    },
                )
                raise RuntimeError(
                    f"Could not fetch mechanism_id={self.mechanism_id} from World."
                ) from e

            if new_ctx is not None:
                self.m_ctx = new_ctx
                self.m = self.m_ctx.mechanism
                self._using_default_mechanism = False

            # TODO raising error if training started and default mechanism is still on - leads to silent error

        # Optional benchmark Reset hook to initialize state and add to observation
        if self._reset is not None:
            self.S_t = getattr(self, self._reset)()
        self._infos = {agent_id: {} for agent_id in self.agents}

        # Start with an empty observation dictionary.
        # The benchmark hook may construct or transform it.
        obs = self.observation({})
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
        # Policy outputs -> normalized semantic actions -> mechanisms.
        actions = self.action(action_dict)

        # Benchmark intrinsic utility using current state + delivered actions.
        if self._reward is not None:
            intrinsic_rewards = getattr(self, self._reward)(actions)
        else:
            intrinsic_rewards = {agent_id: 0.0 for agent_id in self.agents}

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

        # Dynamics
        S_t = self.S_t.copy()

        if self._transition is not None:
            self.S_t = getattr(self, self._transition)(A_t=actions, S_t=S_t.copy())

        rewards = self.mechanism.reward(intrinsic_rewards, env=self, action_after=actions)
        [
            self.logger.push(key=("by_agent", aid, "reward"), value=r)
            for aid, r in rewards.items()
        ]

        obs = self.observation({})

        time_limit = self.horizon is not None and (self._t + 1) >= self.horizon

        terminated = {aid: False for aid in self.agents}
        terminated["__all__"] = False

        truncated = {aid: time_limit for aid in self.agents}
        truncated["__all__"] = time_limit

        self._update_infos(key="intrinsic_utility", values=intrinsic_rewards)

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

    @override(MultiAgentEnv)
    def action(
        self,
        action_dict: MultiAgentDict,
    ) -> MultiAgentDict:
        # apply normalization to action components
        # TODO provide wrapper hooks for benchmark envs
        action_dict = {aid: self._normalize_action(action) for aid, action in action_dict.items()}

        if self._action is not None:
            action_dict = getattr(self, self._action)(action_dict)

        # TODO choose the action components
        return self.mechanism.action(action_dict, env=self)

    @override(MultiAgentEnv)
    def reward(
        self,
        reward_dict: MultiAgentDict,
    ) -> MultiAgentDict:
        if self._reward is not None:
            reward_dict = getattr(self, self._reward)(reward_dict)

        # TODO provide wrapper hooks for benchmark envs
        return self.mechanism.action(
            reward_dict,
            env=self,
        )

    @override(MultiAgentEnv)
    def observation(
        self,
        observation_dict: MultiAgentDict,
    ) -> MultiAgentDict:
        """o_i = O_i(S_t, theta)"""
        if self._observation is not None:
            observation_dict = getattr(
                self,
                self._observation,
            )(observation_dict)

        # move this to mechanism
        theta = self.mechanism.to_vector()
        obs_with_theta = np.concatenate([observation_dict, theta], axis=0)

        # TODO provide wrapper hooks for benchmark envs
        return self.mechanism.action(
            obs_with_theta,
            env=self,
        )

    # TODO these could be decorators for sub_envs!
    # TODO : unnecessary ?
    # @abstractmethod
    # def _step(
    #     self, action: ActType = None
    # ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
    #     """Run one timestep of the environment's dynamics using the agent actions."""
    #     raise NotImplementedError
