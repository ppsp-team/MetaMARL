"""Single-agent environment base publishing every step to the ``World``.

``BaseEnv`` is a gymnasium ``Env`` implementing the template method: the
public ``step``/``reset`` call the abstract ``_step``/``_reset``/``_pre_reset``
hooks, apply the overridable ``action``/``observation``/``reward`` transforms
and publish an ``EnvStepContext`` to the shared ``World`` actor. The regulated
inner environments (:class:`core.envs.regulated.RegulatedEnv`,
:class:`core.envs.marl_regulated.MultiAgentRegulatedEnv`) and the outer
``RegulatorEnv`` build on it.
"""

from abc import abstractmethod
from typing import Any, Optional, SupportsFloat

import numpy as np
import ray
from gymnasium import Env
from gymnasium.core import ActType, ObsType, WrapperActType, WrapperObsType

from core.annotations import override
from core.mechanism.space import MechanismSpace
from core.metrics.logger import MetricLogger
from core.metrics.schemas import MetricSchema
from core.reporting.base import Reporter
from core.reporting.config import ReporterConfig
from core.reporting.query import Query
from core.types import OptimizerID
from core.world.base import World
from core.world.context import Context, ContextSchema, EnvStepContext, MechanismStatus


class BaseEnv(Env):
    """Base environment that directly interacts with the World.

    Parameters
    ----------
    world : World
        Ray actor handle of the shared blackboard.
    opt_id : OptimizerID, optional
        Identifier of the optimizer owning this env (set on published contexts).
    env_name : str, optional
        Label prefix of the env-level reporter.
    horizon : int, optional
        Episode length in steps.
    mechanism_space : MechanismSpace or type[MechanismSpace], optional
        Geometry of the mechanism manifold (encode/decode, default mechanism).
        A class is instantiated without arguments.
    seed, policy_seed : int, optional
        Environment RNG seed and seed of the associated policy.
    mode : {"train", "eval"}
        Lifecycle status stamped on published contexts.
    reporter_cfg : ReporterConfig, optional
        Builds the env-level ``Reporter``; ``None`` disables reporting.
    queries : tuple[AnyQuery, ...], optional
        Queries rendered by the env-level reporter.
    schema : type[MetricSchema], optional
        Metric schema of the env logger; ``None`` disables logging.
    """

    def __init__(
        self,
        *,
        world: World,
        opt_id: Optional[OptimizerID] = None,
        env_name: Optional[str] = None,
        horizon: Optional[int] = None,
        mechanism_space: MechanismSpace = None,
        seed: Optional[int] = None,
        policy_seed: Optional[int] = None,
        mode: Optional[str] = "train",
        reporter_cfg: Optional[ReporterConfig] = None,
        queries: Optional[tuple[Query]] = None,
        schema: Optional[MetricSchema] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        self.world = world
        self._opt_id = opt_id
        self.horizon = horizon
        self.seed = seed
        self.policy_seed = policy_seed
        self.rng = np.random.default_rng(seed)
        self._t = 0
        self.env_id = None
        self.mode = MechanismStatus(mode)

        # Mechanism space
        if isinstance(mechanism_space, type):
            self.m_space: MechanismSpace = mechanism_space()
        else:
            self.m_space: MechanismSpace = mechanism_space

        # observation map
        self.obs_map: Optional[dict[int, str]] = None

        # logger
        self.logger: Optional[MetricLogger] = (
            MetricLogger.from_schema(schema) if schema else None
        )

        # reporter
        mechanism_id = getattr(self, "mechanism_id", None)
        reporting_env_id = (
            f"{env_name}"
            f"|mode={mode}"
            f"{f'|m={mechanism_id}' if mechanism_id is not None else ''}"
            f"|ps={policy_seed}"
            f"|ss={self.seed}"
        )
        self.reporter: Reporter = reporter_cfg.build(label=reporting_env_id)
        self.reporter.schema = schema

        self.reporter.add_query(*(queries or ()))

    # Setter
    def set_opt_id(self, opt_id: OptimizerID) -> None:
        """Set the optimizer identifier stamped on every context this env publishes."""

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

    @abstractmethod
    def _step(
        self, action: ActType = None
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Run one timestep of the environment's dynamics using the agent actions."""

        raise NotImplementedError

    @abstractmethod
    def _pre_reset(self, seed: Optional[int] = None) -> None:
        pass

    @abstractmethod
    def _reset(self):
        raise NotImplementedError

    @override(Env)
    def step(
        self, action: ActType = None
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Advance the environment by one step and publish the transition.

        The template method applies :meth:`action` to ``action``, runs the
        abstract ``_step``, then applies :meth:`observation` and :meth:`reward`
        to the raw outputs. The transformed observation and reward, together
        with the original ``action`` and ``info``, are published to the
        ``World`` as an ``EnvStepContext`` before the step counter is advanced
        and logged under ``("iter",)``.

        Returns
        -------
        tuple
            ``(obs, reward, terminated, truncated, info)`` in the gymnasium
            convention.
        """

        raw_obs, raw_reward, terminated, truncated, info = self._step(
            self.action(action)
        )
        obs = self.observation(raw_obs)
        reward = self.reward(raw_reward)

        # Publish env context to World
        self._publish(
            EnvStepContext(
                env_id=self.env_id,
                seed=self.seed,
                policy_seed=self.policy_seed,
                status=MechanismStatus(self.mode),
                mechanism=getattr(self, "mechanism_id", None),
                observation=obs,
                observation_map=self.obs_map,
                reward=reward,
                action=action,
                info=info,
            )
        )

        self._t += 1

        if self.logger is not None:
            self.logger.push(key=("iter",), value=self._t)

        return obs, reward, terminated, truncated, info

    @override(Env)
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Start a new episode and publish its initial observation.

        The per-call ``seed`` is ignored: the environment seed is fixed at
        construction and ``_pre_reset`` always receives that one. The step
        counter is reset to zero and logged, the abstract ``_reset`` produces
        the first observation, and an ``EnvStepContext`` with zero reward and
        no action is published to the ``World``. ``options`` is accepted for
        gymnasium compatibility and unused.

        Returns
        -------
        tuple[ObsType, dict]
            The initial observation and an empty info dictionary.
        """

        # Option to pass seed directly to env --> sequential
        # TODO what are the options used for ?

        if seed is not None and self.seed is not None and seed != self.seed:
            pass  # do not mutate seed after construction

        # if seed is not None and and seed != self.seed;
        #     self.seed = seed
        #     self.rng = np.random.default_rng(seed)
        self._t = 0

        self.logger.push(key=("iter",), value=self._t)
        self._pre_reset(seed=self.seed)

        obs = self._reset()

        self._publish(
            EnvStepContext(
                env_id=self.env_id,
                seed=self.seed,
                policy_seed=self.policy_seed,
                status=MechanismStatus(self.mode),
                mechanism=getattr(self, "mechanism_id", None),
                observation=obs,
                observation_map=self.obs_map,
                reward=0.0,
                action=None,
                info={},
            )
        )

        return obs, {}

    def observation(self, observation: ObsType) -> WrapperObsType:
        """Returns a modified observation.

        Args:
            observation: The :attr:`env` observation

        Returns:
            The modified observation
        """

        return observation

    def reward(self, reward: SupportsFloat) -> SupportsFloat:
        """Returns a modified environment ``reward``.

        Args:
            reward: The :attr:`env` :meth:`step` reward

        Returns:
            The modified `reward`
        """

        return reward

    def action(self, action: WrapperActType) -> ActType:
        """Returns a modified action before :meth:`step` is called.

        Args:
            action: The original :meth:`step` actions

        Returns:
            The modified actions
        """

        return action
