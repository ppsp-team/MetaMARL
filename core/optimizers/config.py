"""Base configuration objects for optimizers.

``OptimizerConfig`` follows the builder pattern of RLlib's ``AlgorithmConfig``:
a mutable object configured through chained method calls (``environment``,
``debugging``, ``training`` ...), then frozen and turned into an ``Optimizer``
by ``build_optimizer``. Concrete configs (``RayOptimizerConfig``, ``ESConfig``)
extend it with backend-specific builders.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Self, Type, Union

import numpy as np
import ray
from gymnasium import Space
from ray.actor import ActorHandle
from ray.rllib.utils.metrics.metrics_logger import DEFAULT_STATS_CLS_LOOKUP

from core.envs.base import BaseEnv
from core.metrics.schemas import MetricSchema
from core.reporting.config import ReporterConfig
from core.reporting.query import Query
from core.types import EnvConfigDict, EnvType
from core.world.base import World

if TYPE_CHECKING:
    from core.optimizers.base import Optimizer


class _Config(ABC):
    """Minimal interface shared by all configuration objects."""

    def to_dict(self) -> dict:
        """Converts this configuration to dict format."""

        raise NotImplementedError


class OptimizerConfig(_Config, ABC):
    """Fluent, freezable configuration from which an ``Optimizer`` is built.

    Contract
    --------
    - *Fluent*: builder methods mutate ``self`` and return it, so calls
      chain. Subclasses implement ``training`` (abstract) and may add more.
    - *Freeze*: ``freeze()`` flips ``_is_frozen``; afterwards any attribute
      assignment raises ``AttributeError`` (enforced in ``__setattr__``).
      Freezing is shallow: nested objects such as ``env_config`` stay
      mutable.
    - *Copy*: ``copy(copy_frozen=...)`` deep-copies the config and sets the
      frozen flag of the copy. ``build_optimizer`` always works on
      ``self.copy(copy_frozen=True)``, so the original stays editable and the
      optimizer owns an immutable snapshot.

    Parameters
    ----------
    opt_class : type[Optimizer], optional
        Optimizer class instantiated by ``build_optimizer``.

    Attributes
    ----------
    env : str or EnvType or None
        Environment class (or registered name) set by ``environment``.
    env_config : dict
        Keyword arguments passed to the environment constructor.
    horizon : int or None
        Episode length; also copied into ``env_config["horizon"]``.
    base_seed : int or None
        Root seed given to ``debugging``.
    seeds : list of int
        Training seeds derived from ``base_seed`` (empty when unseeded).
    eval_seeds : list of int or None
        Evaluation seeds; set by subclasses (see
        ``RayOptimizerConfig.evaluation``).
    evaluation_config : OptimizerConfig or None
        Optional nested evaluation config; ``copy(copy_frozen=False)`` also
        unfreezes it.
    stats_cls_lookup : dict
        RLlib ``Stats`` class lookup handed to the optimizer's
        ``MetricsLogger``.
    """

    # TODO registry to allow opt_class str
    # TODO runtime checking of opt_class
    def __init__(self, opt_class: Optional[Type[Optimizer]] = None):
        """Initializes an OptimizerConfig instance.

        Args:
            optimizer_class: An optional Optimizer class that this config class belongs to.
                Used (if provided) to build a respective Optimizer instance from this
                config.
        """

        self.opt_class = opt_class

        # -- lifecycle --
        self._is_frozen = False

        # --- world / environment ---
        self.env: Optional[Union[str, EnvType]] = None
        self.env_config: dict = {}
        self.horizon: int = None  # TODO default value

        # --- debugging ---
        self.base_seed: Optional[int] = None
        self.seeds: list[int] = []

        # --- eval ---
        self.evaluation_config: Optional["OptimizerConfig"] = None
        self.eval_seeds: Optional[list[int]] = None

        # TODO
        # --- reporting ---
        self.stats_cls_lookup = DEFAULT_STATS_CLS_LOOKUP
        self._reporter_cfg: Optional[ReporterConfig] = None
        self._reporting_schema: Optional[type[MetricSchema]] = None
        self._reporting_queries: Optional[tuple[Query, ...]] = None
        self._reporting_schema_env: Optional[type[MetricSchema]] = None
        self._reporting_queries_env: Optional[tuple[Query]] = None

    @property
    def reporter_cfg(self) -> Optional[ReporterConfig]:
        """Reporter configuration used by ``_build_reporter``; ``None`` disables reporting."""

        return self._reporter_cfg

    @reporter_cfg.setter
    def reporter_cfg(self, reporter_cfg: ReporterConfig) -> None:
        """Attach a reporter configuration (``BilevelConfig`` copies one to each level)."""

        self._reporter_cfg = reporter_cfg

    def __setattr__(self, name, value):
        if hasattr(self, "_is_frozen") and self._is_frozen:
            if name not in ["_is_frozen"]:
                raise AttributeError(
                    f"Cannot set attribute ({name}) of an already frozen "
                    "OptimizerConfig!"
                )

        super().__setattr__(name, value)

    # TODO generalize this function
    def _merge_env_config(self, extra: dict) -> Self:
        """Shallow-merge ``extra`` into ``env_config`` (``extra`` wins).

        Note that ``environment`` resets ``env_config`` to ``{}`` before
        filling it, so merges done before that call are lost.

        Returns
        -------
        OptimizerConfig
            ``self`` for chaining.
        """

        self.env_config = {
            **(self.env_config or {}),
            **extra,
        }

        return self

    # TODO freezing for nested configs
    def freeze(self) -> None:
        """Freeze this config object, such that no attributes can be set anymore.

        Optimizers should use this method to make sure their config objects
        remain read-only after this.
        """

        if self._is_frozen:
            return

        self._is_frozen = True

    def copy(self, copy_frozen: Optional[bool] = None) -> Self:
        """Creates a deep copy of this config and (un)freezes if necessary.

        Args:
            copy_frozen: Whether the created deep copy is frozen or not, If None,
                keep the same frozen status that 'self' currently has.

        Returns:
            A deep copy of 'self' that is (un)frozen.
        """

        cp = copy.deepcopy(self)

        if copy_frozen is True:
            cp.freeze()
        elif copy_frozen is False:
            cp._is_frozen = False

            if isinstance(cp.evaluation_config, OptimizerConfig):
                cp.evaluation_config._is_frozen = False

        return cp

    # TODO review this
    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Serialization from dict"""

        cfg = cls()

        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

        return cfg

    # TODO review this
    @classmethod
    def from_yaml(cls, path: str) -> Self:
        """Serialization from yaml"""

        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    def _env_creator(
        self,
        **env_ctx,
    ) -> BaseEnv:
        """Instantiate ``self.env`` with the given keyword arguments.

        Parameters
        ----------
        **env_ctx
            Constructor arguments: ``build_optimizer`` passes ``world``,
            ``opt_id``, ``optimizer`` and the ``env_config`` entries;
            ``RayOptimizerConfig`` adds ``agents``, ``mechanism_id``, ``seed``,
            ``policy_seed`` and ``mode``.

        Returns
        -------
        BaseEnv
            A new environment instance. ``self.env`` must be a callable class;
            string specifiers are not resolved here.
        """

        return self.env(**env_ctx)

    # TODO deep copy allows on may be toggled later with use_copy
    # TODO build_optimizer() to accept logger_creator: Optional[Callable[[], Logger]] = None,
    # TODO move optimizer registration to executor in future
    # TODO enable multiple world registration
    def build_optimizer(
        self,
        *,
        world: Optional[ActorHandle[World]] = None,
        inner_opt: Optional[Optimizer] = None,
        **kwargs: Any,
    ) -> Optimizer:
        """Build an ``Optimizer`` from a frozen deep copy of this config.

        The optimizer is registered with ``world`` (when given) to obtain its
        ``opt_id``, receives the optimizer-level reporter, and its environment
        is instantiated once through ``_env_creator`` with ``world``,
        ``opt_id``, ``inner_opt`` and the ``env_config`` entries. Extra
        keyword arguments are accepted for subclass compatibility and ignored.
        """

        cfg = self.copy(copy_frozen=True)

        if cfg.opt_class is None:
            raise ValueError("OptimizerConfig has no opt_class")

        # TODO remove this in the future and create registry for world and optimizer. keep for now as safety guard
        opt: Optimizer = cfg.opt_class(config=cfg)

        opt.world = world

        # Build reporter
        reporter = self._reporter_cfg.build(label=self.opt_class.__name__)
        reporter.schema = self._reporting_schema

        reporter.add_query(*(self._reporting_queries or ()))

        opt.reporting = reporter

        # register optimizer in world to link contexts to optimizers
        opt_id = None

        if world is not None:
            opt_id = ray.get(world._set_new_opt_id.remote(opt_id=opt.opt_id))

            opt.set_id(opt_id)

        env = cfg._env_creator(
            world=world,
            opt_id=opt_id,
            optimizer=inner_opt,
            reporter_cfg=cfg.reporter_cfg.copy()
            if cfg.reporter_cfg is not None
            else None,
            queries=cfg._reporting_queries_env,
            schema=cfg._reporting_schema_env,
            **self.env_config,
        )
        opt.env = env

        return opt

    # TODO EnvConfigDict
    def environment(
        self,
        env: Optional[Union[str, EnvType]] = None,
        train_iters: Optional[int] = None,
        horizon: Optional[int] = None,
        queries: Optional[tuple[Query]] = None,
        schema: Optional[type[MetricSchema]] = None,
        *,
        env_config: Optional[EnvConfigDict] = None,
        observation_space: Optional[Space] = None,
        action_space: Optional[Space] = None,
        disable_env_checking: Optional[bool] = None,
    ) -> Self:
        """Set the environment class and the keyword arguments it is built with.

        Calling this resets ``env_config`` to an empty dict before filling it,
        so ``_merge_env_config`` calls made earlier are lost.

        Parameters
        ----------
        env : str or EnvType, optional
            Environment class instantiated by ``_env_creator`` (string
            specifiers are stored but not resolved by the base class).
        train_iters : int, optional
            Stored in ``env_config["train_iters"]``; the regulator environment
            reads it as the number of inner training iterations per candidate.
        horizon : int, optional
            Episode length in steps, stored in ``env_config["horizon"]``.
        queries : tuple of Query, optional
            Env-level queries rendered by the environment's own reporter.
        schema : type[MetricSchema], optional
            Metric schema the environment logs into.
        env_config : dict, optional
            Extra constructor arguments merged into ``env_config``.
        observation_space, action_space : gymnasium.Space, optional
            Stored in ``env_config`` for environments that take them.
        disable_env_checking : bool, optional
            Stored as an attribute; RLlib-backed subclasses forward it.

        Returns
        -------
        OptimizerConfig
            ``self`` for chaining.
        """

        self.env_config: dict = {}

        if env is not None:
            self.env = env

        if train_iters is not None:
            self.env_config.update({"train_iters": train_iters})

        if observation_space is not None:
            self.env_config.update({"observation_space": observation_space})

        if action_space is not None:
            self.env_config.update({"action_space": action_space})

        if horizon is not None:
            self.env_config.update({"horizon": horizon})

        if env_config is not None:
            self.env_config.update(env_config)

        if disable_env_checking is not None:
            self.disable_env_checking = disable_env_checking

        if queries is not None:
            self._reporting_queries_env = tuple(queries)

        if schema is not None:
            self._reporting_schema_env = schema

        return self

    @abstractmethod
    def training(self) -> Self:
        """Set the optimizer's training hyperparameters (backend specific).

        Subclasses define the accepted keyword arguments and return ``self``.
        """

        raise NotImplementedError

    def debugging(
        self,
        *,
        seed: Optional[int] = None,  # base seed
        num_seeds: int = 3,
    ) -> Self:
        """Derive the training seeds from a base seed.

        Parameters
        ----------
        seed : int or None, optional
            Base seed stored in ``base_seed``. A
            ``numpy.random.SeedSequence(seed)`` is created and
            ``generate_state(num_seeds)`` yields ``num_seeds`` independent
            32-bit integers, stored as Python ints in ``seeds``. The same base
            seed always yields the same list, and different base seeds give
            well-separated streams (the SeedSequence hashing spreads them).
            ``None`` clears ``seeds`` to ``[]``.
        num_seeds : int, optional
            Number of training seeds to derive (default 3).

        Returns
        -------
        OptimizerConfig
            ``self`` for chaining.
        """

        if seed is not None:
            self.base_seed = seed
            ss = np.random.SeedSequence(seed)
            self.seeds = ss.generate_state(num_seeds).tolist()
        else:
            self.seeds = []

        return self

    def reporting(
        self,
        queries: Optional[tuple[Query]],
        schema: Optional[type[MetricSchema]] = None,
    ) -> Self:
        """Declare the optimizer-level metric schema and the queries to render from it."""

        if schema is not None:
            self._reporting_schema = schema

        if queries is not None:
            self._reporting_queries = tuple(queries)

        return self

    # TODO Docstring explanation
    # @abstractmethod
    # def ressources(self):
    #     raise NotImplementedError

    # # TODO Docstring explanation
    # @abstractmethod
    # def evaluation(self):
    #     raise NotImplementedError

    # # TODO Docstring explanation
    # @abstractmethod
    # def reporting(self):
    #     raise NotImplementedError

    # # TODO Docstring explanation
    # @abstractmethod
    # def checkpointing(self):
    #     raise NotImplementedError

    # # TODO Docstring explanation
    # @abstractmethod
    # def fault_tolerance(self):
    #     raise NotImplementedError

    # # TODO Docstring explanation
    # @abstractmethod
    # def experimental(self):
    #     raise NotImplementedError
