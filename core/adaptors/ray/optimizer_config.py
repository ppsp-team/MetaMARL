"""Fluent configuration for an RLlib-backed inner optimizer.

``RayOptimizerConfig`` mirrors the builder methods of RLlib's
``AlgorithmConfig`` (``training``, ``env_runners``, ``evaluation`` ...) but does
not touch an ``AlgorithmConfig`` while the user is chaining calls. Each builder
decorated with ``rllib_config_mutator`` only records an ``RLlibConfigOp`` (the
target function plus its arguments) in ``_cfg_ops``, keyed by method name. The
ops are replayed, in insertion order, on ``algo_class.get_default_config()`` the
first time ``build_optimizer`` runs. This deferral lets later calls such as
``debugging`` or ``build_optimizer`` patch the recorded keyword arguments of an
earlier call (for example scaling ``num_envs_per_env_runner`` by the number of
seeds) before RLlib ever validates them.

Policies are laid out as one RLModule per ``(mechanism candidate, training
seed)``, named ``<base_policy>_m<mechanism_idx>_s<seed>``. Environments are
laid out to match: inside one env runner, environment ``k`` runs mechanism
``k % num_mechanisms`` with seed index ``k // num_mechanisms``. The link between
an episode and its RLModule is carried in the episode ID, which the
``tag_episode_with_env_idx`` callback rewrites as
``env=<idx>|m=<mechanism>|ps=<policy_seed>|ss=<env_seed>|raw=<id>``;
``policy_mapping_fn`` parses that string (``_parse_episode_identity``) to pick
the module. During evaluation the same parsing applies, but the environments
are created with ``env_seed`` drawn from the evaluation seeds while
``policy_seed`` still names the trained module to test.
"""

import uuid
from dataclasses import dataclass
from typing import Any, Callable, Concatenate, Optional, ParamSpec, Self, TypeAlias

import numpy as np
import ray
import torch
from gymnasium import Space
from ray.actor import ActorHandle
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.env.multi_agent_episode import MultiAgentEpisode
from ray.rllib.utils.typing import AgentID
from ray.tune.registry import register_env

from core.adaptors.ray.optimizer import RayOptimizer
from core.annotations import override
from core.callbacks import _evaluate_with_fixed_duration_once
from core.metrics.schemas import MetricSchema
from core.optimizers.config import OptimizerConfig
from core.reporting.query import Query
from core.utils import generate_uuid
from core.world.base import World

# TODO override environment to attach docstrings
P = ParamSpec("P")


@dataclass
class AgentSpec:
    """Declaration of one agent type in a multi-agent environment.

    Attributes
    ----------
    count : int
        Number of agent instances of this type; instances are named
        ``"<agent_type>:<i>"``.
    policy : str
        Base policy name. The actual RLModule IDs are derived from it as
        ``<policy>_m<mechanism_idx>_s<seed>``.
    observation_space, action_space : gymnasium.Space
        Spaces shared by all instances of the type.

    Notes
    -----
    ``RayOptimizerConfig.agents`` and ``_apply_agents_to_rllib`` currently
    read the specs as plain dicts (``spec["count"]``, ``spec.get(...)``), so
    callers pass dicts with these keys rather than ``AgentSpec`` instances.
    """

    count: int
    policy: str
    observation_space: Space
    action_space: Space


FnID: TypeAlias = str


@dataclass
class RLlibConfigOp:
    """Deferred call ``fn(cfg, *args, **kwargs)`` on an ``AlgorithmConfig``.

    Attributes
    ----------
    fn : Callable[..., AlgorithmConfig]
        Function taking the config as first argument and returning the
        updated config.
    args : tuple
        Positional arguments recorded at builder-call time.
    kwargs : dict
        Keyword arguments recorded at builder-call time. Mutable on purpose:
        ``debugging`` and ``build_optimizer`` edit them before replay.
    """

    fn: Callable[..., AlgorithmConfig]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]

    def __call__(self, cfg: AlgorithmConfig) -> AlgorithmConfig:
        return self.fn(cfg, *self.args, **self.kwargs)


class RayOptimizerConfig(OptimizerConfig):
    """``OptimizerConfig`` that assembles an RLlib ``AlgorithmConfig`` lazily.

    Subclasses must set ``algo_class`` (e.g. ``APPO``); its
    ``get_default_config()`` is the base on which the recorded ops are
    replayed. See the module docstring for the deferral mechanism and the
    policy/environment layout.

    Attributes
    ----------
    _cfg_ops : dict[str, RLlibConfigOp]
        Recorded builder calls keyed by method name; one entry per method, a
        second call to the same builder overwrites the first.
    rllib_cfg : AlgorithmConfig or None
        Resolved RLlib config; ``None`` until ``build_optimizer`` replays the
        ops, and cached afterwards.
    agent_specs : dict or None
        Agent declarations set through ``agents``.
    num_mechanisms : int or None
        Mechanism candidates per env runner (before seed multiplication).
    world_name : str or None
        Name of the World actor, stored by ``build_optimizer``.
    eval_episodes, rollout_fragment_length : int or None
        Reserved; not set by the current builders.
    """

    # TODO review this
    # must be overriden in subclasses
    algo_class: Optional[type[Algorithm]] = None

    def __init__(self):
        if self.algo_class is None:
            raise ValueError(f"{self.__class__.__name__} must define `algo_class`")

        super().__init__(opt_class=RayOptimizer)

        # TODO termporary setting until find out how to share world context accross runners
        self._cfg_ops: dict[FnID, RLlibConfigOp] = {}
        self.rllib_cfg: AlgorithmConfig | None = None
        self.agent_specs: Optional[dict] = None  # TODO default
        self.world_name: Optional[str] = None
        self.num_mechanisms: Optional[int] = None
        self.eval_episodes: Optional[int] = None
        self.rollout_fragment_length: Optional[int] = None

        # self._result_mapper: ResultMapper = None

    # TODO let mutator accept an explicit ID
    def rllib_config_mutator(
        fn: Callable[Concatenate[AlgorithmConfig, P], AlgorithmConfig],
    ) -> Callable[Concatenate[Self, P], Self]:
        """Turn a builder into a recorder of a deferred ``RLlibConfigOp``.

        The decorated function has the signature ``fn(cfg, *args, **kwargs)``
        and is *not* executed when the builder is called. Instead the call is
        stored in ``self._cfg_ops[fn.__name__]`` and ``self`` is returned so
        that builders chain. Defined inside the class body without
        ``@staticmethod``; it works as a decorator at class-definition time
        but is also exposed as an (unusable) instance method.
        """

        def wrapper(self: Self, *args: P.args, **kwargs: P.kwargs) -> Self:
            """Record the call and return ``self`` for chaining."""

            self._cfg_ops[fn.__name__] = RLlibConfigOp(fn=fn, args=args, kwargs=kwargs)

            return self

        return wrapper

    @rllib_config_mutator
    def validate(cfg: AlgorithmConfig, **kwargs: Any) -> AlgorithmConfig:
        """Deferred ``AlgorithmConfig.validate``."""

        return cfg.validate(**kwargs)

    @rllib_config_mutator
    def get_config_for_module(cfg: AlgorithmConfig, **kwargs: Any) -> AlgorithmConfig:
        """Deferred ``AlgorithmConfig.get_config_for_module``."""

        return cfg.get_config_for_module(**kwargs)

    @rllib_config_mutator
    def python_environment(cfg: AlgorithmConfig, **kwargs: Any) -> AlgorithmConfig:
        """Sets the config's python environment settings.

        Args:
            extra_python_environs_for_driver: Any extra python env vars to set in the
                algorithm's process, e.g., {"OMP_NUM_THREADS": "16"}.
            extra_python_environs_for_worker: The extra python environments need to set
                for worker processes.
        """

        return cfg.python_environment(**kwargs)

    @rllib_config_mutator
    def resources(cfg: AlgorithmConfig, **kwargs: Any) -> AlgorithmConfig:
        """Specifies resources allocated for an Algorithm and its ray actors/workers."""

        return cfg.resources(**kwargs)

    @rllib_config_mutator
    def framework(cfg: AlgorithmConfig, **kwargs: Any) -> AlgorithmConfig:
        """Sets the config's DL framework settings."""

        return cfg.framework(**kwargs)

    @rllib_config_mutator
    def api_stack(cfg: AlgorithmConfig, **kwargs: Any) -> AlgorithmConfig:
        """Sets the config's API stack settings."""

        return cfg.api_stack(**kwargs)

    def model(self, **kwargs: Any) -> Self:
        """Record model keyword arguments to merge into ``cfg.model`` at build time."""

        def _set_model(cfg):
            """Merge the recorded kwargs into ``cfg.model``."""

            cfg.model.update(kwargs)

            return cfg

        self._cfg_ops["model"] = RLlibConfigOp(fn=_set_model, args=(), kwargs={})

        return self

    @rllib_config_mutator
    def _env_runners(cfg, **kwargs: Any) -> AlgorithmConfig:
        """Sets the rollout worker configuration."""

        return cfg.env_runners(**kwargs)

    def env_runners(self, **kwargs: Any) -> Self:
        """Deferred ``AlgorithmConfig.env_runners`` that records the env count.

        ``num_envs_per_env_runner`` is read as the number of mechanism
        candidates evaluated per runner and stored in ``num_mechanisms``.
        ``debugging`` later multiplies the recorded kwarg by the number of
        training seeds, so this builder must be called before ``debugging``
        for the multiplication to happen.
        """

        self.num_mechanisms = kwargs.get("num_envs_per_env_runner", 1)

        return self._env_runners(**kwargs)

    @rllib_config_mutator
    def learners(cfg: AlgorithmConfig, **kwargs: Any) -> AlgorithmConfig:
        """Sets LearnerGroup and Learner worker related configurations."""

        return cfg.learners(**kwargs)

    @rllib_config_mutator
    def callbacks(cfg: AlgorithmConfig, **kwargs: Any) -> AlgorithmConfig:
        """Deferred ``AlgorithmConfig.callbacks`` (episode and training hooks)."""

        return cfg.callbacks(**kwargs)

    # def evaluation(self, *, episodes: int = None, rollout_fragment_length: int, base_seed: Optional[int]=None, **kwargs) -> Self:
    #     if episodes is not None:
    #         self.eval_episodes = episodes
    #     if base_seed is not None:
    #         self.eval_base_seed = base_seed
    #     # TODO to infer from horizon
    #     if rollout_fragment_length is not None:
    #         self.rollout_fragment_length = rollout_fragment_length
    #     return self

    @rllib_config_mutator
    def _evaluation_rllib(cfg, **kwargs: Any) -> AlgorithmConfig:
        """Deferred ``AlgorithmConfig.evaluation`` (raw pass-through)."""

        return cfg.evaluation(**kwargs)

    def evaluation(
        self,
        base_seed: Optional[int] = None,
        num_seeds: Optional[int] = None,
        seeds: Optional[list[int]] = None,
        **kwargs: Any,
    ) -> Self:
        """Configure evaluation seeds and record the RLlib evaluation call.

        Parameters
        ----------
        base_seed : int or None, optional
            Root of a ``numpy.random.SeedSequence``; ``num_seeds`` children are
            spawned and their first ``uint32`` state word becomes an
            evaluation seed.
        num_seeds : int or None, optional
            Number of evaluation seeds to derive from ``base_seed`` (default
            1). Requires ``base_seed`` unless ``seeds`` is given.
        seeds : list of int or None, optional
            Explicit evaluation seeds; takes precedence over ``base_seed``.
        **kwargs
            Forwarded to ``AlgorithmConfig.evaluation``. The
            ``evaluation_config.env_config`` sub-dict is created if needed and
            gets ``mode="eval"`` so ``env_creator`` builds eval environments;
            ``evaluation_interval`` is forced to ``None`` and
            ``evaluation_parallel_to_training`` to ``False`` so evaluation
            only happens on explicit ``evaluate()`` calls.

        Returns
        -------
        RayOptimizerConfig
            ``self`` for chaining.

        Raises
        ------
        ValueError
            If ``seeds`` is empty, or ``num_seeds`` is given without
            ``base_seed`` or ``seeds``.

        Notes
        -----
        ``build_optimizer`` later overrides ``evaluation_num_env_runners``,
        ``evaluation_duration`` and ``custom_evaluation_function`` in the
        recorded kwargs; values passed here for those keys are replaced.
        """

        eval_config: dict = kwargs.setdefault("evaluation_config", {})
        eval_env_config: dict = eval_config.setdefault("env_config", {})

        eval_env_config["mode"] = "eval"

        if seeds is not None:
            if len(seeds) == 0:
                raise ValueError("`seeds` must contain at least one seed.")

            self.eval_seeds = [int(seed) for seed in seeds]
        elif base_seed is not None:
            num_seeds = num_seeds or 1
            seed_sequence = np.random.SeedSequence(base_seed)
            self.eval_seeds = [
                int(child.generate_state(1, dtype=np.uint32)[0])
                for child in seed_sequence.spawn(num_seeds)
            ]
        elif num_seeds is not None:
            raise ValueError(
                "`num_seeds` requires `base_seed`, unless explicit `seeds` are provided."
            )

        kwargs["evaluation_interval"] = None
        kwargs["evaluation_parallel_to_training"] = False

        return self._evaluation_rllib(**kwargs)

    @rllib_config_mutator
    def offline_data(cfg: AlgorithmConfig, **kwargs: Any) -> AlgorithmConfig:
        """Deferred ``AlgorithmConfig.offline_data``."""

        return cfg.offline_data(**kwargs)

    @rllib_config_mutator
    def multi_agent(cfg: AlgorithmConfig, **kwargs: Any) -> AlgorithmConfig:
        """Sets the config's multi-agent settings."""

        return cfg.multi_agent(**kwargs)

    @rllib_config_mutator
    def _reporting_rllib(cfg, **kwargs: Any) -> AlgorithmConfig:
        """Deferred ``AlgorithmConfig.reporting`` (raw pass-through)."""

        return cfg.reporting(**kwargs)

    @override(OptimizerConfig)
    def reporting(
        self,
        queries: Optional[tuple[Query, ...]],
        schema: Optional[type[MetricSchema]],
        **kwargs: Any,
    ) -> Self:
        """Declare the optimizer-level metrics and record the RLlib reporting call.

        Parameters
        ----------
        queries : tuple of Query or None
            Queries rendered by the optimizer-level reporter, stored through
            ``OptimizerConfig.reporting``.
        schema : type[MetricSchema] or None
            Metric schema attached to the optimizer-level reporter.
        **kwargs
            Forwarded to the deferred ``AlgorithmConfig.reporting`` (RLlib's
            own reporting knobs such as ``metrics_num_episodes_for_smoothing``).

        Returns
        -------
        RayOptimizerConfig
            ``self`` for chaining.
        """

        super().reporting(queries=queries, schema=schema)

        return self._reporting_rllib(**kwargs)

    @rllib_config_mutator
    def checkpointing(cfg: AlgorithmConfig, **kwargs: Any) -> AlgorithmConfig:
        """Deferred ``AlgorithmConfig.checkpointing``."""

        return cfg.checkpointing(**kwargs)

    @rllib_config_mutator
    def fault_tolerance(cfg: AlgorithmConfig, **kwargs: Any) -> AlgorithmConfig:
        """Deferred ``AlgorithmConfig.fault_tolerance`` (worker restart policy)."""

        return cfg.fault_tolerance(**kwargs)

    @rllib_config_mutator
    def rl_module(cfg: AlgorithmConfig, **kwargs: Any) -> AlgorithmConfig:
        """Deferred ``AlgorithmConfig.rl_module`` (RLModule spec and model settings)."""

        return cfg.rl_module(**kwargs)

    @rllib_config_mutator
    def experimental(cfg: AlgorithmConfig, **kwargs: Any) -> AlgorithmConfig:
        """Deferred ``AlgorithmConfig.experimental`` (RLlib experimental flags)."""

        return cfg.experimental(**kwargs)

    def _parse_episode_identity(self, episode_id: str) -> dict[str, str]:
        """Parse a tagged episode ID into its ``key=value`` fields.

        Parameters
        ----------
        episode_id : str
            Episode ID rewritten by ``tag_episode_with_env_idx``, i.e.
            ``"env=<idx>|m=<mechanism>|ps=<policy_seed>|ss=<env_seed>|raw=<id>"``.
            Segments without ``=`` are ignored.

        Returns
        -------
        dict[str, str]
            Field values as strings (no int conversion), e.g.
            ``{"env": "3", "m": "1", "ps": "101", "ss": "101", "raw": "..."}``.

        Raises
        ------
        RuntimeError
            If any of ``env``, ``m``, ``ps``, ``ss`` is missing, which means the
            ``on_episode_created`` callback is not wired into the config.
        """

        parts = episode_id.split("|")
        identity = {}

        for part in parts:
            if "=" not in part:
                continue

            key, value = part.split("=", 1)
            identity[key] = value

        required = {"env", "m", "ps", "ss"}
        missing = required - identity.keys()

        if missing:
            raise RuntimeError(
                f"Episode id is missing identity keys {missing}: {episode_id}"
            )

        return identity

    def _seeded_xavier_uniform(self, seed: Optional[int]):
        """Build a deterministic Xavier-uniform initializer for one seed.

        Parameters
        ----------
        seed : int or None
            Base seed. ``None`` returns the string ``"xavier_uniform_"`` so
            RLlib uses its default (unseeded) initializer.

        Returns
        -------
        callable or str
            An ``init_(tensor, **kwargs)`` function. Each call seeds torch's
            CPU generator with ``seed + i`` (``i`` counting calls on this
            closure), applies ``torch.nn.init.xavier_uniform_`` and restores
            the previous RNG state, so the global stream is left untouched.
            Because a fresh closure (with its own counter) is created per
            RLModule, two modules built with the same seed and the same layer
            order receive identical weights, which is what makes the same
            policy seed comparable across mechanism candidates.
        """

        if seed is None:
            return "xavier_uniform_"

        counter = {"i": 0}

        def init_(tensor, **kwargs):
            """Seeded in-place Xavier init; advances the per-closure counter."""

            layer_seed = int(seed) + counter["i"]
            counter["i"] += 1
            state = torch.random.get_rng_state()

            torch.manual_seed(layer_seed)
            torch.nn.init.xavier_uniform_(tensor, **kwargs)
            torch.random.set_rng_state(state)

        return init_

    def _apply_agents_to_rllib(self) -> list[AgentID]:
        """Expand ``agent_specs`` into per-(mechanism, seed) RLModules.

        For every agent type and every ``(seed, mechanism_idx)`` pair an
        RLModule ``<policy>_m<idx>_s<seed>`` is declared with a seeded Xavier
        initializer (``_seeded_xavier_uniform``) and no shared value layers.
        The observation/action spaces of each agent instance are written into
        ``env_config`` under ``observation_spaces`` / ``action_spaces``, and
        ``rllib_cfg`` receives the ``MultiRLModuleSpec`` plus a
        ``multi_agent`` block whose ``policy_mapping_fn`` reads the mechanism
        index and policy seed from the tagged episode ID.

        Returns
        -------
        list[AgentID]
            Agent instance IDs ``"<agent_type>:<i>"``, passed to the env
            creator.

        Raises
        ------
        ValueError
            If ``num_envs_per_env_runner`` is not a multiple of the number of
            training seeds.

        Notes
        -----
        Must run after the ops have been replayed into ``rllib_cfg`` (it reads
        ``rllib_cfg.num_envs_per_env_runner``). The number of mechanisms is
        recomputed here as ``num_envs // num_seeds`` rather than read from
        ``num_mechanisms``.
        """

        policies = {}
        agent_type_map = {}

        agents: list[AgentID] = []

        observation_spaces = {}
        action_spaces = {}

        # Get number of envs and seeds
        num_envs = self.rllib_cfg.num_envs_per_env_runner or 1
        num_seeds = len(self.seeds) if self.seeds is not None else 1

        if num_envs % num_seeds != 0:
            raise ValueError(
                f"num_envs_per_env_runner={num_envs} must be divisible by num_seeds={num_seeds}"
            )

        # Get number of mechanisms (one policy per mechanism, per seed)
        num_mechanisms = num_envs // num_seeds
        module_specs = {}

        for agent_type, spec in self.agent_specs.items():
            obs_space = spec.get("observation_space")
            act_space = spec.get("action_space")
            base_policy = spec.get("policy")

            # TODO (nadinemgh) this does not guarantee tht different mechanism's policy will be
            # initiated with the same seed !
            # what we want :
            # run mechanism 0, seed 101
            # run mechanism 1, seed 101
            # run mechanism 2, seed 101
            # run mechanism 0, seed 202
            # run mechanism 1, seed 202
            # run mechanism 2, seed 202
            for seed in self.seeds:
                # TODO verify case when null seed
                for m_idx in range(num_mechanisms):
                    policy_id = f"{base_policy}_m{m_idx}_s{seed}"
                    policies[policy_id] = (
                        None,
                        spec.get("observation_space"),
                        spec.get("action_space"),
                        {},
                    )
                    module_specs[policy_id] = RLModuleSpec(
                        observation_space=obs_space,
                        action_space=act_space,
                        model_config=DefaultModelConfig(
                            vf_share_layers=False,
                            fcnet_kernel_initializer=self._seeded_xavier_uniform(seed),
                            fcnet_bias_initializer="zeros_",
                            head_fcnet_kernel_initializer=self._seeded_xavier_uniform(
                                seed
                            ),
                            head_fcnet_bias_initializer="zeros_",
                        ),
                    )

            for i in range(spec.get("count")):
                agent_id = f"{agent_type}:{i}"

                agents.append(agent_id)

                agent_type_map[agent_id] = base_policy
                observation_spaces[agent_id] = obs_space
                action_spaces[agent_id] = act_space

        self.rllib_cfg = self.rllib_cfg.rl_module(
            rl_module_spec=MultiRLModuleSpec(rl_module_specs=module_specs)
        )

        self.env_config.update({"observation_spaces": observation_spaces})
        self.env_config.update({"action_spaces": action_spaces})

        def policy_mapping_fn(agent_id, episode: MultiAgentEpisode, *_, **__):
            """Route an agent to ``<policy>_m<m>_s<ps>`` from the episode ID."""

            base_policy = agent_type_map[agent_id]

            # Old Api Stack Route to policy based on environment index
            # TODO this is depregated !
            # env_idx = getattr(episode, "env_id", None)

            # New API
            identity = self._parse_episode_identity(episode.id_)
            mechanism_id = identity["m"]
            policy_seed = identity["ps"]
            policy_id = f"{base_policy}_m{mechanism_id}_s{policy_seed}"

            if policy_id not in policies:
                raise RuntimeError(
                    f"Unknown policy generated by policy_mapping_fn: "
                    f"policy_id={policy_id}, "
                    f"episode_id={episode.id_}, "
                    f"mechanism_id={mechanism_id}, "
                    f"policy_seed={policy_seed}, "
                    f"available_policies={list(policies.keys())}"
                )

            return policy_id

        all_policies = list(policies.keys())
        self.rllib_cfg = self.rllib_cfg.multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=all_policies,
        )

        return agents

    # TODO agent spec for stricter schema enforcement
    def agents(self, agents: dict[str, AgentSpec]) -> Self:
        """Declare the agent types of the environment.

        Parameters
        ----------
        agents : dict[str, AgentSpec]
            Mapping ``agent_type -> spec``. In practice the specs are plain
            dicts with ``count``, ``policy``, ``observation_space`` and
            ``action_space`` keys, because ``_apply_agents_to_rllib`` indexes
            them with ``spec.get(...)``.

        Returns
        -------
        RayOptimizerConfig
            ``self`` for chaining.
        """

        self.agent_specs = agents

        return self

    # lazy resolution : better encapsulation ?
    # @cached_property
    # def result_mapper(self) -> ResultMapper:
    #     return self._resolve_result_mapper()

    # def _resolve_result_mapper(cfg: AlgorithmConfig) -> ResultMapper:
    #     uses_new_stack = bool(
    #         getattr(cfg, "enable_rl_module_and_learner", False)
    #         and getattr(cfg, "enable_env_runner_and_connector_v2", False)
    #     )
    #     return from_new_api if uses_new_stack else from_old_api

    @override(OptimizerConfig)
    def build_optimizer(
        self,
        *,
        world: ActorHandle[World],
        world_name: Optional[str] = None,
    ) -> RayOptimizer:
        """Resolve the RLlib config, register the env and build a ``RayOptimizer``.

        Steps, in order:

        1. If ``evaluation`` was called, patch its recorded kwargs so that
           there is one evaluation env runner per ``(eval seed, train seed)``
           pair, ``evaluation_duration`` equals ``runners * num_mechanisms``
           episodes, and ``_evaluate_with_fixed_duration_once`` is the custom
           evaluation function.
        2. Replay all recorded ops onto ``algo_class.get_default_config()``
           (only the first time; ``rllib_cfg`` is cached afterwards).
        3. Draw a fresh optimizer ID from the World's registry.
        4. Expand ``agent_specs`` into RLModules (``_apply_agents_to_rllib``).
        5. Register a uniquely named env with ``ray.tune`` whose creator maps
           each sub-environment to a mechanism index and a pair of seeds (see
           ``env_creator`` below), and point ``rllib_cfg`` at it.
        6. Freeze a deep copy of this config and hand it to ``RayOptimizer``;
           the ``Algorithm`` itself is built later inside ``PolicyActor``.
        7. Build the optimizer-level reporter through ``_build_reporter``
           (``None`` when no ``reporter_cfg`` was set) and attach it as
           ``opt.reporting``. Each environment receives a copy of
           ``reporter_cfg`` plus the env-level queries and schema so it can
           build its own reporter.

        Parameters
        ----------
        world : ActorHandle[World]
            Shared world actor, given to every environment.
        world_name : str, optional
            Required when ``world`` is given; stored in ``world_name``.

        Returns
        -------
        RayOptimizer
            Optimizer with ``world``, ``reporting`` and ``opt_id`` set.

        Raises
        ------
        ValueError
            If ``world`` is given without ``world_name`` or ``opt_class`` is
            unset.

        Notes
        -----
        ``opt_id`` and ``agents`` are only bound when ``world`` is not ``None``
        and ``agent_specs`` is set; otherwise the env creator raises
        ``NameError`` when RLlib first instantiates an environment.
        """

        evaluation_op = self._cfg_ops.get("_evaluation_rllib")

        if evaluation_op is not None:
            num_eval_seeds = len(self.eval_seeds) if self.eval_seeds else 1
            num_train_seeds = len(self.seeds) if self.seeds else 1
            num_eval_runners = num_eval_seeds * num_train_seeds
            num_eval_episodes = num_eval_runners * self.num_mechanisms
            evaluation_op.kwargs["evaluation_num_env_runners"] = num_eval_runners

            # TODO verify this
            evaluation_op.kwargs["evaluation_duration"] = num_eval_episodes
            evaluation_op.kwargs["custom_evaluation_function"] = (
                _evaluate_with_fixed_duration_once
            )

        if self.rllib_cfg is None:
            self.rllib_cfg = self.algo_class.get_default_config()

            for op in self._cfg_ops.values():
                self.rllib_cfg = op(self.rllib_cfg)

        if self.opt_class is None:
            raise ValueError("OptimizerConfig has no opt_class")

        env_name = f"regulated_env_{uuid.uuid4().hex}"

        if world is not None:
            if world_name is None:
                raise ValueError(
                    "world_name must be provided when using Ray world actor"
                )

            self.world_name = world_name
            registry = ray.get(world.get_opt_registry.remote())

            # Register the new ID and get the result
            opt_id = ray.get(
                world._set_new_opt_id.remote(opt_id=generate_uuid(registry))
            )

        if self.agent_specs:
            agents = self._apply_agents_to_rllib()

        env_counter = {"train": 0, "eval": 0}

        def env_creator(env_ctx):
            """Build one sub-environment and assign it a mechanism and seeds.

            RLlib calls this once per vectorised sub-environment, in each env
            runner process. ``env_ctx["mode"]`` selects the layout:

            - ``train`` (default): with ``num_envs = num_envs_per_env_runner``
              and ``M = num_mechanisms``, the ``k``-th environment created
              gets ``mechanism_idx = (k % num_envs) % M`` and training seed
              index ``(k % num_envs) // M``; the environment seed equals the
              policy seed, so each seeded policy trains on a matching seeded
              environment.
            - ``eval``: seeds come from the runner, mechanisms from the local
              counter. With ``runner_idx = worker_index - 1``, the policy
              (training) seed index is ``runner_idx % num_train_seeds``, the
              evaluation seed index ``runner_idx // num_train_seeds`` and
              ``mechanism_idx = k % M``. The environment therefore runs under
              an evaluation seed while ``policy_seed`` still names the trained
              module to test.

            ``seed`` and ``policy_seed`` are written into ``env_ctx`` and the
            environment is instantiated through ``OptimizerConfig._env_creator``
            with ``world``, ``opt_id``, ``agents`` and ``mechanism_id``.
            Counters are per process (the closure is pickled to each runner).
            """

            mode = env_ctx.get("mode", "train")
            num_mechanisms = self.num_mechanisms or 1
            train_seeds = self.seeds or [None]
            eval_seeds = self.eval_seeds or [None]
            num_train_seeds = len(train_seeds)
            local_env_idx = env_counter[mode]
            env_counter[mode] += 1

            if mode == "eval":
                runner_idx = env_ctx.worker_index - 1
                train_seed_idx = runner_idx % num_train_seeds
                eval_seed_idx = runner_idx // num_train_seeds
                mechanism_idx = local_env_idx % num_mechanisms
                policy_seed = train_seeds[train_seed_idx]
                env_seed = eval_seeds[eval_seed_idx]
            else:
                num_envs = self.rllib_cfg.num_envs_per_env_runner or 1
                env_idx = local_env_idx % num_envs
                mechanism_idx = env_idx % num_mechanisms
                train_seed_idx = env_idx // num_mechanisms
                policy_seed = train_seeds[train_seed_idx]
                env_seed = policy_seed

            env_ctx["seed"] = env_seed
            env_ctx["policy_seed"] = policy_seed

            return self._env_creator(
                world=world,
                opt_id=opt_id,
                env_name=env_name,
                agents=agents,
                mechanism_id=mechanism_idx,
                reporter_cfg=self._reporter_cfg.copy()
                if self._reporter_cfg is not None
                else None,
                queries=self._reporting_queries_env,
                schema=self._reporting_schema_env,
                **dict(env_ctx),
            )

        register_env(env_name, env_creator)

        self.rllib_cfg = self.rllib_cfg.environment(
            env=env_name, env_config=self.env_config
        )

        # Building defferred to policyActor
        # algo = self.rllib_cfg.build_algo(**kwargs)

        cfg = self.copy(copy_frozen=True)

        # TODO do not give world to ray optimizer. temp solution until environment factory
        opt = RayOptimizer(config=cfg)
        opt.world = world

        # Build reporter
        reporter = self._reporter_cfg.build(label=self.opt_class.__name__)
        reporter.schema = self._reporting_schema

        reporter.add_query(*(self._reporting_queries or ()))

        opt.reporting = reporter

        # register optimizer in world to link contexts to optimizers
        if world is not None:
            opt.set_id(opt_id)

        return opt

    @rllib_config_mutator
    @override(OptimizerConfig)
    def freeze(cfg: AlgorithmConfig, **kwargs: Any) -> AlgorithmConfig:
        """Deferred ``AlgorithmConfig.freeze``.

        Records a freeze of the *RLlib* config for replay; it does not freeze
        this ``RayOptimizerConfig`` (``build_optimizer`` does that through
        ``copy(copy_frozen=True)``).
        """

        return cfg.freeze(**kwargs)

    @rllib_config_mutator
    @override(OptimizerConfig)
    def training(cfg: AlgorithmConfig, **kwargs: Any) -> AlgorithmConfig:
        """Deferred ``AlgorithmConfig.training`` (algorithm hyperparameters)."""

        return cfg.training(**kwargs)

    @rllib_config_mutator
    def _debugging_rllib(
        cfg: AlgorithmConfig, seed: Optional[int] = None, **kwargs: Any
    ) -> AlgorithmConfig:
        """Deferred ``AlgorithmConfig.debugging`` (raw pass-through)."""

        return cfg.debugging(seed=seed, **kwargs)

    @override(OptimizerConfig)
    def debugging(
        self,
        *,
        seed: Optional[int] = None,  # base seed
        num_seeds: int = 3,
        **kwargs: Any,
    ) -> Self:
        """Set the training seeds and scale the env count accordingly.

        Parameters
        ----------
        seed : int or None, optional
            Base seed. ``OptimizerConfig.debugging`` derives ``num_seeds``
            training seeds from it with a ``SeedSequence``.
        num_seeds : int, optional
            Number of training seeds (default 3).
        **kwargs
            Forwarded to the deferred ``AlgorithmConfig.debugging``, together
            with ``seed`` (RLlib's own seed).

        Returns
        -------
        RayOptimizerConfig
            ``self`` for chaining.

        Notes
        -----
        When a seed is given and ``env_runners`` was already called, the
        recorded ``num_envs_per_env_runner`` is taken as the number of
        mechanisms (stored in ``num_mechanisms``) and multiplied by
        ``num_seeds`` so each mechanism candidate gets one environment per
        seed. If ``env_runners`` is called *after* ``debugging``, no scaling
        happens and ``num_mechanisms`` is whatever ``env_runners`` records.
        """

        super().debugging(seed=seed, num_seeds=num_seeds)

        if seed is not None:
            env_runners_op = self._cfg_ops.get("_env_runners")

            if env_runners_op is not None:
                self.num_mechanisms = env_runners_op.kwargs["num_envs_per_env_runner"]
                env_runners_op.kwargs["num_envs_per_env_runner"] = (
                    env_runners_op.kwargs.get("num_envs_per_env_runner", 1) * num_seeds
                )

        # Lazy construction
        return self._debugging_rllib(seed=seed, **kwargs)
