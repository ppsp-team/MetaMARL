"""Context schemas exchanged through the shared ``World`` actor.

A ``Context`` is the unit of communication between the two levels of the
bilevel loop. The outer (regulator) optimizer publishes ``MechanismContext``
candidates, the inner RL environments consume them and publish one
``EnvStepContext`` per ``reset``/``step``, and the regulator reads those step
contexts back to score each candidate. ``MechanismStatus`` records where a
published mechanism stands in that cycle.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, SupportsFloat

from gymnasium.core import ActType, ObsType
from pydantic import BaseModel, SkipValidation
from ray.rllib.utils.typing import MultiAgentDict

from core.mechanism.base import Mechanism
from core.types import ContextID, OptimizerID


class MechanismStatus(Enum):
    """Lifecycle state of a mechanism candidate stored in the ``World``.

    The nominal path is ``published -> train -> eval -> done``:

    - ``published``: the regulator appended the candidate to the World (one
      entry per ``(candidate index, seed)``) and nobody has claimed it yet.
    - ``train``: a training environment fetched it through
      ``World.get_mechanism_by_id(mode=train)``. Only ``published`` entries can
      move here, so each ``(index, seed)`` entry is handed out exactly once.
    - ``eval``: an evaluation environment fetched it. Both ``train`` and
      ``eval`` entries qualify, so several evaluation seeds can share one
      trained mechanism. The regulator flushes ``eval`` entries between inner
      runs (``World.flush(status=eval)``).
    - ``done``: written by regulators when they publish the aggregated fitness
      of a candidate (``mechanism=None``, ``metrics`` filled in).

    Two values sit outside that path. ``assigned`` is set by the legacy
    ``World.get_mechanism`` / ``try_get_mechanism`` accessors, which hand out
    any published mechanism without matching an index or a seed. ``init`` is
    a placeholder for a context created before publication; the core code
    never assigns it.

    The same enum also stamps ``EnvStepContext.status`` with the mode of the
    producing environment (``train`` or ``eval``), which is how the reporting
    utilities separate training rollouts from evaluation rollouts.
    """

    init = "init"
    published = "published"
    assigned = "assigned"
    train = "train"
    eval = "eval"
    done = "done"


# TODO some world contexts are singletons (mutable) others are simply mutable.
# TODO for now singleton/or no is deffered to world
# TODO Enums for Context to access different Context Schemas.


class ContextSchema(BaseModel):
    """Base schema for shared world context."""

    model_config = {"arbitrary_types_allowed": True}


class MechanismContext(ContextSchema):
    """One mechanism candidate, as published by the regulator.

    Attributes
    ----------
    index : int
        Position of the candidate in the regulator's current batch. Training
        environments are built with a matching ``mechanism_id`` and fetch
        their candidate by this index.
    env_id : str or None
        Identifier of the environment that produced the context. Publication
        from the regulator leaves it ``None``. ``World.set_new_context`` and
        ``World.update_context`` reject a ``None`` value; ``append_context``,
        the path actually used by ``BaseEnv._publish``, does not.
    seed : int or None
        Policy (training) seed this copy of the candidate is meant for. The
        regulator publishes one copy per training seed so that each seeded
        policy trains against its own instance.
    eval_seed : int or None
        Environment seed used when the same candidate is evaluated; ``None``
        during training.
    status : MechanismStatus
        Lifecycle state, see ``MechanismStatus``.
    mechanism : Mechanism
        The candidate itself. Pydantic validation is skipped because
        mechanisms are arbitrary user classes. ``None`` on ``done`` contexts.
    metrics : ContextSchema or None
        Aggregated fitness payload; filled only on ``done`` contexts.
    """

    index: int
    env_id: Optional[str]
    seed: Optional[int]
    eval_seed: Optional[int] = None
    status: MechanismStatus
    mechanism: SkipValidation[Mechanism]
    metrics: Optional[ContextSchema]


# TODO strict type annotations rm Any
class EnvStepContext(ContextSchema):
    """Snapshot of one environment transition, published on every step.

    ``BaseEnv.reset`` and ``BaseEnv.step`` both append one of these to the
    World; the reset record carries ``reward=0.0`` and ``action=None``. The
    regulator collects them to score a mechanism, and the reporting utilities
    reduce them into per-episode curves.

    Attributes
    ----------
    env_id : int or None
        Index of the sub-environment inside its vectorised env runner. Set by
        the ``tag_episode_with_env_idx`` callback when the first episode is
        created; ``None`` until then.
    seed : int or None
        Seed of the environment dynamics (``BaseEnv.seed``). During training it
        equals ``policy_seed``; during evaluation it is one of the configured
        evaluation seeds.
    policy_seed : int or None
        Seed identifying which trained policy acts in this environment.
        Together with ``mechanism`` it selects the RLModule named
        ``<policy>_m<mechanism>_s<policy_seed>``.
    status : MechanismStatus
        Mode of the producing environment: ``train`` or ``eval``.
    mechanism : int or None
        Index of the mechanism candidate the environment runs
        (``RegulatedEnv.mechanism_id``), not the mechanism object itself.
    observation : ObsType or MultiAgentDict
        Observation returned to the agent(s) after the transition.
    observation_map : list of str or None
        Optional names for the entries of the observation vector, used by the
        reporting utilities. Note that ``BaseEnv.obs_map`` is typed as a
        ``dict[int, str]`` while this field is typed as a list.
    reward : SupportsFloat or MultiAgentDict or list of float
        Reward of the transition (``0.0`` on the reset record).
    action : ActType or MultiAgentDict
        Action that produced the transition (``None`` on the reset record).
    info : dict or MultiAgentDict or None
        The ``info`` dictionary returned by the environment.
    """

    env_id: Optional[int]
    seed: Optional[int]
    policy_seed: Optional[int]
    status: MechanismStatus
    mechanism: Optional[int]
    observation: ObsType | MultiAgentDict
    observation_map: Optional[list[str]]
    reward: SupportsFloat | MultiAgentDict | list[float]
    action: ActType | MultiAgentDict
    info: dict | MultiAgentDict | None


@dataclass
class Context:
    """
    Runtime instance of a context
    """

    id: ContextID | None
    opt_id: OptimizerID
    step: int
    env: str
    payload: ContextSchema
