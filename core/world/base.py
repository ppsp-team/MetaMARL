"""Ray actor holding the shared state of a bilevel optimisation run.

The ``World`` is the one object both levels of the optimisation talk to. The
outer regulator publishes ``MechanismContext`` entries into it, the inner RLlib
environments (living in env-runner processes) fetch those entries and push back
one ``EnvStepContext`` per transition, and the regulator reads the step contexts
to compute each candidate's fitness. Because ``World`` is a Ray actor, every
method is called as ``world.<method>.remote(...)`` and its return value crosses
a Ray boundary; callers must not rely on mutating a returned object to update
the World.

Three registries are kept in sync: ``_contexts`` (all contexts by
``ContextID``), ``_mechanism_registry`` (the ``MechanismContext`` payloads,
keyed by the ``ContextID`` of their context, not by mechanism index) and
``_opt_ctx_map`` (context IDs owned by each optimizer).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, KeysView, Optional

import ray
from core.types import ContextID, OptimizerID
from core.utils import generate_uuid
from core.world.context import (
    Context,
    ContextSchema,
    EnvStepContext,
    MechanismContext,
    MechanismStatus,
)

if TYPE_CHECKING:
    pass


@ray.remote
class World:
    """
    Shared runtime container for optimizer-produced contexts.

    The World acts as the single source of truth for:
    - Context lifecycles (creation, update, removal)
    - Association between optimizers and their contexts
    - Enforcing global constraints (e.g. singleton contexts)

    The World does NOT own optimizers or environments.
    It only tracks identifiers and context payloads.
    """

    # TODO the reporting type annotation to add
    def __init__(self):
        # Maps optimizer IDs to the list of context IDs they own
        # TODO replace with registry
        self._opt_ctx_map: dict[OptimizerID, list[ContextID]] = {}

        # Maps context IDs to Context objects
        self._contexts: dict[ContextID, Context] = {}

        # Mechanism registry
        self._mechanism_registry: dict[int, MechanismContext] = {}
        self._env_step_ctx_cursor: dict[OptimizerID | None, int] = {}

    def __deepcopy__(self, memo):
        return self

    def __copy__(self):
        return self

    # Accessors
    def get_ctx_registry(self) -> dict[ContextID, Context]:
        """Return the whole context registry.

        Returns
        -------
        dict[ContextID, Context]
            Every registered context, keyed by ID, in insertion order. The
            regulator uses it to collect step contexts before flushing them.
        """

        return self._contexts

    def get_mechanism_registry(self) -> dict[int, MechanismContext]:
        """Return the mechanism registry.

        Returns
        -------
        dict[ContextID, MechanismContext]
            Mechanism payloads keyed by the ``ContextID`` of the context that
            carried them. Despite the ``int`` annotation, keys are UUID
            strings; the mechanism's batch position lives in
            ``MechanismContext.index``.
        """

        return self._mechanism_registry

    def get_opt_registry(self) -> KeysView[OptimizerID]:
        """Return a view of the optimizer IDs known to the world.

        Returns
        -------
        KeysView[OptimizerID]
            Keys of the optimizer-to-contexts map. Used by
            ``RayOptimizerConfig.build_optimizer`` to draw a fresh unique ID.
        """

        return self._opt_ctx_map.keys()

    def get_context(self, ctx_id: ContextID) -> Context | None:
        """Access a context stored in world with an ID"""

        return self._contexts.get(ctx_id, None)

    def get_opt_ctx_ids(self, opt_id: OptimizerID) -> list[ContextID]:
        """
        Return all context IDs registered under a given optimizer.
        """

        return list(self._opt_ctx_map.get(opt_id, []))

    def get_ctx_ids(self) -> set[ContextID]:
        """
        Return all context IDs registered in the world.
        """

        return set(self._contexts.keys())

    def get_opt_ids(self) -> set[OptimizerID]:
        """
        Return all optimizer IDs known to the world.
        """

        return set(self._opt_ctx_map.keys())

        # ADDED: helpers for reduced env plotting

    def get_env_step_contexts(
        self,
        opt_id: Optional[OptimizerID] = None,
    ) -> list[Context]:
        """
        Return all EnvStepContext objects for a given optimizer.
        If opt_id is None, return all EnvStepContext objects in the world.
        Order is preserved according to insertion order.
        """

        if opt_id is None:
            ctxs = list(self._contexts.values())
        else:
            ctx_ids = self._opt_ctx_map.get(opt_id, [])
            ctxs = [self._contexts[cid] for cid in ctx_ids if cid in self._contexts]

        return [
            ctx
            for ctx in ctxs
            if ctx is not None and isinstance(ctx.payload, EnvStepContext)
        ]

    # ADDED: helpers for reduced env plotting
    def get_latest_env_step_contexts(
        self,
        opt_id: Optional[OptimizerID] = None,
    ) -> list[Context]:
        """
        Return only the latest contiguous env-step episode for an optimizer.

        We assume env-step contexts are appended in order and that step resets
        to 0 at the beginning of a new episode. We walk backward from the most
        recent EnvStepContext until we hit step == 0.
        """

        env_ctxs = self.get_env_step_contexts(opt_id=opt_id)

        if not env_ctxs:
            return []

        latest: list[Context] = []

        for ctx in reversed(env_ctxs):
            latest.append(ctx)

            if int(ctx.step) == 0:
                break

        latest.reverse()

        return latest

    def get_new_env_step_contexts(
        self,
        opt_id: Optional[OptimizerID] = None,
    ) -> list[Context]:
        """
        Return all EnvStepContext objects appended since the last call
        for this optimizer, then advance the cursor.

        This is used for reduced env plotting so we capture both train
        and eval contexts produced during one optimizer run().
        """

        ctx_ids = (
            self._opt_ctx_map.get(opt_id, [])
            if opt_id is not None
            else list(self._contexts.keys())
        )
        cursor = self._env_step_ctx_cursor.get(opt_id, 0)
        new_ctx_ids = ctx_ids[cursor:]
        self._env_step_ctx_cursor[opt_id] = len(ctx_ids)

        return [
            self._contexts[cid]
            for cid in new_ctx_ids
            if cid in self._contexts
            and isinstance(self._contexts[cid].payload, EnvStepContext)
        ]

    def get_mechanism(self) -> MechanismContext:
        """Claim the first published mechanism, regardless of index or seed.

        Legacy accessor: the first entry with status ``published`` is moved to
        ``assigned`` and returned. The current environments use
        ``get_mechanism_by_id`` instead, which matches index and seed.

        Returns
        -------
        MechanismContext
            The claimed mechanism, now in status ``assigned``.

        Raises
        ------
        RuntimeError
            If no mechanism is in status ``published``.
        """

        for m_ctx in self._mechanism_registry.values():
            if m_ctx.status == MechanismStatus.published:
                m_ctx.status = MechanismStatus.assigned

                return m_ctx

        raise RuntimeError("no available mechanisms to train")

    # Use the contextID as mechanismID
    def get_mechanism_by_id(
        self, mechanism_id: int, seed: int, mode: MechanismStatus
    ) -> MechanismContext:
        """Fetch the mechanism for ``(mechanism_id, seed)`` and advance its status.

        Called by ``RegulatedEnv._pre_reset`` at the start of every episode.
        The first registry entry whose ``index`` equals ``mechanism_id``, whose
        ``seed`` equals ``seed`` and whose status is a valid predecessor of
        ``mode`` is switched to ``mode`` and returned. Valid predecessors are
        ``published`` for ``mode=train`` and ``{train, eval}`` for
        ``mode=eval``.

        Parameters
        ----------
        mechanism_id : int
            Batch position of the candidate (``MechanismContext.index``), not
            a registry key.
        seed : int
            Policy seed the environment was built with
            (``MechanismContext.seed``).
        mode : MechanismStatus
            Target status: ``MechanismStatus.train`` or ``MechanismStatus.eval``.

        Returns
        -------
        MechanismContext or None
            The matching mechanism on the first successful fetch. On later
            calls with the same arguments the entry is no longer in a
            predecessor status, so ``None`` is returned; environments treat
            that as "keep the mechanism you already have". The return
            annotation does not mention ``None`` but it is a routine outcome.

        Raises
        ------
        TypeError
            If ``mode`` is neither ``train`` nor ``eval``: the predecessor set
            lookup yields ``None`` and the ``in`` test fails.
        """

        required_status = {
            MechanismStatus.train: {MechanismStatus.published},
            MechanismStatus.eval: {
                MechanismStatus.train,
                MechanismStatus.eval,
            },
        }
        target_prev_status = required_status.get(mode)

        for m_ctx in self._mechanism_registry.values():
            if (
                mechanism_id == m_ctx.index
                and seed == m_ctx.seed
                and m_ctx.status in target_prev_status
            ):
                m_ctx.status = mode

                return m_ctx

        return None

    def try_get_mechanism(self) -> MechanismContext | None:
        """Try to get a published mechanism, return None if none available."""

        for m_ctx in self._mechanism_registry.values():
            if m_ctx.status == MechanismStatus.published:
                m_ctx.status = MechanismStatus.assigned

                return m_ctx

        return None

    # TODO fix this function. now the primary key is ctx_id
    def get_mechanism_by_index(self, index: int) -> MechanismContext:
        """Return the mechanism stored under registry key ``index``.

        Despite the name and the ``int`` annotation, the registry is keyed by
        the ``ContextID`` string of the publishing context, so this only works
        when passed that ID; an integer batch index raises ``KeyError``. The
        TODO above records the mismatch. Use ``get_mechanism_by_id`` to look
        up a candidate by its batch position.

        Raises
        ------
        KeyError
            If ``index`` is not a registry key.
        """

        return self._mechanism_registry[index]

    def _validate_ctx_schema_exists(self, schema: type[ContextSchema]) -> None:
        """
        Ensure a singleton ContextSchema is not already present in the world.
        """

        for ctx in self._contexts.values():
            if type(ctx.payload) is schema:
                raise ValueError(
                    f"Singleton Context Schema {schema.__name__} already exists in world."
                )

    # Mutators
    def append_context(self, ctx: Context, *, singleton: bool = False) -> ContextID:
        """Register a context, assigning it a fresh ID.

        This is the path used by ``BaseEnv._publish`` for every mechanism and
        step context. The context is stored in ``_contexts``; a
        ``MechanismContext`` payload is additionally indexed in the mechanism
        registry; and the ID is appended to the owning optimizer's list,
        creating that optimizer entry on the fly if needed.

        Parameters
        ----------
        ctx : Context
            Context to store. ``ctx.id`` is always overwritten with a new UUID,
            so a caller-supplied ID is not preserved; the duplicate check that
            precedes the overwrite can only fire if the caller passed an ID
            that already exists.
        singleton : bool, optional
            If ``True``, raise when another context with the same payload type
            is already registered.

        Returns
        -------
        ContextID
            The generated ID now stored in ``ctx.id``.

        Raises
        ------
        ValueError
            If ``singleton`` is requested and violated, or if a caller-supplied
            ``ctx.id`` already exists.

        Notes
        -----
        Unlike ``set_new_context``, this method does not require
        ``MechanismContext.env_id`` to be set, which is why regulators can
        publish candidates with ``env_id=None``.
        """

        # Enforce singleton schemas if requested
        if singleton:
            self._validate_ctx_schema_exists(type(ctx.payload))

        if ctx.id is not None and ctx.id in self._contexts:
            raise ValueError(f"ContextID '{ctx.id}' already exists")

        ctx.id = generate_uuid(self._contexts)
        self._contexts[ctx.id] = ctx

        if isinstance(ctx.payload, MechanismContext):
            self._mechanism_registry[ctx.id] = ctx.payload

        # Track optimizer → context mapping
        if ctx.opt_id is not None:
            if ctx.opt_id not in self._opt_ctx_map:
                self._set_new_opt_id(ctx.opt_id)

            self._opt_ctx_map[ctx.opt_id].append(ctx.id)

        # if env-step-context call reporter actor
        # if self.reporting is not None and isinstance(ctx.payload, EnvStepContext):
        #     self.reporting.plot_env_step.remote(
        #         ctx=ctx,
        #         obs_keys_skip=(
        #             "fixed_quota",
        #             "prop_quota",
        #             "min_stock",
        #             "target_stock",
        #             "fine_amount",
        #             "risk_penalty_scale",
        #             "risk_penalty_power",
        #         ),
        #     )

        return ctx.id

    def _set_new_opt_id(self, opt_id: OptimizerID) -> OptimizerID:
        """Ensure an optimizer ID exists in the map and return it.

        Parameters
        ----------
        opt_id : OptimizerID or None
            Identifier to register. ``None`` draws a fresh UUID.

        Returns
        -------
        OptimizerID
            The registered identifier. Calling twice with the same ID is a
            no-op that keeps the existing context list.
        """

        if opt_id is None:
            opt_id = generate_uuid(registry=self._opt_ctx_map.keys())

        if opt_id not in self._opt_ctx_map:
            self._opt_ctx_map[opt_id] = []

        return opt_id

    def set_new_context(self, ctx: Context, singleton: bool = False) -> ContextID:
        """
        Register a new context in the world.

        Args:
            ctx: Context object containing optimizer ID and schema payload
            singleton: Whether this context schema must be unique globally

        Returns:
            The generated ContextID
        """

        if singleton:
            self._validate_ctx_schema_exists(type(ctx.payload))

        if ctx.id is not None:
            if ctx.id in self._contexts:
                raise ValueError(f"ContextID '{ctx.id}' already exists")
        else:
            ctx.id = generate_uuid(registry=self._contexts.keys())

        self._contexts[ctx.id] = ctx

        # Track latest mechanism globally
        # Track per-env mechanism
        if isinstance(ctx.payload, MechanismContext):
            if ctx.payload.env_id is None:
                raise ValueError("MechanismContext must include env_id")

            self._mechanism_registry[ctx.id] = ctx.payload

        if ctx.opt_id is not None:
            if ctx.opt_id not in self._opt_ctx_map:
                self._set_new_opt_id(ctx.opt_id)

            self._opt_ctx_map[ctx.opt_id].append(ctx.id)

        return ctx.id

    def update_context(self, ctx: Context) -> None:
        """
        Update an existing context payload.
        """

        if ctx.id not in self._contexts:
            raise KeyError(f"Context {ctx.id} not registered")

        self._contexts[ctx.id] = ctx

        if isinstance(ctx.payload, MechanismContext):
            if ctx.payload.env_id is None:
                raise ValueError("MechanismContext must include env_id")

            self._mechanism_registry[ctx.id] = ctx.payload

    def remove_context(self, ctx: Context) -> None:
        """
        Remove a context from the world.
        """

        if ctx.id in self._contexts:
            self._contexts.pop(ctx.id)

        if ctx.opt_id in self._opt_ctx_map:
            lst = self._opt_ctx_map[ctx.opt_id]

            if ctx.id in lst:
                lst.remove(ctx.id)

            if not lst:
                del self._opt_ctx_map[ctx.opt_id]

    # TODO fix this function. now the primary key is ctx_id
    def flush(self, status: Optional[MechanismStatus] = None) -> None:
        """Drop mechanisms from the mechanism registry.

        Parameters
        ----------
        status : MechanismStatus or None, optional
            Only remove mechanisms in this status. ``None`` removes all of
            them. The regulator calls ``flush(status=eval)`` between inner
            runs so evaluated candidates are not fetched again.

        Notes
        -----
        Only ``_mechanism_registry`` is touched. The ``Context`` objects that
        carried the mechanisms stay in ``_contexts`` and in ``_opt_ctx_map``
        until ``flush_ctx`` removes them.
        """

        to_delete = []

        for ctx_id, m_ctx in self._mechanism_registry.items():
            if status is not None and m_ctx.status != status:
                continue

            to_delete.append(ctx_id)

        for ctx_id in to_delete:
            del self._mechanism_registry[ctx_id]

    def flush_ctx(self, ctx_ids: list[ContextID]) -> None:
        """Remove contexts from the context registry.

        Parameters
        ----------
        ctx_ids : list[ContextID]
            IDs to drop; unknown IDs are ignored. The regulator passes the
            full key set of ``get_ctx_registry`` after scoring a batch.

        Notes
        -----
        Only ``_contexts`` is touched. The IDs remain listed in
        ``_opt_ctx_map`` (so ``get_opt_ctx_ids`` can return IDs that no longer
        resolve) and any mechanism payload remains in the mechanism registry
        until ``flush`` removes it.
        """

        if not ctx_ids:
            return

        for cid in ctx_ids:
            self._contexts.pop(cid, None)
