"""Single-agent environment regulated by a mechanism fetched from the ``World``.

``RegulatedEnv`` extends :class:`core.envs.base.BaseEnv` with the mechanism
lifecycle of the inner level: at every reset it asks the ``World`` for the
candidate published under its ``mechanism_id`` and keeps it for the rest of
the run, falling back to ``m_space.default()`` until one is published. Its
reward is the regulated form ``u - lambda(M) * v``: the base reward minus a
mechanism-dependent penalty weight times a violation signal, both supplied by
the concrete benchmark. The multi-agent variant is
:class:`core.envs.marl_regulated.MultiAgentRegulatedEnv`.
"""

import logging
from abc import abstractmethod
from typing import Any, Optional, SupportsFloat

import ray

from core.annotations import override
from core.envs.base import BaseEnv
from core.mechanism.base import Mechanism
from core.world.context import MechanismContext

logger = logging.getLogger(__name__)


class RegulatedEnv(BaseEnv):
    """Inner environment whose reward is shaped by a published mechanism.

    Parameters
    ----------
    mechanism_id : str
        Identifier of the candidate mechanism this env instance trains
        against; used to fetch its ``MechanismContext`` from the ``World``.
    **kwargs
        Forwarded to :class:`BaseEnv` (``world``, ``mechanism_space``,
        ``horizon``, ``seed``, ...).
    """

    def __init__(
        self,
        *,
        mechanism_id: str,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        # mechanism_space can be a class or an instance
        self.mechanism_id = mechanism_id
        self.m_ctx: MechanismContext = None
        self.m: Mechanism = None
        self._using_default_mechanism = True

    @property
    def mechanism(self) -> Mechanism:
        """Mechanism in force: the published candidate, else the space default."""

        if self.m is not None:
            return self.m

        return self.m_space.default()

    @property
    def published_mechanism_assigned(self) -> bool:
        """Whether a candidate fetched from the ``World`` (not the default) is in force."""

        return self.m is not None and not self._using_default_mechanism

    @override(BaseEnv)
    def _pre_reset(self, seed: Optional[int] = None):
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
                        mechanism_id=self.mechanism_id,
                        seed=self.policy_seed,
                        mode=self.mode,
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

    @abstractmethod
    def violation_signal(self, **kwargs: Any) -> float:
        """``v = V(a, S_t, M)``: how far the current behaviour departs from the mechanism."""

        raise NotImplementedError

    @abstractmethod
    def penalty(self, **kwargs: Any) -> float:
        """``lambda = lambda(M)``: penalty weight attached to the mechanism."""

        raise NotImplementedError

    @override(BaseEnv)
    def reward(self, reward: SupportsFloat, **kwargs: Any) -> SupportsFloat:
        """Regulated reward ``reward - penalty(M) * violation_signal``.

        ``kwargs`` are forwarded to :meth:`penalty` and :meth:`violation_signal`.
        """

        return reward - self.penalty(**kwargs) * self.violation_signal(**kwargs)
