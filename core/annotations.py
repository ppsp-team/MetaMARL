"""Decorators providing lightweight checks on class hierarchies."""

from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def override(parent_cls: type) -> Callable[[F], F]:
    """Decorator for documenting method overrides.

    Parameters
    ----------
    parent_cls : type
        The superclass that provides the overridden method.

    Returns
    -------
    callable
        A decorator that returns the decorated method unchanged after checking
        that ``parent_cls`` exposes an attribute with the same name.

    Raises
    ------
    NameError
        If ``parent_cls`` has no attribute named like the decorated method.

    Notes
    -----
    Only the name check is effective. The inner ``OverrideCheck`` descriptor is
    meant to verify, through ``__set_name__``, that the owning class is a
    subclass of ``parent_cls``. However ``decorator`` returns the original
    ``method`` rather than the ``OverrideCheck`` instance, so the descriptor is
    never bound to the class and ``__set_name__`` never runs. The subclass
    check is therefore dead code and no ``TypeError`` is raised when the owning
    class is unrelated to ``parent_cls``.
    """

    class OverrideCheck:
        """Descriptor intended to validate the owner class (see Notes above)."""

        def __init__(self, func, expected_parent_cls):
            self.func = func
            self.expected_parent_cls = expected_parent_cls

        def __set_name__(self, owner, name):
            if not issubclass(owner, self.expected_parent_cls):
                raise TypeError(
                    f"When using the @override decorator, {owner.__name__} must be a "
                    f"subclass of {parent_cls.__name__}!"
                )

            setattr(owner, name, self.func)

    def decorator(method):
        """Check that ``parent_cls`` has an attribute named like ``method``."""

        if method.__name__ not in dir(parent_cls):
            raise NameError(
                f"When using the @override decorator, {method.__name__} must override "
                f"the respective method (with the same name) of {parent_cls.__name__}!"
            )

        OverrideCheck(method, parent_cls)

        return method

    return decorator
