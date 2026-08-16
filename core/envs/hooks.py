from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def _hook(func: F, name: str) -> F:
    setattr(func, name, True)
    return func


def action(func: F) -> F:
    return _hook(func, "action")


def observation(func: F) -> F:
    return _hook(func, "observation")


def reset(func: F) -> F:
    return _hook(func, "reset")


def reward(func: F) -> F:
    return _hook(func, "reward")


def transition(func: F) -> F:
    return _hook(func, "transition")