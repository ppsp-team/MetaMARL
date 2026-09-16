"""Enumerations of the reporting layer."""

from enum import Enum


class ReporterType(Enum):
    """Reporting backend named in a configuration.

    ``wandb`` streams figures to Weights & Biases; ``local`` writes to disk
    (CSV or TensorBoard event files).
    """

    wandb: str = "wandb"
    local: str = "local"


class Resolution(Enum):
    """Time axis of a report.

    ``env`` counts environment steps, ``inner`` counts inner-loop training
    iterations and ``outer`` counts outer-loop generations.
    """

    env: str = "env_steps"
    inner: str = "train_iters"
    outer: str = "generation"
