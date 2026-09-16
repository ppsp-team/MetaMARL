"""Declarative selection of metric series to report.

A :class:`Query` names an x path and one or several y paths inside a
``MetricSchema`` tree. Paths are tuples of field names; at a *dynamic* node
(``dict[ID, MetricSchema]``) a component may be the wildcard ``"*"``, which
expands to every runtime key in sorted order::

    Query(title="Fish biomass by mechanism",
          x=("iter",),
          y=("train", "rollout", "by_mechanism", "*", "by_seed", "*", "fish_norm"),
          reduce="mean", error="std")
"""

from dataclasses import dataclass
from typing import Literal, Optional, TypeAlias, cast

from core.metrics.enums import ReduceProtocol

Path: TypeAlias = tuple[str | ReduceProtocol, ...]
PlotMode: TypeAlias = Literal["lines", "markers", "lines+markers"]


@dataclass(frozen=True, slots=True)
class Query:
    """Defines a reporting query over a MetricSchema."""

    title: str
    x: Path
    y: Path | tuple[Path, ...]
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    legend_labels: Optional[tuple[str, ...]] = None
    plot_modes: Optional[tuple[PlotMode, ...]] = None
    show_group_labels: bool = True
    color: Optional[Path] = None
    color_label: Optional[str] = None
    colorscale: Optional[str] = None
    error: Literal["none", "std"] = "none"
    error_path: Optional[Path] = None

    def __post_init__(self) -> None:
        if not self.x:
            raise ValueError("Query x path cannot be empty.")

        if not self.y:
            raise ValueError("Query y path cannot be empty.")

        if self.legend_labels is not None and len(self.legend_labels) != len(
            self.y_paths
        ):
            raise ValueError("legend_labels must have the same length as y paths.")

        if self.plot_modes is not None and len(self.plot_modes) != len(self.y_paths):
            raise ValueError("plot_modes must have the same length as y paths.")

        if self.color is None:
            if self.color_label is not None:
                raise ValueError("color_label requires a color path.")

            if self.colorscale is not None:
                raise ValueError("colorscale requires a color path.")

        if self.error not in ("none", "std"):
            raise ValueError(f"Unsupported error statistic: {self.error!r}.")

        if self.error == "none":
            if self.error_path is not None:
                raise ValueError("error_path requires an error statistic.")

            return

        if not self.error_path:
            raise ValueError(f"error={self.error!r} requires error_path.")

        if isinstance(
            self.error_path[-1],
            ReduceProtocol,
        ):
            raise ValueError(
                "error_path must point to the dynamic dimension, not its reduction operator."
            )

        target = self.error_path + (ReduceProtocol.MEAN,)

        if not any(path[: len(target)] == target for path in self.y_paths):
            raise ValueError(
                f"error_path {self.error_path} must be followed by ReduceProtocol.MEAN in one "
                "of the query's y paths."
            )

    @property
    def y_paths(self) -> tuple[Path, ...]:
        if self.y and isinstance(self.y[0], tuple):
            return cast(
                tuple[Path, ...],
                self.y,
            )

        return (cast(Path, self.y),)
