from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np

from core.metrics.enums import ReduceProtocol
from core.metrics.metric.base import PrimitiveType
from core.metrics.schemas import MetricSchema
from core.reporting.query import Path, Query

Group: TypeAlias = tuple[tuple[str, str], ...]
Resolved: TypeAlias = dict[Group, list[PrimitiveType]]


@dataclass(frozen=True, slots=True)
class PathResolution:
    values: Resolved
    errors: Resolved


class Reporter(ABC):
    """Base interface for reporting reduced metric results.

    A Reporter receives populated MetricSchema objects, resolves its configured
    queries against those schemas, and delegates the resulting data to a
    backend-specific reporting implementation.

    Reporter views are write-once: they may be configured after construction,
    but cannot be replaced once set.
    """

    # TODO how to store data in the results reporter ?
    _queries: tuple[Query, ...] = ()
    _schema: type[MetricSchema] | None = None

    @property
    def queries(self) -> tuple[Query, ...]:
        """Return the queries registered with this reporter."""

        return self._queries

    def add_query(self, *queries: Query) -> None:
        """Register one or more reporting queries.

        Args:
            *queries: Queries to register with this reporter.
        """

        self._queries += queries

    @property
    def schema(self) -> type[MetricSchema] | None:
        return self._schema

    @schema.setter
    def schema(self, schema: type[MetricSchema]) -> None:
        if self._schema is not None:
            raise AttributeError(
                "Reporter schema has already been set and cannot be changed."
            )

        self._schema = schema

    # TODO remove reduction logic from reporting
    def _resolve_path(
        self,
        path: Path,
        metrics: MetricSchema | dict | list[Any],
        *,
        index: int = 0,
        group: Group = (),
        junction: str | None = None,
        error: Literal["none", "std"] = "none",
        error_path: Path | None = None,
    ) -> PathResolution:
        """
        Returns the Metric object following the Path in a metric schema
        """

        if index >= len(path):
            if not isinstance(metrics, list):
                raise KeyError(f"Path does not point to a metric series: {path}")

            if any(isinstance(value, list) for value in metrics):
                raise ValueError(
                    "Path resolves to a nested metric series. A series reduction is required: {path}"
                )

            return PathResolution(values={group: metrics}, errors={})

        token = path[index]

        if isinstance(metrics, dict):
            if token == ReduceProtocol.SERIES:
                values: Resolved = {}
                errors: Resolved = {}

                for dynamic_id, child in sorted(
                    metrics.items(), key=lambda item: str(item[0])
                ):
                    child_group = group + (
                        (
                            junction or "dict",
                            str(dynamic_id),
                        ),
                    )
                    child_result = self._resolve_path(
                        path=path,
                        metrics=child,
                        index=index + 1,
                        group=child_group,
                        error=error,
                        error_path=error_path,
                    )

                    values.update(child_result.values)
                    errors.update(child_result.errors)

                return PathResolution(values=values, errors=errors)

            if token == ReduceProtocol.MEAN:
                branches = [
                    self._resolve_path(
                        path=path,
                        metrics=child,
                        index=index + 1,
                        group=group,
                        error=error,
                        error_path=error_path,
                    )
                    for child in metrics.values()
                ]

                if not branches:
                    return PathResolution(values={}, errors={})

                groups = set(branches[0].values)

                if any(set(branch.values) != groups for branch in branches[1:]):
                    raise ValueError(
                        "Cannot compute mean across branches with different SERIES groups."
                    )

                reduction_path = path[:index]
                capture_error = error != "none" and error_path == reduction_path

                if not capture_error and any(branch.errors for branch in branches):
                    raise ValueError(
                        "error_path is nested below another MEAN reduction. Error propagation "
                        "through additional reductions is not defined."
                    )

                reduced: Resolved = {}
                errors: Resolved = {}

                for branch_group in groups:
                    series = [branch.values[branch_group] for branch in branches]
                    lengths = {len(values) for values in series}

                    if len(lengths) != 1:
                        raise ValueError(
                            "Cannot compute pointwise mean over series with different lengths: "
                            f"{sorted(lengths)}."
                        )

                    values = np.asarray(
                        series,
                        dtype=np.float64,
                    )
                    reduced[branch_group] = np.mean(
                        values,
                        axis=0,
                    ).tolist()

                    if capture_error:
                        if error == "std":
                            errors[branch_group] = np.std(
                                values,
                                axis=0,
                            ).tolist()

                return PathResolution(
                    values=reduced,
                    errors=errors,
                )

            if isinstance(token, ReduceProtocol):
                raise NotImplementedError(
                    f"Dictionary query reduction {token} is not supported."
                )

            try:
                child = metrics[token]
            except KeyError:
                raise KeyError(f"Unknown metric path: {path}") from None

            return self._resolve_path(
                path=path,
                metrics=child,
                index=index + 1,
                group=group,
                error=error,
                error_path=error_path,
            )

        """MetricSchema"""

        if isinstance(token, ReduceProtocol):
            raise TypeError(
                f"{token} cannot be applied to {type(metrics).__name__} in path {path}."
            )

        try:
            child = getattr(metrics, token)
        except AttributeError:
            raise KeyError(f"Unknown metric path: {path}") from None

        return self._resolve_path(
            path=path,
            metrics=child,
            index=index + 1,
            group=group,
            junction=token if isinstance(child, dict) else None,
            error=error,
            error_path=error_path,
        )

    def _resolve_query(
        self,
        metrics: MetricSchema,
        query: Query,
    ) -> tuple[
        Resolved,
        list[Resolved],
        list[Resolved],
        Resolved | None,
    ]:
        """Resolve a query against a populated metric schema."""

        x_result = self._resolve_path(
            path=query.x,
            metrics=metrics,
        )
        xs = x_result.values
        y_results = [
            self._resolve_path(
                path=path,
                metrics=metrics,
                error=query.error,
                error_path=query.error_path,
            )
            for path in query.y_paths
        ]
        yss = [result.values for result in y_results]
        error_yss = [result.errors for result in y_results]

        colors: Resolved | None = None

        if query.color is not None:
            colors = self._resolve_path(
                path=query.color,
                metrics=metrics,
            ).values

        for path, ys, errors in zip(
            query.y_paths,
            yss,
            error_yss,
        ):
            if set(xs) == {()}:
                x = xs[()]

                for group, y in ys.items():
                    if len(x) != len(y):
                        raise ValueError(
                            "Query series must have equal length: x={query.x} ({len(x)}), "
                            f"y={path}, group={group} ({len(y)})."
                        )
            else:
                if set(xs) != set(ys):
                    raise ValueError(
                        "Dynamic x and y groups do not match: "
                        f"x={set(xs)}, y={set(ys)}."
                    )

                for group in xs:
                    if len(xs[group]) != len(ys[group]):
                        raise ValueError(
                            "Query series must have equal "
                            f"length for group {group}: x={len(xs[group])}, y={len(ys[group])}."
                        )

            for group, values in errors.items():
                if group not in ys:
                    raise ValueError(
                        f"Error group {group} has no corresponding y series."
                    )

                if len(values) != len(ys[group]):
                    raise ValueError(
                        f"Error series must have the same length as y for group {group}."
                    )

            if colors is not None:
                for group, y in ys.items():
                    if () in colors:
                        color_values = colors[()]
                    else:
                        try:
                            color_values = colors[group]
                        except KeyError:
                            raise ValueError(
                                f"No color series exists for group {group}."
                            ) from None

                    if len(color_values) != len(y):
                        raise ValueError(
                            f"Color series must have the same length as y for group {group}: "
                            f"color={len(color_values)}, y={len(y)}."
                        )

        return (
            xs,
            yss,
            error_yss,
            colors,
        )

    @abstractmethod
    def _report(
        self,
        query: Query,
        x: Resolved,
        ys: list[Resolved],
        errors: list[Resolved],
        colors: Resolved | None,
    ) -> None:
        """Report one resolved query using the concrete reporting backend.

        Args:
            query: Query defining how the resolved values should be represented.
            x: Resolved values for the query's x dimension.
            y: Resolved values for the query's y dimension.
        """

        ...

    def report(self, metrics: MetricSchema) -> None:
        """Report all applicable configured views for a metric schema.

        Each configured query is resolved against `metrics`. Queries whose
        required values are available are forwarded to the backend-specific
        reporting implementation.

        Args:
            metrics: Reduced metric schema to report.
        """

        for query in self._queries:
            x, ys, errors, colors = self._resolve_query(metrics, query)

            self._report(query, x, ys, errors, colors)

    @abstractmethod
    def close(self) -> None:
        """Close the reporter instance"""

        ...
