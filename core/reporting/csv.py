"""CSV reporter: one long-form file per query, rewritten on every report.

Each row is:

    (query, x, series, value, error, color)

``series`` is the resolved y-series label. ``error`` contains the pointwise
standard deviation when requested by the query, and ``color`` contains the
resolved per-point color value when a color path is configured. Missing
error/color values are written as empty cells.
"""

from __future__ import annotations

import csv
from enum import Enum
from pathlib import Path
from typing import Optional

from core.metrics.enums import ReduceProtocol
from core.reporting.base import Group, Reporter, Resolved
from core.reporting.config import ReporterConfig
from core.reporting.query import Path as QueryPath
from core.reporting.query import Query
from core.utils import sanitize_key


class CSVConfig(ReporterConfig):
    def __init__(
        self,
        *,
        project: str,
        output_dir: str = "results",
    ) -> None:
        super().__init__(project=project)

        self.output_dir = Path(output_dir)

    def build(
        self,
        *,
        label: Optional[str] = None,
    ) -> CSVReporter:
        name = f"{self.world}-{label}" if label is not None else self.world

        return CSVReporter(
            output_dir=self.output_dir / self.project_name / name,
        )


class CSVReporter(Reporter):
    HEADER = (
        "query",
        "x",
        "series",
        "value",
        "error",
        "color",
    )

    def __init__(
        self,
        *,
        output_dir: Path,
    ) -> None:
        self._output_dir = Path(output_dir)

        self._output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    def path_for(
        self,
        query: Query,
    ) -> Path:
        return self._output_dir / f"{sanitize_key(query.title)}.csv"

    @staticmethod
    def _path_name(
        path: QueryPath,
    ) -> str:
        return "/".join(
            str(token.value) if isinstance(token, Enum) else token
            for token in path
            if not isinstance(token, ReduceProtocol)
        )

    @classmethod
    def _series_label(
        cls,
        path: QueryPath,
        group: Group,
        label: Optional[str] = None,
    ) -> str:
        name = label if label is not None else cls._path_name(path)

        if not group:
            return name

        group_name = ", ".join(
            f"{junction}={dynamic_id}" for junction, dynamic_id in group
        )

        return f"{name} [{group_name}]"

    def _report(
        self,
        query: Query,
        x: Resolved,
        ys: list[Resolved],
        errors: list[Resolved],
        colors: Resolved | None,
    ) -> None:
        if not any(ys):
            return

        labels = (
            query.legend_labels
            if query.legend_labels is not None
            else (None,) * len(query.y_paths)
        )

        with self.path_for(query).open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)

            writer.writerow(self.HEADER)

            for path, resolved_y, resolved_errors, path_label in zip(
                query.y_paths,
                ys,
                errors,
                labels,
                strict=True,
            ):
                for group, y_values in resolved_y.items():
                    if () in x:
                        x_values = x[()]
                    else:
                        try:
                            x_values = x[group]
                        except KeyError:
                            raise ValueError(
                                f"No x series exists for group {group}."
                            ) from None

                    error_values = resolved_errors.get(group)

                    if error_values is None:
                        error_values = [""] * len(y_values)

                    color_values = None

                    if colors is not None:
                        if () in colors:
                            color_values = colors[()]
                        else:
                            try:
                                color_values = colors[group]
                            except KeyError:
                                raise ValueError(
                                    f"No color series exists for group {group}."
                                ) from None

                    if color_values is None:
                        color_values = [""] * len(y_values)

                    label = self._series_label(
                        path,
                        group,
                        label=path_label,
                    )

                    for (
                        x_value,
                        y_value,
                        error_value,
                        color_value,
                    ) in zip(
                        x_values,
                        y_values,
                        error_values,
                        color_values,
                        strict=True,
                    ):
                        writer.writerow(
                            (
                                query.title,
                                x_value,
                                label,
                                y_value,
                                error_value,
                                color_value,
                            )
                        )

    def close(self) -> None:
        pass
