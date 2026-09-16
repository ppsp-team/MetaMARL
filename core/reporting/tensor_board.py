"""TensorBoard reporter.

Each resolved y series is logged as one scalar tag indexed by the integer x
axis. Standard-deviation series are logged under a ``/std`` suffix.

TensorBoard scalar plots do not support per-point colour, so query ``color``
values are ignored.
"""

from __future__ import annotations

import logging
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from core.metrics.enums import ReduceProtocol
from core.metrics.metric.base import PrimitiveType
from core.reporting.base import Group, Reporter, Resolved
from core.reporting.config import ReporterConfig
from core.reporting.query import Path as QueryPath
from core.reporting.query import Query
from core.utils import sanitize_key

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class TensorBoardConfig(ReporterConfig):
    def __init__(
        self,
        *,
        project: str,
        log_dir: str = "runs",
    ) -> None:
        super().__init__(project=project)

        self.log_dir = Path(log_dir)

    def build(
        self,
        *,
        label: Optional[str] = None,
    ) -> TensorBoardReporter:
        name = f"{self.world}-{label}" if label is not None else self.world

        return TensorBoardReporter(
            log_dir=self.log_dir / self.project_name / name,
        )


class TensorBoardReporter(Reporter):
    def __init__(
        self,
        *,
        log_dir: Path,
    ) -> None:
        self._log_dir = Path(log_dir)
        self._writer: SummaryWriter | None = None

    @property
    def log_dir(self) -> Path:
        return self._log_dir

    def _get_writer(self) -> SummaryWriter:
        if self._writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as e:
                raise ImportError(
                    "TensorBoardReporter needs the 'tensorboard' package: "
                    "uv sync --extra tensorboard"
                ) from e

            self._writer = SummaryWriter(
                log_dir=str(self._log_dir),
            )

        return self._writer

    @staticmethod
    def _path_name(path: QueryPath) -> str:
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

    @staticmethod
    def _step(
        x_value: PrimitiveType,
        query: Query,
    ) -> int:
        try:
            step = int(x_value)
        except (TypeError, ValueError) as e:
            raise TypeError(
                "TensorBoard x-axis must be integer-valued: "
                f"{query.x} contains {x_value!r}."
            ) from e

        if step != x_value:
            raise TypeError(
                "TensorBoard x-axis must be integer-valued: "
                f"{query.x} contains {x_value!r}."
            )

        return step

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

        if colors is not None:
            logger.info(
                "TensorBoardReporter ignores the color path %s "
                "of query %r because TensorBoard scalars do not "
                "support per-point colour.",
                query.color,
                query.title,
            )

        writer = self._get_writer()
        title = sanitize_key(query.title)
        labels = (
            query.legend_labels
            if query.legend_labels is not None
            else (None,) * len(query.y_paths)
        )

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

                label = self._series_label(
                    path,
                    group,
                    label=path_label,
                )
                tag = f"{title}/{sanitize_key(label)}"

                for x_value, y_value in zip(
                    x_values,
                    y_values,
                    strict=True,
                ):
                    writer.add_scalar(
                        tag=tag,
                        scalar_value=y_value,
                        global_step=self._step(
                            x_value,
                            query,
                        ),
                    )

                if group in resolved_errors:
                    for x_value, error_value in zip(
                        x_values,
                        resolved_errors[group],
                        strict=True,
                    ):
                        writer.add_scalar(
                            tag=f"{tag}/std",
                            scalar_value=error_value,
                            global_step=self._step(
                                x_value,
                                query,
                            ),
                        )

        writer.flush()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()

            self._writer = None
