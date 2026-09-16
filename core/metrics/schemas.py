"""Root of the metric schema hierarchy.

Every logged structure derives from :class:`MetricSchema`; it carries the
``iter`` counter (last value kept) shared by all loggers.
"""

from typing import Optional

from pydantic import BaseModel, Field

from core.metrics.enums import ReduceProtocol


class MetricSchema(BaseModel):
    """Base class of every logged structure.

    Subclasses declare the metrics of one component as pydantic fields: a
    primitive field is a leaf reduced according to its ``reduce`` extra
    (``MEAN`` by default), a nested ``MetricSchema`` is a static sub-tree and
    a ``dict[ID, MetricSchema]`` is a dynamic node populated at runtime.
    ``iter`` is the only field shared by all schemas: the index of the
    iteration that produced the values (an integer counter, no unit), reduced
    with ``LAST`` so a populated schema always reports the most recent one.
    """

    iter: Optional[int] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.LAST}
    )
