"""Typed metric accumulation driven by pydantic schemas.

A ``MetricSchema`` declares *what* is logged: each field is a leaf metric whose
reducer comes from ``Field(json_schema_extra={"reduce": ReduceProtocol.X})``
(MEAN by default), a nested ``MetricSchema`` is a static sub-tree, and a
``dict[ID, MetricSchema]`` is a *dynamic* node whose children are created on
first use (one per agent, policy, candidate, ...). :class:`MetricLogger` builds
the matching tree of :class:`~core.metrics.metric.base.Metric` objects,
accumulates values through :meth:`~MetricLogger.push` / :meth:`~MetricLogger.push_data`,
and returns populated schema instances through :meth:`~MetricLogger.peek`
(non-destructive, raw histories) or :meth:`~MetricLogger.reduce` (destructive,
reduced values). Pushing a *subclass* of a declared schema specializes that
sub-tree at runtime, which is how an ES logger ends up holding the concrete
RLlib and environment schemas of the inner level.
"""

from __future__ import annotations

from abc import ABC
from typing import Any, ClassVar, TypeAlias, get_args, get_origin

import tree  # dm_tree

from core.metrics.enums import ReduceProtocol
from core.metrics.metric.base import Metric
from core.metrics.metric.factory import MetricFactory
from core.metrics.schemas import MetricSchema

Path: TypeAlias = tuple[str, ...]


class Node(dict[str, "Node | Metric"]):
    """One level of the metric tree built from a ``MetricSchema``.

    A node maps field names (or runtime ids for a *dynamic* node) to child
    nodes or :class:`Metric` leaves. ``schema`` is the pydantic class that the
    node rebuilds in :meth:`construct`; ``dynamic`` marks a
    ``dict[ID, MetricSchema]`` field whose children are created on first push.
    """

    schema: type[MetricSchema]
    dynamic: bool = False
    subtree_reduce: ReduceProtocol | None = None

    def __init__(
        self,
        *args,
        schema: type[MetricSchema] | None = None,
        dynamic: bool = False,
        subtree_reduce: ReduceProtocol | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.schema = schema
        self.dynamic = dynamic
        self.subtree_reduce = subtree_reduce

    def construct(self, data: dict[str, Any]) -> MetricSchema | dict[str, Any]:
        """Rebuild a schema instance from values laid out like this node.

        ``data`` mirrors the node: one entry per child, holding either a leaf
        value or the nested dictionary of a child node. A dynamic node returns
        a plain ``dict`` keyed by runtime id; a static node returns
        ``schema.model_construct(**values)``, so no pydantic validation runs.

        Raises
        ------
        RuntimeError
            If a static node was built without a schema.
        """

        if self.dynamic:
            return {
                dynamic_id: child.construct(data[dynamic_id])
                for dynamic_id, child in self.items()
            }

        values: dict[str, Any] = {}

        for field_name, child in self.items():
            value = data[field_name]

            if isinstance(child, Metric):
                values[field_name] = value
            else:
                values[field_name] = child.construct(value)

        if self.schema is None:
            raise RuntimeError("Runtime Node has no schema.")

        return self.schema.model_construct(**values)


class MetricLogger(ABC):
    """Tree of metric accumulators mirroring a ``MetricSchema``.

    Build one with :meth:`from_schema` (direct instantiation raises
    ``TypeError``), feed it with :meth:`push` (one path) or :meth:`push_data`
    (a whole schema instance), then read it back with :meth:`peek` (raw
    histories, non-destructive) or :meth:`reduce` (reduced values, clears the
    accumulators). The logger is the second layer of the reporting stack:
    ``MetricSchema`` declares, ``MetricLogger`` accumulates,
    :class:`~core.reporting.query.Query` selects and
    :class:`~core.reporting.base.Reporter` renders.
    """

    _TOKEN: ClassVar[object] = object()
    _schema: type[MetricSchema]
    _refs: dict[Path, Metric]
    _root: str
    _tree: Node

    def __new__(cls, *, _token: object | None = None) -> "MetricLogger":
        if _token is not cls._TOKEN:
            raise TypeError(
                "MetricLogger cannot be instantiated directly. "
                "Use MetricLogger.from_schema(schema)"
            )

        return super().__new__(cls)

    def __init__(self, *, _token: object | None = None) -> None:
        if _token is not self._TOKEN:
            raise TypeError(
                "MetricLogger cannot be instantiated directly. "
                "Use MetricLogger.from_schema(schema)"
            )

    # TODO immutability
    @classmethod
    def from_schema(cls, schema: type[MetricSchema]) -> "MetricLogger":
        """Create a logger whose tree mirrors ``schema``.

        This is the only supported constructor: it builds the ``Node`` tree and
        the flat ``path -> Metric`` index that :meth:`push` looks up first.
        """

        tree, refs = cls._build_from_schema(schema)
        self = cls.__new__(cls, _token=cls._TOKEN)
        self._schema = schema
        self._root = schema.__name__
        self._tree = tree
        self._refs = refs

        return self

    @classmethod
    def _build_from_schema(
        cls,
        schema: type[MetricSchema],
        *,
        prefix: Path = (),
        dynamic: bool = False,
        subtree_reduce: ReduceProtocol | None = None,
    ) -> tuple[Node, dict[Path, Metric]]:
        # TODO guardrails when Metric isnt well formatted
        refs: dict[Path, Metric] = {}

        node = Node(schema=schema, dynamic=dynamic, subtree_reduce=subtree_reduce)

        for field_name, field in schema.__pydantic_fields__.items():
            path = prefix + (field_name,)
            ann = field.annotation

            # Unwrap Optional[T] / T | None.
            args = get_args(ann)

            if type(None) in args:
                non_none = tuple(arg for arg in args if arg is not type(None))

                if len(non_none) == 1:
                    ann = non_none[0]

            extra = field.json_schema_extra or {}
            field_protocol = extra.get("reduce")
            field_override = extra.get("subtree_reduce")
            reduce = subtree_reduce if subtree_reduce is not None else field_override

            if isinstance(ann, type) and issubclass(ann, MetricSchema):
                child, child_ref = cls._build_from_schema(
                    ann, prefix=path, subtree_reduce=reduce
                )

                if not node.dynamic:
                    node[field_name] = child

                    refs.update(child_ref)

                continue

            # CASE WHEN dict[ID, MetricSchema]
            if get_origin(ann) is dict:
                _, value_ann = get_args(ann)

                if not (
                    isinstance(value_ann, type) and issubclass(value_ann, MetricSchema)
                ):
                    raise TypeError(
                        f"{schema.__name__}.{field_name} must be "
                        f"dict[ID, MetricSchema], got {ann!r}"
                    )

                node[field_name] = Node(
                    schema=value_ann, dynamic=True, subtree_reduce=reduce
                )

                continue

            protocol = (
                subtree_reduce
                if subtree_reduce is not None
                else field_protocol or ReduceProtocol.MEAN
            )
            metric = MetricFactory.create(protocol)

            if not node.dynamic:
                node[field_name] = metric
                refs[path] = metric

        return node, refs

    def _resolve_path(
        self, path: Path, *, node: Node, index: int, prefix: Path
    ) -> Metric:
        if node.dynamic:
            if index >= len(path):
                raise KeyError(f"Expected dynamic ID at path: {path}")

            if node.schema is None:
                raise RuntimeError(f"Dynamic node at {prefix} has no schema.")

            dynamic_id = path[index]
            prefix = prefix + (dynamic_id,)
            runtime_child = node.get(dynamic_id)

            if runtime_child is None:
                runtime_child, refs = self._build_from_schema(
                    node.schema,
                    prefix=prefix,
                    subtree_reduce=node.subtree_reduce,
                )
                node[dynamic_id] = runtime_child

                self._refs.update(refs)

            return self._resolve_path(
                path=path,
                node=runtime_child,
                index=index + 1,
                prefix=prefix,
            )

        if index >= len(path):
            raise KeyError(f"Logger path does not point to a metric: {path}")

        field_name = path[index]

        try:
            child = node[field_name]
        except KeyError:
            raise KeyError(f"Unknown logger path: {path}") from None

        child_path = prefix + (field_name,)

        # leaf
        if isinstance(child, Metric):
            if index != len(path) - 1:
                raise KeyError(f"Path continues beyond metric leaf: {path}")

            return child

        return self._resolve_path(
            node=child,
            path=path,
            index=index + 1,
            prefix=child_path,
        )

    # TODO refactor into one push function
    def push_data(
        self, data: MetricSchema, prefix: Path = (), node: Node | None = None
    ) -> None:
        """Push all leaf values of a MetricSchema into their correspondi ng metrics."""

        if node is None:
            if not prefix and type(data) is not self._schema:
                raise TypeError(
                    f"Expected {self._schema.__name__}, got {type(data).__name__}."
                )

            node = self._tree

        for field_name in type(data).model_fields:
            value = getattr(data, field_name)

            if value is None:
                continue

            path = prefix + (field_name,)

            try:
                child_node = node[field_name]
            except KeyError:
                raise KeyError(
                    f"Unknown logger field {field_name!r} at {prefix} for {type(data).__name__}."
                ) from None

            if isinstance(value, MetricSchema):
                if not isinstance(child_node, Node):
                    raise TypeError(
                        f"Expected Node at {path}, got {type(child_node).__name__}."
                    )

                runtime_schema = type(value)
                declared_schema = child_node.schema

                if declared_schema is None:
                    raise RuntimeError(f"Node at {path} has no declared schema.")

                if runtime_schema is not declared_schema:
                    if not issubclass(runtime_schema, declared_schema):
                        raise TypeError(
                            f"{runtime_schema.__name__} is not a subclass of "
                            f"{declared_schema.__name__} at {path}."
                        )

                    subtree_reduce = child_node.subtree_reduce
                    child_node, refs = self._build_from_schema(
                        runtime_schema,
                        prefix=path,
                        subtree_reduce=subtree_reduce,
                    )
                    node[field_name] = child_node

                    self._refs.update(refs)

                self.push_data(value, prefix=path, node=child_node)
                continue

            if isinstance(value, dict):
                if not isinstance(child_node, Node) or not child_node.dynamic:
                    raise TypeError(f"Expected dynamic Node at {path}.")

                declared_schema = child_node.schema

                if declared_schema is None:
                    raise RuntimeError(
                        f"Dynamic node at {path} has no declared schema."
                    )

                for dynamic_id, dynamic_child in value.items():
                    if not isinstance(
                        dynamic_child, MetricSchema
                    ):  # TODO only verify it is a metric schema not that its env
                        raise TypeError(
                            f"Expected MetricSchema at {path + (dynamic_id,)}, "
                            f"got {type(dynamic_child).__name__}."
                        )

                    runtime_schema = type(dynamic_child)

                    if not issubclass(runtime_schema, declared_schema):
                        raise TypeError(
                            f"{runtime_schema.__name__} is not a subclass of "
                            f"{declared_schema.__name__} at {path}."
                        )

                    runtime_path = path + (dynamic_id,)
                    runtime_node = child_node.get(dynamic_id)

                    if runtime_node is None:
                        runtime_node, refs = self._build_from_schema(
                            runtime_schema,
                            prefix=runtime_path,
                            subtree_reduce=child_node.subtree_reduce,
                        )
                        child_node[dynamic_id] = runtime_node

                        self._refs.update(refs)
                    elif not isinstance(runtime_node, Node):
                        raise TypeError(
                            f"Expected runtime Node at {runtime_path}, got "
                            f"{type(runtime_node).__name__}."
                        )
                    elif runtime_node.schema is not runtime_schema:
                        raise TypeError(
                            f"Runtime schema changed at {runtime_path}: "
                            f"{runtime_node.schema.__name__} -> {runtime_schema.__name__}."
                        )

                    self.push_data(
                        dynamic_child,
                        prefix=runtime_path,
                        node=runtime_node,
                    )

                continue

            if not isinstance(child_node, Metric):
                raise TypeError(
                    f"Expected Metric at {path}, got {type(child_node).__name__}."
                )

            child_node.push(value)

    def push(self, key: Path, value: Any) -> None:
        # TODO narrow down Any to stricter type annotation
        """Logs a new value or item under a (strictly existing) path to the logger"""

        metric = self._refs.get(key)

        if metric is None:
            metric = self._resolve_path(path=key, node=self._tree, index=0, prefix=())

        metric.push(value)

    def peek_value(self, key: Path) -> Any:
        # TODO narrow down Any to stricter type annotation
        # NOTE this does not work for sub trees as of now !
        """
        Reads a metric value given its path without destructively reducing it
        """

        metric = self._refs.get(key)

        if metric is None:
            raise KeyError(f"Unknown logger path: {key}")

        return metric.peek()

    def peek(self) -> MetricSchema:
        """
        Returns all accumulated values as a MetricSchema
        without destructively reducing them.
        """

        def _peek(path: Path, metric: Metric):
            try:
                return metric.peek(compile=False)
            except Exception as e:
                raise ValueError(
                    f"Error peeking metric {metric} at path {path}."
                ) from e

        peeked = tree.map_structure_with_path(
            _peek,
            self._tree,
        )

        return self._tree.construct(peeked)

    def reduce(self) -> MetricSchema:
        """
        Reduces all logged values based on their settings and returns a MetricSchema object.
        """

        def _reduce(path: Path, metric: Metric):
            try:
                return metric.reduce(compile=True)

            # TODO custom exceptions
            except Exception as e:
                raise ValueError(
                    f"Error reducing metrics {metric} at path {path}."
                ) from e

        reduced = tree.map_structure_with_path(
            _reduce,
            self._tree,
        )

        return self._tree.construct(reduced)

    def compile(self) -> dict:
        """
        Compiles all current values and throughputs into a single dictionary.
        """

        return self.reduce().model_dump(serialize_as_any=True)

    def reset(self) -> None:
        """
        Resets all data stored in this MetricLogger.
        """

        for path, metric in self._refs.items():
            try:
                metric.flush()

            # TODO custom exceptions
            except Exception as e:
                raise ValueError(
                    f"Error flushing metrics {metric} at path {path}."
                ) from e

    def flush(self, key: Path) -> None:
        """
        Flush all accumulated values for the metric at `key`.
        """

        metric = self._refs.get(key)

        if metric is None:
            raise KeyError(f"Unknown logger path: {key}")

        try:
            metric.flush()
        except Exception as e:
            raise ValueError(f"Error flushing metric {metric} at path {key}.") from e
