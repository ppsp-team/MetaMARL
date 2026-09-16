"""Small numeric and string helpers shared across the package.

Two families live here: tolerant scalar conversions used by the reporting
layer (``to_float``, ``finite``, ``safe_ratio``, ``flatten_numeric``,
``sanitize_key``) and smooth, differentiable stand-ins for ``max``/``min``/
clipping used when shaping rewards (``sigmoid``, ``smooth_positive``,
``smooth_min``, ``smooth_cap_01``, ``smooth_positive_zero_at_origin``).
"""

import re
import uuid
from collections.abc import Mapping
from typing import AbstractSet, Any, Optional

import numpy as np

# Generic utils

# TODO restrict Any type annotation.
EPS = 1e-8


def generate_uuid(registry: AbstractSet[Any]) -> str:
    """Generate a UUID4 string that is not already present in ``registry``.

    Parameters
    ----------
    registry : AbstractSet[Any]
        Existing identifiers to avoid (any container supporting ``in``).

    Returns
    -------
    str
        A fresh ``uuid.uuid4()`` string absent from ``registry``.

    Examples
    --------
    >>> uid = generate_uuid({"a", "b"})
    >>> uid not in {"a", "b"}
    True
    """

    while True:
        id = str(uuid.uuid4())

        if id not in registry:
            return id


def safe_ratio(num: Any, den: Any) -> Optional[float]:
    """Divide two scalars, returning ``None`` instead of raising.

    Parameters
    ----------
    num, den : Any
        Values convertible by ``finite``.

    Returns
    -------
    float or None
        ``num / den``, or ``None`` if either operand is missing, non-finite,
        or ``den`` is zero.

    Examples
    --------
    >>> safe_ratio(1.0, 0.0) is None
    True
    """

    fnum = finite(num)
    fden = finite(den)

    if fnum is None or fden is None or fden == 0.0:
        return None

    return float(fnum) / float(fden)


def to_float(x: Any) -> Optional[float]:
    """Best-effort conversion of a scalar (incl. NumPy scalars) to ``float``.

    Parameters
    ----------
    x : Any
        Value to convert. ``numpy.generic`` instances go through ``.item()``.

    Returns
    -------
    float or None
        The converted value, or ``None`` if ``x`` is ``None`` or the
        conversion raises. NaN and inf are returned as-is; use ``finite`` to
        reject them.

    Examples
    --------
    >>> to_float(np.float32(2.5)), to_float("abc")
    (2.5, None)
    """

    if x is None:
        return None

    try:
        if isinstance(x, np.generic):
            return float(x.item())

        return float(x)
    except Exception:
        return None


def finite(x: Any) -> Optional[float]:
    """Convert to ``float`` and reject NaN/inf.

    Parameters
    ----------
    x : Any
        Value passed to ``to_float``.

    Returns
    -------
    float or None
        The finite float, or ``None`` if conversion failed or the value is
        NaN or infinite.
    """

    fx = to_float(x)

    if fx is None or not np.isfinite(fx):
        return None

    return fx


def sanitize_key(s: str) -> str:
    """Make a string safe to use as a metric key (e.g. for W&B).

    Runs of characters outside ``[0-9a-zA-Z_-]`` are collapsed into a single
    underscore. The input is coerced with ``str`` first.

    Examples
    --------
    >>> sanitize_key("info/stock level (t)")
    'info_stock_level_t_'
    """

    # keep alnum, underscore, dash; replace everything else with underscore
    return re.sub(r"[^0-9a-zA-Z_\-]+", "_", str(s))


def is_mapping(x: Any) -> bool:
    """Return ``True`` if ``x`` is a ``collections.abc.Mapping``."""

    return isinstance(x, Mapping)


def flatten_numeric(value: Any) -> list[float]:
    """Flatten a scalar, sequence or array into a flat list of floats.

    Parameters
    ----------
    value : Any
        Anything ``numpy.asarray`` accepts: a scalar, a nested list, or an
        array of any shape.

    Returns
    -------
    list of float
        Row-major flattening of ``value``; a scalar yields a one-element list.

    Examples
    --------
    >>> flatten_numeric([[1, 2], [3, 4]])
    [1.0, 2.0, 3.0, 4.0]
    """

    import numpy as np

    arr = np.asarray(value)

    if arr.ndim == 0:
        return [float(arr)]

    return [float(x) for x in arr.reshape(-1)]


def sigmoid(x: float) -> float:
    """Numerically stable logistic function ``1 / (1 + exp(-x))``.

    The two-branch form avoids overflow in ``exp`` for large ``|x|``.

    Examples
    --------
    >>> sigmoid(0.0)
    0.5
    """

    x = float(x)

    if x >= 0.0:
        z = np.exp(-x)

        return float(1.0 / (1.0 + z))

    z = np.exp(x)

    return float(z / (1.0 + z))


def smooth_positive(x: float, width: float) -> float:
    """Smooth approximation of ``max(x, 0)`` (softplus with a width scale).

    Computes ``width * log(1 + exp(x / width))``. The result is strictly
    positive everywhere and equals ``width * ln 2`` at ``x = 0``; use
    ``smooth_positive_zero_at_origin`` when an exact zero at the origin is
    required.

    Parameters
    ----------
    x : float
        Input value.
    width : float
        Transition scale, in the same units as ``x``. Clamped to at least
        ``EPS``; smaller widths approach the hard hinge.

    Returns
    -------
    float
        Softplus value, always ``> 0``.
    """

    width = max(float(width), EPS)

    return float(
        width
        * np.logaddexp(
            0.0,
            float(x) / width,
        )
    )


def smooth_min(
    a: float,
    b: float,
    width: float,
) -> float:
    """Smooth transition between ``a`` and ``b`` approximating ``min(a, b)``.

    Returns a sigmoid-weighted blend ``(1 - w) * a + w * b`` where
    ``w = sigmoid((a - b) / width)``, so the output leans towards ``b`` when
    ``a > b`` and towards ``a`` otherwise.

    Parameters
    ----------
    a, b : float
        Values to blend.
    width : float
        Transition scale, in the same units as ``a`` and ``b``. Clamped to at
        least ``EPS``.

    Returns
    -------
    float
        Blended value, between ``min(a, b)`` and ``max(a, b)``.
    """

    a = float(a)
    b = float(b)
    width = max(float(width), EPS)
    weight_on_b = sigmoid((a - b) / width)

    return float((1.0 - weight_on_b) * a + weight_on_b * b)


def smooth_cap_01(x: float) -> float:
    """Smooth saturation of a non-negative value into ``[0, 1)``.

    Computes ``1 - exp(-x)``: zero at the origin, slope one there, and
    asymptotically one. Negative inputs give negative outputs; the caller is
    expected to pass ``x >= 0``.
    """

    x = float(x)

    return float(1.0 - np.exp(-x))


def smooth_positive_zero_at_origin(
    x: float,
    width: float,
) -> float:
    """Smooth approximation of ``max(x, 0)`` that is exactly zero at ``x=0``.

    Shifts ``smooth_positive`` down by ``width * ln 2`` so that the origin maps
    to zero, then clamps at zero from below (the shifted softplus is negative
    for ``x < 0``).

    Parameters
    ----------
    x : float
        Input value.
    width : float
        Transition scale, in the same units as ``x``. Clamped to at least
        ``EPS``.

    Returns
    -------
    float
        Non-negative value, ``0.0`` for ``x <= 0``.
    """

    width = max(float(width), EPS)
    value = width * (np.logaddexp(0.0, float(x) / width) - np.log(2.0))

    return float(max(0.0, value))
