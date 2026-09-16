"""Process-wide Ray bootstrap: environment variables, logging and ``ray.init``.

``RayRuntimeConfig`` gathers the knobs that must be set before Ray and torch
start (device visibility, thread counts, log verbosity) and ``RayRuntime``
applies them once per process. The bilevel optimizer calls
``RayRuntime.ensure_initialized`` before building any actor.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional

import ray
import torch

DeviceType = Literal["cpu", "cuda", "mps"]


@dataclass
class RayRuntimeConfig:
    """Settings applied to the process before ``ray.init``.

    Attributes
    ----------
    device : {"cpu", "cuda", "mps"}
        Torch default device, also used to hide or expose accelerators
        through environment variables (see ``_apply_env_vars``).
    num_cpus : int or None
        If set, forwarded to ``ray.init(num_cpus=...)``.
    num_gpus : int or None
        If set, exported as ``RLLIB_NUM_GPUS`` and forwarded to
        ``ray.init(num_gpus=...)``.
    omp_threads : int
        Exported as ``OMP_NUM_THREADS`` to stop torch from oversubscribing
        cores when many env runners share the machine. Default ``1``.
    disable_mps : bool
        When ``device="cpu"``, also export ``RAY_USE_MPS=0``.
    disable_cuda : bool
        When ``device="cuda"``, hide the GPUs with ``CUDA_VISIBLE_DEVICES=""``.
        The default ``True`` therefore disables CUDA even when it is
        requested; set it to ``False`` to actually use a GPU.
    logging_level : str
        Passed to ``ray.init(logging_level=...)``.
    runtime_env : dict or None
        Passed to ``ray.init(runtime_env=...)``.
    init_kwargs : dict
        Extra keyword arguments forwarded verbatim to ``ray.init``.
    ray_debug : bool
        Export ``RAY_DEBUG=1`` so remote breakpoints are honoured.
    """

    device: DeviceType = "cpu"
    num_cpus: Optional[int] = None
    num_gpus: Optional[int] = None
    omp_threads: int = 1
    disable_mps: bool = True
    disable_cuda: bool = True
    logging_level: str = "ERROR"
    runtime_env: Optional[Dict[str, Any]] = None
    init_kwargs: Dict[str, Any] = field(default_factory=dict)
    ray_debug: bool = True

    def _apply_env_vars(self):
        """Export device, threading and logging variables, then set torch device.

        Must run before torch or Ray workers are spawned, because child
        processes inherit the environment at fork time. Besides the
        device-specific variables, it always silences Ray's stderr mirroring
        and tune's automatic callback loggers, and finishes with
        ``torch.set_default_device(self.device)``.
        """

        if self.device == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["RLLIB_NUM_GPUS"] = "0"
            os.environ["USE_CUDA"] = "0"
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

            if self.disable_mps:
                os.environ["RAY_USE_MPS"] = "0"
        elif self.device == "mps":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            os.environ["RLLIB_NUM_GPUS"] = "0"
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        elif self.device == "cuda":
            os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"

            if self.disable_cuda:
                os.environ["CUDA_VISIBLE_DEVICES"] = ""

        if self.num_gpus is not None:
            os.environ["RLLIB_NUM_GPUS"] = str(self.num_gpus)

        if self.omp_threads is not None:
            os.environ["OMP_NUM_THREADS"] = str(self.omp_threads)

        if self.ray_debug:
            os.environ["RAY_DEBUG"] = "1"

        # logging behaviour for workers
        os.environ["RAY_LOG_TO_STDERR"] = "0"
        os.environ["RAY_BACKEND_LOG_LEVEL"] = "error"
        os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"

        torch.set_default_device(self.device)

    def initialize(self) -> None:
        """Apply environment variables, quieten library loggers and ``ray.init``.

        Ray is started in ``local_mode=True`` with the dashboard and metrics
        collection disabled and ``log_to_driver=False``. Local mode runs every
        actor and task in the driver process; the source comment marks this
        as intentional for debugging and asks not to turn it off. Before
        ``ray.init``, Ray's ``uv run`` runtime-env hook
        (``RAY_ENABLE_UV_RUN_RUNTIME_ENV``) is disabled because ``local_mode``
        rejects the runtime env it would inject. ``num_cpus`` and ``num_gpus``
        are forwarded to ``ray.init`` when they are not ``None``.
        """

        self._apply_env_vars()

        # Silence noisy loggers globally
        # # logging.getLogger().setLevel(logging.WARNING)
        logging.getLogger("ray").setLevel(logging.WARNING)
        logging.getLogger("ray.rllib").setLevel(logging.WARNING)
        logging.getLogger("ray.tune").setLevel(logging.WARNING)
        logging.getLogger("tensorboardX").setLevel(logging.ERROR)
        logging.getLogger("asyncio").setLevel(logging.ERROR)
        ray.init(
            ignore_reinit_error=True,
            logging_level=self.logging_level,
            runtime_env=self.runtime_env,
            local_mode=True,  # Turn on for debugging only (DO NOT TURN OFF)
            num_cpus=self.num_cpus,
            num_gpus=self.num_gpus,
            log_to_driver=False,
            include_dashboard=False,
            _system_config={
                "metrics_report_interval_ms": 0,
                "enable_metrics_collection": False,
            },
            **self.init_kwargs,
        )


class RayRuntime:
    """Idempotent entry point for starting Ray once per process."""

    _initialized = False

    @classmethod
    def ensure_initialized(cls, cfg: RayRuntimeConfig) -> None:
        """Initialise Ray with ``cfg`` unless it is already running.

        Parameters
        ----------
        cfg : RayRuntimeConfig
            Settings to apply. Ignored when ``ray.is_initialized()`` is
            already ``True``, so a second call with a different config has no
            effect. The ``_initialized`` flag is only set here and never
            consulted; ``ray.is_initialized()`` is the actual guard.
        """

        if not ray.is_initialized():
            cfg.initialize()

            cls._initialized = True
