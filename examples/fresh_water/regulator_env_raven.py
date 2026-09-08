import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium.core import ObsType

from core.annotations import override
from core.envs.regulator import RegulatorEnv
from core.world.context import (
    Context,
    EnvStepContext,
    MechanismContext,
    MechanismStatus,
)

EPS = 1e-8

logger = logging.getLogger(__name__)


class WaterRegulatorRavenEnv(RegulatorEnv):
    """
    Outer-loop environment for water mechanism optimization.

    Mirrors the fishery regulator but adapted to water-level semantics. Keeps the
    same context processing and FitnessContext production so downstream tooling
    (visualization, loader) can remain unchanged.
    """

    def __init__(
        self,
        *,
        ecology_cfg: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        env_cfg = ecology_cfg or {}
        self.sustainability_weight = env_cfg.get("sus_weight", 5.0)
        self.sustainability_threshold = env_cfg.get("sus_threshold", 0.2)
        self.max_water = env_cfg.get("max_water", 100.0)
        # Denormalized threshold for visualization
        self.raw_sustainability_threshold = (
            self.sustainability_threshold * self.max_water
        )
        self.trajectories: dict[int, list[dict[str, Any]]] = {}

        self.economic_weight = env_cfg.get("economic_weight", 1.0)
        self.sustainability_weight = env_cfg.get("sustainability_weight", 1.0)

        target_status = env_cfg.get("aggregation_status", "train")
        self.aggregation_status = MechanismStatus(target_status)

    @override(RegulatorEnv)
    def observation(self, obs: ObsType) -> ObsType:
        return 0.0

    @override(RegulatorEnv)
    def aggregate_rewards(self, ctxs: list[Context]) -> list[float]:
        """
        Compute per-mechanism fitness from step-level EnvStepContexts.
        Implementation mirrors `FisheryRegulatorEnv.aggregate_rewards`.
        """

        # Must be able to get the seed and run the baseline dynamically for the no extraction
        step_ctxs = [
            ctx
            for ctx in ctxs
            if isinstance(ctx.payload, EnvStepContext)
            and ctx.payload.status == self.aggregation_status
        ]
        if not step_ctxs:
            logger.warning("[Regulator] No EnvStepContext received")
            return []

        by_run: dict[tuple[int, int], list[Context]] = defaultdict(list)

        # TODO group by seed as well
        # for now we assume aggregation over seed
        for ctx in step_ctxs:
            s = ctx.payload
            by_run[(s.mechanism, s.seed)].append(ctx)

        economic_by_m: dict[int, list[float]] = defaultdict(list)
        deviation_by_m: dict[int, list[float]] = defaultdict(list)
        seeds_by_m: dict[int, list[float]] = defaultdict(list)

        for (m_idx, seed), steps in by_run.items():
            steps = sorted(steps, key=lambda c: c.step)
            rewards = []
            for ctx in steps:
                r = ctx.payload.reward
                # TODO for now we average accross agents but later another solution required!
                if isinstance(r, dict):
                    rewards.append(float(np.mean(list(r.values()))))
                else:
                    rewards.append(float(r))
            economic_score = float(np.mean(rewards))
            deviation = compute_streamflow_deviation_from_step_ctx(
                steps[-1].payload, streamflow_col="Belwood_Lake (res. inflow) [m3/s]"
            )["streamflow_deviation"]
            economic_by_m[m_idx].append(economic_score)
            deviation_by_m[m_idx].append(deviation)
            seeds_by_m[m_idx].append(seed)

        # TODO take into consideration economic vs susteinability reward
        max_idx = max(economic_by_m.keys())
        fitness = np.full(max_idx + 1, -np.inf, dtype=np.float32)

        for m_idx in economic_by_m:
            mean_economic = float(np.mean(economic_by_m[m_idx]))
            mean_deviation = float(np.mean(deviation_by_m[m_idx]))
            sustainability_score = 1.0 / (1.0 + mean_deviation)
            objective = (
                self.economic_weight * mean_economic
                + self.sustainability_weight * sustainability_score
            )
            fitness[m_idx] = objective

            self._publish(
                MechanismContext(
                    index=m_idx,
                    env_id=self.env_id,
                    seed=seed,
                    status=MechanismStatus.done,
                    mechanism=None,  # TODO add the mechanism object
                    metrics={
                        "objective": objective,
                        "economic_score": mean_economic,
                        "streamflow_deviation": mean_deviation,
                        "sustainability_score": sustainability_score,
                        "economic_weight": self.economic_weight,
                        "streamflow_weight": self.sustainability_weight,
                    },
                )
            )

        return fitness.tolist()


def read_hydrograph_series(
    output_dir: str,
    column_name: str,
) -> dict[str, float]:
    csv_path = Path(output_dir) / "ohms_canshield_Hydrographs.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing hydrograph file: {csv_path}")

    series = {}

    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)

        for row in reader:
            row = {k.strip(): v for k, v in row.items()}

            date = (
                row.get("date") or row.get("Date") or row.get("time") or row.get("Time")
            )

            if date is None:
                date = next(iter(row.values()))

            if column_name not in row:
                match = next(
                    (k for k in row if k.strip() == column_name.strip()),
                    None,
                )
                if match is None:
                    raise KeyError(
                        f"Column '{column_name}' not found in {csv_path}. "
                        f"Available columns: {list(row.keys())}"
                    )
                column_name = match

            raw = row[column_name]

            if raw not in ("", "---", None):
                series[str(date)] = float(raw)

    return series


def compute_streamflow_deviation(
    *,
    mechanism_output_dir: str,
    baseline_output_dir: str,
    streamflow_col: str,
) -> float:
    q_m = read_hydrograph_series(
        output_dir=mechanism_output_dir,
        column_name=streamflow_col,
    )

    q_0 = read_hydrograph_series(
        output_dir=baseline_output_dir,
        column_name=streamflow_col,
    )

    common_dates = sorted(set(q_m) & set(q_0))

    if not common_dates:
        return float("nan")

    numerator = sum(abs(q_m[t] - q_0[t]) for t in common_dates)
    denominator = sum(abs(q_0[t]) for t in common_dates)

    return numerator / max(EPS, denominator)


def compute_streamflow_deviation_from_step_ctx(
    payload,
    *,
    streamflow_col: str = "West_Montrose [m3/s]",
) -> dict[str, float]:
    first_agent_info = next(iter(payload.info.values()))

    baseline_output_dir = first_agent_info["baseline_ref"]

    baseline_path = Path(baseline_output_dir)

    prepared_runs_dir = baseline_path.parents[1]
    baseline_run_name = baseline_path.parents[0].name

    mechanism_run_name = baseline_run_name.replace("baseline_", "", 1)

    mechanism_output_dir = prepared_runs_dir / mechanism_run_name / "3_Model_output"

    deviation = compute_streamflow_deviation(
        mechanism_output_dir=str(mechanism_output_dir),
        baseline_output_dir=baseline_output_dir,
        streamflow_col=streamflow_col,
    )

    return {
        "env_id": payload.env_id,
        "seed": payload.seed,
        "mechanism": payload.mechanism,
        "streamflow_col": streamflow_col,
        "streamflow_deviation": deviation,
    }
