import csv
import logging
import os
import shutil
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, SupportsFloat

import numpy as np
from gymnasium.core import ActType
from ray.rllib.utils.typing import AgentID, MultiAgentDict

from core.annotations import override
from core.envs.marl_regulated import MultiAgentRegulatedEnv

logger = logging.getLogger(__name__)

EPS = 1e-8

CORN_GRAIN_KC = {
    "initial": 0.40,
    "development": 0.80,
    "mid": 1.15,
    "late": 0.70,
    "offseason": 0.0,
}

P_BY_MONTH_45N = {
    1: 0.20,
    2: 0.23,
    3: 0.27,
    4: 0.30,
    5: 0.34,
    6: 0.35,
    7: 0.34,
    8: 0.32,
    9: 0.28,
    10: 0.24,
    11: 0.21,
    12: 0.20,
}


CORN_WILTING_FRAC = 0.35


class WaterRegulatedEdHsEnv(MultiAgentRegulatedEnv):
    def __init__(
        self,
        *,
        ecology_cfg: dict,
        use_raven: bool = True,
        raven_cmd: Optional[str] = None,
        raven_cwd: str = "raven",
        raven_freq: int = 1,
        raven_stage_col: str = "Belwood_Lake",
        raven_streamflow_col: str = "Belwood_Lake (res. inflow) [m3/s]",
        raven_outflow_col: str = "Belwood_Lake [m3/s]",
        raven_precip_col: str = "precip [mm/day]",
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Normally pulled out of Raven
        self.full_stage_m = ecology_cfg.get("full_stage_m")
        self.max_depth_m = ecology_cfg.get("max_depth_m")
        self.lake_area_m2 = ecology_cfg.get("lake_area_m2")

        # Config params
        self.max_farm_area_m2 = ecology_cfg.get("max_farm_area_m2")

        # internal params
        self._planting_month: Optional[int] = None
        self._planting_day: Optional[int] = None

        # Simplified corn-specific crop-stress proxy
        self.permanent_wilting_fraction = ecology_cfg.get(
            "corn_permanent_wilting_fraction",
            0.35,
        )

        # self.streamflow_init = ecology_cfg.get("streamflow_init", 124.724)
        # self.streamflow_init_sigma = ecology_cfg.get("streamflow_init_sigma", 0.05)
        # self.streamflow_ref = ecology_cfg.get("streamflow_ref", self.streamflow_init)
        # self.default_precip_mm_day = ecology_cfg.get("default_precip_mm_day", 2.5)
        # self.default_temperature_c = ecology_cfg.get("default_temperature_c", 22.0)
        # self.temperature_sigma = ecology_cfg.get("temperature_sigma", 2.0)
        # self.precip_sigma = ecology_cfg.get("precip_sigma", 0.35)
        # self.underuse_penalty_scale = ecology_cfg.get("underuse_penalty_scale", 0.15)
        # self.underuse_penalty_power = ecology_cfg.get("underuse_penalty_power", 2.0)

        self.use_raven = use_raven
        self.raven_cmd = raven_cmd or "raven"
        self.raven_cwd = raven_cwd
        self.raven_freq = max(0, raven_freq)
        self.raven_stage_col = raven_stage_col
        self.raven_streamflow_col = raven_streamflow_col
        self.raven_outflow_col = raven_outflow_col
        self.raven_precip_col = raven_precip_col
        # self.raven_temp_col = raven_precip_col

        self.withdrawal_history_m3s: list[tuple[datetime, float]] = []

        self.key = f"m_{self.mechanism_id}_seed_{self.seed}_"
        self.run_root: Optional[str] = None

        self.obs_map = [
            "reservoir_level_norm",
            # "usage_norm",
            "release_pressure",
            "effective_quota",
            "total_usage_norm",
        ]
        self.full_required_m3_day = 0.0

        self._baseline_ref: Optional[str] = None

    @override(MultiAgentRegulatedEnv)
    def _reset(self):
        self.withdrawal_history_m3s = []

        # Seeded randomization of planting day
        planting_day_of_year = int(
            self.rng.integers(
                low=121,
                high=274,
            )
        )
        planting_date = datetime(1980, 1, 1) + timedelta(days=planting_day_of_year)
        self._planting_day = planting_date.day
        self._planting_month = planting_date.month

        # Run raven to get initial inflow and outflow, precip, temp etc.
        # assume no usage in step 1 ?
        # Start reading at initial da
        self._run_raven(date=planting_date)
        self.baseline_ref = self._prepare_no_extraction_baseline()
        reservoir_stage_init = self._read_raven_reservoir_stage(self.raven_stage_col)
        reservoir_level_norm_init = (
            reservoir_stage_init - (self.full_stage_m - self.max_depth_m)
        ) / self.max_depth_m

        # streamflow = inflow
        streamflow_m3s_init = self._read_raven_streamflow(self.raven_streamflow_col)
        outflow_m3s_init = self._read_raven_streamflow(self.raven_outflow_col)
        precip_mm_day_init = self._read_raven_precip(self.raven_precip_col)
        # temp_c_init = self._read_raven_temp(self.raven_precip_col)
        temp_c_init = self._estimate_temp_c(date=planting_date)

        release_pressure = min(
            1.0, max(0.0, outflow_m3s_init / max(EPS, streamflow_m3s_init))
        )

        self.S_t = {
            "date": planting_date,
            "reservoir_stage": reservoir_stage_init,
            "reservoir_level_norm": reservoir_level_norm_init,
            "streamflow_m3s": streamflow_m3s_init,
            "precip_mm_day": precip_mm_day_init,
            "temp_c": temp_c_init,
            "release_pressure": release_pressure,
            "last_usage_m3_day": 0.0,
        }

        return {
            agent_id: self.observation(agent_id, self.S_t) for agent_id in self.agents
        }

    @override(MultiAgentRegulatedEnv)
    def _is_truncated(self) -> bool:
        return self.horizon is not None and (self._t + 1) >= self.horizon

    def _action_to_float(self, action: ActType) -> float:
        if hasattr(action, "item"):
            return float(action.item())
        return float(action)

    def _crop_stage(self, date: datetime) -> str:
        planting_date = datetime(date.year, self._planting_month, self._planting_day)
        dap = (date - planting_date).days

        if dap < 0:
            return "offseason"
        if dap <= 30:
            return "initial"
        if dap <= 70:
            return "development"
        if dap <= 110:
            return "mid"
        if dap <= 150:
            return "late"
        return "offseason"

    # TODO  temporary since raven does not output temperature
    def _estimate_temp_c(self, date: datetime) -> float:
        return {
            1: -5.0,
            2: -3.0,
            3: 2.0,
            4: 8.0,
            5: 15.0,
            6: 20.0,
            7: 23.0,
            8: 22.0,
            9: 17.0,
            10: 10.0,
            11: 4.0,
            12: -2.0,
        }[date.month]

    # The mechanism controls the fraction of full_required_m3_day allowed.
    # This avoids collapse to the floor when reservoir_level_norm is close
    # to fixed_quota.
    def _quota_stress(self, reservoir_level_norm: float) -> float:
        return float(
            np.clip(
                (reservoir_level_norm - self.mechanism.fixed_quota)
                / max(EPS, 1.0 - self.mechanism.fixed_quota),
                0.0,
                1.0,
            )
        )

    # allowed_frac smoothly interpolates from min_demand_frac to max_demand_frac.
    def _allowed_frac(self, reservoir_level_norm: float) -> float:
        stress = self._quota_stress(reservoir_level_norm)

        return float(
            self.mechanism.min_demand_frac
            + stress * (self.mechanism.max_demand_frac - self.mechanism.min_demand_frac)
        )

    # allowed_m3_day is now demand-scaled instead of storage-scaled.
    def _allowed_m3_day(self, reservoir_level_norm: float) -> float:
        return self._allowed_frac(reservoir_level_norm) * self.full_required_m3_day

    @override(MultiAgentRegulatedEnv)
    def intrinsic_utility(self, A_t: dict[AgentID, ActType]) -> MultiAgentDict:
        precip_mm_day = self.S_t["precip_mm_day"]
        date: datetime = self.S_t["date"]
        temp_c = self._estimate_temp_c(date=date)
        month = date.month

        crop_stage = self._crop_stage(date)

        eto_mm_day = P_BY_MONTH_45N[month] * ((0.46 * temp_c) + 8.0)
        eto_mm_day = max(0.0, eto_mm_day)

        etcrop_mm_day = eto_mm_day * CORN_GRAIN_KC[crop_stage]
        deficit_mm_day = max(0.0, etcrop_mm_day - precip_mm_day)

        # TODO must also retreive the time of the day to water in order to normalize
        # per seconds
        full_required_m3_day = (
            deficit_mm_day / 1000.0 * self.max_farm_area_m2
            # / 86400.0
        )
        self.full_required_m3_day = full_required_m3_day

        crop_water_need_m3_day = etcrop_mm_day / 1000.0 * self.max_farm_area_m2

        precip_water_m3_day = precip_mm_day / 1000.0 * self.max_farm_area_m2

        crop_satisfaction = {}

        for agent_id, action in A_t.items():
            irrigation_frac = np.clip(self._action_to_float(action), 0.0, 1.0)
            requested_m3_day = irrigation_frac * full_required_m3_day

            reservoir_level_norm = (
                self.S_t["reservoir_stage"] - (self.full_stage_m - self.max_depth_m)
            ) / self.max_depth_m

            # Old storage-based quota replaced by demand-fraction quota.
            allowed_m3_day = self._allowed_m3_day(reservoir_level_norm)
            delivered_m3_day = min(
                requested_m3_day,
                allowed_m3_day,
            )

            if crop_water_need_m3_day <= EPS:
                crop_satisfaction[agent_id] = 1.0
            else:
                crop_satisfaction[agent_id] = min(
                    1.0,
                    (delivered_m3_day + precip_water_m3_day) / crop_water_need_m3_day,
                )

        self._update_infos(key="crop_stage", values=crop_stage)
        self._update_infos(key="eto_mm_day", values=eto_mm_day)
        self._update_infos(key="etcrop_mm_day", values=etcrop_mm_day)
        self._update_infos(key="deficit_mm_day", values=deficit_mm_day)
        self._update_infos(key="full_required_m3_day", values=full_required_m3_day)
        self._update_infos(key="precip_mm_day", values=precip_mm_day)
        self._update_infos(key="temp_c", values=temp_c)
        self._update_infos(key="crop_satisfaction", values=crop_satisfaction)

        return crop_satisfaction

    @override(MultiAgentRegulatedEnv)
    def violation_signal(
        self,
        u_i: SupportsFloat,  # crop satisfaction
        agent_id: AgentID,
        *,
        A_t: MultiAgentDict,
    ) -> SupportsFloat:

        requested_m3_day = (
            np.clip(self._action_to_float(A_t[agent_id]), 0.0, 1.0)
            * self.full_required_m3_day
        )
        reservoir_level_norm = (
            self.S_t["reservoir_stage"] - (self.full_stage_m - self.max_depth_m)
        ) / self.max_depth_m

        # Old storage-based quota replaced by demand-fraction quota.
        allowed_frac = self._allowed_frac(reservoir_level_norm)
        allowed_m3_day = allowed_frac * self.full_required_m3_day
        delivered_m3_day = min(
            requested_m3_day,
            allowed_m3_day,
        )

        # TODO review this
        quota_violation_m3_day = max(0.0, requested_m3_day - allowed_m3_day)
        quota_penalty = min(
            1.0,
            self.mechanism.fine_amount
            * (quota_violation_m3_day / max(EPS, self.full_required_m3_day)),
        )

        # risk_penalty_power is now actually used.
        # This makes high extraction fractions increasingly costly when release pressure is high.
        requested_frac = requested_m3_day / max(EPS, self.full_required_m3_day)
        flow_penalty = (
            self.mechanism.risk_penalty_scale
            * self.S_t["release_pressure"]
            * (requested_frac**self.mechanism.risk_penalty_power)
        )

        # minimum_crop_need_m3_day = (
        #     CORN_WILTING_FRAC,
        #     * self.full_required_m3_day
        # )

        # under_irrigation_m3_day = max(
        #     0.0,
        #     minimum_crop_need_m3_day - requested_m3_day,
        # )

        # under_irrigation_penalty = min(
        #     1.0,
        #     self.mechanism.under_irrigation_penalty_scale
        #     * under_irrigation_m3_day
        #     / max(EPS, minimum_crop_need_m3_day),
        # )

        total_penalty = min(1.0, quota_penalty + flow_penalty)
        # quota_violation_m3_day = 0
        # quota_penalty = 0

        self._update_infos(key="requested_m3_day", values={agent_id: requested_m3_day})
        self._update_infos(key="allowed_m3_day", values={agent_id: allowed_m3_day})
        self._update_infos(key="requested_frac", values={agent_id: requested_frac})
        self._update_infos(key="delivered_m3_day", values={agent_id: delivered_m3_day})
        self._update_infos(
            key="quota_violation_m3", values={agent_id: quota_violation_m3_day}
        )
        self._update_infos(key="quota_penalty", values={agent_id: quota_penalty})
        self._update_infos(key="flow_penalty", values={agent_id: flow_penalty})
        self._update_infos(
            key="quota_stress",
            values={agent_id: self._quota_stress(reservoir_level_norm)},
        )
        self._update_infos(
            key="min_demand_frac", values={agent_id: self.mechanism.min_demand_frac}
        )
        self._update_infos(
            key="max_demand_frac", values={agent_id: self.mechanism.max_demand_frac}
        )

        return total_penalty

    @override(MultiAgentRegulatedEnv)
    def penalty(self, u_i: SupportsFloat, **kwargs) -> SupportsFloat:
        return 1.0

    def transition_kernel(
        self,
        *,
        A_t: MultiAgentDict,
        S_t: dict[str, float],
    ) -> dict[str, float]:

        delivered_m3_day = {}

        for agent_id, action in A_t.items():
            irrigation_frac = np.clip(self._action_to_float(action), 0.0, 1.0)
            requested_m3_day = irrigation_frac * self.full_required_m3_day

            reservoir_level_norm = (
                S_t["reservoir_stage"] - (self.full_stage_m - self.max_depth_m)
            ) / self.max_depth_m

            # Old storage-based quota replaced by demand-fraction quota.
            allowed_m3_day = self._allowed_m3_day(reservoir_level_norm)

            delivered_m3_day[agent_id] = min(
                requested_m3_day,
                allowed_m3_day,
            )

        total_usage_m3_day = sum(delivered_m3_day.values())
        total_usage_m3s = total_usage_m3_day / 86400.0
        self._update_infos(key="total_usage_m3s", values=total_usage_m3s)

        old_date: datetime = S_t["date"]
        next_date = old_date + timedelta(days=1)

        if self.use_raven and self.raven_freq > 0 and (self._t % self.raven_freq == 0):
            try:
                self.withdrawal_history_m3s.append((next_date, total_usage_m3s))
                self._run_raven(date=next_date)

                # based on the current usage get the stage at the end of day (midnight)
                eod_reservoir_stage = self._read_raven_reservoir_stage(
                    self.raven_stage_col
                )
                eod_streamflow_m3s = self._read_raven_streamflow(
                    self.raven_streamflow_col
                )
                eod_outflow_m3s = self._read_raven_streamflow(self.raven_outflow_col)
                eod_precip_mm_day = self._read_raven_precip(self.raven_precip_col)

                eod_02GA041_streamflow_m3s = self._read_raven_streamflow(
                    "02GA041 [m3/s]"
                )
                eod_02GA041_streamflow_m3s_observed = self._read_raven_streamflow(
                    "02GA041 (observed) [m3/s]"
                )
                eod_02GA014_streamflow_m3s = self._read_raven_streamflow(
                    "02GA014 [m3/s]"
                )
                eod_02GA014_streamflow_m3s_observed = self._read_raven_streamflow(
                    "02GA014 (observed) [m3/s]"
                )

                eod_West_Montrose_streamflow_m3s = self._read_raven_streamflow(
                    "West_Montrose [m3/s]"
                )
                eod_West_Montrose_streamflow_m3s_observed = self._read_raven_streamflow(
                    "West_Montrose (observed) [m3/s]"
                )

                # eod_temp_c = self._read_raven_temp(self.raven_temp_col)
                eod_temp_c = self._estimate_temp_c(date=next_date)

                if eod_reservoir_stage is not None:
                    eod_reservoir_level_norm = (
                        float(eod_reservoir_stage)
                        - (self.full_stage_m - self.max_depth_m)
                    ) / self.max_depth_m

                # TODO review this
                # compute flow penalty
                # TODO this may need to be capped or may explode
                release_pressure = min(
                    1.0, max(0.0, eod_outflow_m3s / max(EPS, eod_streamflow_m3s))
                )

                # Residence Time
                eod_reservoir_m3 = (
                    eod_reservoir_stage - (self.full_stage_m - self.max_depth_m)
                ) / self.max_depth_m
                q_ref_m3s = max(eod_streamflow_m3s, eod_outflow_m3s)
                residence_time_days = eod_reservoir_m3 / max(EPS, q_ref_m3s) / 86400

            except Exception:
                logger.exception(
                    "Raven integration failed; falling back to internal dynamics"
                )

        new_state = {
            "date": next_date,
            "reservoir_stage": eod_reservoir_stage,
            "reservoir_level_norm": eod_reservoir_level_norm,
            "streamflow_m3s": eod_streamflow_m3s,
            "precip_mm_day": eod_precip_mm_day,
            "temp_c": eod_temp_c,
            "last_usage_m3_day": total_usage_m3_day,
            "release_pressure": release_pressure,
            "residence_time_days": residence_time_days,
        }
        # TODO this updates every time step - find way to update once at reset
        self._update_infos(key="baseline_ref", values=self.baseline_ref)
        self._update_infos(key="reservoir_stage", values=eod_reservoir_stage)
        self._update_infos(key="reservoir_level_norm", values=eod_reservoir_level_norm)
        self._update_infos(key="streamflow_m3s", values=eod_streamflow_m3s)
        self._update_infos(key="outflow_m3s", values=eod_outflow_m3s)
        self._update_infos(key="precip_mm_day", values=eod_precip_mm_day)
        self._update_infos(key="temp_c", values=eod_temp_c)
        self._update_infos(key="release_pressure", values=release_pressure)
        self._update_infos(key="residence_time_days", values=residence_time_days)
        self._update_infos(
            key="02GA041_streamflow_m3s", values=eod_02GA041_streamflow_m3s
        )
        self._update_infos(
            key="02GA041_streamflow_m3s_observed",
            values=eod_02GA041_streamflow_m3s_observed,
        )
        self._update_infos(
            key="02GA014_streamflow_m3s", values=eod_02GA014_streamflow_m3s
        )
        self._update_infos(
            key="02GA014_streamflow_m3s_observed",
            values=eod_02GA014_streamflow_m3s_observed,
        )
        self._update_infos(
            key="West_Montrose_streamflow_m3s", values=eod_West_Montrose_streamflow_m3s
        )
        self._update_infos(
            key="West_Montrose_streamflow_m3s_observed",
            values=eod_West_Montrose_streamflow_m3s_observed,
        )

        self.S_t = new_state
        return self.S_t

    def _observation(self, agent_id: AgentID, S_t: dict[str, float]):
        reservoir_level_norm = float(S_t["reservoir_level_norm"])
        total_usage_norm = float(S_t.get("last_usage_m3_day", 0.0))

        # TODO Q : what can we normalize streamflow_m3s with ?
        # streamflow_m3s = float(S_t.get("streamflow_m3s", 0.0))
        # streamflow_norm = streamflow_m3s / max(EPS, self.streamflow_ref_m3s)
        release_pressure = S_t["release_pressure"]

        # effective_quota now means the current demand fraction available to agents.
        effective_quota = self._allowed_frac(reservoir_level_norm)
        return np.array(
            [
                reservoir_level_norm,
                # streamflow_norm,
                release_pressure,
                effective_quota,
                total_usage_norm,
            ],
            dtype=np.float32,
        )

    def _prepare_raven_run(self) -> str:
        if self.run_root is not None:
            return self.run_root

        src = os.path.abspath(self.raven_cwd)

        cache_root = os.path.abspath(
            os.path.join(self.raven_cwd, ".cache", "prepared_runs")
        )
        os.makedirs(cache_root, exist_ok=True)

        self.run_root = os.path.join(cache_root, self.key)

        if os.path.exists(self.run_root):
            shutil.rmtree(self.run_root)

        shutil.copytree(
            src,
            self.run_root,
            ignore=shutil.ignore_patterns(".cache"),
            dirs_exist_ok=False,
        )
        logger.info("Prepared Raven run at %s", self.run_root)
        return self.run_root

    def _append_extraction_to_rvt(
        self,
        run_dir: str,
        date: datetime,
    ) -> int:
        rvt_path = Path(run_dir) / "input" / "Extraction.rvt"

        lines = rvt_path.read_text(encoding="utf-8").splitlines()

        header_idx = next(
            i
            for i, line in enumerate(lines)
            if line.strip().startswith("1980-01-01 00:00:00")
        )

        end_idx = next(
            i
            for i in range(header_idx + 1, len(lines))
            if lines[i].strip() in [":EndObservationData", ":EndData"]
        )

        end_token = lines[end_idx].strip()

        start = datetime(1980, 1, 1)
        n_days = (date.date() - start.date()).days + 1

        values = ["\t0.0"] * n_days

        for usage_date, usage_m3s in self.withdrawal_history_m3s:
            idx = (usage_date.date() - start.date()).days

            if 0 <= idx < n_days:
                usage = float(usage_m3s)
                if abs(usage) < 1e-12:
                    values[idx] = "\t0.0"
                else:
                    values[idx] = f"\t{-usage:.10f}"

        lines[header_idx] = f"\t1980-01-01 00:00:00\t1\t{n_days}"

        new_lines = lines[: header_idx + 1] + values + [end_token]

        rvt_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")

        return n_days

    def _patch_raven_end_date(self, run_dir: str, date: datetime) -> None:
        rvi_path = Path(run_dir) / "2_Raven" / "ohms_canshield.rvi"

        lines = rvi_path.read_text(encoding="utf-8").splitlines()

        end_str = date.strftime("%Y-%m-%d 00:00:00")

        patched = []
        for line in lines:
            if line.strip().startswith(":EndDate"):
                patched.append(f":EndDate         {end_str}")
            else:
                patched.append(line)

        rvi_path.write_text("\n".join(patched) + "\n", encoding="utf-8")

    def _run_raven(self, date: datetime) -> None:
        if not self.use_raven:
            return

        run_dir = self._prepare_raven_run()

        self._append_extraction_to_rvt(
            run_dir=run_dir,
            date=date,
        )

        self._patch_raven_end_date(
            run_dir=run_dir,
            date=date,
        )

        out_dir = "3_Model_output"
        os.makedirs(Path(run_dir) / out_dir, exist_ok=True)

        subprocess.run(
            [self.raven_cmd, "2_Raven/ohms_canshield", "-o", out_dir],
            cwd=run_dir,
            check=False,
        )

    def _prepare_no_extraction_baseline(self) -> str:

        if not self.use_raven:
            return ""

        if self.run_root is None:
            raise RuntimeError(
                "Cannot prepare baseline reference before mechanism Raven run exists."
            )

        src = Path(self.run_root)
        baseline_key = f"baseline_m_{self.mechanism_id}_seed_{self.seed}_"

        cache_root = Path(self.raven_cwd).resolve() / ".cache" / "prepared_runs"
        baseline_run_root = cache_root / baseline_key

        if baseline_run_root.exists():
            shutil.rmtree(baseline_run_root)

        shutil.copytree(src, baseline_run_root)

        logger.info(
            "Copied current zero-extraction Raven run as baseline reference: %s -> %s",
            src,
            baseline_run_root,
        )

        return str(baseline_run_root / "3_Model_output")

    def _read_raven_reservoir_stage(self, column_name: str) -> Optional[float]:
        if self.run_root is None:
            return None

        csv_path = os.path.join(
            self.run_root,
            "3_Model_output",
            "ohms_canshield_ReservoirStages.csv",
        )

        if not os.path.exists(csv_path):
            logger.warning("Raven output not found: %s", csv_path)
            return None

        try:
            with open(csv_path, newline="") as fh:
                reader = csv.DictReader(fh)
                rows = [{k.strip(): v for k, v in row.items()} for row in reader]

            if not rows:
                return None

            row = rows[-1]

            if column_name not in row:
                match = next((k for k in row.keys() if k.startswith(column_name)), None)

                if match is None:
                    logger.warning(
                        "ReservoirStages CSV missing column '%s'. Available: %s",
                        column_name,
                        list(row.keys()),
                    )
                    return None

                column_name = match

            raw = row.get(column_name, "")
            return float(raw) if raw not in ["", "---"] else None

        except Exception:
            logger.exception("Failed to read Raven ReservoirStages CSV")
            return None

    def _read_raven_streamflow(self, column_name: str) -> Optional[float]:
        if self.run_root is None:
            return None

        csv_path = os.path.join(
            self.run_root,
            "3_Model_output",
            "ohms_canshield_Hydrographs.csv",
        )

        if not os.path.exists(csv_path):
            logger.warning("Raven hydrograph output not found: %s", csv_path)
            return None

        try:
            with open(csv_path, newline="") as fh:
                reader = csv.DictReader(fh)
                rows = [{k.strip(): v for k, v in row.items()} for row in reader]

            if not rows:
                return None

            row = rows[-1]

            if column_name not in row:
                match = next(
                    (k for k in row.keys() if k.strip() == column_name.strip()),
                    None,
                )

                if match is None:
                    logger.warning(
                        "Hydrographs CSV missing column '%s'. Available: %s",
                        column_name,
                        list(row.keys()),
                    )
                    return None

                column_name = match

            raw = row.get(column_name, "")
            return float(raw) if raw not in ["", "---"] else None

        except Exception:
            logger.exception("Failed to read Raven Hydrographs CSV")
            return None

    def _read_raven_precip(self, column_name: str) -> Optional[float]:
        return self._read_raven_hydrograph_value(column_name)

    # def _read_raven_temp(self, column_name: Optional[str]) -> Optional[float]:
    #     if column_name is None:
    #         return self.default_temp_c
    #     return self._read_raven_hydrograph_value(column_name)

    def _read_raven_hydrograph_value(self, column_name: str) -> Optional[float]:
        if self.run_root is None:
            return None

        csv_path = os.path.join(
            self.run_root,
            "3_Model_output",
            "ohms_canshield_Hydrographs.csv",
        )

        if not os.path.exists(csv_path):
            logger.warning("Raven hydrograph output not found: %s", csv_path)
            return None

        try:
            with open(csv_path, newline="") as fh:
                reader = csv.DictReader(fh)
                rows = [{k.strip(): v for k, v in row.items()} for row in reader]

            if not rows:
                return None

            row = rows[-1]

            if column_name not in row:
                match = next(
                    (k for k in row.keys() if k.strip() == column_name.strip()),
                    None,
                )
                if match is None:
                    logger.warning(
                        "Hydrographs CSV missing column '%s'. Available: %s",
                        column_name,
                        list(row.keys()),
                    )
                    return None
                column_name = match

            raw = row.get(column_name, "")
            return float(raw) if raw not in ["", "---"] else None

        except Exception:
            logger.exception("Failed to read Raven Hydrographs CSV value")
            return None
