from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from core.annotations import override
from core.mechanism.base import Mechanism
from core.mechanism.space import MechanismSpace


@dataclass(frozen=True)
class WaterMechanism(Mechanism):
    fixed_quota: float

    # CHANGED:
    # Replaced prop_quota + min_stock with demand-fraction quota parameters.
    #
    # Old behavior:
    #   allowed_m3_day was based on excess reservoir storage:
    #       prop_quota * excess_norm * max_depth_m * lake_area_m2
    #   This often collapsed to ~0 when reservoir_level_norm <= fixed_quota,
    #   forcing the mechanism to always use the tiny floor.
    #
    # New behavior:
    #   allowed_m3_day is directly a fraction of the crop water deficit:
    #       allowed_frac * full_required_m3_day
    #   This makes the mechanism interpretable and prevents the quota from
    #   being clipped to near-zero absolute volumes.
    min_demand_frac: float
    max_demand_frac: float
    fine_amount: float
    risk_penalty_scale: float
    risk_penalty_power: float
    under_irrigation_penalty_scale: float
    max_farm_area_m2: float

    def __post_init__(self) -> None:
        # fixed_quota = protected reservoir fullness threshold.
        # If reservoir_level_norm <= fixed_quota, agents receive only min_demand_frac.
        # If reservoir_level_norm approaches 1.0, agents approach max_demand_frac.
        assert 0.6 <= self.fixed_quota <= 0.95

        # CHANGED:
        # These are fractions of crop irrigation demand, not fractions of reservoir storage.
        assert 0.0 <= self.min_demand_frac <= 1.0
        assert 0.05 <= self.max_demand_frac <= 1.0
        assert self.min_demand_frac <= self.max_demand_frac
        assert 0.0 <= self.fine_amount <= 0.1
        assert 0.0 <= self.risk_penalty_scale <= 1.0
        assert 1.0 <= self.risk_penalty_power <= 5.0
        assert 0.0 <= self.under_irrigation_penalty_scale <= 1.0
        assert 100_000 <= self.max_farm_area_m2 <= 20_000_000

    @override(Mechanism)
    def to_vector(self) -> np.ndarray:
        return np.array(
            [
                self.fixed_quota,
                self.min_demand_frac,
                self.max_demand_frac,
                self.fine_amount,
                self.risk_penalty_scale,
                (self.risk_penalty_power - 1.0) / 4.0,  # maps [1,5] -> [0,1]
                self.under_irrigation_penalty_scale,
                self.max_farm_area_m2,
            ],
            dtype=np.float32,
        )

    def param_names(self) -> list[str]:
        return [
            "fixed_quota",
            "min_demand_frac",
            "max_demand_frac",
            "fine_amount",
            "risk_penalty_scale",
            "risk_penalty_power",
            "under_irrigation_penalty_scale",
            "max_farm_area_m2",
        ]


class WaterMechanismSpace(MechanismSpace):
    # CHANGED:
    # prop_quota and min_stock removed.
    # min_demand_frac and max_demand_frac added.
    ALL_PARAMS = [
        "fixed_quota",
        "min_demand_frac",
        "max_demand_frac",
        "fine_amount",
        "risk_penalty_scale",
        "risk_penalty_power",
        "under_irrigation_penalty_scale",
        "max_farm_area_m2",
    ]

    def __init__(
        self,
        use_stochastic_rounding: bool = True,
        optimize_params: list[str] | None = None,
        default_fixed_quota: float = 0.85,
        # CHANGED:
        # Defaults now mean:
        #   under stress: at least 5% of full required irrigation can be delivered
        #   when reservoir is healthy: up to 100% of full required irrigation can be delivered
        default_min_demand_frac: float = 0.05,
        default_max_demand_frac: float = 1.0,
        default_fine_amount: float = 0.05,
        default_risk_penalty_scale: float = 0.5,
        default_risk_penalty_power: float = 2.0,
        default_under_irrigation_penalty_scale: float = 0.25,
        default_max_farm_area_m2: float = 500_000,
    ):
        super().__init__()

        self.use_stochastic_rounding = use_stochastic_rounding
        self.optimize_params = optimize_params or [
            "fixed_quota",
            "min_demand_frac",
            "max_demand_frac",
            "fine_amount",
            "risk_penalty_scale",
            "risk_penalty_power",
            "under_irrigation_penalty_scale",
            "max_farm_area_m2",
        ]
        self.dimension = len(self.optimize_params)
        self.full_dimension = len(self.ALL_PARAMS)
        self.defaults = {
            "fixed_quota": default_fixed_quota,
            "min_demand_frac": default_min_demand_frac,
            "max_demand_frac": default_max_demand_frac,
            "fine_amount": default_fine_amount,
            "risk_penalty_scale": default_risk_penalty_scale,
            "risk_penalty_power": default_risk_penalty_power,
            "under_irrigation_penalty_scale": default_under_irrigation_penalty_scale,
            "max_farm_area_m2": default_max_farm_area_m2,
        }

    def _denormalize_param(self, name: str, value: float, u: np.ndarray) -> float | int:
        if name == "fixed_quota":
            return 0.6 + value * (0.95 - 0.6)

        # CHANGED:
        # min_demand_frac controls the floor of allowed irrigation demand.
        if name == "min_demand_frac":
            return value * 0.35  # maps [0,1] -> [0,0.35]

        # CHANGED:
        # max_demand_frac controls the ceiling of allowed irrigation demand.
        # We enforce max_demand_frac >= min_demand_frac after decoding.
        if name == "max_demand_frac":
            return 0.35 + value * (1.0 - 0.35)  # maps [0,1] -> [0.35,1.0]

        if name == "fine_amount":
            return value * 0.10

        if name == "risk_penalty_scale":
            return value

        if name == "risk_penalty_power":
            return 1.0 + 4.0 * value

        if name == "under_irrigation_penalty_scale":
            return value

        if name == "max_farm_area_m2":
            return 100_000 + value * (20_000_000 - 100_000)

        return value

    def _denormalize(self, u: np.ndarray) -> dict:
        result = {}

        for i, name in enumerate(self.optimize_params):
            result[name] = self._denormalize_param(name, float(u[i]), u)

        # CHANGED:
        # Keep the quota curve monotonic.
        if "min_demand_frac" in result and "max_demand_frac" in result:
            result["max_demand_frac"] = max(
                result["max_demand_frac"],
                result["min_demand_frac"],
            )

        return result

    def _normalize_param(self, name: str, value: float | int) -> float:
        if name == "fixed_quota":
            return (float(value) - 0.6) / (0.95 - 0.6)

        # CHANGED:
        # Normalize demand-fraction quota parameters.
        if name == "min_demand_frac":
            return float(value) / 0.35

        if name == "max_demand_frac":
            return (float(value) - 0.35) / (1.0 - 0.35)

        if name == "fine_amount":
            return float(value) / 0.10

        if name == "risk_penalty_scale":
            return float(value)

        if name == "risk_penalty_power":
            return (float(value) - 1.0) / 4.0

        if name == "under_irrigation_penalty_scale":
            return float(value)

        if name == "max_farm_area_m2":
            return (float(value) - 100_000) / (20_000_000 - 100_000)

        return float(value)

    def default(self) -> WaterMechanism:
        return WaterMechanism(
            fixed_quota=self.defaults["fixed_quota"],
            min_demand_frac=self.defaults["min_demand_frac"],
            max_demand_frac=self.defaults["max_demand_frac"],
            fine_amount=self.defaults["fine_amount"],
            risk_penalty_scale=self.defaults["risk_penalty_scale"],
            risk_penalty_power=self.defaults["risk_penalty_power"],
            under_irrigation_penalty_scale=self.defaults[
                "under_irrigation_penalty_scale"
            ],
            max_farm_area_m2=self.defaults["max_farm_area_m2"],
        )

    def encode(self, m: WaterMechanism) -> NDArray[np.float32]:
        values = []

        for name in self.optimize_params:
            raw = getattr(m, name)

            values.append(self._normalize_param(name, raw))

        return np.array(values, dtype=np.float32)

    def decode(self, x: NDArray[np.float32]) -> Mechanism:
        u = np.clip(self._validate(x), 0.0, 1.0)
        params = dict(self.defaults)

        for i, name in enumerate(self.optimize_params):
            params[name] = self._denormalize_param(name, float(u[i]), u)

        # CHANGED:
        # Make sure min <= max even when only one of the two params is optimized.
        params["max_demand_frac"] = max(
            params["max_demand_frac"],
            params["min_demand_frac"],
        )
        mech = WaterMechanism(
            fixed_quota=params["fixed_quota"],
            min_demand_frac=params["min_demand_frac"],
            max_demand_frac=params["max_demand_frac"],
            fine_amount=params["fine_amount"],
            risk_penalty_scale=params["risk_penalty_scale"],
            risk_penalty_power=params["risk_penalty_power"],
            under_irrigation_penalty_scale=params["under_irrigation_penalty_scale"],
            max_farm_area_m2=params["max_farm_area_m2"],
        )

        return self.clip(mech)

    def clip(self, m: WaterMechanism) -> WaterMechanism:
        min_demand_frac = float(np.clip(m.min_demand_frac, 0.0, 1.0))
        max_demand_frac = float(np.clip(m.max_demand_frac, 0.05, 1.0))

        # CHANGED:
        # Preserve monotonicity after clipping.
        max_demand_frac = max(max_demand_frac, min_demand_frac)

        return WaterMechanism(
            fixed_quota=float(np.clip(m.fixed_quota, 0.6, 0.95)),
            min_demand_frac=min_demand_frac,
            max_demand_frac=max_demand_frac,
            fine_amount=float(np.clip(m.fine_amount, 0.0, 0.1)),
            risk_penalty_scale=float(np.clip(m.risk_penalty_scale, 0.0, 1.0)),
            risk_penalty_power=float(np.clip(m.risk_penalty_power, 1.0, 5.0)),
            under_irrigation_penalty_scale=float(
                np.clip(m.under_irrigation_penalty_scale, 0.0, 1.0)
            ),
            max_farm_area_m2=float(np.clip(m.max_farm_area_m2, 100_000, 20_000_000)),
        )

    def from_dict(self, cfg: dict) -> WaterMechanism:
        # CHANGED:
        # Backward-compatible migration for old configs.
        # If old prop_quota/min_stock appear, ignore them and use the new defaults.
        cfg = dict(cfg)

        cfg.pop("prop_quota", None)
        cfg.pop("min_stock", None)

        if "min_demand_frac" not in cfg:
            cfg["min_demand_frac"] = self.defaults["min_demand_frac"]

        if "max_demand_frac" not in cfg:
            cfg["max_demand_frac"] = self.defaults["max_demand_frac"]

        if "risk_penalty_scale" not in cfg:
            cfg["risk_penalty_scale"] = self.defaults["risk_penalty_scale"]

        if "risk_penalty_power" not in cfg:
            cfg["risk_penalty_power"] = self.defaults["risk_penalty_power"]

        if "under_irrigation_penalty_scale" not in cfg:
            cfg["under_irrigation_penalty_scale"] = self.defaults[
                "under_irrigation_penalty_scale"
            ]

        if "max_farm_area_m2" not in cfg:
            cfg["max_farm_area_m2"] = self.defaults["max_farm_area_m2"]

        mech = WaterMechanism(**cfg)

        return self.clip(mech)
