import logging
import numpy as np

from core.envs.marl_regulated import MultiAgentRegulatedEnv
from core.envs.hooks import observation, reset, reward, transition
from core.types import MultiAgentDict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

EPS = 1e-8


class FisheryRegulatedEnv(MultiAgentRegulatedEnv):
    def __init__(
        self,
        *,
        ecology_cfg: dict,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.r = ecology_cfg.get("r", 0.3)
        self.K = max(ecology_cfg.get("K", ecology_cfg.get("max_fish", 1000.0)), EPS)
        self.p = max(ecology_cfg.get("p", 1.0), EPS)
        self.fish_init = ecology_cfg.get("fish_init", ecology_cfg.get("B0", self.K))

        # stochasticity
        self.sigma = ecology_cfg.get("sigma", 0.05)
        self.initial_stock_log_sigma = float(ecology_cfg.get("initial_stock_log_sigma", 0.05))

        self.B_msy = max(self.K * (1.0 / (self.p + 1.0)) ** (1.0 / self.p), EPS)
        self.MSY = self.r * self.K / (self.p + 1.0) ** ((self.p + 1.0) / self.p)
        self.F_msy = self.MSY / max(self.B_msy, EPS)


        # TODO move collapse also out of env or should be a mechanism
        self.unregulated_f_multiplier = ecology_cfg.get("unregulated_f_multiplier", 2.0)
        self.obs_map = ["fish_norm", "total_usage_norm"] #unnecessary
        

    def _full_required_harvest(
        self,
        fish: float,
    ) -> float:
        return (
            self.unregulated_f_multiplier
            * self.F_msy
            * fish
            / len(self.agents)
        )

    
    @reset
    def reset_fishery(self) -> dict[str, float]:
        if self.initial_stock_log_sigma == 0.0:
            initial_fish = self.fish_init
        else:
            initial_fish = self.rng.lognormal(
                mean=np.log(max(self.fish_init, EPS)),
                sigma=self.initial_stock_log_sigma, #sigma around sampling from lognormal distribution
            )
        return {
            "fish": np.clip(initial_fish, EPS, self.K),
            "last_usage": 0.0,
        }

    @transition
    def pella_tomlinson(
        self,
        *,
        A_t: MultiAgentDict,
        S_t: MultiAgentDict,
        **kwargs
    ) -> dict[str, float]:
        fish = S_t["fish"]
        full_required_harvest = self._full_required_harvest(fish)
        delivered_harvest = {
            agent_id: action * full_required_harvest  for agent_id, action in A_t.items()
        }

        H = sum(delivered_harvest.values())
        B = max(fish, EPS)

        noise = self.sigma * self.rng.normal() * fish

        biological_growth = (self.r / self.p) * B * (1.0 - (B / self.K) ** self.p)
        growth = biological_growth + noise + kwargs["restoration"]
        available = max(B + growth, 0.0)

        H_realized = min(H, available)
        fish_next = available - H_realized
        fish_next = float(np.clip(fish_next, 0.0, K)) # TODO remove clipping

        new_state = {
            "fish": fish_next,
            "last_usage": H_realized,
        }

        self._update_infos(key="fish", values=fish)
        self._update_infos(key="fish_next", values=fish_next)
        self._update_infos(key="fish_norm", values=fish / max(self.K, EPS))
        self._update_infos(key="fish_norm_next", values=fish_next / max(self.K, EPS))
        self._update_infos(key="growth", values=growth)
        self._update_infos(key="growth_noise", values=noise)
        self._update_infos(key="H_attempted", values=H)
        self._update_infos(key="H_realized", values=H_realized)
        self._update_infos(key="total_usage_norm", values=H_realized / max(EPS, self.K))
        self._update_infos(key="B_msy", values=self.B_msy)
        self._update_infos(key="MSY", values=self.MSY)
        self._update_infos(key="F_msy", values=self.F_msy)

        self.S_t = new_state
        return self.S_t

    @observation
    def fishery_observation(
        self,
        observation_dict: MultiAgentDict,
    ) -> MultiAgentDict:
        fish_norm = self.S_t["fish"] / max(self.K, EPS)
        total_usage_norm = self.S_t.get("last_usage", 0.0) / max(EPS, self.K)
        observation = np.array([fish_norm, total_usage_norm])
        return {agent_id: observation.copy() for agent_id in self.agents}