from examples.bilevel_fishery.deprecated.mechanism import (
    FisheryMechanism as FisheryMechanism,
)
from examples.bilevel_fishery.deprecated.mechanism import FisheryMechanismSpace
from examples.bilevel_fishery.deprecated.regulated_env import FisheryRegulatedEnv
from examples.bilevel_fishery.deprecated.regulated_env_v1 import (
    FisheryRegulatedEnv as FisheryRegulatedEnvV1,
)
from examples.bilevel_fishery.mechanism_v1 import FisheryMechanism as FisheryMechanismV1
from examples.bilevel_fishery.mechanism_v1 import (
    FisheryMechanismSpace as FisheryMechanismSpaceV1,
)
from examples.bilevel_fishery.regulated_env_shaefer import (
    FisheryRegulatedEnv as FisheryRegulatedEnvSchaefer,
)
from examples.bilevel_fishery.regulator_env import FisheryRegulatorEnv
from examples.fresh_water.deprecated.regulated_env import WaterRegulatedEnv
from examples.fresh_water.deprecated.regulator_env import WaterRegulatorEnv

# Water-usage example components
from examples.fresh_water.regulated_env_ed_hs import WaterRegulatedEdHsEnv
from examples.fresh_water.regulator_env_raven import WaterRegulatorRavenEnv

REGISTRY = {
    "mechanism_space": {
        "FisheryMechanismSpace": FisheryMechanismSpace,
        "FisheryMechanismSpaceV1": FisheryMechanismSpaceV1,
        # "WaterMechanismSpace": WaterMechanismSpace,
    },
    "mechanism": {
        "FisheryMechanism": FisheryMechanism,
        "FisheryMechanismV1": FisheryMechanismV1,
        # "WaterMechanism": WaterMechanism,
    },
    "env": {
        "FisheryRegulatorEnv": FisheryRegulatorEnv,
        "FisheryRegulatedEnv": FisheryRegulatedEnv,
        "FisheryRegulatedEnvV1": FisheryRegulatedEnvV1,
        "FisheryRegulatedEnvSchaefer": FisheryRegulatedEnvSchaefer,
        "WaterRegulatorEnv": WaterRegulatorEnv,
        "WaterRegulatedEnv": WaterRegulatedEnv,
        # Generic bilevel aliases (map to water example by default)
        "RegulatorEnv": WaterRegulatorEnv,
        "WaterRegulatorRavenEnv": WaterRegulatorRavenEnv,
        "RegulatedEnv": WaterRegulatedEnv,
        "WaterRegulatedEdHsEnv": WaterRegulatedEdHsEnv,
    },
}
