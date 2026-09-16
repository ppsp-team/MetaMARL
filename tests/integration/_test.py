import numpy as np
import ray
from gymnasium import spaces

# core optimizers
from core.optimizers.bilevel import BilevelConfig
from core.optimizers.es.config import ESConfig
from core.optimizers.ppo.config import PPOptimizerConfig

# Fishery-specific objects
from examples.bilevel_fishery.deprecated.mechanism import FisheryMechanism, FisheryMechanismSpace
from examples.bilevel_fishery.deprecated.regulated_env import FisheryRegulatedEnv
from examples.bilevel_fishery.regulator_env import FisheryRegulatorEnv

# TODO the default mechanism config and fisherman, and observation spaces and action spaces part of config
# TODO where to do ray initialization ? gpu vs cpu - needs to happen when we build optimizer
# TODO num_fisherman
# TODO wire up the evaluation cfg
# TODO seeding API
# TODO experimentation helpers
# TODO review ray configz

ray.shutdown()

bilevel_opt_cfg: BilevelConfig = (
    BilevelConfig()
    .world(world_name="fishery_world")
    .mechanism(
        space=FisheryMechanismSpace,
        default=FisheryMechanism(
            fixed_quota=0.2,
            prop_quota=0.1,
            min_stock=0.1,
            fine_amount=1.0,
            ban_period=2,
        ),
    )
    .training(outer_iters=1)  # 20
    .ray(
        device="cpu",
        num_cpus=4,
        omp_threads=1,
        logging_level="ERROR",
        runtime_env={
            "excludes": [
                "wandb/",
                ".git/",
                ".venv/",
                "__pycache__/",
                "ray_results/",
                "runs/",
            ]
        },
    )
    .outer(
        ESConfig()
        .training(  # TODO dimension inferred from mechanism ?
            sigma=0.15,
            mean_lr=0.1,
            sigma_lr=0.05,
            min_sigma=1e-3,
            max_sigma=0.5,
        )
        .environment(
            env=FisheryRegulatorEnv,
            env_config={
                "ecology_cfg": {
                    "sus_weight": 5.0,
                    "sus_threshold": 0.1,
                },
            },
            horizon=3,  # 200
            train_iters=1,
        )
    )
    .inner(
        PPOptimizerConfig()
        .resources(
            num_cpus_for_main_process=1,
        )
        .framework(
            framework="torch",
        )
        .api_stack(
            enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False
        )
        .environment(
            env=FisheryRegulatedEnv,
            env_config={
                "ecology_cfg": {
                    "algae_init": 1.0,
                    "fish_init": 0.5,
                    "max_fish": 2.0,
                    "max_algae": 2.0,
                    "alpha": 0.5,
                    "beta": 0.1,
                    "delta": 0.1,
                    "gamma": 0.5,
                    "dt": 0.01,
                    # "horizon": 200,
                },
                "seed": 0,
            },
            horizon=3,  # must be the same as regulator 200
        )
        .env_runners(
            num_env_runners=1,
            num_cpus_per_env_runner=1,
            num_gpus_per_env_runner=0,
            num_envs_per_env_runner=2,  # batch evaluated mechanism or population size for ES 16
            rollout_fragment_length=2,  # must be same as env horizon 200
            batch_mode="complete_episodes",
        )
        .training(
            gamma=0.99,
            lr=3e-4,
            train_batch_size=20,  # 3200
            minibatch_size=1,  # 512
        )
        .evaluation(
            evaluation_interval=None,
            evaluation_duration=2000,  # rollout_fragment_length X num_episodes
            evaluation_duration_unit="timesteps",
            evaluation_num_env_runners=1,
            # evaluation_parallel_to_training=False,  # keep it simple/deterministic
            evaluation_config={
                "explore": False,  # greedy eval actions
                "seed": 1234,
                "num_envs_per_env_runner": 16,  # same as training
                "rollout_fragment_length": 200,  # same as training
                "batch_mode": "complete_episodes",  # same as training
            },
        )
        .agents(
            {
                "fisher": {
                    "count": 3,
                    "policy": "fisher_policy",
                    "observation_space": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(
                            2 + FisheryMechanismSpace().dimension,
                        ),  # fish and alage #mechanism conditioned-RL
                        dtype=np.float32,
                    ),
                    "action_space": spaces.Box(
                        low=0.0,
                        high=0.3,
                        shape=(1,),
                        dtype=np.float32,
                    ),
                }
            }
        )
        .fault_tolerance(restart_failed_env_runners=False)
    )
)

# TODO
# custom_evaluation_function

# run the experiment
bilevel_opt = bilevel_opt_cfg.build_optimizer()
bilevel_opt.run()

# TODO add this after run done
ray.shutdown()
