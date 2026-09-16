import numpy as np
import ray
from gymnasium import spaces

from core.callbacks import tag_episode_with_env_idx
from core.optimizers.appo.config import APPOptimizerConfig
from core.optimizers.bilevel import BilevelConfig
from core.optimizers.es.config import ESConfig
from examples.fresh_water.mechanism import WaterMechanismSpace
from examples.fresh_water.regulated_env_ed_hs_v4 import WaterRegulatedEdHsEnv
from examples.fresh_water.regulator_env_raven import WaterRegulatorRavenEnv

ray.shutdown()

bilevel_opt_cfg: BilevelConfig = (
    BilevelConfig()
    .world(world_name="water_world")
    .reporting(
        reporter="wandb",
        project_name="bilevel",
        settings_dict={
            "x_disable_stats": True,
            "x_disable_meta": True,
            "quiet": True,
            "max_end_of_run_summary_metrics": 0,
            "max_end_of_run_history_metrics": 0,
        },
    )
    .mechanism(
        # TODO adding defaults
        space=WaterMechanismSpace(),
    )
    .training(outer_iters=100)
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
        .training(
            sigma=0.5,
            mean_lr=0.2,
            sigma_lr=0.05,
            min_sigma=0.01,
            max_sigma=0.6,
        )
        .environment(
            env=WaterRegulatorRavenEnv,
            env_config={
                "ecology_cfg": {
                    "sus_weight": 1.0,
                    "sus_threshold": 0.1,
                    "max_water": 100.0,
                },
            },
            horizon=100,
            train_iters=200,
        )
        .debugging(
            seed=42,
            num_seeds=1,
        )
    )
    .inner(
        APPOptimizerConfig()
        .resources(
            num_cpus_for_main_process=1,
        )
        .framework(
            framework="torch",
        )
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(
            env=WaterRegulatedEdHsEnv,
            env_config={
                "ecology_cfg": {
                    "max_farm_area_m2": 1_000_000.0,
                    # TODO move this to Raven helper
                    "full_stage_m": 420.41,
                    "max_depth_m": 11.0,
                    "lake_area_m2": 5756935.89615,
                },
                "use_raven": True,
                "raven_cwd": "/Users/nadine/src/github.com/nadinemgh/bilevel-fishery/examples/fresh_water/raven",
                "raven_cmd": "/Users/nadine/src/github.com/nadinemgh/bilevel-fishery/examples/fresh_water/raven/2_Raven/Raven.exe",
                "raven_freq": 1,
                "seed": 0,
            },
            horizon=150,
            disable_env_checking=False,
        )
        .env_runners(
            num_env_runners=0,
            num_cpus_per_env_runner=1,
            num_gpus_per_env_runner=0,
            num_envs_per_env_runner=1,
            rollout_fragment_length=150,
            batch_mode="truncate_episodes",
        )
        .learners(
            num_learners=0,
            num_gpus_per_learner=0,
        )
        .callbacks(
            on_episode_created=tag_episode_with_env_idx,
        )
        .training(
            vtrace=True,
            circular_buffer_num_batches=4,
            circular_buffer_iterations_per_batch=1,
            broadcast_interval=1,
            timeout_s_sampler_manager=10,
            timeout_s_aggregator_manager=10,
            gamma=0.99,
            lr=0.001,
            train_batch_size_per_learner=150,
            minibatch_size=150,
            num_epochs=1,
            entropy_coeff=0.001,
            grad_clip=40.0,
        )
        .evaluation(
            evaluation_interval=1,
            evaluation_duration=1,
            evaluation_duration_unit="episodes",
            evaluation_num_env_runners=0,
            evaluation_parallel_to_training=False,
            evaluation_config={
                "explore": False,
                "rollout_fragment_length": 150,
                "batch_mode": "complete_episodes",
            },
        )
        .agents(
            {
                "utilizer": {
                    "count": 500,
                    "policy": "utilizer_policy",
                    "observation_space": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(4 + WaterMechanismSpace().full_dimension,),
                        dtype=np.float32,
                    ),
                    "action_space": spaces.Box(
                        # Action is the fraction of the agent's maximum pull capacity.
                        #
                        # action = 0.0 -> requests 0% of max_pull_fraction
                        # action = 0.5 -> requests 50% of max_pull_fraction
                        # action = 1.0 -> requests 100% of max_pull_fraction
                        #
                        # Actual requested flow:
                        # requested_m3s = action * max_pull_fraction * current_streamflow
                        #
                        # With max_pull_fraction = 0.005:
                        # action high=1.0 -> up to 0.5% of current streamflow
                        # action high=0.5 -> up to 0.25% of current streamflow
                        # action high=0.2 -> up to 0.10% of current streamflow
                        # action high=0.1 -> up to 0.05% of current streamflow
                        #
                        # Rough agent interpretation:
                        # high=0.05 -> individual/small farm-scale user
                        # high=0.1  -> large farm / small irrigation user
                        # high=0.2  -> irrigation district / small utility
                        # high=0.5  -> municipality / industrial user
                        # high=1.0  -> large municipality / regional user
                        #
                        # NOTE:
                        # This does not change max_pull_fraction itself. It changes how much
                        # of that maximum capacity the policy is allowed to request.
                        low=0.0,
                        high=1,
                        shape=(1,),
                        dtype=np.float32,
                    ),
                }
            }
        )
        .fault_tolerance(
            restart_failed_env_runners=False,
        )
        .debugging(
            seed=42,
            num_seeds=1,
        )
        .reporting(
            min_time_s_per_iteration=0,
            min_sample_timesteps_per_iteration=0,
            min_train_timesteps_per_iteration=0,
        )
    )
)

bilevel_opt = bilevel_opt_cfg.build_optimizer()

bilevel_opt.run()
ray.shutdown()
