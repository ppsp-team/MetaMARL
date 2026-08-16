"""Bilevel fishery experiment with the typed metrics/reporting stack.

The outer Evolution Strategies optimizer searches ``fixed_quota`` and
``restoration_subsidy``; the inner APPO optimizer trains the fishers against
each candidate. Every level logs a typed ``MetricSchema`` and renders the
queries of :mod:`examples.bilevel_fishery.queries` through the configured
reporter (Weights & Biases by default, CSV with ``--reporter csv``).

Smoke configuration::

    WANDB_MODE=offline uv run python -m examples.bilevel_fishery.debug \\
        --outer-iters 2 --train-iters 2 --num-agents 2 --horizon 20

Full configuration: the defaults.
"""

import numpy as np
import ray
from gymnasium import spaces


from core.adaptors.ray.schema import RaySchema
from core.callbacks import log_and_report_episode_metrics, tag_episode_with_env_idx
from core.optimizers.appo.config import APPOptimizerConfig
from core.optimizers.bilevel import BilevelConfig
from core.optimizers.es.config import ESConfig
from core.optimizers.es.schema import ESSchema
from core.reporting.wandb import WandbConfig
from examples.bilevel_fishery.metric_schema import FisheryMetricSchema
from core.callbacks import tag_episode_with_env_idx
from core.mechanism.algorithms.quota import QuotaMechanism
from core.mechanism.algorithms.social_influence import SocialInfluenceMechanism
from core.mechanism.algorithms.subsidy import SubsidyMechanism
from core.mechanism.algorithms.threhold_penalty import ThresholdPenaltyMechanism
from core.mechanism.composition.chained_mechanism import ChainedMechanism
from core.optimizers.bilevel import BilevelConfig
from core.optimizers.es.config import ESConfig
from core.optimizers.appo.config import APPOptimizerConfig
from examples.bilevel_fishery.regulated_env import FisheryRegulatedEnv
from examples.bilevel_fishery.regulator_env import FisheryRegulatorEnv
from examples.bilevel_fishery.queries import (
    ES_QUERIES,
    # FISHERY_ENV_QUERIES,
    INNER_QUERIES,
)

ray.shutdown()

EPS = 1e-8

bilevel_opt_cfg: BilevelConfig = (
    BilevelConfig()
    .world(world_name="fishery_world")
    .reporter(
        config=WandbConfig(
            project="bilevel",
            x_disable_stats=True,
            x_disable_meta=True,
            quiet=True,
            max_end_of_run_summary_metrics=0,
            max_end_of_run_history_metrics=0,
        )
    )
    .mechanism(
        mechanism = ChainedMechanism(
            children=(
                QuotaMechanism(
                    action_component=0,
                    bindings={
                        "resource_level": lambda env: (
                            env.S_t["fish"] / max(env.K, EPS)
                        ),
                    },
                    optimize_params=["fixed_quota"],
                    default_fixed_quota=0.56224, #0.90 #0.52
                    default_max_demand_frac=1.0,
                    default_fine_amount=0.20, 
                    default_risk_penalty_scale=0.0, #1.0
                    default_risk_penalty_power=1.0,

                ),
                SubsidyMechanism(
                    action_component=1,
                    optimize_params=["restoration_subsidy"],
                    default_restoration_subsidy=0.10,
                ),
                ThresholdPenaltyMechanism(
                    threshold=0.20,
                    penalty_amount=0.10,
                    transition_width=0.03,
                    bindings={
                        "resource_level": lambda env: (
                            env.S_t["fish"] / max(env.K, EPS)
                        ),
                    },
                ),
                SocialInfluenceMechanism(
                    influence_weight=...,
                    bindings={
                        "previous_actions": lambda env: (
                            env.previous_actions
                        ),
                        "agent_ids": lambda env: (
                            tuple(env.agents)
                        ),
                    },
                )
            )
        )
    )
    .training(outer_iters=1000)
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
            sigma=0.15,
            mean_lr=0.10,
            sigma_decay=1.0,
            sigma_lr=0.00,
            min_sigma=0.15,
            max_sigma=0.15,
        )
        .environment(
            env=FisheryRegulatorEnv,
            env_config={
                "ecology_cfg": {
                    "sustainability_weight": 2,  # assert between 0 and 5
                    "sustainability_threshold": 0.20,
                    "K": 5_000,  # HAS to match environmnet K
                },
            },
            horizon=100,
            train_iters=50,
        )
        .debugging(
            seed=42,
            num_seeds=1,
        )
        .reporting(
            schema=ESSchema,
            queries=ES_QUERIES,
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
            env=FisheryRegulatedEnv,
            env_config={
                "ecology_cfg": {
                    # Pella-Tomlinson / Schaefer single-stock dynamics
                    "r": 0.3,
                    "K": 5_000,
                    "p": 1.0,
                    "B0": 4_000,
                    "fish_init": 4_000,
                    # Env stochasticity
                    "sigma": 0.02,
                    "initial_stock_log_sigma": 0.05,
                    "unregulated_f_multiplier": 2.0,
                },
                "seed": 0,
            },
            horizon=100,
            disable_env_checking=False,
            schema=FisheryMetricSchema,
            queries=(),
        )
        .env_runners(
            num_env_runners=0,
            num_cpus_per_env_runner=1,
            num_gpus_per_env_runner=0,
            num_envs_per_env_runner=4,
            rollout_fragment_length=100,
            batch_mode="truncate_episodes",
            max_requests_in_flight_per_env_runner=1,
        )
        .learners(
            num_learners=0,
            num_gpus_per_learner=0,
        )
        .callbacks(
            on_episode_created=tag_episode_with_env_idx,
            on_episode_end=log_and_report_episode_metrics,
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
            train_batch_size_per_learner=100,
            minibatch_size=100,
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
                "rollout_fragment_length": 100,
                "batch_mode": "complete_episodes",
                "max_requests_in_flight_per_env_runner": 1,
            },
            base_seed=42,
            num_seeds=3,
        )
        .agents(
            {
                "utilizer": {
                    "count": 10,
                    "policy": "fisher_policy",
                    "observation_space": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(3 + FisheryMechanismSpace().full_dimension,),
                        dtype=np.float32,
                    ),
                    "action_space": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(2,), # TODO action shape must match the mechanism space components -> dynamically initiate this
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
            schema=RaySchema,
            queries=INNER_QUERIES,
            # TODO test queries agg over mechanisms (or other dynamic fields)
            # TODO test queries with y keys from reduced (env)
        )
    )
)

bilevel_opt = bilevel_opt_cfg.build_optimizer()

bilevel_opt.run()
ray.shutdown()
