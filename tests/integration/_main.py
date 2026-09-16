import numpy as np
from gymnasium import spaces
from ray.rllib.utils.from_config import NotProvided

# core optimizers
from core.optimizers.bilevel import BilevelConfig
from core.optimizers.es.config import ESConfig
from core.optimizers.ppo.config import PPOptimizerConfig

# Fishery-specific objects
from examples.bilevel_fishery.deprecated.mechanism import FisheryMechnanismSpace
from examples.bilevel_fishery.deprecated.regulated_env import FisheryRegulatedEnv
from examples.bilevel_fishery.regulator_env import FisheryRegulatorEnv

# TODO the default mechanism config and fisherman, and observation spaces and action spaces part of config
# TODO where to do ray initialization ? gpu vs cpu - needs to happen when we build optimizer
# TODO num_fisherman
# TODO wire up the evaluation cfg
# TODO seeding API
# TODO experimentation helpers

bilevel_opt_cfg: BilevelConfig = (
    BilevelConfig()
    .world(world_name="fishery_world")
    .outer(
        ESConfig()
        .training(
            dimension=5,  # TODO dimension inferred from mechanism ?
            pop_size=4,
            sigma=0.15,
            mean_lr=0.1,
            sigma_lr=0.05,
            min_sigma=1e-3,
            max_sigma=0.5,
        )
        .environment(
            env=FisheryRegulatorEnv,
            env_config={
                "mechanism_space": FisheryMechnanismSpace().from_dict(
                    cfg={
                        "fixed_quota": 0.2,  # Fixed harvest quota
                        "prop_quota": 0.1,  # Proportional quota factor
                        "min_stock": 0.1,  # Minimum stock threshold
                        "fine_amount": 1.0,  # Fine per unit over-harvest
                        "ban_period": 2,  # Periods banned after violation
                    }
                ),
                "ecology_cfg": {"sus_weight": 5.0, "sus_threshold": 0.1},
            },
            train_iters=2,
        )
    )
    .inner(
        PPOptimizerConfig()
        .python_environment(
            extra_python_environs_for_driver=NotProvided,
            extra_python_environs_for_worker=NotProvided,
        )
        .resources(
            num_cpus_for_main_process=1,
            placement_strategy=NotProvided,
        )
        .framework(
            framework="torch",
            eager_tracing=NotProvided,
            eager_max_retraces=NotProvided,
            tf_session_args=NotProvided,
            local_tf_session_args=NotProvided,
            torch_compile_learner=NotProvided,
            torch_compile_learner_what_to_compile=NotProvided,
            torch_compile_learner_dynamo_mode=NotProvided,
            torch_compile_learner_dynamo_backend=NotProvided,
            torch_compile_worker=NotProvided,
            torch_compile_worker_dynamo_backend=NotProvided,
            torch_compile_worker_dynamo_mode=NotProvided,
            torch_ddp_kwargs=NotProvided,
            torch_skip_nan_gradients=NotProvided,
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
                    "alpha": 1.0,
                    "beta": 0.5,
                    "delta": 0.5,
                    "gamma": 1.0,
                    "dt": 0.05,
                },
                "seed": 0,
            },
            observation_space=spaces.Dict(
                {
                    "fish": spaces.Box(
                        low=0.0,
                        high=np.finfo(np.float32).max,
                        shape=(1,),
                        dtype=np.float32,
                    ),
                    "algae": spaces.Box(
                        low=0.0,
                        high=np.finfo(np.float32).max,
                        shape=(1, 0),
                        dtype=np.float32,
                    ),
                }
            ),
            action_space=spaces.Dict(
                {
                    "fish_harvest": spaces.Box(
                        low=0.0,
                        high=np.finfo(np.float32).max,
                        shape=(1,),
                        dtype=np.float32,
                    )
                }
            ),
            horizon=30,
            render_env=NotProvided,
            clip_rewards=NotProvided,
            normalize_actions=NotProvided,
            clip_actions=NotProvided,
            disable_env_checking=NotProvided,
            is_atari=NotProvided,
            action_mask_key=NotProvided,
        )
        .env_runners(
            env_runner_cls=NotProvided,
            num_env_runners=1,  # num_workers
            create_local_env_runner=NotProvided,
            create_env_on_local_worker=NotProvided,
            num_envs_per_env_runner=NotProvided,
            gym_env_vectorize_mode=NotProvided,
            num_cpus_per_env_runner=1,
            num_gpus_per_env_runner=0,
            custom_resources_per_env_runner=NotProvided,
            validate_env_runners_after_construction=NotProvided,
            sample_timeout_s=NotProvided,
            max_requests_in_flight_per_env_runner=NotProvided,
            env_to_module_connector=NotProvided,
            module_to_env_connector=NotProvided,
            add_default_connectors_to_env_to_module_pipeline=NotProvided,
            add_default_connectors_to_module_to_env_pipeline=NotProvided,
            episode_lookback_horizon=NotProvided,
            merge_env_runner_states=NotProvided,
            broadcast_env_runner_states=NotProvided,
            use_worker_filter_stats=NotProvided,
            update_worker_filter_stats=NotProvided,
            rollout_fragment_length=NotProvided,
            batch_mode=NotProvided,
            explore=NotProvided,
            episodes_to_numpy=NotProvided,
            compress_observations=NotProvided,
        )
        .learners(
            num_learners=NotProvided,
            num_cpus_per_learner=NotProvided,
            num_gpus_per_learner=NotProvided,
            num_aggregator_actors_per_learner=NotProvided,
            max_requests_in_flight_per_aggregator_actor=NotProvided,
            local_gpu_idx=NotProvided,
            max_requests_in_flight_per_learner=NotProvided,
            learner_class=NotProvided,
            learner_connector=NotProvided,
            add_default_connectors_to_learner_pipeline=NotProvided,
            learner_config_dict=NotProvided,
        )
        .training(
            gramma=0.99,
            lr=3e-4,
            grad_clip=NotProvided,
            grad_clip_by=NotProvided,
            train_batch_size=200,
            train_batch_size_per_learner=NotProvided,
            num_epochs=NotProvided,
            minibatch_size=64,
            shuffle_batch_per_epoch=NotProvided,
            model=NotProvided,
            optimizer=NotProvided,
        )
        .evaluation(
            evaluation_interval=1,
            evaluation_duration=2,  # eval iters
            evaluation_duration_unit="episodes",
            evaluation_auto_duration_min_env_steps_per_sample=NotProvided,
            evaluation_auto_duration_max_env_steps_per_sample=NotProvided,
            evaluation_sample_timeout_s=NotProvided,
            evaluation_parallel_to_training=False,
            evaluation_force_reset_envs_before_iteration=NotProvided,
            evaluation_config={"explore": False},
            off_policy_estimation_methods=NotProvided,
            ope_split_batch_by_episode=NotProvided,
            evaluation_num_env_runners=NotProvided,
            custom_evaluation_function=NotProvided,
            offline_evaluation_interval=NotProvided,
            num_offline_eval_runners=NotProvided,
            offline_evaluation_type=NotProvided,
            offline_eval_runner_class=NotProvided,
            offline_loss_for_module_fn=NotProvided,
            offline_eval_batch_size_per_runner=NotProvided,
            dataset_num_iters_per_offline_eval_runner=NotProvided,
            offline_eval_rl_module_inference_only=NotProvided,
            num_cpus_per_offline_eval_runner=NotProvided,
            num_gpus_per_offline_eval_runner=NotProvided,
            custom_resources_per_offline_eval_runner=NotProvided,
            offline_evaluation_timeout_s=NotProvided,
            max_requests_in_flight_per_offline_eval_runner=NotProvided,
            broadcast_offline_eval_runner_states=NotProvided,
            validate_offline_eval_runners_after_construction=NotProvided,
            restart_failed_offline_eval_runners=NotProvided,
            ignore_offline_eval_runner_failures=NotProvided,
            max_num_offline_eval_runner_restarts=NotProvided,
            offline_eval_runner_health_probe_timeout_s=NotProvided,
            offline_eval_runner_restore_timeout_s=NotProvided,
        )
        .reporting(
            keep_per_episode_custom_metrics=NotProvided,
            metrics_episode_collection_timeout_s=NotProvided,
            metrics_num_episodes_for_smoothing=NotProvided,
            min_time_s_per_iteration=NotProvided,
            min_train_timesteps_per_iteration=NotProvided,
            min_sample_timesteps_per_iteration=NotProvided,
            log_gradients=NotProvided,
            custom_stats_cls_lookup=NotProvided,
        )
        .agents(
            {
                "fisher": {
                    "count": 2,
                    "policy": "fisher_policy",
                }
            }
        )
        .fault_tolerance(restart_failed_env_runners=False)
    )
    .training(outer_iters=10)
    .ray(
        device="cpu",
        num_cpus=8,
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
)


# run the experiment
bilevel_opt = bilevel_opt_cfg.build_optimizer()
bilevel_opt.run()

# .evaluation(
#     evaluation_interval=0,
#     evaluation_duration=5,  # eval iters
#     evaluation_duration_unit="episodes",
#     evaluation_num_env_runners=2,  # TODO dynamically build this from training config
#     evaluation_sample_timeout_s=NotProvided,
#     evaluation_parallel_to_training=False,
#     evaluation_force_reset_envs_before_iteration=True, #TODO review this - fix :
#     evaluation_config={"explore": False},
# )
