import numpy as np
import ray
from gymnasium import spaces
from ray.rllib.models import ModelCatalog

# core optimizers
from core.adaptors.ray.mps_model import MPSFullyConnectedNetwork
from core.callbacks import tag_episode_with_env_idx
from core.optimizers.appo.config import APPOptimizerConfig
from core.optimizers.bilevel import BilevelConfig
from core.optimizers.es.config import ESConfig
from examples.cartpole.regulated_env import CartpoleRegulatedEnv

# Fishery-specific objects
from examples.dummy.mechanism import DummyMechanismSpace
from examples.dummy.regulator_env import DummyRegulatorEnv

# TODO the default mechanism config and fisherman, and observation spaces and action spaces part of config
# TODO where to do ray initialization ? gpu vs cpu - needs to happen when we build optimizer
# TODO num_fisherman
# TODO wire up the evaluation cfg
# TODO seeding API
# TODO experimentation helpers
# TODO review ray configz

ray.shutdown()

# TODO move this to the config !
# Register custom MPS model
ModelCatalog.register_custom_model("mps_fcnet", MPSFullyConnectedNetwork)

bilevel_opt_cfg: BilevelConfig = (
    BilevelConfig()
    .world(world_name="cartpole_world")
    .mechanism(
        space=DummyMechanismSpace(),
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
        .training(  # TODO dimension inferred from mechanism ?
            sigma=0.5,
            mean_lr=0.2,
            sigma_lr=0.05,
            min_sigma=0.01,
            max_sigma=0.6,
        )
        .environment(
            env=DummyRegulatorEnv,
            env_config={},
            horizon=1000,
            train_iters=200,  # TODO implement early stop for plateau
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
        # TODO fix this its using old api stack
        # .model(custom_model="mps_fcnet")
        # TODO use the new api stack and better custom model integration
        .api_stack(
            enable_rl_module_and_learner=True, enable_env_runner_and_connector_v2=True
        )
        .environment(
            env=CartpoleRegulatedEnv,
            env_config={
                "seed": 0,
            },
            horizon=1000,
            disable_env_checking=False,
        )
        .env_runners(
            num_env_runners=0,
            num_cpus_per_env_runner=1,
            num_gpus_per_env_runner=0,
            num_envs_per_env_runner=1,  # batch evaluated mechanism or population size for ES 16
            rollout_fragment_length=200,  # must be same as env horizon 200
            batch_mode="truncate_episodes",
        )
        .learners(num_learners=1, num_gpus_per_learner=0)
        .callbacks(
            on_episode_created=tag_episode_with_env_idx  # New API stack
        )
        .training(
            vtrace=True,
            circular_buffer_num_batches=2,  # TODO review
            circular_buffer_iterations_per_batch=1,  # TODO review
            # minibatch_buffer_size=200,
            broadcast_interval=5,
            # learner_queue_size=64,
            # learner_queue_timeout=300,
            timeout_s_sampler_manager=300,
            timeout_s_aggregator_manager=300,
            gamma=0.99,
            lr=0.001,
            train_batch_size=200,  # 3200
            minibatch_size=200,  # 512
            entropy_coeff=0.01,
            # entropy_coeff_schedule=[
            #     [0, 0.01],
            #     [200_000, 0.001],
            #     [1_000_000, 0.0],
            # ],
            grad_clip=40.0,
            # lr_schedule=[
            #     [0, 1e-3],
            #     [300_000, 3e-4],
            #     [1_000_000, 1e-4],
            # ]
        )
        .evaluation(
            evaluation_interval=None,
            evaluation_duration=1000,  # rollout_fragment_length X num_episodes
            evaluation_duration_unit="timesteps",
            evaluation_num_env_runners=1,
            # evaluation_parallel_to_training=False,  # keep it simple/deterministic
            evaluation_config={
                "explore": False,  # greedy eval actions
                "seed": 42,
                "num_envs_per_env_runner": 1,  # same as training
                "rollout_fragment_length": 1000,  # same as training
                "batch_mode": "complete_episodes",  # same as training
                "minibatch_size": None,
            },
        )
        .agents(
            {
                "agent_0": {
                    "count": 1,
                    "policy": "cartpole_policy",
                    "observation_space": spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(4,),
                        dtype=np.float32,
                    ),
                    "action_space": spaces.Discrete(2),
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
