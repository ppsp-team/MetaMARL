---
title: "Bilevel-Fishery: A Framework for Mechanism Design in Multi-Agent Reinforcement Learning"
tags:
  - Python
  - reinforcement learning
  - multi-agent systems
  - mechanism design
  - sustainability
  - simulation
authors:
  - name: Nadine Mohamed
    affiliation: 1
  - name: Guillaume Dumas
    affiliation: # TODO
  - name: Elise Devoie
    affiliation: # TODO
  - name: Benjamin Rossman
    affiliation: # TODO
  - name: Christofer Brandtner
    affiliation: # TODO
affiliations:
  - name: Mila - Quebec AI Institute, Université de Montréal, Canada
    index: 1
date: 2026
bibliography: paper.bib

---
# Summary

Sustainable resource management problems often involve multiple agents interacting within shared environments, where individual incentives can conflict with long-term system stability.

`Bilevel-Fishery` is a Python framework for studying mechanism design in multi-agent reinforcement learning (MARL) using a bilevel optimization approach. The framework combines:

- an **inner loop** where agents learn policies via reinforcement learning,
- an **outer loop** where regulatory mechanisms governing the environment are optimized.

The package provides a configurable simulation environment for renewable resource systems (e.g., fisheries), along with tools for defining and optimizing regulatory mechanisms such as quotas, fines, and enforcement policies.

---

# Statement of Need

Most reinforcement learning frameworks assume fixed environments and reward structures. However, many real-world systems require optimizing the rules of the environment itself, not just agent behavior.

Mechanism design in reinforcement learning is particularly challenging due to:

- delayed system-level feedback,
- misaligned short-term versus long-term incentives,
- non-stationary dynamics arising from learning agents,
- difficulty evaluating long-horizon outcomes.

`Bilevel-Fishery` addresses these challenges by providing:

- parameterized mechanism design,
- integration with multi-agent reinforcement learning,
- outer-loop optimization of regulatory policies,
- tools for analyzing emergent system behavior.

To our knowledge, no existing open-source framework integrates multi-agent reinforcement learning with mechanism optimization in a modular and extensible way.

---

# State of the Field

Existing frameworks such as RLlib [@liang2018rllib] and Stable-Baselines3 [@raffin2021stable] support agent training but do not address optimization of environment-level mechanisms.

Modern policy optimization methods such as Proximal Policy Optimization (PPO) [@schulman2017ppo] are widely used for agent learning, while black-box optimization approaches such as Evolution Strategies [@salimans2017evolution] provide scalable solutions for optimizing high-dimensional parameter spaces.

`Bilevel-Fishery` provides a computational framework bridging reinforcement learning and mechanism design.

---

# Software Design

The framework is structured around three main components:

## Environment

Implements renewable resource dynamics (e.g., fish–algae systems), supporting:

- continuous state variables (e.g., real-valued fish and algae population levels evolving over time),
- stochastic or deterministic transitions,
- configurable ecological parameters.

## Mechanism Space

Defines regulatory policies via parameterized spaces, including:

- quotas (fixed or proportional),
- penalties (fines),
- enforcement probability,
- restrictions (bans).

## Bilevel Optimization

- **Inner loop**: reinforcement learning (e.g., PPO/APPO via RLlib)  
- **Outer loop**: optimization over mechanisms (e.g., Evolution Strategies)

---

# Example Usage

Below is a minimal working example of running a bilevel experiment.

```python
import ray
from core.optimizers.bilevel import BilevelConfig
from core.optimizers.es.config import ESConfig
from core.optimizers.appo.config import APPOptimizerConfig

from examples.bilevel_fishery.mechanism import FisheryMechanismSpace
from examples.bilevel_fishery.regulated_env import FisheryRegulatedEnv
from examples.bilevel_fishery.regulator_env import FisheryRegulatorEnv

ray.shutdown()

config = (
    BilevelConfig()
    .world(world_name="fishery_world")

    .reporting(
        reporter="wandb",
        project_name="bilevel",
    )

    .mechanism(
        space=FisheryMechanismSpace(
            max_fine=10.0,
            max_ban=200,
            default_fixed_quota=1.0,
            default_prop_quota=1.0,
            default_min_stock=0.10,
            default_fine_amount=0.5,
            default_ban_period=0,
            default_catch_prob=1.0,
        ),
    )

    .training(outer_iters=100)

    .outer(
        ESConfig()
        .training(
            sigma=0.5,
            mean_lr=0.2,
        )
        .environment(
            env=FisheryRegulatorEnv,
            horizon=1000,
            train_iters=50,
        )
    )

    .inner(
        APPOptimizerConfig()
        .framework(framework="torch")
        .environment(
            env=FisheryRegulatedEnv,
            env_config={"seed": 0},
            horizon=1000,
        )
        .training(
            gamma=0.99,
            lr=1e-3,
            train_batch_size=200,
        )
    )
)

optimizer = config.build_optimizer()
optimizer.run()

ray.shutdown()
```
---
# Installation

Clone the repository and install:

    ```bash
    git clone https://github.com/<your-username>/bilevel-fishery.git
    cd bilevel-fishery
    pip install -e .
    ```

Install dependencies:

    ```bash
    pip install ray[rllib] wandb gymnasium
    ```

---

## Running Experiments

Experiments are defined via a `BilevelConfig` object and executed with:

    ```python
    optimizer = config.build_optimizer()
    optimizer.run()
    ```

Each run executes the following pipeline:

1. Sample mechanism parameters (outer loop)
2. Train agents under the mechanism (inner loop)
3. Evaluate system behavior
4. Update mechanism distribution (Evolution Strategies)
5. Repeat

---

## Configuration Overview

Experiments are composed of:

- **Mechanism space** → defines regulatory policies  
- **Inner loop** → agent learning (RLlib PPO/APPO)  
- **Outer loop** → mechanism optimization (Evolution Strategies)  
- **Environment** → ecological simulation  

---

## Logging and Visualization (Weights & Biases)

Experiments are tracked using **Weights & Biases (W&B)**.

### Setup
    ```bash
    wandb login
    ```

Enable logging in configuration:

    ```python
    .reporting(
        reporter="wandb",
        project_name="bilevel",
    )
    ```

---

## Visualization

W&B enables:

- training curves (reward, entropy)  
- ecosystem dynamics (fish vs algae trajectories)  
- comparison across mechanisms  
- multi-agent behavior tracking  

These visualizations are critical for identifying:

- over-exploitation regimes  
- delayed recovery effects  
- differences between reactive vs preventive regulation strategies  

---
# Research Impact

The framework enables:
- study of sustainability in multi-agent systems,
- analysis of long-term ecological dynamics,
- evaluation of adaptive regulatory strategies.

It serves as a research tool for reinforcement learning, computational economics, and sustainability modeling.

---

# Acknowledgements

This work was supported in part by the CIFAR Catalyst Fund (CF-0360) under the project *“Cooperative AI for Climate Action Coordination”*, led by :contentReference[oaicite:2]{index=2}.

The project brought together an interdisciplinary collaboration spanning machine learning, computational neuroscience, environmental science, and organizational research, including contributions from researchers affiliated with CIFAR programs.

The author also acknowledges discussions and collaborative exchanges with members of this project, as well as participants from workshops held in Lausanne (2025) and Toronto (2026), which contributed to the development of the ideas underlying this software.

---
# References
TODO
