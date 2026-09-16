"""Optimizer layer: fluent configs and the optimizers they build.

``core.optimizers.config`` holds the base ``OptimizerConfig`` builder,
``core.optimizers.base`` the ``Optimizer`` contract, ``core.optimizers.bilevel``
the composition root that pairs an outer ``ESOptimizer`` with an inner
RLlib learner (``core.optimizers.appo`` / ``core.optimizers.ppo``). Nothing is
re-exported here; import from the submodules.
"""
