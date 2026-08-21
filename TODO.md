# Visualization feature branch — handoff TODO

This document is the implementation handoff for the visualization/metrics
feature branch.

The target architecture is:

```text
WHAT DATA EXISTS        -> MetricSchema
HOW DATA ACCUMULATES    -> Metric / MetricLogger
WHAT DATA TO SELECT     -> Query
HOW IT IS DISPLAYED     -> Reporter
```

The branch is not complete until W&B reproduces the dev visualizations,
dynamic runtime keys are queryable, ES advanced plots are supported, tests are
in place, and CSV/TensorBoard reporters are complete.

# Mechanism abstraction branch — research engineer handoff TODO

## Scope

This branch redesigns regulation as explicit mechanism algorithms that intervene
on the agent/environment loop through three channels:

\[
\mathcal M_\theta
=
\left(
\mathcal M_\theta^O,
\mathcal M_\theta^A,
\mathcal M_\theta^R
\right)
\]

with:

\[
o_t^\* = \mathcal M_\theta^O(s_t, o_t),
\qquad
a_t^\* = \mathcal M_\theta^A(s_t, a_t),
\qquad
r_t^\* = \mathcal M_\theta^R(r_t, s_t, a_t^\*, s_{t+1}).
\]

The implementation has been designed and mostly written, but **has not yet been
validated end-to-end against `dev`**. Treat the current code as an integration
branch, not as a validated replacement.

Primary concerned files:

```text
core/envs/hooks.py
core/envs/marl_regulated.py
core/envs/regulated.py
core/mechanism/algorithms/penalty.py
core/mechanism/algorithms/quota.py
core/mechanism/algorithms/social_influence.py
core/mechanism/algorithms/subsidy.py
core/mechanism/base.py
core/mechanism/composition/chained_mechanism.py
core/mechanism/composition/parallel_mechanism.py
core/mechanism/space.py
core/types.py
```

The first goal is **not new features**. The first goal is to make this complete
abstraction run end-to-end, cover the concerned modules with tests, and verify
that the new abstraction preserves benchmark behavior.
>>>>>>> c9c15fb (feat: TODO and tutorial notebooks)

---

# 0. Definition of done

- [ ] Environment plots on the feature branch reproduce the dev plots/data.
- [ ] Inner optimizer plots reproduce the dev plots/data.
- [ ] ES plots reproduce the dev plots/data.
- [ ] Mechanism IDs, seed IDs, episode IDs, policy IDs, agent IDs, and ES
      parameter names do not need to be hard-coded into user queries.
- [ ] `Query` supports runtime dict keys with `"*"`.
- [ ] Mean ±1 std across seeds works per mechanism.
- [ ] Train-vs-eval shaded plots work per mechanism.
- [ ] ES cumulative parameter scatter plots work across every candidate and
      generation.
- [ ] ES parallel-coordinates plot is supported.
- [ ] Unit tests cover MetricLogger, schema polymorphism, Query resolution,
      reporters, environment/Ray/ES integration, and dynamic wildcards.
- [ ] CSV export is implemented and tested.
- [ ] TensorBoard reporting is implemented and tested.
- [ ] Legacy dev W&B plotting utilities can be removed only after parity is
      proven.

---

# 1. P0 — validate exact dev plot parity

## 1.1 Environment-level plots

The environment reporter is attached to `FisheryMetricSchema` and should
render the environment horizon before the episode logger is destructively
reduced.

Required current queries:

```python
FISHERY_ENV_QUERIES = [
    Query(
        title="Fish biomass",
        x=("iter",),
        y=("fish_norm",),
    ),
    Query(
        title="Next fish biomass",
        x=("iter",),
        y=("fish_norm_next",),
    ),
    Query(
        title="Fish stock",
        x=("iter",),
        y=("fish_stock",),
    ),
    Query(
        title="Next fish stock",
        x=("iter",),
        y=("fish_stock_next",),
    ),
    Query(
        title="Biological growth",
        x=("iter",),
        y=("growth",),
    ),
    Query(
        title="Growth noise",
        x=("iter",),
        y=("growth_noise",),
    ),
    Query(
        title="Attempted harvest",
        x=("iter",),
        y=("H_attempted",),
    ),
    Query(
        title="Realized harvest",
        x=("iter",),
        y=("H_realized",),
    ),
    Query(
        title="Allowed harvest",
        x=("iter",),
        y=("allowed_harvest",),
    ),
    Query(
        title="Total usage normalized",
        x=("iter",),
        y=("total_usage_norm",),
    ),
    Query(
        title="Quota stress",
        x=("iter",),
        y=("quota_stress",),
    ),
    Query(
        title="Biomass at MSY",
        x=("iter",),
        y=("B_msy",),
    ),
    Query(
        title="Maximum sustainable yield",
        x=("iter",),
        y=("MSY",),
    ),
    Query(
        title="Fishing mortality at MSY",
        x=("iter",),
        y=("F_msy",),
    ),
]
```

Current per-agent helper:

```python
def fishery_agent_queries(agent_id: str) -> list[Query]:
    base = ("by_agent", agent_id)

    return [
        Query(
            title=f"Reward — {agent_id}",
            x=("iter",),
            y=base + ("reward",),
        ),
        Query(
            title=f"Action — {agent_id}",
            x=("iter",),
            y=base + ("action",),
        ),
        Query(
            title=f"Observation — {agent_id}",
            x=("iter",),
            y=base + ("observation",),
        ),
        Query(
            title=f"Intrinsic utility — {agent_id}",
            x=("iter",),
            y=base + ("intrinsic_utility",),
        ),
        Query(
            title=f"Violation signal — {agent_id}",
            x=("iter",),
            y=base + ("violation_signal",),
        ),
        Query(
            title=f"Requested harvest — {agent_id}",
            x=("iter",),
            y=base + ("requested_harvest",),
        ),
        Query(
            title=f"Delivered harvest — {agent_id}",
            x=("iter",),
            y=base + ("delivered_harvest",),
        ),
        Query(
            title=f"Requested harvest fraction — {agent_id}",
            x=("iter",),
            y=base + ("requested_frac",),
        ),
        Query(
            title=f"Quota violation — {agent_id}",
            x=("iter",),
            y=base + ("quota_violation",),
        ),
        Query(
            title=f"Quota penalty — {agent_id}",
            x=("iter",),
            y=base + ("quota_penalty",),
        ),
        Query(
            title=f"Risk penalty — {agent_id}",
            x=("iter",),
            y=base + ("risk_penalty",),
        ),
    ]
```

### Environment acceptance checks

- [ ] `env.logger.peek()` produces a horizon-length x series and aligned y
      series before `reduce()`.
- [ ] `env.reporter.report(env.logger.peek())` does not mutate or clear the
      logger.
- [ ] After reporting, `env.logger.reduce()` still returns the correct compiled
      episode metrics.
- [ ] Agent query traces match the agent values shown on dev.
- [ ] No environment-specific field silently disappears when the runtime
      `FisheryMetricSchema` subtype is materialized.
- [ ] No `FisheryAgentMetricSchema` field disappears at the deeper `by_agent`
      runtime subtype.

### Data/schema gaps versus old dev environment plotting

The dev environment plotting code referenced some fields not present in the
provided current `FisheryMetricSchema`, including:

- `full_required_harvest`
- `min_demand_frac`
- `max_demand_frac`
- some environment-level `intrinsic_utility` usage
- named observation/info components

For exact parity:

- [ ] Decide whether each missing metric is environment-level or agent-level.
- [ ] Add shared values to `FisheryMetricSchema`.
- [ ] Add agent-specific values to `FisheryAgentMetricSchema`.
- [ ] Add named observation fields if exact old observation plots are required.
- [ ] Add queries only after the schema field exists.

---

## 1.2 Inner optimizer: raw RLlib rollout plots

Required queries:

```python
RAY_ROLLOUT_QUERIES = [
    Query(
        title="Train reward",
        x=("iter",),
        y=(
            ("train", "rollout", "aggregate", "reward_mean"),
            ("train", "rollout", "aggregate", "reward_min"),
            ("train", "rollout", "aggregate", "reward_max"),
        ),
    ),
    Query(
        title="Train episode length",
        x=("iter",),
        y=(
            ("train", "rollout", "aggregate", "episode_len_mean"),
            ("train", "rollout", "aggregate", "episode_len_min"),
            ("train", "rollout", "aggregate", "episode_len_max"),
        ),
    ),
    Query(
        title="Train episodes",
        x=("iter",),
        y=("train", "rollout", "aggregate", "num_episodes"),
    ),
    Query(
        title="Train episodes lifetime",
        x=("iter",),
        y=("train", "rollout", "aggregate", "num_episodes_lifetime"),
    ),
    Query(
        title="Eval reward",
        x=("iter",),
        y=(
            ("eval", "rollout", "aggregate", "reward_mean"),
            ("eval", "rollout", "aggregate", "reward_min"),
            ("eval", "rollout", "aggregate", "reward_max"),
        ),
    ),
    Query(
        title="Eval episode length",
        x=("iter",),
        y=(
            ("eval", "rollout", "aggregate", "episode_len_mean"),
            ("eval", "rollout", "aggregate", "episode_len_min"),
            ("eval", "rollout", "aggregate", "episode_len_max"),
        ),
    ),
    Query(
        title="Eval episodes",
        x=("iter",),
        y=("eval", "rollout", "aggregate", "num_episodes"),
    ),
]
```

Acceptance:

- [ ] Train reward mean/min/max are numerically equivalent to dev.
- [ ] Eval reward mean/min/max are numerically equivalent to dev.
- [ ] Episode lengths and episode counts are equivalent.
- [ ] x-axis uses the intended RLlib iteration and is monotonic.

---

## 1.3 Inner optimizer: performance plots

Required queries:

```python
RAY_PERFORMANCE_QUERIES = [
    Query(
        title="Train environment steps",
        x=("iter",),
        y=(
            ("train", "performance", "env_steps_this_iter"),
            ("train", "performance", "env_steps_lifetime"),
        ),
    ),
    Query(
        title="Train agent steps",
        x=("iter",),
        y=(
            ("train", "performance", "agent_steps_this_iter_sum"),
            ("train", "performance", "agent_steps_lifetime_sum"),
        ),
    ),
    Query(
        title="Environment throughput",
        x=("iter",),
        y=("train", "performance", "env_steps_throughput"),
    ),
    Query(
        title="Training timing",
        x=("iter",),
        y=(
            ("train", "performance", "training_iteration_s"),
            ("train", "performance", "training_step_s"),
            ("train", "performance", "sample_s"),
            ("train", "performance", "learner_update_s"),
        ),
    ),
    Query(
        title="Weights sequence number",
        x=("iter",),
        y=("train", "performance", "weights_seq_no"),
    ),
    Query(
        title="Eval environment steps",
        x=("iter",),
        y=(
            ("eval", "performance", "env_steps_this_iter"),
            ("eval", "performance", "env_steps_lifetime"),
        ),
    ),
    Query(
        title="Eval agent steps",
        x=("iter",),
        y=(
            ("eval", "performance", "agent_steps_this_iter_sum"),
            ("eval", "performance", "agent_steps_lifetime_sum"),
        ),
    ),
    Query(
        title="Eval weights sequence number",
        x=("iter",),
        y=("eval", "performance", "weights_seq_no"),
    ),
]
```

Old dev also had timing/throughput fields such as sync-weight time and
learn-throughput depending on the RLlib result version.

- [ ] Compare current `PerformanceSchema` to the exact dev fields.
- [ ] Add missing fields only if exact dev parity requires them.
- [ ] Do not read raw RLlib dictionaries directly from the reporter once the
      schema adaptor owns those mappings.

---

## 1.4 Inner optimizer: per-policy learner plots

Current helper:

```python
def ray_policy_queries(policy_id: str) -> list[Query]:
    base = ("train", "learner", "by_policy", policy_id)

    return [
        Query(
            title=f"Batch size — {policy_id}",
            x=("iter",),
            y=base + ("batch_size",),
        ),
        Query(
            title=f"Total loss — {policy_id}",
            x=("iter",),
            y=base + ("total_loss",),
        ),
        Query(
            title=f"Residual variance — {policy_id}",
            x=("iter",),
            y=base + ("residual_variance",),
        ),
        Query(
            title=f"Sample staleness — {policy_id}",
            x=("iter",),
            y=base + ("sample_staleness",),
        ),
        Query(
            title=f"Policy loss — {policy_id}",
            x=("iter",),
            y=base + ("policy_loss",),
        ),
        Query(
            title=f"Policy entropy — {policy_id}",
            x=("iter",),
            y=base + ("policy_entropy",),
        ),
        Query(
            title=f"Policy entropy coefficient — {policy_id}",
            x=("iter",),
            y=base + ("policy_entropy_coeff",),
        ),
        Query(
            title=f"Policy relative entropy — {policy_id}",
            x=("iter",),
            y=base + ("policy_relative_entropy",),
        ),
        Query(
            title=f"Entropy pressure — {policy_id}",
            x=("iter",),
            y=base + ("entropy_pressure",),
        ),
        Query(
            title=f"Policy KL — {policy_id}",
            x=("iter",),
            y=base + ("policy_kl",),
        ),
        Query(
            title=f"Policy KL coefficient — {policy_id}",
            x=("iter",),
            y=base + ("policy_kl_coeff",),
        ),
        Query(
            title=f"Value loss — {policy_id}",
            x=("iter",),
            y=base + ("value_loss",),
        ),
        Query(
            title=f"Value mean — {policy_id}",
            x=("iter",),
            y=base + ("value_mean",),
        ),
        Query(
            title=f"Value target — {policy_id}",
            x=("iter",),
            y=base + ("value_target",),
        ),
        Query(
            title=f"Gradient norm — {policy_id}",
            x=("iter",),
            y=base + ("gradient_norm",),
        ),
        Query(
            title=f"Gradient noise — {policy_id}",
            x=("iter",),
            y=base + ("gradient_noise",),
        ),
    ]
```

Acceptance:

- [ ] Every policy in `LearnerSchema.by_policy` can be plotted.
- [ ] Policies do not have to be manually hard-coded after wildcard support.
- [ ] `__all_modules__` / aggregate learner entries are handled intentionally:
      either include with a clear label or exclude explicitly.
- [ ] Loss/entropy/KL/gradient values match dev for the same deterministic run.

Old dev learner fields included names such as `kl`, `entropy`, `vf_loss`,
`policy_loss`, `total_loss`, `vf_explained_var`, `grad_gnorm`, `cur_lr`,
and `cur_kl_coeff`.

Current schema uses normalized names such as `policy_kl`, `policy_entropy`,
`value_loss`, and `gradient_norm`.

- [ ] Make a documented mapping table from dev metric names to new schema names.
- [ ] Add genuinely missing metrics if exact parity requires them.
- [ ] Do not duplicate equivalent metrics under two names without a reason.

---

# 2. P0 — dynamic dictionary key support in Query

Current limitation:

```python
Query(
    ...,
    y=(
        "train",
        "rollout",
        "by_mechanism",
        "0",       # hard-coded
        "by_seed",
        "100",     # hard-coded
        ...
    ),
)
```

Target:

```python
Query(
    ...,
    y=(
        "train",
        "rollout",
        "by_mechanism",
        "*",
        "by_seed",
        "*",
        ...
    ),
)
```

## 2.1 Required wildcard semantics

- [ ] `"*"` matches keys only at dynamic dict nodes.
- [ ] Static schema fields are not accidentally wildcarded.
- [ ] One wildcard expands to one trace per matched runtime key.
- [ ] Multiple wildcards retain their bindings.
- [ ] Expansion order is deterministic (sort keys or preserve a documented
      insertion order).
- [ ] Missing dynamic branches produce a useful error or an empty match
      according to an explicit policy.
- [ ] An exact concrete key continues to work unchanged.
- [ ] Wildcard expansion does not mutate the schema/MetricLogger.
- [ ] Wildcard resolution works on runtime subtype nodes.

## 2.2 x/y binding

This is critical for ES scatter plots.

Given:

```python
x=(
    "by_mechanism",
    "*",
    "by_parameter",
    "fixed_quota",
    "value",
)

y=(
    "by_mechanism",
    "*",
    "fitness",
)
```

the same wildcard binding must be used on both sides:

```text
x m0 <-> y m0
x m1 <-> y m1
x m2 <-> y m2
x m3 <-> y m3
```

Never form a Cartesian product of candidate x values and candidate y values.

Tests must make candidate values obviously distinct so a binding error cannot
pass accidentally.

---

# 3. P0 — mechanism mean ±std across seeds

Desired user-facing query:

```python
Query(
    title="Train fish biomass by mechanism ±1 std across seeds",
    x=("iter",),
    y=(
        "train",
        "rollout",
        "by_mechanism",
        "*",
        "by_seed",
        "*",
        "by_episode",
        "*",
        "fish_norm",
    ),
    reduce="mean",
    error="std",
)
```

Expected output:

```text
m0 mean + shaded ±1 std
m1 mean + shaded ±1 std
m2 mean + shaded ±1 std
...
```

The implementation must distinguish:

- grouping dimension: mechanism;
- replicate/reduction dimensions: seed and, if needed, episode.

A naïve wildcard implementation that averages every mechanism and every seed
into one line is incorrect.

## 3.1 Resolve the `SeedRolloutSchema.aggregate` question

The requested older-style path was:

```python
(
    "by_mechanism",
    "0",
    "by_seed",
    "100",
    "aggregate",
    "fish_norm",
)
```

But the supplied schema is:

```python
class SeedRolloutSchema(MetricSchema):
    by_episode: dict[EpisodeID, EpisodeRolloutSchema]
```

There is no `aggregate`.

Choose one:

### Option A — add seed aggregate

```python
class SeedRolloutSchema(MetricSchema):
    aggregate: EpisodeRolloutSchema
    by_episode: dict[EpisodeID, EpisodeRolloutSchema] = Field(
        default_factory=dict
    )
```

Then update the Ray adaptor to populate it.

### Option B — no schema change

Keep `by_episode` only and let Query resolution aggregate wildcard-matched
episode leaves.

Acceptance:

- [ ] The choice is documented.
- [ ] Tests use the final supported path only.
- [ ] No tutorial/example advertises a nonexistent `aggregate` field.

---

# 4. P0 — train-vs-eval shaded mechanism plots

Target query:

```python
Query(
    title="Fish biomass: train vs eval by mechanism ±1 std across seeds",
    x=("iter",),
    y=(
        (
            "train",
            "rollout",
            "by_mechanism",
            "*",
            "by_seed",
            "*",
            "by_episode",
            "*",
            "fish_norm",
        ),
        (
            "eval",
            "rollout",
            "by_mechanism",
            "*",
            "by_seed",
            "*",
            "by_episode",
            "*",
            "fish_norm",
        ),
    ),
    reduce="mean",
    error="std",
)
```

Dev behavior to reproduce:

- [ ] train and eval appear in the same figure;
- [ ] one mean curve per phase × mechanism;
- [ ] ±1 std shaded band across seeds;
- [ ] mechanism identity is distinguishable;
- [ ] train/eval identity is distinguishable;
- [ ] deterministic legend order;
- [ ] horizon version uses environment step;
- [ ] over-training version uses RLlib training iteration;
- [ ] no W&B-specific grouping logic is required in the optimizer.

Required target queries for the primary fishery metrics:

```python
TARGET_TRAIN_EVAL_QUERIES = [
    Query(
        title="Fish biomass: train vs eval by mechanism ±1 std",
        x=("iter",),
        y=(
            (
                "train", "rollout", "by_mechanism", "*",
                "by_seed", "*", "by_episode", "*", "fish_norm",
            ),
            (
                "eval", "rollout", "by_mechanism", "*",
                "by_seed", "*", "by_episode", "*", "fish_norm",
            ),
        ),
        reduce="mean",
        error="std",
    ),
    Query(
        title="Realized harvest: train vs eval by mechanism ±1 std",
        x=("iter",),
        y=(
            (
                "train", "rollout", "by_mechanism", "*",
                "by_seed", "*", "by_episode", "*", "H_realized",
            ),
            (
                "eval", "rollout", "by_mechanism", "*",
                "by_seed", "*", "by_episode", "*", "H_realized",
            ),
        ),
        reduce="mean",
        error="std",
    ),
    Query(
        title="Reward: train vs eval by mechanism ±1 std",
        x=("iter",),
        y=(
            (
                "train", "rollout", "by_mechanism", "*",
                "by_seed", "*", "by_episode", "*", "reward_mean",
            ),
            (
                "eval", "rollout", "by_mechanism", "*",
                "by_seed", "*", "by_episode", "*", "reward_mean",
            ),
        ),
        reduce="mean",
        error="std",
    ),
    Query(
        title="Quota stress: train vs eval by mechanism ±1 std",
        x=("iter",),
        y=(
            (
                "train", "rollout", "by_mechanism", "*",
                "by_seed", "*", "by_episode", "*", "quota_stress",
            ),
            (
                "eval", "rollout", "by_mechanism", "*",
                "by_seed", "*", "by_episode", "*", "quota_stress",
            ),
        ),
        reduce="mean",
        error="std",
    ),
]
```

Extend the same pattern to the remaining fishery environment fields only when
they are useful and present in the schema.

---

# 5. P0 — ES exact dev plot parity

The old dev ES visualization had one accumulated history row per evaluated
candidate:

```text
generation
mechanism_idx
fitness
sigma
parameter_0
parameter_1
...
```

It produced:

1. fitness over outer generations;
2. one cumulative fitness-vs-parameter scatter per optimized parameter;
3. cumulative parallel coordinates;
4. scalar series for sigma;
5. search mean per parameter;
6. global-best fitness;
7. global-best candidate parameter values;
8. generation-best parameter values;
9. all-generations table.

The new logger should remain the source of truth; do not reintroduce a separate
global `_ES_HISTORY_TABLES` history cache.

---

## 5.1 ES schema verification

The supplied `ESSchema` includes:

- `sigma`
- `population_size`
- `fitness_mean`
- `fitness_best`
- `best_mechanism_idx`
- `best_fitness_global`
- `by_mechanism`
- `search_mean`
- `global_best`
- `inner`

### Required checks

- [ ] `generation` is explicitly present and is `ReduceProtocol.SERIES`, or the
      entire implementation consistently uses inherited `iter`.
- [ ] The optimizer and Query use the same x field.
- [ ] Add `generation_best` for exact dev parity.
- [ ] Consider renaming the type alias used for `search_mean` and
      `global_best` keys from `MechanismID` to `ParameterName`; those dict keys
      are parameter names, not mechanism IDs.

Recommended additions if not already present:

```python
ParameterName: TypeAlias = str

class ESSchema(MetricSchema):
    generation: Optional[int] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.SERIES},
    )

    generation_best: dict[ParameterName, ESParameterSchema] = Field(
        default_factory=dict
    )
```

Populate:

```python
generation_best={
    parameter_name: ESParameterSchema(
        value=float(population[best_idx, parameter_idx])
    )
    for parameter_idx, parameter_name in enumerate(parameter_names)
}
```

---

## 5.2 ES fitness-over-generations queries

Fishery acceptance fixture:

```python
ES_CANDIDATE_IDS = ("0", "1", "2", "3")
```

Required query data:

```python
Query(
    title="Fitness over outer optimization iterations",
    x=("generation",),
    y=(
        ("by_mechanism", "0", "fitness"),
        ("by_mechanism", "1", "fitness"),
        ("by_mechanism", "2", "fitness"),
        ("by_mechanism", "3", "fitness"),
        ("fitness_mean",),
        ("fitness_best",),
    ),
)

Query(
    title="Global best fitness",
    x=("generation",),
    y=("best_fitness_global",),
)

Query(
    title="ES sigma",
    x=("generation",),
    y=("sigma",),
)
```

Exact visual parity:

- [ ] candidate fitness = marker traces;
- [ ] generation mean = line + markers;
- [ ] generation best = line + markers;
- [ ] candidate hover contains outer generation, candidate/mechanism index, and
      fitness;
- [ ] figure is cumulative over all completed generations.

This likely requires trace style metadata or a specialized plot query. Data
selection alone is not enough to reproduce marker-vs-line semantics.

---

## 5.3 ES search mean and global best

Required:

```python
Query(
    title="ES search mean",
    x=("generation",),
    y=(
        ("search_mean", "fixed_quota", "value"),
        ("search_mean", "restoration_subsidy", "value"),
    ),
)

Query(
    title="Global-best mechanism parameters",
    x=("generation",),
    y=(
        ("global_best", "fixed_quota", "value"),
        ("global_best", "restoration_subsidy", "value"),
    ),
)
```

After `generation_best` is added:

```python
Query(
    title="Generation-best mechanism parameters",
    x=("generation",),
    y=(
        ("generation_best", "fixed_quota", "value"),
        ("generation_best", "restoration_subsidy", "value"),
    ),
)
```

Parameter names are runtime-defined by `MechanismSpace`.

Target dynamic form:

```python
Query(
    title="ES search mean",
    x=("generation",),
    y=("search_mean", "*", "value"),
)

Query(
    title="Global-best mechanism parameters",
    x=("generation",),
    y=("global_best", "*", "value"),
)

Query(
    title="Generation-best mechanism parameters",
    x=("generation",),
    y=("generation_best", "*", "value"),
)
```

- [ ] No optimized parameter name must be hard-coded after wildcard support.

---

## 5.4 ES fitness-vs-parameter scatter

Current concrete smoke-test generator:

```python
def es_parameter_fitness_queries(
    candidate_ids=("0", "1", "2", "3"),
    parameter_names=("fixed_quota", "restoration_subsidy"),
):
    queries = []

    for parameter_name in parameter_names:
        for candidate_id in candidate_ids:
            queries.append(
                Query(
                    title=f"Fitness vs {parameter_name} — candidate {candidate_id}",
                    x=(
                        "by_mechanism",
                        candidate_id,
                        "by_parameter",
                        parameter_name,
                        "value",
                    ),
                    y=(
                        "by_mechanism",
                        candidate_id,
                        "fitness",
                    ),
                )
            )

    return queries
```

This verifies that the data paths work, but it does **not** reproduce the dev
figure exactly.

Exact target:

```python
Query(
    title="Fitness vs fixed_quota",
    x=(
        "by_mechanism",
        "*",
        "by_parameter",
        "fixed_quota",
        "value",
    ),
    y=("by_mechanism", "*", "fitness"),
)
```

Required semantics:

- [ ] all candidates in one cumulative scatter;
- [ ] all generations included;
- [ ] x/y wildcard bindings aligned by candidate ID;
- [ ] each point carries generation metadata;
- [ ] point color represents outer generation, as on dev;
- [ ] colorbar title identifies outer iteration;
- [ ] hover includes outer iteration, candidate/mechanism index, parameter
      value, and fitness;
- [ ] one figure per runtime optimized parameter.

The current `Query` has no z/color metadata. Implement one of:

- [ ] optional query metadata path for color/group;
- [ ] `ScatterQuery`;
- [ ] generic named-dimension query consumed by the W&B reporter.

Do not bury generation lookup inside a W&B-only helper if the same semantic
plot should be portable to another backend.

---

## 5.5 ES parallel coordinates

Dev behavior:

- every evaluated candidate is one line;
- each optimized mechanism parameter is one axis;
- fitness is the final axis;
- fitness also controls line color;
- axis ranges are dynamically padded;
- all generations accumulate.

The current `Query(x, y)` API cannot represent this.

Implement one of:

### Option A

```python
ParallelCoordinatesQuery(
    title="Parallel coordinates of evaluated mechanisms",
    dimensions=(
        ("by_mechanism", "*", "by_parameter", "*", "value"),
        ("by_mechanism", "*", "fitness"),
    ),
    color=("by_mechanism", "*", "fitness"),
)
```

### Option B

A generic table/multidimensional query that resolves a row per evaluated
candidate and lets reporters choose a parallel-coordinate renderer.

Acceptance:

- [ ] all optimized parameters appear exactly once;
- [ ] fitness appears exactly once as final axis;
- [ ] line color = fitness;
- [ ] rows remain aligned across parameter dimensions;
- [ ] cumulative data across generations;
- [ ] fixed-mode ES still uses the full default mechanism vector;
- [ ] empty/constant ranges do not crash.

---

# 6. P0 — schema extension rules must be documented and tested

## Add shared environment metric

Subclass `EpisodeRolloutSchema`:

```python
class FisheryMetricSchema(EpisodeRolloutSchema):
    new_env_metric: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )
```

## Add per-agent metric

Subclass `AgentEnvStepSchema` and override `by_agent`:

```python
class FisheryAgentMetricSchema(AgentEnvStepSchema):
    new_agent_metric: Optional[float] = Field(
        default=None,
        json_schema_extra={"reduce": ReduceProtocol.MEAN},
    )

class FisheryMetricSchema(EpisodeRolloutSchema):
    by_agent: dict[str, FisheryAgentMetricSchema] = Field(
        default_factory=dict
    )
```

## Add learner metric

Add to `PolicyLearnerSchema`.

## Add runtime policy dimension

Use:

```python
by_policy: dict[PolicyID, PolicyLearnerSchema]
```

Do not add one static field per policy.

## Add performance metric

Add to `PerformanceSchema`.

## Add ES generation metric

Add to `ESSchema`, generally as `SERIES` if the plot needs the full outer
history.

## Add ES candidate metric

Add to `ESCandidateSchema`.

## Add ES parameter metric

Add to `ESParameterSchema`.

## Add inner optimizer-specific schema under ES

Keep:

```python
inner: Optional[MetricSchema] = None
```

and rely on runtime subtype binding.

Tests must prove:

```text
ESSchema.inner declared MetricSchema
              runtime RaySchema
                     -> train/eval
                     -> by_mechanism
                     -> by_seed
                     -> by_episode runtime FisheryMetricSchema
                     -> by_agent runtime FisheryAgentMetricSchema
```

---

# 7. P1 — MetricLogger unit tests

Create focused unit tests, not only integration tests.

Suggested files:

```text
tests/metrics/test_metric_logger_schema_build.py
tests/metrics/test_metric_logger_push_data.py
tests/metrics/test_metric_logger_dynamic.py
tests/metrics/test_metric_logger_peek_reduce.py
```

Required tests:

## 7.1 Schema build

- [ ] static nested `MetricSchema` builds the correct node tree;
- [ ] `dict[ID, MetricSchema]` becomes a dynamic node;
- [ ] leaf reducer metadata creates the correct Metric subclass;
- [ ] `_refs` contains every materialized leaf path.

## 7.2 Push leaf values

- [ ] push scalar into existing leaf;
- [ ] skip `None`;
- [ ] reject unknown field;
- [ ] reject incompatible value/schema.

## 7.3 Dynamic dict materialization

- [ ] first dynamic ID materializes its subtree;
- [ ] second dynamic ID materializes independently;
- [ ] runtime subclass of declared schema is accepted;
- [ ] unrelated schema is rejected;
- [ ] runtime schema cannot silently change for an already-bound ID.

## 7.4 Static nested runtime subtype binding

This is the ES inner-optimizer regression test.

- [ ] `ESSchema.inner` starts declared as `MetricSchema`;
- [ ] first `RaySchema` push replaces/materializes the inner subtree;
- [ ] `train` and `eval` fields exist;
- [ ] second `RaySchema` push reuses the same subtree;
- [ ] second push accumulates metrics instead of resetting to length 1;
- [ ] switching to an incompatible concrete subtype in the same logger raises.

## 7.5 Deep runtime polymorphism

Push:

```text
ESSchema
 -> inner RaySchema
 -> train rollout
 -> by_mechanism["0"]
 -> by_seed["seed"]
 -> by_episode["episode"]
 -> FisheryMetricSchema
 -> by_agent["utilizer:0"]
 -> FisheryAgentMetricSchema
```

Assert deep fields such as:

```text
fish_norm
quota_stress
requested_harvest
quota_penalty
```

are registered and receive values.

## 7.6 `peek()`

- [ ] non-destructive;
- [ ] two consecutive peeks are equal;
- [ ] SERIES history remains intact;
- [ ] calling reporter after peek does not change logger contents.

## 7.7 `reduce()`

- [ ] destructive according to current Metric semantics;
- [ ] resulting typed schema is correct;
- [ ] empty reducer semantics are correct:
      Series `[]`, Mean `None`, Min `None`, Max `None`, Last `None`,
      Sum `0`, Count `0`.

## 7.8 `_refs`

- [ ] `_refs[path]` is the same leaf object as the corresponding `_tree` leaf;
- [ ] dynamic materialization updates `_refs`;
- [ ] reduction does not leave stale aliases.

---

# 8. P1 — Query unit tests

Suggested files:

```text
tests/reporting/test_query.py
tests/reporting/test_query_resolution.py
tests/reporting/test_query_wildcards.py
```

## 8.1 Constructor / validation

- [ ] one-element path tuple is supported;
- [ ] one y path;
- [ ] multiple y paths;
- [ ] `error="std"` with `reduce="none"` raises;
- [ ] malformed empty path raises or has documented behavior.

## 8.2 Static resolution

- [ ] root leaf;
- [ ] nested leaf;
- [ ] multiple y paths;
- [ ] x/y length mismatch produces a clear error.

## 8.3 Mean/std

Synthetic series:

```python
seed_1 = [1.0, 2.0, 3.0]
seed_2 = [3.0, 4.0, 5.0]
```

Expected:

```python
mean = [2.0, 3.0, 4.0]
std = [1.0, 1.0, 1.0]
```

Assert exact output before testing W&B rendering.

## 8.4 Wildcards

- [ ] one wildcard;
- [ ] two nested wildcards;
- [ ] concrete key + wildcard;
- [ ] deterministic match order;
- [ ] no matches;
- [ ] wildcard only applies to dynamic dict nodes.

## 8.5 Wildcard grouping

For:

```text
mechanism -> seed -> value
```

assert:

- [ ] one group per mechanism;
- [ ] reduction across seeds only;
- [ ] mechanism groups do not get averaged together.

## 8.6 Wildcard x/y alignment

Synthetic candidate data:

```text
m0 parameter=0.1 fitness=10
m1 parameter=0.2 fitness=20
m2 parameter=0.3 fitness=30
```

Resolved scatter rows must be:

```text
(0.1, 10)
(0.2, 20)
(0.3, 30)
```

Never:

```text
(0.1, 20)
...
```

---

# 9. P1 — Reporter base tests

Suggested:

```text
tests/reporting/test_reporter_base.py
```

- [ ] Reporter resolves every registered query against the configured schema.
- [ ] Empty query list is a no-op.
- [ ] Missing path includes the full path in the error.
- [ ] Multiple y series preserve labels/identity.
- [ ] Reduced mean/std data has expected shape.
- [ ] Wildcard metadata/bindings survive until backend `_report`.
- [ ] Reporter does not mutate input `MetricSchema`.
- [ ] Same accumulated SERIES may be reported repeatedly as it grows.

---

# 10. P1 — W&B reporter tests

Use mocks/fakes; unit tests should not require a network connection.

Suggested:

```text
tests/reporting/test_wandb_reporter.py
```

Required:

- [ ] simple line query logs under stable key;
- [ ] multiple raw y series produce expected trace count;
- [ ] mean/std creates mean + band;
- [ ] dynamic wildcard trace labels contain mechanism/seed/policy/agent ID;
- [ ] repeated report with growing SERIES updates using the complete current
      history;
- [ ] ES fitness plot trace count and types match dev;
- [ ] ES parameter scatter point count =
      generations × population size;
- [ ] ES scatter x/y candidate correspondence is exact;
- [ ] parallel coordinates dimensions are exact;
- [ ] constant fitness/parameter values do not crash range calculation;
- [ ] no global W&B history table is required as source of truth.

For dev parity, assert figure structure where practical:

```text
trace names
trace count
trace mode
x arrays
y arrays
band upper/lower
hover metadata fields
```

A screenshot comparison can be a secondary integration check, but numeric
trace assertions should be primary.

---

# 11. P1 — environment integration tests

Suggested:

```text
tests/integration/test_fishery_visualization.py
```

Small deterministic fixture:

```text
2 mechanisms
2 seeds
2 agents
short horizon
few training iterations
```

- [ ] environment logger contains expected horizon length;
- [ ] `FisheryMetricSchema` values match direct env values;
- [ ] `by_agent` contains both agents;
- [ ] horizon reporter receives data before episode reduction;
- [ ] reduction afterward still works.

---

# 12. P1 — Ray/inner optimizer integration tests

Use a short deterministic run or a synthetic adaptor payload when full RLlib
would be too expensive.

- [ ] `RaySchema.train` populated;
- [ ] `RaySchema.eval` populated after explicit evaluation;
- [ ] all mechanism IDs present;
- [ ] all seed IDs present;
- [ ] stable episode IDs present;
- [ ] per-policy learner IDs present;
- [ ] performance fields present;
- [ ] train/eval query outputs match manually computed values;
- [ ] mechanism mean/std across seeds matches NumPy calculation.

---

# 13. P1 — ES integration tests

Use a deterministic synthetic environment if possible.

Fixture:

```text
population size = 4
dimension = 2
3 generations
parameter names = fixed_quota, restoration_subsidy
```

- [ ] one ES payload per generation;
- [ ] generation series length grows 1 -> 2 -> 3;
- [ ] candidate fitness series length grows 1 -> 2 -> 3;
- [ ] search mean series grows;
- [ ] global best is updated after current population evaluation;
- [ ] logged mean/sigma correspond to the pre-update distribution that sampled
      the population;
- [ ] `inner` contains the concrete inner schema;
- [ ] second generation does not reconstruct/reset `inner`;
- [ ] fitness plot has `3 * 4 = 12` candidate points;
- [ ] each parameter scatter has 12 points;
- [ ] parallel coordinates has 12 lines;
- [ ] global-best trajectory is monotonic non-decreasing for maximization.

Fixed-mode regression:

- [ ] ES dimension 0 does not crash reporting;
- [ ] plotting payload uses the full default mechanism vector;
- [ ] parameter names match the default mechanism vector.

---

# 14. P1 — CSV reporter implementation

CSV export is part of the feature definition and must be completed.

Suggested file:

```text
core/reporting/csv.py
```

The CSV reporter must consume the same `Query` contract.

## 14.1 Required behavior

- [ ] implement Reporter subclass;
- [ ] configure output directory/path through `ReporterConfig`;
- [ ] create directories safely;
- [ ] stable file naming from query title/key;
- [ ] append/update semantics documented;
- [ ] no W&B dependency;
- [ ] scalar series export;
- [ ] multiple raw series export;
- [ ] mean/std export;
- [ ] dynamic wildcard labels exported;
- [ ] train/eval/mechanism/seed dimensions preserved as columns;
- [ ] ES candidate/parameter metadata preserved;
- [ ] flush/close lifecycle;
- [ ] safe behavior if process exits after partial run.

Recommended long-form representation:

```text
query
x
series
value
error_std
phase
mechanism
seed
episode
policy
agent
parameter
generation
```

Only populate dimensions relevant to the query.

For simple queries, a wide CSV may also be convenient:

```text
iter, reward_mean, reward_min, reward_max
```

Do not throw away dynamic identity just to force everything into wide format.

## 14.2 CSV tests

- [ ] temp directory fixture;
- [ ] single series;
- [ ] multi-series;
- [ ] mean/std;
- [ ] wildcard series labels;
- [ ] repeated report appends/updates correctly;
- [ ] no duplicate header;
- [ ] NaN/None policy documented;
- [ ] output can be loaded by pandas and reconstruct expected series.

---

# 15. P1 — TensorBoard reporter implementation

TensorBoard support must be completed.

Suggested file:

```text
core/reporting/tensorboard.py
```

Use the same resolved Query result, not raw optimizer dictionaries.

## 15.1 Required behavior

- [ ] Reporter subclass;
- [ ] `SummaryWriter` lifecycle;
- [ ] stable tag naming;
- [ ] single scalar/series using `add_scalar`;
- [ ] multiple related series using `add_scalars` where appropriate;
- [ ] dynamic mechanism/policy/agent labels represented in tags;
- [ ] mean/std behavior documented;
- [ ] train/eval grouping represented consistently;
- [ ] flush and close;
- [ ] no W&B imports.

Complex figures:

- line/scatter/shaded plots can be logged either as:
  - scalar families that TensorBoard renders natively; or
  - a rendered figure/image where native scalar APIs are insufficient.
- parallel coordinates likely requires image/figure rendering because
  TensorBoard does not have a native parallel-coordinate primitive.

- [ ] choose and document the complex-figure representation;
- [ ] avoid adding a heavy conversion dependency unless justified.

## 15.2 TensorBoard tests

- [ ] temporary logdir;
- [ ] event file created;
- [ ] expected scalar tags exist;
- [ ] repeated iterations produce multiple steps;
- [ ] dynamic tags are stable;
- [ ] writer flush/close works;
- [ ] complex figure path has a test if supported.

---

# 16. P2 — exact dev environment tables / post-hoc analysis

The old dev environment module also emitted:

- raw long-form timestep table;
- raw wide timestep table;
- derived wide table;
- correlation matrix;
- distribution summary;
- training metrics table.

Decide whether these belong in the generic visualization API or in CSV/post-hoc
analysis.

- [ ] Do not silently drop them if they are still required.
- [ ] Prefer CSV/table export over forcing them into line `Query`.
- [ ] Keep generic reporting backend-agnostic.
- [ ] Water-specific observed-vs-simulated plots should remain domain-specific
      unless generalized deliberately.

---

# 17. P2 — clean up legacy dev plotting only after parity

Legacy modules currently hold W&B-specific history/state such as accumulated
tables.

Once the feature branch passes parity tests:

- [ ] remove obsolete direct W&B calls from optimizers;
- [ ] remove duplicate ES history caches;
- [ ] remove dead `plot_population`, `plot_parameter_names`, `plot_mean`, and
      `plot_best_candidate` preparation if no longer used;
- [ ] remove old plotting entry points only after screenshots/data are compared;
- [ ] leave migration notes or deprecation stubs if other examples import them.

Do not delete dev reference code before the parity test is complete.

---

# 18. P2 — optional-schema presence semantics

Current schema design has optional nested branches such as:

```python
class RaySchema(MetricSchema):
    train: Optional[TrainSchema] = None
    eval: Optional[EvalSchema] = None
```

Verify that an absent optional branch remains `None` instead of being
constructed as an empty schema during `peek()`/`reduce()`.

Do not infer absence from leaf reduced values because legitimate empty reducer
values include:

```text
Sum   -> 0
Count -> 0
Series -> []
```

If needed, add explicit node presence tracking.

- [ ] test train-only payload;
- [ ] test eval-only payload;
- [ ] test train + eval payload;
- [ ] test destructive reduce/reset behavior.

---

# 19. P2 — Ray serialization boundary regression

The optimizer-local `MetricLogger` should not have to cross into the `World`
actor merely to register an optimizer ID.

- [ ] base optimizer config sends only optimizer ID/registry data to World;
- [ ] Ray optimizer config follows the same ownership model;
- [ ] regression test proves an optimizer containing an unpicklable local
      object can still register if only its ID crosses the actor boundary.

This keeps logger/reporter runtime state local to the optimizer that owns it.

---

# 20. Recommended implementation order

1. [ ] Verify/add `ESSchema.generation`.
2. [ ] Add `ESSchema.generation_best`.
3. [ ] Lock the current non-wildcard environment, Ray, and ES query smoke tests.
4. [ ] Implement wildcard path expansion.
5. [ ] Implement wildcard x/y binding.
6. [ ] Implement mechanism grouping + seed mean/std.
7. [ ] Implement train-vs-eval grouped shaded rendering.
8. [ ] Reproduce ES fitness-over-generations trace modes.
9. [ ] Reproduce cumulative ES parameter scatter with generation color.
10. [ ] Implement parallel-coordinate query/renderer.
11. [ ] Run deterministic dev-vs-feature parity validation.
12. [ ] Complete unit/integration tests.
13. [ ] Complete CSV reporter.
14. [ ] Complete TensorBoard reporter.
15. [ ] Remove legacy W&B-specific plotting/cache code.
16. [ ] Finalize docs/tutorial examples.

---

# 21. Final acceptance run

Use one small deterministic fishery configuration and preserve its config/seed
in the test documentation.

Recommended:

```text
4 ES candidates
2 optimized parameters:
    fixed_quota
    restoration_subsidy
2-3 environment seeds
>= 3 outer generations
short horizon for CI
explicit evaluation
```

Capture both dev and feature outputs.

Compare:

## Environment

- [ ] same horizon x values;
- [ ] same fish biomass;
- [ ] same harvest;
- [ ] same quota stress;
- [ ] same agent reward/action/harvest/penalty values.

## Inner optimizer

- [ ] same rollout aggregates;
- [ ] same performance values;
- [ ] same per-policy learner values;
- [ ] same mechanism × seed values;
- [ ] same train/eval grouping;
- [ ] same mean/std calculation.

## ES

- [ ] same candidate-to-fitness correspondence;
- [ ] same generation mean/best;
- [ ] same global best;
- [ ] same search mean/sigma semantics;
- [ ] same parameter scatter point set;
- [ ] same parallel-coordinate row set.

Plot styling can differ only where intentionally documented. The goal for this
branch request is to reproduce the dev plots exactly unless a difference is
explicitly approved.

---

# 22. Feature status summary

Already supported / expected to work now:

- [x] Environment horizon plots.
- [x] Raw RLlib rollout plots.
- [x] Performance plots.
- [x] Per-policy learner data/plots with concrete IDs.
- [x] Per-agent environment data/plots with concrete IDs.
- [x] ES SERIES accumulation through `push_data -> peek -> report`.
- [x] Nested ES `inner: MetricSchema` runtime specialization to `RaySchema`.
- [x] Deep runtime specialization to fishery episode and agent schemas.

Still required:

- [ ] Dynamic mechanism IDs.
- [ ] Dynamic seed IDs.
- [ ] Dynamic episode IDs.
- [ ] Dynamic policy IDs.
- [ ] Dynamic agent IDs.
- [ ] Dynamic ES parameter keys.
- [ ] Mechanism mean ±std across seeds through Query API.
- [ ] Train-vs-eval shaded mechanism plots through Query API.
- [ ] ES exact candidate/mean/best mixed trace styling.
- [ ] ES all-candidate parameter scatter in one plot.
- [ ] ES generation color metadata.
- [ ] ES parallel coordinates.
- [ ] Generation-best ES parameter schema/queries.
- [ ] Full unit tests.
- [ ] Full integration parity tests.
- [ ] CSV reporter.
- [ ] TensorBoard reporter.
- [ ] Legacy visualization cleanup after parity.
- [ ] The fishery benchmark builds and runs end-to-end with the new mechanism abstraction.
- [ ] A quota-only run completes training and evaluation.
- [ ] A quota + subsidy run completes.
- [ ] A quota + subsidy + social-observation run completes.
- [ ] Chained composition works for action, observation, and reward channels.
- [ ] Parallel composition has one coherent API and tests.
- [ ] All concrete `Mechanism` implementations satisfy the abstract base class.
- [ ] Mechanism optimizer vectors encode/decode correctly.
- [ ] Action and observation spaces agree with transformed values.
- [ ] Unit tests cover every concerned mechanism/env/composition module.
- [ ] Integration tests cover the benchmark + mechanism lifecycle.
- [ ] Reproducibility against `dev` is checked where practical.
- [ ] Quota behavior is numerically compared against the dev fishery benchmark if time permits.
- [ ] The tutorial notebooks run after the P0 integration fixes are merged.

---

# 1. P0 — make the current implementation internally consistent

There are several API mismatches in the supplied branch that should be fixed
before trying to compare results.

## 1.1 Reconcile `BilevelConfig.mechanism(...)`

The supplied config method currently has the shape:

```python
def mechanism(
    self,
    *,
    space: MechanismSpace,
    default: Mechanism = None,
    **kwargs,
) -> Self:
    ...
```

but the proposed benchmark config uses:

```python
.mechanism(
    mechanism=ChainedMechanism(...)
)
```

Choose and document one public API.

Recommended target:

```python
.mechanism(
    mechanism=mechanism,
    space=optional_space,
)
```

or, if `MechanismSpace` remains the owner of default construction:

```python
.mechanism(
    space=space,
    default=mechanism,
)
```

Acceptance:

- [ ] exactly one supported builder signature;
- [ ] examples and tutorials use that signature;
- [ ] `BilevelConfig.build_optimizer()` injects the same mechanism/space into inner and outer components;
- [ ] fixed mechanisms work without an unnecessary optimizer space;
- [ ] optimized mechanisms expose an optimizer dimension unambiguously.

---

## 1.2 Make every concrete mechanism instantiable

`Mechanism` currently declares these abstract methods/properties:

```text
dimension
encode
decode
clip
param_names
to_vector
```

plus identity implementations for:

```text
action
observation
reward
```

Audit every concrete class.

### `QuotaMechanism`

Currently shown:

```text
to_vector       YES
param_names     YES
action          YES
observation     YES

dimension       MISSING in supplied code
encode          MISSING
decode          MISSING
clip            MISSING
```

- [ ] implement missing abstract API or move common parameterized behavior into a reusable base.

### `SubsidyMechanism`

Currently shown:

```text
to_vector       YES
param_names     YES
reward          YES

dimension       MISSING
encode          MISSING
decode          MISSING
clip            MISSING
```

- [ ] implement missing abstract API.

### `SocialInfluenceMechanism`

Currently shown only with `observation(...)`.

- [ ] implement fixed/optimized parameter API;
- [ ] decide whether `influence_weight` is optimized or fixed;
- [ ] if fixed, `dimension == 0`;
- [ ] if optimized, define normalized encode/decode bounds.

### `ThresholdPenaltyMechanism`

Currently has `dimension`, `encode`, `decode`, `param_names`, `reward`.

- [ ] verify/implement `clip`;
- [ ] verify/implement `to_vector`;
- [ ] decide whether threshold/penalty are fixed or optimizer-controlled.

### `ChainedMechanism`

- [ ] verify/implement `clip`;
- [ ] define `to_vector` for the semantic vector exposed to agents;
- [ ] test concatenation/slicing of child optimizer vectors.

### `ParallelMechanism`

- [ ] same abstract-method audit;
- [ ] same vector semantics audit.

No concrete mechanism should remain abstract accidentally.

---

## 1.3 Remove stale constructor arguments from examples

The proposed example still uses old `MechanismSpace`-style arguments:

```python
QuotaMechanism(
    optimize_params=["fixed_quota"],
    default_fixed_quota=0.56224,
    default_max_demand_frac=1.0,
)
```

but the shown new dataclass is:

```python
QuotaMechanism(
    fixed_quota: float,
    bindings: ...,
    action_component: int = 0,
    ...
)
```

Likewise `SubsidyMechanism` currently declares:

```python
SubsidyMechanism(
    subsidy: float,
    cost: float,
    action_component: int = 1,
)
```

not old `optimize_params/default_*` arguments.

- [ ] update all examples to the new object model;
- [ ] keep optimization selection in one place only;
- [ ] do not duplicate defaults in both mechanism objects and spaces.

---

# 2. P0 — fix environment mechanism dispatch

The regulated env should have one explicit lifecycle:

```text
policy output
    ↓
normalize action
    ↓
benchmark action hook (optional)
    ↓
mechanism action transform
    ↓
benchmark intrinsic/base reward
    ↓
benchmark transition
    ↓
mechanism reward transform
    ↓
benchmark observation
    ↓
mechanism observation transform
    ↓
publish context
```

Audit `MultiAgentRegulatedEnv` against this.

## 2.1 Wrong mechanism method calls

The supplied `reward(...)` method returns:

```python
return self.mechanism.action(
    reward_dict,
    env=self,
)
```

- [ ] call `self.mechanism.reward(...)`.

The supplied `observation(...)` method returns:

```python
return self.mechanism.action(
    obs_with_theta,
    env=self,
)
```

- [ ] call `self.mechanism.observation(...)`.

---

## 2.2 Observation concatenation is currently dict-unsafe

The supplied code contains:

```python
theta = self.mechanism.to_vector()
obs_with_theta = np.concatenate(
    [observation_dict, theta],
    axis=0,
)
```

`observation_dict` is a multi-agent dictionary, not an array.

Target:

```python
obs_with_theta = {
    agent_id: np.concatenate(
        [
            np.asarray(observation, dtype=np.float32).reshape(-1),
            theta,
        ]
    ).astype(np.float32, copy=False)
    for agent_id, observation in observation_dict.items()
}
```

Then pass the dict through `mechanism.observation(...)`.

- [ ] add a regression test for this exact failure mode.

---

## 2.3 Avoid bypassing the public reward/observation pipeline

`step()` currently directly calls mechanism reward and public observation.

Decide whether public `reward(...)` is the pipeline method or remove it.
There should not be two overlapping reward paths.

Recommended:

```python
intrinsic_rewards = benchmark_reward(delivered_actions)
...
rewards = self.reward(
    intrinsic_rewards,
    action_after=delivered_actions,
)
obs = self.observation({})
```

- [ ] one reward path only;
- [ ] one observation path only;
- [ ] one action path only.

---

## 2.4 Fix "no published mechanism" fallback path

The supplied branch calls:

```python
self.observation(agent_id, self.S_t)
```

even though `observation(...)` accepts one `observation_dict`.

- [ ] make fallback reset/step behavior use the same observation pipeline;
- [ ] add a test where the world has not published a non-default mechanism.

---

# 3. P0 — fix the fishery benchmark under the new action semantics

The benchmark currently has a 2-component action:

```text
component 0 = harvest fraction
component 1 = restoration effort
```

but the transition currently treats the full vector as harvest:

```python
delivered_harvest = {
    agent_id: action * full_required_harvest
    for agent_id, action in A_t.items()
}
```

Target:

```python
harvest_fraction = float(action[0])
restoration_effort = float(action[1])
```

Then:

```python
requested_harvest_i = harvest_fraction_i * full_required_harvest
```

- [ ] extract action components deliberately;
- [ ] document the semantic component map;
- [ ] avoid implicit whole-vector arithmetic.

## 3.1 Restoration dynamics are currently disconnected

The supplied transition uses:

```python
growth = biological_growth + noise + kwargs["restoration"]
```

but the shown env step does not pass `restoration`.

Recommended:

```python
restoration = restoration_effectiveness * sum(
    action[1] for action in A_t.values()
)
```

and either pass it explicitly to the transition hook or derive it there.

The subsidy mechanism should modify reward; ecological restoration belongs in
the benchmark transition.

- [ ] connect restoration action to fish dynamics;
- [ ] keep ecology and incentive shaping separate.

## 3.2 Fix `K` reference

The transition contains:

```python
fish_next = float(np.clip(fish_next, 0.0, K))
```

- [ ] use `self.K` or deliberately remove the upper clipping;
- [ ] add boundary tests.

## 3.3 Define the base reward

The shown `FisheryRegulatedEnv` does not include a `@reward` hook.

- [ ] add or verify the benchmark base reward;
- [ ] test reward before any mechanism;
- [ ] test reward after subsidy/penalty.

---

# 4. P0 — fix `SubsidyMechanism`

Intended reward:

\[
r_{i,t}^{*}
=
r_{i,t}
-
c e_{i,t}^{2}
+
\sigma_\theta e_{i,t}.
\]

The supplied implementation contains:

```python
actions[agent_id[self.action_component]]
```

Fix to:

```python
actions[agent_id][self.action_component]
```

Recommended implementation:

```python
effort = float(
    actions[agent_id][self.action_component]
)

reward
+ self.subsidy * effort
- self.cost * effort**2
```

Acceptance:

- [ ] zero effort -> no subsidy/cost;
- [ ] positive effort -> exact analytical reward;
- [ ] component selection tested;
- [ ] reward type remains `float`;
- [ ] public bounds use `ValueError`, not only `assert`.

---

# 5. P0 — finish `SocialInfluenceMechanism`

Full social influence from the project slides:

\[
c_t^i
=
\sum_{j\neq i}
D_{KL}
\left[
\pi_j(a_t^j \mid a_t^i, s_t^j)
\|
\pi_j(a_t^j \mid s_t^j)
\right]
\]

and:

\[
r_t^i = r_{i,t} + \beta c_t^i.
\]

The supplied implementation currently implements only observation shaping:

\[
o_{i,t}^{*}
=
[
o_{i,t},
a_{1,t-1},
\dots,
a_{j,t-1},
\dots
].
\]

- [ ] document that this is observation augmentation, not the full Jacques et al. KL bonus;
- [ ] `influence_weight` is currently unused in the shown implementation;
- [ ] either implement the KL reward term or scope/rename the class;
- [ ] add `bindings` to the dataclass if constructor-injected bindings are intended;
- [ ] test peer-action ordering;
- [ ] test self-action exclusion;
- [ ] test observation dimensionality.

---

# 6. P0 — quota mechanism numerical tests

The quota computes:

\[
L = \sigma((0-q)/w_q),
\quad
U = \sigma((1-q)/w_q),
\quad
C_t = \sigma((b_t-q)/w_q)
\]

and:

\[
\alpha_t = \frac{C_t-L}{U-L}.
\]

Requested fraction \(u_{i,t}\) becomes:

\[
u_{i,t}^{*}
=
u_{i,t}
-
\operatorname{smooth}_{+}
\left(
u_{i,t}-\alpha_t;
w_u
\right).
\]

Tests:

- [ ] resource close to 0 -> allowed fraction near lower end;
- [ ] resource close to 1 -> allowed fraction near 1;
- [ ] resource near `fixed_quota` -> expected sigmoid transition;
- [ ] request below allowed fraction remains approximately unchanged;
- [ ] request above allowed fraction is smoothly capped;
- [ ] non-target action components are unchanged;
- [ ] input arrays are not mutated in place;
- [ ] per-agent mapping preserved;
- [ ] `allowed_frac` is available to the quota observation transform.

---

# 7. P0/P1 — optional quota reproducibility against `dev`

Preferred but not blocking if time is limited.

Create a deterministic quota-only fixture using the same:

```text
r
K
p
fish_init/B0
sigma
policy seed
environment seed
action trajectory
fixed quota
transition widths
```

Compare old dev and new quota transforms before involving RLlib.

At each step compare:

```text
resource_level
allowed_frac/effective quota
requested harvest fraction
delivered harvest fraction
fish stock
fish_norm
H_attempted
H_realized
reward if reward semantics are unchanged
```

Suggested tolerance:

```python
np.testing.assert_allclose(
    new,
    dev,
    rtol=1e-6,
    atol=1e-7,
)
```

If a difference is intentional, document whether it comes from normalization,
smoothing, reward semantics, or dynamics.

---

# 8. P1 — hook discovery tests

`core/envs/hooks.py` attaches markers and
`MultiAgentRegulatedEnv.__init_subclass__` discovers them.

Tests:

- [ ] `@reset` registers reset hook;
- [ ] `@action` registers action hook;
- [ ] `@reward` registers reward hook;
- [ ] `@observation` registers observation hook;
- [ ] `@transition` registers transition hook;
- [ ] inherited hooks behave intentionally;
- [ ] multiple hooks of one type either raise or have documented deterministic behavior.

Recommendation: fail fast rather than silently letting the last same-type hook
win.

---

# 9. P1 — mechanism binding tests

A binding is:

```python
binding: env -> runtime context value
```

Example:

```python
bindings={
    "resource_level": lambda env: (
        env.S_t["fish"] / max(env.K, EPS)
    ),
}
```

Tests:

- [ ] `resolve(env)` returns configured keys;
- [ ] missing required binding raises at construction;
- [ ] quota receives normalized resource level;
- [ ] social observation receives `previous_actions` and `agent_ids`;
- [ ] child bindings in compositions resolve against the correct env;
- [ ] bindings remain serializable in Ray/cloudpickle integration.

---

# 10. P1 — chained composition tests

For:

```python
ChainedMechanism(
    children=(m1, m2, m3)
)
```

contract:

\[
x^\*
=
M_3(M_2(M_1(x))).
\]

Tests:

- [ ] action order exactly follows child tuple order;
- [ ] reward order exactly follows child tuple order;
- [ ] observation order exactly follows child tuple order;
- [ ] each child receives previous child's transformed output;
- [ ] each child resolves its own env bindings;
- [ ] dimension is sum of child dimensions;
- [ ] encode is concatenation;
- [ ] decode slices correctly;
- [ ] parameter names preserve child identity/order;
- [ ] zero-dimension children do not break slicing.

Interaction test:

```text
child 1: multiply by 2
child 2: add 1

expected chain:
2x + 1
```

---

# 11. P1 — parallel composition API repair and tests

The supplied `ParallelMechanism` calls:

```python
child.apply_action(...)
child.apply_reward(...)
child.apply_observation(...)
```

while `Mechanism` exposes:

```python
action(...)
reward(...)
observation(...)
```

- [ ] reconcile this before use.

Recommended target:

```python
context = child.resolve(env)
child.action(copy, **context)
```

Parallel contract:

\[
M_{\parallel}^{A}(x)
=
\Gamma_A
\left(
x,
M_1^A(x),
\dots,
M_k^A(x)
\right).
\]

Each child receives the same original input.

Tests:

- [ ] every child sees the same original input;
- [ ] no child sees another child's output;
- [ ] merge receives original + tuple of outputs;
- [ ] merge ordering is documented;
- [ ] deep copies prevent cross-child mutation;
- [ ] action/reward/observation merge functions tested separately;
- [ ] dimensions/encode/decode tested.

---

# 12. P1 — action and observation spaces

The current example hard-codes:

```python
action_space = Box(shape=(2,))
```

and an observation shape based on an older mechanism-space abstraction.

Expected fishery features include:

```text
base observation:
    fish_norm
    total_usage_norm

quota augmentation:
    allowed_frac/effective quota

social augmentation:
    previous peer actions

optional mechanism parameter vector:
    theta
```

For 10 agents and 2-D actions, social influence adds:

```text
(10 - 1) * 2 = 18
```

features per agent.

- [ ] compute/validate final observation dimension;
- [ ] decide whether mechanisms expose `observation_dimension_delta`;
- [ ] decide whether `to_vector()` is always appended;
- [ ] remove dependencies on obsolete `FisheryMechanismSpace().full_dimension` where inappropriate;
- [ ] assert actual observation shape matches declared space;
- [ ] assert normalized action shape matches declared action space.

---

# 13. P1 — context publishing tests

Each step should preserve:

```text
env_id
seed
policy_seed
mode/status
mechanism_id
observation
reward
action
info
```

- [ ] values correspond to regulated action/reward/observation actually used;
- [ ] seeds remain immutable for an env instance;
- [ ] mechanism ID matches the published mechanism;
- [ ] publication does not modify behavior.

---

# 14. P1 — recommended test layout

```text
tests/envs/test_hooks.py
tests/envs/test_marl_regulated.py

tests/mechanism/test_base.py
tests/mechanism/algorithms/test_quota.py
tests/mechanism/algorithms/test_subsidy.py
tests/mechanism/algorithms/test_penalty.py
tests/mechanism/algorithms/test_social_influence.py

tests/mechanism/composition/test_chained_mechanism.py
tests/mechanism/composition/test_parallel_mechanism.py
tests/mechanism/test_space.py

tests/examples/test_fishery_regulated_env.py
tests/integration/test_fishery_mechanisms.py
```

Coverage target:

- [ ] meaningful branch coverage for all concerned files;
- [ ] aim for >=90% line coverage on pure mechanism/composition modules;
- [ ] every mechanism dispatch path covered even if distributed integration coverage is lower.

Suggested command:

```bash
pytest \
  tests/envs \
  tests/mechanism \
  tests/examples/test_fishery_regulated_env.py \
  tests/integration/test_fishery_mechanisms.py \
  --cov=core.envs \
  --cov=core.mechanism \
  --cov-report=term-missing
```

---

# 15. P1 — staged end-to-end smoke runs

Do not begin with 1000 outer iterations.

## Stage A — environment only

```text
2 agents
deterministic seed
fixed mechanism
horizon 5
```

## Stage B — inner optimizer only

```text
2 agents
1 mechanism
1 seed
horizon 10
1-2 train iterations
```

## Stage C — quota-only bilevel

```text
2 candidates
1 seed
2 outer generations
```

## Stage D — quota + subsidy

Verify restoration dynamics and reward incentive.

## Stage E — quota + subsidy + social observation

Verify observation-space growth.

## Stage F — evaluation

```text
2-3 explicit eval seeds
deterministic evaluation policy
```

Only then restore the larger benchmark configuration.

---

# 16. P2 — clarify stateful mechanism context

`QuotaMechanism` stores transient values in `_context`.

A frozen dataclass can still mutate a contained dict, but this makes the
mechanism stateful.

Decide:

- [ ] Is one mechanism object shared across multiple env instances?
- [ ] Could vectorized envs overwrite one another's `_context`?
- [ ] Should mechanism state reset per episode?
- [ ] Should step context live on the env instead?

Recommended default:

```text
Mechanism definition = immutable
Per-step mechanism context = environment-local
```

If mechanism-local context remains, tests must prove instances are not shared
across concurrently stepping envs.

---

# 17. P2 — public validation should not rely on `assert`

Examples use:

```python
assert 0.0 <= self.fixed_quota <= 1.0
```

- [ ] use explicit `ValueError` for public configuration;
- [ ] keep assertions for internal invariants only.

---

# 18. P2 — documentation acceptance

Ship:

```text
tutorials/mechanism_benchmarks_tutorial.py
tutorials/custom_benchmark_mechanism_tutorial.py
MECHANISM_ABSTRACTION_TODO.md
```

Tutorial examples must reflect the final merged public API.

---

# 19. Recommended implementation order

1. [ ] Reconcile `BilevelConfig.mechanism` public API.
2. [ ] Make all mechanism classes concretely instantiable.
3. [ ] Fix reward/observation dispatch in `MultiAgentRegulatedEnv`.
4. [ ] Fix per-agent observation concatenation.
5. [ ] Fix fishery 2-component action decomposition.
6. [ ] Connect restoration action to transition dynamics.
7. [ ] Fix subsidy indexing bug.
8. [ ] Scope/finish social influence behavior.
9. [ ] Repair `ParallelMechanism` method API.
10. [ ] Add unit tests for hooks and transforms.
11. [ ] Add composition tests.
12. [ ] Add deterministic fishery tests.
13. [ ] Run quota-only smoke benchmark.
14. [ ] Run quota + subsidy smoke benchmark.
15. [ ] Run social observation smoke benchmark.
16. [ ] Add evaluation smoke test.
17. [ ] Optional/preferred: numerical quota parity against `dev`.
18. [ ] Update tutorials to final API.
19. [ ] Run coverage and close remaining untested branches.
