# Trajectory data

The forward interface of `Flows.jl` mutates a state in place and returns nothing else. Most algorithms need *more*: a sampled history of a derived quantity, the full state at intermediate times, or the exact stage values of every accepted step. The package provides three orthogonal data structures that cover these needs:

| Object                                | Records                                              | Typical use                              |
|---------------------------------------|------------------------------------------------------|------------------------------------------|
| [`Monitor`](@ref)                     | A user-defined observable, one sample per step       | Time series for plotting / convergence / running averages |
| [`RAMStorage`](@ref)                  | The state (or a derived field) at each accepted step | Drives a *continuous* linearised solve   |
| [`RAMStageCache`](@ref)               | Every internal stage value of every step             | Drives a *discretely consistent* linearised solve |

A specialised [`StoreNFromLast`](@ref) monitor records a single observation at a fixed offset before the end of an integration. All four are passed to a [`Flows.Flow`](@ref Flows.Flow) as an extra positional argument, after the state and the time span.

This page covers each in detail, the relationship between them, and the design choices that put them on this page together.

## Monitors

A [`Monitor`](@ref) records a single user-defined observable along a trajectory and stores the samples in an in-memory storage. The observable is any callable `f(t, x)` whose return value is what is actually stored — strings, named tuples, plain numbers, arrays, whatever the user finds useful.

### Construction

```julia
mon = Monitor(x, (t, x)->norm(x))
```

The first argument is a *template* of the system state (used only to infer the type of `f(0.0, x)` so the backing storage can be preallocated). The second is the observable. There are six keyword arguments:

| Keyword            | Default            | Purpose                                                                  |
|--------------------|--------------------|--------------------------------------------------------------------------|
| `oneevery`         | `1`                | Decimation. Store one sample every `oneevery` accepted steps.            |
| `savebetween`      | `(-Inf, Inf)`      | Only store samples whose time falls in this closed interval.             |
| `skipfirst`        | `false`            | Drop the very first observation (often the user-supplied initial state). |
| `sizehint`         | `0`                | Preallocate capacity in the backing storage.                             |
| `io`               | `devnull`          | Output stream for the [Logger](@ref Trajectory-printing-with-Logger).    |
| `logevery`         | `1`                | Decimation of the printed output.                                        |

### Driving from a flow

```julia
x = rand(3)
F(x, (0.0, 10.0), mon)

times(mon)     # the times at which the observable was sampled
samples(mon)   # the corresponding sample values
```

Inside the propagation loop, the schedule is:

- One observation is pushed at the **initial time** with `force=true`, before the first step.
- After each accepted step the monitor is pushed; the `oneevery` decimation drops all-but-one-in-`oneevery` of these, except that the **final step is always forced** so the recorded series ends at the user-requested endpoint.
- `savebetween` and `skipfirst` are then applied on top.

### Using monitors as callbacks

The observable does not need to be a pure function. Because it is called every accepted step, it can mutate the state — a useful pattern for re-normalisation or projection onto a constraint:

```julia
mon = Monitor(zeros(5), (t, x)->(x ./= norm(x); norm(x)); oneevery=10)
```

The mutation is performed *before* the observable's return value is stored, so the recorded value reflects the post-mutation state.


### `StoreNFromLast`: a specialised monitor

A common pattern in optimisation is to need the state **a fixed number of steps before the end** of an integration: for instance, to seed an adjoint integration from a slightly earlier point so the terminal-condition gradient is well-defined.

[`StoreNFromLast{N}`](@ref) records exactly one observation, $N$ steps before the final step. It is dispatched by the propagation loop with a separate code path so the decimation logic is exact (no off-by-one errors):

```julia
m = StoreNFromLast{2}(zeros(3))      # record state 2 steps before the end
F(x, (0.0, 1.0), m)
m.t                                  # the recorded time
m.x                                  # the recorded state
```


## Trajectory printing with `Logger`

`Monitor` carries an internal [`Flows.Logger`](@ref Flows.Logger) that pretty-prints the running observable to an `io` stream. The default is `devnull` (silent); pass `io=stdout` to monitor a long run on the terminal.

The logger introspects the observable's return type once at construction (via the `f(0.0, x)` it computes for storage typing) and builds a `Printf.Format` row matching the shape:

- a single float → one float column,
- a tuple of mixed types → one column per element (floats, ints, complex numbers; arrays and other "unprintable" entries render as `?`).

The `logevery` keyword decimates the *printed* output independently of the `oneevery` decimation of the *stored* output, which is useful when you want to see one line per minute of wallclock on a long run but still keep every sample on disk.

## Storages

A [`RAMStorage`](@ref) records the system state at every accepted step and exposes it both as a vector of snapshots and as a callable interpolator. It is the data source for the **continuous** linearised integration paths.

### Construction

```julia
store = RAMStorage(x)                       # infer X from `x`
store = RAMStorage(typeof(x))               # equivalent
store = RAMStorage(x; degree=5, period=2π)  # tuned for a periodic problem
```

Keyword arguments:

| Keyword       | Default     | Purpose                                                                        |
|---------------|-------------|--------------------------------------------------------------------------------|
| `ttype`       | `Float64`   | Element type of the time vector.                                               |
| `degree`      | `3`         | Odd Lagrange polynomial degree (`≥ 3`) used by the interpolator.               |
| `period`      | `0.0`       | Positive period if the data is periodic; zero (default) means non-periodic.    |
| `storelast`   | `true`      | Whether the propagation loop should push the final sample of an integration.   |

### Driving from a flow

```julia
x = rand(3)
F(x, (0.0, 10.0), store)

times(store)       # the sampled times
samples(store)     # the snapshots
timespan(store)    # (first, last) — for periodic storages, (first, period)
```

A storage is also the data source for the *continuous* tangent / adjoint integration paths; see [Linearised dynamics](linearised.md).

### Interpolation

A populated storage is **callable**:

```julia
out = similar(samples(store)[1])
store(out, t)               # function value at t
store(out, t, Val(1))       # first derivative at t (via central differences)
```

The interpolation is Lagrange of degree `DEG` set at construction. The query time `t` must lie inside `timespan(store)` — extrapolation throws an `ArgumentError`. The buffer `out` must match the snapshot type; the interpolator overwrites it in place and returns it.

The implementation in `src/storage.jl` is allocation-free in the hot path:

1. `searchsortedlast(ts, t)` locates the stencil center;
2. boundary indices are shifted so the stencil stays inside the data;
3. the Lagrange weights are computed at code-generation time via a `@generated` helper, so the inner loop is a fixed-length weighted sum.

### Periodic storages

When the recorded data is one period of a periodic signal, set `period` to the period at construction and **do not** push the duplicated endpoint. The interpolator then wraps around the endpoints automatically:

```julia
store = RAMStorage(zeros(3); period=2π, storelast=false)
for t in range(0, stop=2π, length=N+1)[1:N]
    push!(store, t, primal_at(t))
end
store(out, 0.0)         # interpolates correctly near the wrap
store(out, 2π - 1e-9)   # also fine
```

The combination `period > 0`, `storelast = false` is the canonical setup for periodic trajectories.


## Stage caches

A [`RAMStageCache`](@ref) records the internal stage values of every accepted step. It is the data source for the **discretely consistent** linearised integration paths.

### Why stages and not states?

The state-at-step-end is what a `RAMStorage` records. But the *discrete* tangent equation depends on the value of the operator at exactly the same intermediate stage points as the primal — at $t_n + c_k\Delta t_n$ for every stage $k$, every step $n$. Recording just the step-end state is not enough; the linearised flow needs each `stages[k]` and the exact `(t_n, Δt_n)` that the primal used.

A `RAMStageCache{NS}` is configured at construction with the stage count `NS` of the primal scheme and records `(t_n, Δt_n, (Y₁,…,Y_{NS}))` triples in three parallel vectors:

```julia
cache = RAMStageCache(nstages(RK4), zeros(3))     # NS = 4 for RK4
F(x, (0.0, 1.0), cache)

cache.ts                                          # Vector{Float64}
cache.Δts                                         # Vector{Float64}
cache.xs                                          # Vector{NTuple{NS, X}}
```

The propagation loop drives the scheme's caching `step!` overload, which preserves the same time grid as a primal integration without a cache. The only cost is the per-step allocation of the `NTuple` of stages — there is no extra arithmetic and no extra dispatch overhead.

### Stage count matching

The propagation loop checks `nstages(method) == nstages(cache)` at every entry point that takes a cache:

```julia
cache = RAMStageCache(2, zeros(3))                # NS = 2
F     = flow(f, RK4(zeros(3)), TimeStepConstant(0.01))  # NS = 4
F(x, (0.0, 1.0), cache)                            # ArgumentError
```

This catches scheme mismatches at integration time rather than failing silently inside `step!`.

### Driving a linearised flow

Once a cache is populated, pair it with [`TimeStepFromCache`](@ref Flows.TimeStepFromCache):

```julia
L = flow(f_tangent, RK4(zeros(3), DiscreteMode(false)), TimeStepFromCache())
L(y, cache)               # marches y through cache.ts forwards
```

or, for the adjoint mode,

```julia
A = flow(f_adjoint, RK4(zeros(3), DiscreteMode(true)), TimeStepFromCache())
A(q, cache)               # marches q through cache.ts in reverse
```

See [Linearised dynamics](linearised.md) for the full discussion.


## Choosing among the three

The choice between [`Monitor`](@ref), [`RAMStorage`](@ref), and [`RAMStageCache`](@ref) is dictated by what the downstream computation needs from the recorded data:

| What you have                                   | What you need to record                                                | Use                       |
|-------------------------------------------------|------------------------------------------------------------------------|---------------------------|
| A scalar / small observable along a trajectory  | One value per step                                                     | [`Monitor`](@ref)         |
| The state at the very end, minus N steps        | A single state                                                         | [`StoreNFromLast`](@ref)  |
| A driver for a continuous linearised solve      | State snapshots dense enough to interpolate to the linearised stages   | [`RAMStorage`](@ref)      |
| A driver for a discretely consistent linearised solve | All stage values of every step                                   | [`RAMStageCache`](@ref)   |

All four can be used simultaneously, in any combination, on the same `Flow` call: the propagation loop accepts at most one of each, and you can pass them in any order as long as you remain consistent with the call signatures of the `Flow` method overloads (`F(x, span, store)`, `F(x, span, cache)`, `F(x, span, mon)`, …).

## Cross-references

- [Linearised dynamics](linearised.md) — full algorithmic context for storages and stage caches.
- [Time stepping](time-stepping.md) — `TimeStepFromStorage` and `TimeStepFromCache` policies that consume these data structures.
- [Internals](internals.md) — Lagrange weight computation and storage memory layout.
