# Time stepping

A [`Flow`](@ref) is the composition of an **integration scheme** (one step of the method) and a **time-stepping policy** (which sequence of `(t, Δt)` pairs cover the integration interval). The two are orthogonal: any policy works with any scheme. This page catalogues the policies, gives the use case for each, and shows the exact `_propagate!` body that each policy drives.

## Why the split

Splitting "what one step does" from "which steps to take" buys two things:

1. **The same primal scheme** (`RK4(zeros(3))`, say) can be driven with a constant step, an adaptive step, or — when paired with a stage cache — *replayed exactly* through a linearised companion. The scheme code is identical in all three cases.
2. **Adding a new policy** does not require modifying any scheme. The schemes implement `step!`; the policies implement the propagation loop.

The four shipped policies cover the four common usage patterns in dynamical-systems and PDE work.

| Policy                                | Use                                                                          |
|---------------------------------------|------------------------------------------------------------------------------|
| [`TimeStepConstant`](@ref)            | Fixed `Δt`. The default everyday choice.                                     |
| [`AbstractTimeStepFromHook`](@ref Flows.AbstractTimeStepFromHook) | User supplies a callable that returns the next `Δt` each step. |
| [`TimeStepFromCache`](@ref Flows.TimeStepFromCache)     | Replay a recorded stage cache. Drives discretely-consistent linearisation. |
| [`TimeStepFromStorage`](@ref Flows.TimeStepFromStorage) | Fixed `Δt` but interpolate the primal state from a storage. Drives continuous linearisation. |

## Constant time step

The everyday default. Marches with a fixed step `Δt` validated to be strictly positive at construction.

```julia
F = flow(f, RK4(zeros(3)), TimeStepConstant(0.01))
F(x, (0.0, 1.0))
```

Construction-time error checking:

```julia
TimeStepConstant(0.0)   # throws: "time step must be positive"
TimeStepConstant(-0.1)  # throws: "time step must be positive"
```

Backward integration is signalled by giving the flow a decreasing `span`; the sign of `Δt` is flipped internally:

```julia
F(x, (1.0, 0.0))   # marches backwards with the same |Δt|
```

When the requested interval is not a multiple of `Δt`, the **last step is shortened** so the endpoint is hit exactly. The mechanism is the [`Flows.Steps`](@ref Flows.Steps) iterator, which is lossy by construction in that case:

```julia
collect(Flows.Steps(0.0, 1.0, 0.3))
#  →  [(0.0, 0.3), (0.3, 0.3), (0.6, 0.3), (0.9, 0.1)]
```

This is the *only* behaviour that breaks the strict-constant-step invariant. It is benign because the schemes are order-$p$ on every step regardless of `Δt`, and the truncated step contributes a single extra $O(\Delta t^{p+1})$ error term to the global error.

```@docs
TimeStepConstant
```

## Adaptive: time step from a user hook

When the time-step size has to react to the state — e.g. a CFL-driven step in a convection problem — the user supplies a **hook**: a concrete subtype of `AbstractTimeStepFromHook` that is callable with signature `hook(g, A, z) -> Δt`. The flow calls the hook once per step with the current right-hand-side, linear operator, and state, and treats the return value as the next `Δt`.

```julia
struct CFLHook <: AbstractTimeStepFromHook
    cmax::Float64    # target CFL number
    dx::Float64      # grid spacing
end

(h::CFLHook)(g, A, z) = h.cmax * h.dx / (maximum(abs, z) + eps())

F = flow(f, RK4(zeros(N)), CFLHook(0.7, dx))
F(x, (0.0, T))
```

The hook is expected to return a **strictly positive** `Δt`; the propagation loop asserts this. When applying the hook would overshoot the requested endpoint, the step is shrunk so that the endpoint is hit exactly — there is no loss of precision at the boundary.

```@docs
Flows.AbstractTimeStepFromHook
```

### Hook design notes

- The hook is called *before* each step. It cannot inspect the state at $t + \Delta t$ because that state does not exist yet.
- The hook receives the explicit and (possibly `nothing`) implicit operators alongside the state, so it can incorporate operator-dependent estimates of the timescale without keeping a reference to the system itself.
- A hook with state of its own (a step controller, a PID, ...) is best implemented as a mutable struct so its state can evolve between calls.

## Replay: time step from a stage cache

A populated [`RAMStageCache`](@ref) records the exact $(t, \Delta t, \mathrm{stages})$ triples produced by a primal integration. The [`TimeStepFromCache`](@ref Flows.TimeStepFromCache) policy is a singleton (no `Δt` parameter) that drives a *linearised* flow through exactly the same time grid, in order. This is the mechanism behind discretely-consistent tangent and adjoint integration.

```julia
# primal: cache the stages
cache = RAMStageCache(nstages(RK4), zeros(3))
F     = flow(f, RK4(zeros(3)), TimeStepConstant(0.01))
F(x, (0.0, 1.0), cache)

# tangent: replay the cache
L = flow(f_tangent, RK4(zeros(3), DiscreteMode(false)), TimeStepFromCache())
L(y, cache)                    # no `span`: the cache contains it
```

The propagation loop walks `cache.ts[i], cache.Δts[i], cache.xs[i]` either forward (tangent) or in reverse (adjoint) depending on the mode tag carried by the scheme.

```@docs
Flows.TimeStepFromCache
```

### When to use it

`TimeStepFromCache` is the right answer whenever a discretely-consistent tangent or adjoint is required. The cookbook *Discrete adjoint sensitivity* in the [Cookbook](@ref Cookbook) walks through a complete example. See [Linearised dynamics](@ref Linearised-dynamics) for the comparison with the continuous alternative.

## Replay: time step from a storage

The continuous counterpart. A primal trajectory is recorded into a [`RAMStorage`](@ref) at the times of the primal integration, and a linearised flow marches with its own fixed `Δt`, interpolating the primal trajectory at every internal-stage time it needs. The interpolation uses a Lagrange polynomial of the degree configured on the storage.

```julia
# primal: record into a storage
store = RAMStorage(zeros(3); degree=5)
F     = flow(f, RK4(zeros(3)), TimeStepConstant(0.01))
F(x, (0.0, 1.0), store)

# tangent / adjoint: continuous integration
L = flow(f_lin, RK4(zeros(3), ContinuousMode(false)), TimeStepFromStorage(0.005))
L(y, store, (0.0, 1.0))
```

Two differences from `TimeStepFromCache` are worth highlighting:

1. The linearised flow chooses its **own** `Δt`. It does not have to match the primal grid. A common pattern is to record on a coarse grid and integrate the linearisation on a finer one.
2. The result is *not* discretely consistent: the linearisation evaluates the operator at an interpolant of the primal, not at the primal's exact stage values. See [Linearised dynamics](@ref Linearised-dynamics) for what this means in practice.

The constructor validates `Δt > 0` like `TimeStepConstant`. Backward integration is again signalled by a decreasing `span`.

```@docs
Flows.TimeStepFromStorage
```

## Putting it together

The four policies cover four positions in a "discretely-consistent ⟷ continuous" times "primal ⟷ adaptive" grid:

|                           | **Fixed step**          | **Adaptive step**         | **Replay from a record**   |
|---------------------------|-------------------------|---------------------------|----------------------------|
| **Primal**                | `TimeStepConstant`      | `AbstractTimeStepFromHook`| —                          |
| **Linearised, continuous**| `TimeStepFromStorage`   | —                         | —                          |
| **Linearised, discrete**  | —                       | —                         | `TimeStepFromCache`        |

Adaptive linearisation and from-cache primal are not provided because they have no natural use in the algorithms `Flows.jl` targets.

## Cross-references

- [Trajectory data](@ref Trajectory-data) — storages and stage caches, the data sources for the replay policies.
- [Linearised dynamics](@ref Linearised-dynamics) — the algorithmic context that the replay policies serve.
- [Internals](@ref Internals) — the `_propagate!` overloads, one per policy.
