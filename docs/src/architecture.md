# Architecture

This page describes how `Flows.jl` is put together: which types own which responsibility, how they fit into one another, and how `F(x, span)` becomes a sequence of `step!` calls. It is the right page to read once if you intend to extend the package, write a custom scheme, or debug performance.

## The flow as a composition

A [`Flow`](@ref Flows.Flow) is a four-field container:

```julia
mutable struct Flow{TS, M, S, SO}
    tstep::TS   # ::AbstractTimeStepping  — how the time grid is built
    meth ::M    # ::AbstractMethod        — one-step scheme + preallocated stages
    sys  ::S    # ::System                — vector field(s) and optional L
    sym  ::SO   # ::Nothing or SymTransform or CoupledTransform
end
```

The four fields are **orthogonal** in the sense that they can be combined freely:

```
Flow ≅ TimeStepping  ⊗  Method  ⊗  System  ⊗  Symmetry
```

| Axis           | Concrete types                                                         | What it decides                                  |
|----------------|------------------------------------------------------------------------|--------------------------------------------------|
| Time stepping  | `TimeStepConstant`, `AbstractTimeStepFromHook`, `TimeStepFromCache`, `TimeStepFromStorage` | The sequence of $(t_n, \Delta t_n)$ pairs        |
| Method         | `RK4`, `CNRK2`, `CB3R2R2`, `CB3R2R3e`, `CB3R2R3c`, `CB4R3R4`           | What one step does, and how many stage buffers it owns  |
| System         | `System{N, DEPS, GT, AT}`                                              | Which right-hand-side(s) and optional linear operator(s) are evaluated |
| Symmetry       | `Nothing`, `SymTransform`, `CoupledTransform`                          | Post-step transformation applied when the user supplies `s` |

The construction entry point is the [`flow`](@ref) factory; it has many overloads only because the constructor signature changes when more axes are present. The internal representation never branches on what the user passed.

## Layered call

Calling a flow walks a fixed pipeline:

```
F(x, (t0, T))
   └─→ _propagate!(meth, tstep, sys, span, x, cache?, store?, mon?)
            └─→ for (t, Δt) in Steps(t0, T, tstep.Δt)
                    step!(meth, sys, t, Δt, x, cache?)
                    (push! into mon? / store?)
            return x
```

Three observations:

1. **The time iterator is a value, not a callback.** [`Flows.Steps`](@ref Flows.Steps) is a thin `AbstractVector` over a Julia `Range`; its iteration produces `(t, Δt)` pairs without any per-iteration heap allocation. The final pair may carry a shorter `Δt` when the span is not a multiple of the nominal step.
2. **`step!` is dispatched on `(method, mode)`.** Every scheme implements several `step!` methods, one per `MODE` it supports: the normal-mode one runs the primal, the `ContinuousMode` ones interpolate from a storage, and the `DiscreteMode` ones replay a tuple of stages. Adding a scheme means writing a new `struct` plus one `step!` per mode.
3. **Monitors, storages, and caches are dispatch-time options.** `_propagate!` is parameterised over the types of `mon`, `store`, and `cache`; when any of them is `Nothing`, the corresponding branch is dead code at the LLVM level. There is no runtime "if monitor != nothing" check.

## The System wrapper

A [`Flows.System`](@ref) bundles the right-hand-side(s) `g` and the optional linear operator(s) `A`. The single-component version just stores `(g, A)`; the multi-component version stores tuples plus a [`CallDependency`](@ref) describing which components flow into which signature.

`System` is callable. For the explicit part it walks `DEPS` at code-generation time and emits a single call to each component:

```julia
@generated function (sys::System{N, DEPS})(t, z::Coupled{N}, dzdt::Coupled{N}) where {N, DEPS}
    expr = quote end
    for i = 1:N
        # build the tuple (z[d], dzdt[d]) for each d in DEPS[i]
        push!(expr.args, :( sys.g[$(Val(i))](t, <unpacked>...) ))
    end
    return expr
end
```

The `@generated` body sees `N` and `DEPS` as compile-time constants, so the resulting code is a flat sequence of calls with no runtime tuple manipulation in the hot path.

For the implicit part, `System` provides three thin shims:

```julia
mul!(out, sys, z)        # action of A on z; falls back to (out .= 0) when A === nothing
ImcA!(sys, c, y, z)      # solve (I - cA) z = y;     falls back to z .= y when A === nothing
ImcA_mul!(sys, c, y, z)  # compute (I - cA) y;       falls back to z .= y when A === nothing
```

The `nothing` fallback is what makes the **quadrature-via-coupling** idiom work: an explicitly-marched quadrature component pairs the user's `f` with `nothing` for `A`, and the IMEX scheme sees the quadrature row as if it were a no-op implicit problem.

## The integration method

Every concrete scheme is a `struct ConcreteMethod{X, MODE, NX} <: AbstractMethod{X, MODE, NS}`:

| Type parameter | Role                                                                                    |
|----------------|-----------------------------------------------------------------------------------------|
| `X`            | The state type. Locked at construction; passing a state of a different type is an error. |
| `MODE`         | The integration mode tag (see below). Locks which `step!` is dispatched.                |
| `NS`           | The stage count. Matched against any `AbstractStageCache` passed at integration time.   |

The scheme stores all preallocated buffers in a single `NTuple{N, X}` field named `store`, where `N` depends on the scheme *and* on the mode: a `ContinuousMode` `RK4` needs an extra buffer to hold the interpolant compared with a `NormalMode` one. See [Integration schemes](@ref Integration-schemes) for the exact counts.

### Mode tags

The mode is a zero-cost type tag — an empty `struct` whose only job is to pick a code path.

| Tag                                       | When                                                     |
|-------------------------------------------|----------------------------------------------------------|
| `NormalMode`                              | Forward nonlinear (primal) integration                   |
| `ContinuousMode(false)` / `(true)`        | Continuous tangent / adjoint, driven by a `RAMStorage`   |
| `DiscreteMode(false)` / `(true)`          | Discretely consistent tangent / adjoint, driven by a `RAMStageCache` |

`isadjoint(MODE)` returns `true` only for the `*Mode(true)` variants and is the predicate the schemes use to decide whether to march forwards or backwards.

## Time-stepping policies

The three propagation paths in `src/integrator.jl` are selected by the `(method, tstep)` pair:

| `tstep` type                       | `_propagate!` body                                                     |
|------------------------------------|------------------------------------------------------------------------|
| `TimeStepConstant`                 | walks `Steps(t0, T, Δt)` and calls `step!(method, sys, t, Δt, z, cache)` |
| `<: AbstractTimeStepFromHook`      | repeatedly queries `hook(g, A, z)` for the next `Δt` until `t == T`     |
| `TimeStepFromCache`                | replays `(ts[i], Δts[i], xs[i])` from a stage cache, no `span` argument |
| `TimeStepFromStorage`              | walks `Steps(t0, T, Δt)` but threads the storage into `step!`           |

The first three are for primal integration; `TimeStepFromCache` and `TimeStepFromStorage` are the entry points used by linearised flows.

## Coupling: the `Coupled` type

A [`Coupled{N, ARGS}`](@ref Flows.Coupled) is a thin wrapper around an `NTuple{N, Any}`. It indexes (`z[i]`), it has length `N`, and — critically — it participates in Julia's broadcasting infrastructure: a dot expression involving `Coupled`s and `Number`s is forwarded down to the components, so the time-stepping kernels do

```julia
y .= x .+ Δt .* k1
```

uniformly regardless of whether `x` is a single state or a coupled state.

The broadcast machinery (`Base.copyto!(::Coupled{N}, ::Broadcasted{CoupledStyle})`) is `@generated` so that, for each statically-known `N`, it expands to exactly `N` calls into per-component `copyto!` with no tuple iteration at run time. The implementation pattern follows MultiScaleArrays.jl.

## Generated dispatch on N

Because the broadcast happens through `@generated`, the same `step!` body works for `N=1, 2, 3, …`. The cost of supporting many `N` is paid once, at compile time, for each `N` the user actually uses. This is the reason coupled-system integration in `Flows.jl` is as fast as hand-written single-system code: there is no per-step tuple unpacking.

## The symmetry wrapper

After `_propagate!` returns, the optional `sym` field of the flow is applied if and only if the user supplied a third positional argument `s`:

```julia
F(x, span)              # _propagate!(...); return x
F(x, span, s)           # I.sym(_propagate!(...), s); return x
```

Both [`Flows.SymTransform`](@ref) and [`Flows.CoupledTransform`](@ref) are themselves callables. The wrapper exists so that the `Flow` carries a uniform type discipline: `F.sym` is either `nothing`, or a callable that maps a state and a parameter to the transformed state. The dispatch on `SO <: NoTransform` (where `NoTransform == Nothing`) is what suppresses the wrapping in the no-symmetry case at zero cost.

## File map

A direct correspondence between the abstractions and the source layout:

| Concept                        | File(s)                                                 |
|--------------------------------|---------------------------------------------------------|
| `Coupled`, broadcasting, sym wrappers | `src/couple.jl`                                  |
| Butcher tableaux + `CB*_I/E` constants | `src/tableaux.jl`                               |
| `Steps` time iterator                  | `src/stepper.jl`                                |
| `System`, `CallDependency`             | `src/system.jl`                                 |
| `RAMStorage`, Lagrange interpolation   | `src/storage.jl`                                |
| `RAMStageCache`                        | `src/stagecache.jl`                             |
| `AbstractMethod`, mode tags            | `src/steps/shared.jl`                           |
| Schemes (one file per scheme)          | `src/steps/{rk4,CNRK2,CB3R2R,CB4R3R}.jl`        |
| `_propagate!`, `Flow` callable surface | `src/integrator.jl`                             |
| `Monitor`, `Logger`                    | `src/monitor.jl`, `src/logger.jl`               |
| `StoreNFromLast` specialised monitor   | `src/storenfromlast.jl`                         |
| `TimeStepConstant`, hook, cache, storage | `src/timestepping.jl`                         |
| `ImcA!`, `ImcA_mul!`, `Diagonal` fallback | `src/imca.jl`                                |
| Stand-alone quadrature rules           | `src/utils.jl`                                  |

The dependency graph is roughly: `couple.jl → tableaux.jl → stagecache.jl → system.jl → storage.jl → steps/* → timestepping.jl → integrator.jl`. The top-level [`Flows.jl`](https://github.com/Davide-Lasagna-s-Lab/Flows.jl/blob/master/src/Flows.jl) `include`s them in that order.
