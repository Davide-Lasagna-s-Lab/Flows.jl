# Internals

This page documents the **implementation** of `Flows.jl`: data layouts, generated code, dispatch tricks, and design rationale. It is intended for contributors, users debugging a performance regression, and curious readers who already understand the user-facing API. The conceptual map of the package is on [Architecture](@ref Architecture); this page goes one level deeper.

## Module layout

`Flows.jl` is a single module split across one source file per concern. Files are `include`d in dependency order from `src/Flows.jl`:

```
couple.jl  →  tableaux.jl  →  stagecache.jl  →  system.jl  →  storage.jl
   ↓
steps/shared.jl  →  steps/{rk4,CNRK2,CB3R2R,CB4R3R}.jl
   ↓
timestepping.jl  →  logger.jl  →  monitor.jl  →  storenfromlast.jl  →  imca.jl
   ↓
stepper.jl  →  integrator.jl  →  utils.jl
```

Each file holds one self-contained piece of functionality. There is no cyclic dependency. The flat module structure is intentional: the package is small enough that submodules would add navigation overhead without buying any encapsulation.

## The `Coupled` broadcast machinery

`src/couple.jl` defines a `CoupledStyle <: Broadcast.ArrayStyle{Coupled}` and a single `@generated` `Base.copyto!`. The generated body unfolds, at compile time, a per-component `copyto!` for each of the `N` slots:

```julia
@generated function Base.copyto!(dest::Coupled{N},
                                   bc::Broadcast.Broadcasted{CoupledStyle}) where {N}
    quote
        $(Expr(:meta, :inline))
        Base.Cartesian.@nexprs $N i -> begin
            @inbounds copyto!(getfield(dest.args, i), unpack(bc, Val(i)))
        end
        return dest
    end
end
```

`unpack(bc, Val(i))` walks the `Broadcasted` tree and replaces every `Coupled` operand by its `i`-th component, leaving `Number`s and other operands untouched. The result is a per-component `Broadcasted` that `copyto!` can dispatch on the underlying component type — which usually falls through to the default `AbstractArray` broadcast.

The pattern is adapted from `MultiScaleArrays.jl`. It is the reason a single hand-written `step!` body like

```julia
y .= x .+ Δt .* k1
```

works uniformly for any `N`, and that the generated code has zero per-step tuple-iteration overhead.

## Tableau storage

`src/tableaux.jl` carries the Butcher tableaux as `Tableau` and `IMEXTableau` structs. The four IMEX tableaux from Cavaglieri & Bewley (2015) are entered as exact `Rational` arrays and converted *once*, at module load, to `Float64`:

```julia
const CB2 = convert(IMEXTableau{Float64, 3}, IMEXTableau(CB2_I, CB2_E))
```

`getindex(tab, :aᴵ, i, j)` and friends return coefficients by their mathematical role. The unicode superscripts make the IMEX bookkeeping in `src/steps/CB*.jl` line up visually with the paper.

The tableau types support `convert(::Type{Tableau{T, NS}}, tab)`, so a tableau in any element type (rationals, `Float32`, `Float64`) can be coerced to whatever the schemes need.

## `Steps`: the time iterator

`src/stepper.jl` defines

```julia
struct Steps{S, R<:AbstractRange{S}} <: AbstractVector{Tuple{S, S}}
        rng::R
          T::S
    isLossy::Bool
end
```

— a thin `AbstractVector` wrapping a Julia `Range`, with a flag for whether the range covers the requested span exactly. Iteration produces `(t, Δt)` pairs:

- For the lossless case (`last(rng) == T`), `Δt` is constant.
- For the lossy case, the final `Δt` is shortened to `T - last(rng)` so that `t + dt == T` exactly.

The constructor accepts a positive `dt` and flips its sign internally when `t0 > T` so that backward integration shares the same iteration code path.

There is **no heap allocation** anywhere in `Steps`. Iteration is just two integer-index lookups into a `Range`. This is why the propagation loops in `src/integrator.jl` are allocation-free.

## `System`: generated dispatch on call dependencies

`src/system.jl` defines `System{N, DEPS, GT, AT}` whose call operator is `@generated` on `(N, DEPS)`. The body builds a flat sequence of calls into the components:

```julia
@generated function (sys::System{N, DEPS})(t, z::Coupled{N}, dzdt::Coupled{N}) where {N, DEPS}
    expr = quote end
    for i = 1:N
        tup = Expr(:tuple)
        for d in DEPS[i]
            append!(tup.args, (:(z[$d]), :(dzdt[$d])))
        end
        push!(expr.args, :(sys.g[$(Val(i))](t, $(tup)...)))
    end
    return expr
end
```

`DEPS` is a `CallDependency{N, INFO}` whose `INFO` is a compile-time tuple-of-tuples. The `@generated` walks `INFO` and emits exactly the right call signature for each component. The result, after specialisation, is a straight-line sequence of calls — no tuple iteration, no runtime dispatch on `DEPS`.

The same idea is used for the implicit-part bookkeeping: `mul!(out, sys, z)` and `ImcA!(sys, c, y, z)` are `@generated` on `AT`, walking the type parameters and emitting `(out[i] .= 0)` or `(z[i] .= y[i])` whenever the matching `A[i]` is `Nothing`. The `nothing` branch costs nothing at run time.

## Mode tags

`src/steps/shared.jl` defines `NormalMode`, `ContinuousMode{ISADJOINT}`, and `DiscreteMode{ISADJOINT}`. They are empty structs whose only role is to dispatch the appropriate `step!` overload of each scheme.

`isadjoint(::Type{NormalMode}) = false` and `isadjoint(::Type{*Mode{ISADJOINT}}) = ISADJOINT` are the predicates used inside the schemes to flip a sign on `Δt` (and to choose whether to iterate the stage cache forwards or backwards in `src/integrator.jl`).

## Integration schemes

Every scheme follows the same layout:

```julia
struct SCHEME{X, MODE, NX} <: AbstractMethod{X, MODE, NS}
    store::NX     # NTuple{N, X} of preallocated stage buffers
end
```

with one outer constructor that takes a template `x::X` and a `mode::AbstractMode`, chooses `N` based on the mode, and allocates `ntuple(i -> similar(x), N)`.

The user-facing `step!(method, sys, t, Δt, x, cache_or_stages)` is implemented as several overloads, one per `MODE`:

- `NormalMode` — primal forward step with optional caching of internal stages.
- `ContinuousMode{ISADJOINT}` — interpolated-primal step, sign of `Δt` determined by `ISADJOINT`.
- `DiscreteMode{ISADJOINT}` — replay-of-cached-stages step, forwards if `false`, backwards-and-transposed if `true`.

The CB schemes additionally use `@eval`-driven code generation in `src/steps/CB3R2R.jl` to produce three concrete types (`CB3R2R2`, `CB3R2R3e`, `CB3R2R3c`) from a single shared `step!` template parameterised on the tableau.

## Propagation loops

`src/integrator.jl` contains the four propagation kernels:

| `_propagate!` overload                                       | Used by                                                         |
|---------------------------------------------------------------|-----------------------------------------------------------------|
| `(method::AbstractMethod{Z, NormalMode}, stepping::TimeStepConstant, ...)` | All primal flows with constant time step                       |
| `(method::AbstractMethod{Z, MODE<:NormalMode}, hook::AbstractTimeStepFromHook, ...)` | Hook-driven primal flows                                       |
| `(method::AbstractMethod{Z, MODE<:DiscreteMode}, system, z, cache, mon)`   | Discrete tangent and adjoint                                  |
| `(method::AbstractMethod{Z, MODE<:ContinuousMode}, stepping::TimeStepFromStorage, ...)` | Continuous tangent and adjoint                                  |

Each loop walks a `Steps` iterator (or the cache's recorded time grid) and calls the scheme's `step!`. The cache, storage, and monitor branches are dispatched on the static types of the corresponding arguments, so `_propagate!` is fully specialised on each combination — no per-step `isa` check.

## `RAMStorage` and Lagrange interpolation

`src/storage.jl` stores times and snapshots in parallel `Vector`s. The interpolator is callable:

```julia
(store::RAMStorage{T, X, DEG})(out::X, t::Real, ::Val{ORD}=Val(0))
```

Three steps:

1. **Stencil selection.** `_interp_indices` does a `searchsortedlast`, then shifts the stencil if necessary to keep it inside the data (non-periodic case) or to wrap it modulo the length (periodic case). The shift logic is unrolled at code-generation time by being inside a `@generated` function with `Val(N)`.
2. **Periodic-wrap adjustment.** `_make_tuple_of_times` constructs a strictly increasing tuple of times by adding `period` to entries that follow the wrap point, and adjusts the query time `t` if it falls outside the new interval.
3. **Lagrange weights and weighted sum.** `_lagr_weights` is generated for each `N` and produces an `NTuple{N, Float64}` of weights via `_prod(t, ts, Val(j))`. `_lagr_interp` then walks the indices and accumulates the result in place into `out`.

The weighted-sum loop is `out .= ws[1] .* xs[rng[1]]; for i in 2:N; out .+= ws[i] .* xs[rng[i]]; end` — fully dot-broadcast, allocation-free, and dispatched on the component type via the standard `AbstractArray` broadcast machinery.

The first-derivative variant `_lagr_weights(t, ts, Val(1))` is a four-point central-difference approximation of the function-value weights. This is a stop-gap until an analytic derivative is implemented; the current accuracy is sufficient for the few use cases that need a derivative interpolant.

## `RAMStageCache`

`src/stagecache.jl` is intentionally minimalist: three parallel `Vector`s for `(ts, Δts, xs)`, with `push!` taking the three pieces and appending in lock-step. The stage tuple `xs[i] :: NTuple{NS, X}` is pushed *as a whole tuple* per step, so the storage layout matches what the replay path consumes.

`nstages(::AbstractStageCache{NS}) = NS` is the predicate used by every propagation loop that takes a cache, to catch scheme/cache mismatches at the entry point.

## `Monitor` and `Logger`

`src/monitor.jl` implements a `mutable struct Monitor` that delegates storage to any `AbstractStorage`. The push policy is implemented in a single `Base.push!(mon, t, x, force)`:

```julia
if force || (mon.count % mon.oneevery == 0)
    if isbetween(t, mon.savebetween...)
        if !(mon.count == 0 && mon.skipfirst)
            push!(mon.store, t, mon.f(t, x))
            …                                  # printf via Logger
        end
    end
end
mon.count += 1
```

`force` is the integration loop's mechanism for guaranteeing that the **final step** is always pushed, regardless of `oneevery` decimation.

The optional `Logger` in `src/logger.jl` is a `mutable struct Logger{N, O, F}` whose call operator prints either a header or a row, depending on whether it is called with no arguments or with `(count, store)`. The `Printf.Format` row is built once at construction by inspecting the type of the observable: floats, integers, and complex numbers each get their own format width; everything else renders as `?`.

## Symmetry wrappers

`src/couple.jl` defines `SymTransform{SO}` and `CoupledTransform{SO}`, both callables. Their constructors collapse to `nothing` when given `nothing`, so the *no symmetry* case is represented by a uniform `Nothing` — `const NoTransform = Nothing`.

The `Flow` callable dispatches on `SO <: NoTransform` to elide the `sym(...)` call when there is no symmetry, keeping the two-argument call `F(x, span)` as cheap as a flow built without a symmetry.

## Allocation discipline

The package's test suite contains `@allocated` regression tests at the boundary of every propagation path:

- `test/test_integrator.jl` checks that `F(x, span)` allocates zero bytes after warmup.
- `test/test_steps.jl` checks that `Flows.step!(scheme, sys, t, Δt, x, nothing)` allocates zero bytes.
- `test/test_quadrature.jl` checks that the coupled-quadrature path is allocation-free.

Two known offenders are tagged `@test_broken`:

- The `Logger`-attached `Monitor` allocates once per push because the `Printf.format` machinery allocates a `String`.
- The CB3R2R `step!` cache branch allocates a temporary `X[]` vector that is then `tuple(...)`-ed before being pushed. This is a known TODO in the code.

These regressions are bounded and localised; they do not affect the hot path of a primal integration with no monitor.

## Where to start when contributing

If you want to extend the package, the entry points are:

| To add                          | Touch                                                                                          |
|---------------------------------|-------------------------------------------------------------------------------------------------|
| A new scheme                    | `src/steps/yourscheme.jl` (plus an `include` in `src/Flows.jl`); add at least `step!` for `NormalMode`. Add `ContinuousMode` and `DiscreteMode` for linearised support. |
| A new linear-operator type      | Write `LinearAlgebra.mul!` and `Flows.ImcA!` for it. No `Flows` change needed.                |
| A new time-stepping policy      | Add a subtype of `AbstractTimeStepping` (or `AbstractTimeStepFromHook`) and a `_propagate!` overload that takes it. |
| A new storage backend           | Subtype `AbstractStorage` and implement `reset!`, `push!`, `times`, `samples`, `timespan`, and the interpolation callable. |
| A new monitor type              | Subtype `AbstractMonitor` and implement `push!(mon, t, x, force)`.                            |

The test suite mirrors the source layout, so new functionality should be paired with new tests in `test/test_<area>.jl`.

## Cross-references

- [Architecture](@ref Architecture) — the four-axis decomposition that this page implements.
- [API](@ref Full-public-API) — the public docstrings.
