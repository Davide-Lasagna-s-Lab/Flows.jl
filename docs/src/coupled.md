# Coupled systems

Many problems are not one differential equation in one state but **a small number of equations in heterogeneous states that need to advance together**. Common examples are

- a primal system plus a tangent (linearised) perturbation,
- a primal system plus an adjoint state,
- a primal system plus several trajectory integrals,
- a primal system plus several independent tangent perturbations.

`Flows.jl` supports these natively via the [`Coupled`](@ref) state type and the [`CallDependency`](@ref) interface. This page explains the motivation, the type, the broadcasting machinery, and the construction of coupled flows.

## Motivation

Let $\mathbf{x}(t)\in\mathcal{X}$ and $\mathbf{y}(t)\in\mathcal{Y}$ be two state components governed by

```math
\left\{
\begin{aligned}
\dot{\mathbf{x}}(t) &= \mathbf{f}(t,\,\mathbf{x}(t))\\
\dot{\mathbf{y}}(t) &= \mathbf{g}(t,\,\mathbf{x}(t),\,\mathbf{y}(t))
\end{aligned}
\right.
```

The dynamics of $\mathbf{x}$ is autonomous in $\mathbf{x}$ alone; the dynamics of $\mathbf{y}$ depends on **both** $\mathbf{x}$ and $\mathbf{y}$.

A naïve strategy is to solve the first equation alone, store its solution at some grid of times, and then drive the second equation with a stored or interpolated trajectory. For very large state spaces (DNS of turbulent flows is the canonical example) storing the entire $\mathbf{x}$-trajectory is prohibitive: $\mathbf{x}$ can be tens of GB per snapshot. The alternative is to **advance both components together**, with the integration scheme treating the composite $\mathbf{z}(t) = (\mathbf{x}(t),\,\mathbf{y}(t))$ as a single state.

`Flows.jl`'s solution is a thin wrapper type, [`Coupled`](@ref), that holds the components as an `NTuple` and forwards every package-relevant operation (broadcast, `copy`, `similar`, indexing) to them.

## The `Coupled` type

A `Coupled{N, ARGS}` wraps an `NTuple{N, Any}`. It is constructed with [`couple`](@ref):

```julia
julia> z = couple([1.0, 2.0, 3.0], [0.0, 4.0])
2-element Coupled{2, Tuple{Vector{Float64}, Vector{Float64}}}:
 [1.0, 2.0, 3.0]
 [0.0, 4.0]

julia> z[1]
3-element Vector{Float64}:
 1.0
 2.0
 3.0

julia> length(z)
2
```

The wrapper is itself immutable; mutating *the components* of `z` is fine, but `z[2] = ...` is not. This invariant lets the schemes treat a `Coupled` state as if it were a fixed-size container.

```@docs
Coupled
couple
couplecopy
```

### Broadcasting

The key property of `Coupled` is that it participates in Julia's dot-broadcasting machinery. The expression

```julia
z .= 2 .* z .+ y
```

is interpreted as

```julia
z[1] .= 2 .* z[1] .+ y[1]
z[2] .= 2 .* z[2] .+ y[2]
```

without any temporary `Coupled` ever being allocated. The mechanism is a custom `Broadcast.BroadcastStyle` and a `@generated` `copyto!` that unfolds the `N` per-component broadcasts at compile time. Numbers in the expression pass through unchanged, so mixing scalars and `Coupled`s in a single fused dot expression is supported.

This is what makes coupled-system integration in `Flows.jl` competitive with hand-written single-system code: every `step!` body of every scheme is written in terms of dot expressions, and the same body works for `N = 1, 2, 3, …`. There is no per-step tuple unpacking at run time.

### Independence of components

`couplecopy(N, x)` is a convenience for `couple(deepcopy(x), deepcopy(x), …)`. The components are guaranteed to be **independent** (no shared storage):

```julia
z = couplecopy(3, zeros(5))
z[1][1] = -1
@assert z[2][1] == 0      # other components are untouched
```

This is useful for, e.g., evolving several independent tangent perturbations of the same primal in lock-step.

## Constructing a coupled flow

Building a [`Flow`](@ref) on a coupled state is mechanically identical to the single-state case, with two changes:

1. The right-hand-side is a `Coupled{N}` of callables, one per component.
2. The integration scheme is constructed with a `Coupled{N}` template state, so its internal buffers are themselves `Coupled{N}`.

```julia
f(t, x, dxdt)              = …            # primal RHS
g(t, x, dxdt, y, dydt)     = …            # second-component RHS

x = zeros(3)
y = zeros(3)

F = flow(couple(f, g),
         RK4(couple(x, y)),
         TimeStepConstant(0.01))

F(couple(x, y), (0.0, 1.0))
```

The signature of `g` here is the **default** one: each component sees itself and every preceding component. This is the right signature for tangent equations, quadrature equations, and most other "primal + extension" workflows.

## Default call dependency

The default expansion of a coupled right-hand-side call is

```julia
g[1](t,  u1, du1dt)
g[2](t,  u1, du1dt,  u2, du2dt)
g[3](t,  u1, du1dt,  u2, du2dt,  u3, du3dt)
g[4](t,  u1, du1dt,  u2, du2dt,  u3, du3dt,  u4, du4dt)
```

i.e. component $i$ depends on components $1, 2, \dots, i$. The default is constructed by `default_dep(N)` and is set up for `N ∈ 1:3` directly; larger `N` is supported via the explicit construction below.

## Custom call dependencies

Some coupling patterns do **not** fit the default. Two independent tangent perturbations $\mathbf{y}_1, \mathbf{y}_2$ of the same primal $\mathbf{x}$ obey

```math
\left\{
\begin{aligned}
\dot{\mathbf{x}}(t)   &= \mathbf{f}(t,\mathbf{x}(t))\\
\dot{\mathbf{y}}_1(t) &= \mathbf{g}_1(t,\mathbf{x}(t),\mathbf{y}_1(t))\\
\dot{\mathbf{y}}_2(t) &= \mathbf{g}_2(t,\mathbf{x}(t),\mathbf{y}_2(t))
\end{aligned}
\right.
```

Note that $\mathbf{g}_1$ does **not** depend on $\mathbf{y}_2$, and vice versa. The default dependency would force the signature of $\mathbf{g}_2$ to receive `y1` whether or not it uses it. This is wasteful, and it is also misleading: the right-hand-side declares dependencies it does not have.

The fix is a [`CallDependency`](@ref):

```julia
deps = CallDependency((1,),     # component 1 depends on itself
                      (1, 2),   # component 2 depends on 1 and itself
                      (1, 3))   # component 3 depends on 1 and itself

F = flow(couple(f, g1, g2),
         deps,
         RK4(couple(zeros(3), zeros(3), zeros(3))),
         TimeStepConstant(0.01))
```

with

```julia
f(t,  x, dxdt)
g1(t, x, dxdt,  y1, dy1dt)
g2(t, x, dxdt,  y2, dy2dt)
```

`CallDependency((spec1,), (spec2,), …)` accepts `N` tuples of positive integers; each tuple `spec_i` is the sorted list of component indices that flow into the signature of component `i`. The constructor enforces:

- every index lies in `1:N`,
- every spec is sorted in increasing order,
- the inner tuples carry **no duplicates** (sorted strict-monotone).

```@docs
CallDependency
```

## Coupled IMEX systems

For IMEX problems the **linear part** is also a `Coupled{N}`. Each component may be either a linear operator (with [`ImcA!`](@ref) and `mul!`) or `nothing`. The `nothing` marker means *advance this component fully explicitly* — the universal idiom for a quadrature component appended to a stiff primal:

```julia
# stiff primal + explicitly-advanced quadrature
f_ex(t, x, dxdt) = …
quad(t, x, dxdt, I, dIdt) = (dIdt[1] = x[1]^2)

L = Diagonal(...)        # linear operator for the primal

F = flow(couple(f_ex, quad),
         couple(L, nothing),
         CNRK2(couple(zeros(3), zeros(1))),
         TimeStepConstant(0.01))
```

The IMEX `step!` implementations check the type of each component of `A`: when `A[i]` is `nothing`, the implicit subproblem for component `i` collapses to `z .= y` (the identity), and the explicit part takes the whole step. When `A[i]` is non-`nothing`, the full IMEX update is applied. The dispatch is done at compile time via `@generated` functions in `src/system.jl`, so the `nothing`-branch costs nothing at run time.

## What `Coupled` is *not*

A few non-uses worth flagging:

- **It is not for parallelism.** Each component still advances on the same thread.
- **It is not a general "tuple of states" abstraction.** The schemes' broadcast rules assume the components share the same shape of broadcast operations. Mixing wildly different state types in one `Coupled` (a `Vector{Float64}` and a `SparseMatrixCSC`, say) may work, but is outside the design envelope.
- **It is not nested.** `Coupled` of `Coupled` is not supported in the broadcasting code paths and is not tested.

## Cross-references

- [States and vector fields](@ref States-and-vector-fields) — single-component requirements.
- [Quadrature equations](@ref Quadrature-equations) — the canonical "primal + extension" coupled use case.
- [Linearised dynamics](@ref Linearised-dynamics) — coupled primal+tangent for forward-mode sensitivity.
- [Internals](@ref Internals) — the `@generated` broadcast machinery in `src/couple.jl`.
