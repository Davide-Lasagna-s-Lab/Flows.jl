# Symmetry transformations

Many dynamical systems are *equivariant* under a continuous group action: a Galilean translation, a phase rotation, a spatial shift. The flow of such a system intertwines with the group action, and any algorithm that walks the state space modulo this action benefits from the ability to *apply the symmetry as part of the flow*. `Flows.jl` provides exactly this: an optional symmetry-transformation callable attached to every [`Flows.Flow`](@ref Flows.Flow) that is applied (or not) at each call.

This page explains the math, the API, and the patterns that show up in practice.

## Equivariance and reduction

Let $g_s : \mathcal{X}\to\mathcal{X}$ be a one-parameter family of invertible transformations indexed by $s\in\mathbb{R}$. The system is **equivariant** under $g_s$ if its flow $\Phi^{\tau}$ commutes with it:

```math
\Phi^{\,\tau}(g_s\,\mathbf{x}) = g_s\,\Phi^{\,\tau}\mathbf{x}\qquad\forall\,\tau,\,s.
```

The canonical example is a translation-invariant PDE: $g_s$ shifts the field by $s$ in the spatial direction, and a primal evolved from a shifted initial condition is the shifted primal.

In a **relative periodic orbit** computation, the goal is to find a state $\mathbf{x}^{\star}$ and a pair $(T^{\star}, s^{\star})$ such that

```math
\mathbf{x}^{\star} = g_{s^{\star}}^{-1}\,\Phi^{\,T^{\star}}\mathbf{x}^{\star}.
```

The orbit is closed *after the action* of $g_{s^{\star}}^{-1}$. Numerical schemes that hunt for $(\mathbf{x}^{\star}, T^{\star}, s^{\star})$ — Newton on a residual involving $g_s^{-1}\Phi^{\,T}\mathbf{x} - \mathbf{x}$ — need to evaluate $g_s^{-1}\Phi^{\,T}\mathbf{x}$ many times. A flow that does the symmetry application internally is the natural primitive.

A second class of uses is **symmetry reduction**: post-composing the flow with $g_s$ for a phase $s = s(\mathbf{x})$ chosen by a "slice condition" $\langle\mathbf{x}, \boldsymbol{\nu}\rangle = 0$. The reduced flow lives on a slice that intersects each group orbit once, which removes the marginally-stable drift direction.

## The API

A symmetry is supplied as a callable

```julia
sym(x, s)        # apply sym parameterised by s to x, in place; return x
```

passed as the **last** positional argument to any [`flow`](@ref) constructor:

```julia
F = flow(f, RK4(zeros(3)), TimeStepConstant(0.01), (x, s) -> x[1] += s)
```

The constructor wraps the callable: for a single state in a [`Flows.SymTransform`](@ref), for a coupled state in a [`Flows.CoupledTransform`](@ref). Both collapse to `nothing` when constructed with `nothing`, so passing `sym=nothing` (the default) is equivalent to omitting the argument.

At the call site, a flow with a non-trivial `sym` is called with an extra positional argument `s`:

```julia
F(x, (0, T), s)          # apply sym(x, s) to the propagated state before returning
```

Without `s`, the symmetry is *not* applied:

```julia
F(x, (0, T))             # primal advance only
```

This is what lets the same `F` serve as both the raw flow and the symmetry-transformed flow, parameterised by the caller's choice of `s`.

## Single-state vs coupled-state symmetries

The two wrappers handle the two state-type cases:

| Wrapper                            | Applied to        | Per-component?    |
|------------------------------------|-------------------|-------------------|
| [`Flows.SymTransform`](@ref)       | A single state    | n/a (one call)    |
| [`Flows.CoupledTransform`](@ref)   | A `Coupled{N}`    | Yes — one call per component |

`SymTransform` simply delegates: `f(x, s) = f.sym(x, s)`. `CoupledTransform` loops over the `N` components and calls `f.sym(x[i], s)` on each. The implication is that the user's `sym` callable should be written *as if it acted on a single state component*, and the wrapper choice is made automatically by `flow` based on whether the components are coupled.

The choice of wrapper is made by the appropriate `flow` overload:

- Non-coupled constructors wrap in `SymTransform`.
- Coupled constructors (those whose `g` is a `Coupled`) wrap in `CoupledTransform`.

You never instantiate either wrapper yourself in normal use; the public API is just the `sym` callable.


## Patterns

### Translation symmetry

```julia
sym_shift(x, s) = (x .= circshift(x, round(Int, s)); x)
F = flow(f, RK4(zeros(N)), TimeStepConstant(Δt), sym_shift)

F(x, (0, T), 10)           # advance, then shift by 10 cells
```

The example uses a discrete `circshift` for simplicity; in spectral codes the same idea is implemented as multiplication by a phase factor in Fourier space.

### Phase rotation

```julia
sym_rotate(x, θ) = (x .= [cos(θ) -sin(θ); sin(θ) cos(θ)] * x; x)
F = flow(f, RK4(zeros(2)), TimeStepConstant(Δt), sym_rotate)

F(x, (0, T), π/4)          # advance, then rotate by π/4
```

### Slice-reduced flow

A common pattern is to make `sym` *self-determining* — to compute the appropriate `s` from the state itself, and ignore the caller's `s`:

```julia
function sym_slice(x, _ignored)
    s = best_phase_for_slice(x)
    apply_phase!(x, s)
    return x
end

F = flow(f, RK4(zeros(N)), TimeStepConstant(Δt), sym_slice)
F(x, (0, T), 0.0)          # `s` is ignored; the slice condition picks the phase
```

This is the *method-of-slices* approach for reducing translational drift from chaotic-attractor visualisations.

## Interaction with other constructs

Symmetries compose cleanly with the rest of the package:

- **Monitors** are pushed *before* the symmetry transformation is applied, so the recorded series is the un-symmetry-reduced trajectory. If you need the reduced trajectory recorded, perform the transformation inside the monitor's observable.
- **Storages and stage caches** also record the un-reduced trajectory. The symmetry transformation is purely a post-processing step on the final state.
- **Linearised flows** do not currently apply the symmetry wrapper. If your application needs a symmetry-reduced linearised flow, write the linearised RHS to encode the reduction directly.

## Single-application semantics

The symmetry is applied **once**, on the *final* state returned by `_propagate!`, not at every internal time step. This means the relation

```math
\Phi^{\,\tau_1+\tau_2} = g_{s_2}\,\Phi^{\,\tau_2}\,\circ\,g_{s_1}\,\Phi^{\,\tau_1}\qquad
(\text{compose two symmetry-applied flow calls})
```

is generally not the same as a single call $g_{s_1+s_2}\,\Phi^{\,\tau_1+\tau_2}$ — the symmetry is applied between the two pieces in the composed call but not in the single one. If your algorithm requires the latter behaviour, sum the `s` values yourself before the single call.

## Cross-references

- [Architecture](architecture.md) — how the symmetry wrapper fits into the four-axis decomposition of `Flow`.
- [Coupled systems](coupled.md) — `CoupledTransform` builds on the `Coupled` machinery.
- [Cookbook](cookbook.md) — symmetry-reduced flow on a 1D PDE.
