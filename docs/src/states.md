# States and vector fields

This page describes the two things the user is responsible for providing: the **state type** that represents one snapshot of the system, and the **callable** that evaluates the right-hand-side of the ODE. Everything else in the package — the schemes, the storages, the time-stepping policies, the linearised integrators — composes around them.

The page covers:

1. The contract a state type must satisfy.
2. The signatures the integrators expect from the vector field.
3. The additional requirements when an IMEX scheme splits off a linear part.
4. The signatures for coupled systems.

## State requirements

`Flows.jl` is deliberately polymorphic over the state type. There is no `AbstractState` supertype; a state is anything that satisfies the following contract.

| Method                          | Why it is needed                                                                                     |
|---------------------------------|------------------------------------------------------------------------------------------------------|
| `Base.similar(x)`               | The schemes preallocate their stage buffers as `similar(x)` at construction time.                    |
| `Base.copy(x)`                  | The `_propagate!` routines push copies into monitors and storages.                                   |
| Dot-broadcasting                | Every scheme's update step is written as `y .= a .* x .+ b .* k`.                                    |

That is the whole contract. There is no requirement to subtype `AbstractArray`, no requirement to implement `length`, `size`, or scalar indexing — and no requirement to be a `Vector`.

### Built-in: plain Julia arrays

Standard `AbstractArray`s of any rank satisfy the contract out of the box. So do `Vector`s of complex numbers, `BitMatrix`es, and (with caveats around mutability) anything else in `Base`.

```julia
F = flow(f, RK4(zeros(64, 64)), TimeStepConstant(1e-3))
F(rand(64, 64), (0, 1))
```

If your state lives in a static-size container (`StaticArrays.SArray`), it is **not** a valid state type because it is immutable: dot-broadcasting allocates a new object rather than overwriting in place. Either use the mutable `StaticArrays.MArray` variant or wrap your data in a plain `Vector`.

### Custom: user-defined state types

For PDE or spectral solvers it is common to wrap the underlying storage in a domain-aware type. The recipe is to forward `similar`, `copy`, and broadcasting to the inner storage. A minimal example:

```julia
struct SpectralField{T<:Complex} <: AbstractVector{T}
    data::Vector{T}
end

# Bare-minimum AbstractArray plumbing
Base.size(u::SpectralField)              = size(u.data)
Base.getindex(u::SpectralField, i::Int)  = u.data[i]
Base.setindex!(u::SpectralField, v, i::Int) = (u.data[i] = v)

# Forward the two methods we promised the package
Base.similar(u::SpectralField) = SpectralField(similar(u.data))
Base.copy(u::SpectralField)    = SpectralField(copy(u.data))
```

Because `SpectralField <: AbstractVector`, broadcasting falls back to the standard `AbstractArray` broadcast machinery and works without further effort. A field type that is *not* an `AbstractArray` will need to define a custom `BroadcastStyle` — see `src/couple.jl` in this package for a worked example.

The motivation for going to the trouble of a custom type is everything you do *outside* `Flows.jl`: custom convenience constructors, derivative methods, IO, pretty-printing, dispatch on physical meaning. The package itself never inspects the type beyond the three requirements above.

## Explicit vector fields

For explicit integrators ([`RK4`](@ref) and the explicit half of any IMEX scheme), the vector field is a callable with signature

```julia
f(t::Real, x, dxdt) -> dxdt
```

| Argument | Direction | Meaning                                                                                          |
|----------|-----------|--------------------------------------------------------------------------------------------------|
| `t`      | input     | Current time.                                                                                    |
| `x`      | **read-only input**   | Current state. **Do not mutate** it; the scheme is still using it.                  |
| `dxdt`   | **write-only output** | Buffer to be overwritten with $\mathrm{d}\mathbf{x}/\mathrm{d}t$ evaluated at `(t, x)`. |

The return value is conventionally `dxdt`, but the package never inspects it. Returning `nothing` is also fine.

`f` can be any callable: a `function`, a closure capturing parameters, a callable `struct`, an anonymous function. The latter two are useful when the right-hand-side carries parameters or owns auxiliary scratch space:

```julia
struct Burgers
    ν::Float64
    k::Vector{Float64}                 # wavenumbers, precomputed
    work::Vector{ComplexF64}           # scratch buffer
end

function (eq::Burgers)(t, u, dudt)
    # ... use eq.ν, eq.k, eq.work to fill dudt in place ...
    return dudt
end
```

The `work` buffer demonstrates a useful pattern: any state your right-hand-side needs but the scheme does not is best owned by the right-hand-side itself.

## IMEX vector fields

When the dynamics split as $\dot{\mathbf{x}} = \mathcal{L}\mathbf{x} + \mathbf{f}_\mathrm{ex}(t,\mathbf{x})$, the user provides both pieces.

### The explicit part

The signature of `f_ex` is exactly the explicit-vector-field signature above:

```julia
f_ex(t, x, dxdt)
```

### The linear part

The linear operator $\mathcal{L}$ is *not* a callable — it is anything that supports two `LinearAlgebra`-like methods:

```julia
LinearAlgebra.mul!(out, A, x)    #  computes out .= A * x
Flows.ImcA!(A, c::Real, y, z)    #  solves (I - c·A) z = y
```

`mul!` is the standard `LinearAlgebra` interface; the package never patches it. [`ImcA!`](@ref) is the central primitive of the IMEX schemes: every implicit stage reduces to one `ImcA!` call with the scheme-dependent constant $c$. Read the name as "I-minus-cee-A".

For the very common case where the stiff operator is diagonal, the package ships fallback methods that satisfy both signatures for `LinearAlgebra.Diagonal`. Stiff problems with a diagonal $\mathcal{L}$ — e.g. spectral discretisations of linear diffusion — work without writing any operator-side code:

```julia
using LinearAlgebra

# semi-discretised diffusion in spectral space:
#   ∂_t û_k = -ν k² û_k + nonlinear(û)
ν = 0.01
k = collect(0:31)
L = Diagonal(-ν .* k.^2)               # works out of the box: see src/imca.jl
f_ex(t, û, dûdt) = (# ... fill dûdt with the nonlinear part)

F = flow(f_ex, L, CNRK2(zeros(ComplexF64, 32)), TimeStepConstant(1e-2))
```

For non-diagonal $\mathcal{L}$, the user provides a `struct` and two methods:

```julia
struct MyOperator
    # whatever data describes A
end

function LinearAlgebra.mul!(out, A::MyOperator, x)
    # out .= A * x
    return out
end

function Flows.ImcA!(A::MyOperator, c::Real, y, z)
    # solve (I - c·A) z = y, in place
    return z
end
```

Three notes:

1. **The same `ImcA!` method is used for every stage of every IMEX scheme.** The constant `c` changes; nothing else does. Write it once, parameterised by `c`.
2. **`ImcA!` should not assume `c > 0`.** Adjoint integrations pass negative `c`. Code your solver as a generic `(I - c·A)` solve.
3. **`ImcA_mul!` has a default.** It computes $(I - c\mathcal{L})\mathbf{y}$ by calling `mul!` and then doing the axpby in place. Override only if you have a faster route.

The cookbook has a [worked stiff example](@ref Cookbook).

## Coupled vector fields

When the state is itself a [`Coupled{N}`](@ref Coupled) of `N` components, the right-hand-side is a `Coupled{N}` of `N` callables, each with a signature that depends on which other components it sees.

The **default** call dependency is "each component sees itself and every previous one":

```julia
g[1](t, u1, du1dt)
g[2](t, u1, du1dt, u2, du2dt)
g[3](t, u1, du1dt, u2, du2dt, u3, du3dt)
…
```

This is the right shape for, e.g., a quadrature appended to a primal system, or a tangent equation appended to a primal system.

```julia
f_primal(t, x, dxdt)              = …
f_tangent(t, x, dxdt, y, dydt)    = …    # depends on x and y

F = flow(couple(f_primal, f_tangent),
         RK4(couple(zeros(3), zeros(3))),
         TimeStepConstant(1e-2))
```

For non-default coupling patterns — e.g. two independent perturbations both driven by the same primal — the user supplies an explicit [`CallDependency`](@ref):

```julia
# component 1 depends on itself; 2 and 3 each depend on 1 and themselves
deps = CallDependency((1,), (1, 2), (1, 3))

F = flow(couple(f, g1, g2), deps,
         RK4(couple(zeros(3), zeros(3), zeros(3))),
         TimeStepConstant(1e-2))
```

The signature of `g1` is then `g1(t, x, dxdt, y1, dy1dt)` (sees components 1 and 2), and similarly for `g2`.

For the IMEX case, the linear part is also a `Coupled{N}` whose components are either a linear operator or `nothing`. `nothing` marks a component that is advanced fully explicitly — the usual case for a quadrature component. See [Coupled systems](@ref Coupled-systems).

## What you do *not* have to do

Three things that other ODE packages require but `Flows.jl` does not:

- **You do not provide an initial-condition object to a constructor.** The state is passed to the flow each time it is called, and the schemes hold only buffers, not solutions.
- **You do not subtype `AbstractODEProblem` or similar.** There is no problem object; the vector field and the linear operator are the abstractions.
- **You do not implement an analytical Jacobian.** The IMEX path uses the (linear) operator, not the Jacobian. The tangent and adjoint paths require a separately-defined linearisation; see [Linearised dynamics](@ref Linearised-dynamics).

With states and vector fields in place, the next page describes how to pick the integration scheme that drives them.
