# Quick start

This page shows the shortest path from "I have a vector field" to "I am running an integration". The example uses the classical Lorenz system, but every step transfers directly to systems of arbitrary size and structure.

If you need a deeper explanation of the design before you start, jump to [Mathematical foundations](@ref Mathematical-foundations) and [Architecture](@ref Architecture). Otherwise, follow the four steps below.

## 1. Define the state

A state in `Flows.jl` is any Julia object that satisfies two contracts: it can be `copy`-ed and `similar`-ed, and it supports in-place dot-broadcasting (`x .= a .* y .+ b .* z`). Plain Julia arrays satisfy both out of the box, so for our Lorenz state we use a length-3 `Vector{Float64}`:

```julia
x = zeros(3)
```

User-defined types (a `SpectralField`, a `Grid`, …) also work as long as they satisfy these two contracts. See [States and vector fields](@ref States-and-vector-fields).

## 2. Define the vector field

The vector field is any callable with signature

```julia
f(t::Real, x, dxdt) -> dxdt
```

It must overwrite `dxdt` with `dx/dt` evaluated at `(t, x)`. It must not mutate `x`. For Lorenz:

```julia
struct Lorenz
    ρ::Float64
end

function (eq::Lorenz)(t, x, dxdt)
    σ, β = 10.0, 8/3
    dxdt[1] = σ * (x[2] - x[1])
    dxdt[2] = eq.ρ * x[1] - x[2] - x[1]*x[3]
    dxdt[3] = -β * x[3] + x[1]*x[2]
    return dxdt
end
```

There is no requirement to use a `struct`: a plain function or a closure works equally well. The `struct` is useful when the right-hand side carries parameters or owns auxiliary buffers.

## 3. Build the flow operator

Bundle a vector field, an integration scheme, and a time-stepping policy into a [`Flow`](@ref):

```julia
using Flows

F = flow(Lorenz(28.0), RK4(zeros(3)), TimeStepConstant(1e-2))
```

The pieces:

- `Lorenz(28.0)` is the right-hand side.
- `RK4(zeros(3))` is the integration scheme. Passing `zeros(3)` tells the scheme to preallocate its internal stage buffers as `Vector{Float64}` of length `3`, matching the state.
- `TimeStepConstant(1e-2)` is the time-stepping policy: march with a fixed `Δt = 0.01`.

`F` is callable.

## 4. Run the integration

```julia
x = rand(3)
F(x, (0.0, 50.0))    # advances x in place from t=0 to t=50
```

`x` is overwritten with the state at `T = 50`. Nothing is allocated on the hot path — the scheme reuses its internal buffers, and the time loop walks an iterator-of-`(t, Δt)` pairs that has no heap footprint of its own.

## Compose with a monitor

To record samples along the trajectory, attach a [`Monitor`](@ref):

```julia
using LinearAlgebra

x   = rand(3)
mon = Monitor(x, (t, x) -> norm(x))   # observable = state norm
F(x, (0.0, 50.0), mon)

samples(mon)   # Vector{Float64} of norms, one per accepted step
times(mon)     # corresponding times
```

The observable is any callable `f(t, x)`. Its return value is what gets stored; the storage element type is inferred at monitor construction time.

## Compose with a storage

If you need to *replay* the trajectory later (e.g. to drive a linearised solve), use a [`RAMStorage`](@ref) instead of (or in addition to) a monitor:

```julia
x     = rand(3)
store = RAMStorage(x)
F(x, (0.0, 50.0), store)

times(store)    # sampled times
samples(store)  # state snapshots
```

A populated storage is callable: `store(out, t)` interpolates the trajectory at any `t` in `timespan(store)`, writing into the preallocated buffer `out`. This is the entry point for the continuous tangent / adjoint integrators of [Linearised dynamics](@ref Linearised-dynamics).

## Next steps

You now have the full picture of the "forward" interface. The rest of the manual unpacks each piece, and adds:

- IMEX schemes for stiff problems — see [Integration schemes](@ref Integration-schemes).
- Coupling several state components — see [Coupled systems](@ref Coupled-systems).
- Recording stage values for discretely consistent linearisation — see [Trajectory data](@ref Trajectory-data) and [Linearised dynamics](@ref Linearised-dynamics).
- Computing trajectory integrals — see [Quadrature equations](@ref Quadrature-equations).
- Worked end-to-end examples — see the [Cookbook](@ref Cookbook).
