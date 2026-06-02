# Flows.jl

```@raw html
<p align="center">
  <img src="assets/logo.svg" alt="Flows.jl logo" width="640">
</p>
```

A composable, allocation-aware, Julia-native framework for **flow-style numerical integration** of ordinary differential equations and semi-discretised partial differential equations.

```math
\mathbf{x}(T) = \Phi^{\,T-t_0}\mathbf{x}(t_0)
```

`Flows.jl` is built around the idea that the natural primitive of many algorithms in dynamical systems theory is **the flow operator itself** — a callable object that maps a state forward by a finite amount of time. Algorithms that fit this idiom include shooting methods, Newton–Krylov continuation of periodic orbits, Floquet analysis, Lyapunov exponent estimation, optimal-control / adjoint-based sensitivity, model-order reduction, and direct numerical simulation drivers. Each of these is more conveniently expressed in terms of repeated, composable calls to a flow than in terms of an opaque "solve-and-return-trajectory" routine.

## What this package is

| Feature | Outcome |
|---|---|
| Callable [`Flows.Flow`](@ref Flows.Flow) objects                  | `F(x, (t₀, T))` advances `x` in place. Composable. |
| Allocation-free hot path                         | Internal stage buffers are preallocated at scheme construction. |
| Explicit and IMEX integrators                    | RK4 plus the low-storage Cavaglieri–Bewley IMEX family. |
| Coupled-state support                            | `Coupled` carries `N` heterogeneous components and broadcasts elementwise. |
| Tangent and adjoint linearisations               | Discretely consistent (stage-cache) and continuous (storage-based). |
| Symmetry handling                                | Optional in-place symmetry transformation applied after each flow call. |
| Trajectory data structures                       | Monitors, storages (with Lagrange interpolation), stage caches. |
| Quadrature-via-coupling                          | Trajectory integrals are first-class, marched alongside the state. |

## What this package is *not*

It is not a general-purpose ODE/DAE/SDE solver. It does not provide event detection, jump processes, automatic algorithm switching, callback chains, or symbolic problem definitions. The intended user already knows their problem class and the scheme they want to use, and just needs a fast, predictable way to build and compose flow operators.

For broad, batteries-included differential-equation solving in Julia, see [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl). `Flows.jl` is deliberately narrower and more imperative.

## Installation

`Flows.jl` is not registered in the Julia `General` registry. Install directly from GitHub: hit `]` to enter Pkg mode, then

```julia
add https://github.com/Davide-Lasagna-s-Lab/Flows.jl
```

The package depends only on `LinearAlgebra`, `MacroTools`, `DataStructures`, and `Printf` from the standard library / minimal ecosystem, so it is fast to install and has no heavy transitive dependencies.

## Reading guide

The documentation is organised as four layers. Read the first two in order; the rest is reference-style and can be browsed on demand.

| Where to read | Why |
|---|---|
| [Quick start](quickstart.md)                                   | Land a working integration in five minutes. |
| [Mathematical foundations](foundations.md)         | Equation classes, flow operators, linearisations, the IMEX splitting. |
| [Architecture](architecture.md)                                 | How `Flow`, `System`, `AbstractMethod`, `AbstractTimeStepping`, and the symmetry wrapper fit together. |
| [States and vector fields](states.md)         | Requirements on user types, and what callables the package expects. |
| [Integration schemes](schemes.md)                   | RK4, CNRK2, and the Cavaglieri–Bewley low-storage IMEX family. |
| [Time stepping](time-stepping.md)                               | Constant, adaptive, stage-cache-driven, storage-driven. |
| [Coupled systems](coupled.md)                           | Heterogeneous multi-component states and call dependencies. |
| [Trajectory data](trajectories.md)                           | Monitors, storages, stage caches, and how they relate. |
| [Linearised dynamics](linearised.md)                   | Tangent and adjoint integration, discrete vs continuous. |
| [Symmetry transformations](symmetry.md)         | Equivariant flows under a continuous group action. |
| [Quadrature equations](quadrature.md)                 | Trajectory integrals as coupled components, plus post-hoc rules. |
| [Cookbook](cookbook.md)                                         | End-to-end worked examples. |
| [Internals](internals.md)                                       | Implementation details and design rationale. |
| [API](api.md)                                       | Docstrings for every exported symbol. |

## Citation

If you use `Flows.jl` in academic work, please cite the package via its GitHub URL until a software paper is available.
