# Cookbook

End-to-end worked examples. Each entry is self-contained: it states the problem, derives whatever mathematics is needed, sets up the `Flows.jl` flow, runs the integration, and reports the result.

The examples are arranged roughly by increasing complexity. The first three exercise the forward interface; the middle three exercise the coupled and linearised paths; the last two exercise symmetry reduction and stiff integration.

| # | Example                                                                   | Concepts                                                                      |
|---|---------------------------------------------------------------------------|-------------------------------------------------------------------------------|
| 1 | [Lorenz attractor — forward integration](@ref)                            | Flow construction, monitor, `times` / `samples`                               |
| 2 | [Lorenz with a running average via quadrature](@ref)                      | Coupled systems, quadrature equation                                          |
| 3 | [Custom adaptive time-step controller](@ref)                              | `AbstractTimeStepFromHook`                                                    |
| 4 | [Sensitivity to initial conditions of the Lorenz system](@ref)            | Coupled forward tangent path                                                  |
| 5 | [Lyapunov exponent of the Lorenz attractor](@ref)                         | Tangent + quadrature + re-normalisation via monitor                           |
| 6 | [Discrete adjoint sensitivity for an optimal-control problem](@ref)       | Stage cache, `DiscreteMode(true)`, `TimeStepFromCache`                        |
| 7 | [Continuous adjoint for a stiff diffusion problem](@ref)                  | `RAMStorage`, `ContinuousMode(true)`, `TimeStepFromStorage`, IMEX             |
| 8 | [Stiff diffusion with CNRK2](@ref)                                        | IMEX setup, `Diagonal` linear operator                                        |

The complete code for each example can be copy-pasted into a Julia REPL with `Flows.jl` installed.

---

## Lorenz attractor — forward integration

The canonical "Hello, world!" for a flow library. We define the Lorenz system and integrate it from a random initial condition, recording the trajectory.

```julia
using Flows, LinearAlgebra

struct Lorenz
    σ::Float64; ρ::Float64; β::Float64
end

function (eq::Lorenz)(t, x, dxdt)
    dxdt[1] = eq.σ * (x[2] - x[1])
    dxdt[2] = eq.ρ * x[1] - x[2] - x[1]*x[3]
    dxdt[3] = -eq.β * x[3] + x[1]*x[2]
    return dxdt
end

F   = flow(Lorenz(10.0, 28.0, 8/3),
           RK4(zeros(3)),
           TimeStepConstant(1e-2))

mon = Monitor(zeros(3), (t, x) -> copy(x))
x   = rand(3)
F(x, (0.0, 50.0), mon)

ts  = times(mon)
xs  = samples(mon)
```

`xs` is a `Vector{Vector{Float64}}` ready to be plotted; `ts` is the matching time vector.

---

## Lorenz with a running average via quadrature

We want the time-average $\langle x_3\rangle = (1/T)\int_0^T x_3(t)\,\mathrm{d}t$ along a Lorenz trajectory. The coupled-quadrature recipe yields a fourth-order accurate result with no quadrature-rule overhead.

```julia
quad(t, x, dxdt, I, dIdt) = (dIdt[1] = x[3]; return dIdt)

F = flow(couple(Lorenz(10.0, 28.0, 8/3), quad),
         RK4(couple(zeros(3), zeros(1))),
         TimeStepConstant(1e-2))

x = rand(3)
I = [0.0]
F(couple(x, I), (0.0, 100.0))

avg = I[1] / 100.0    # time-averaged ⟨x₃⟩
```

To monitor the running average $\bar g(\tau) = I(\tau)/\tau$ along the trajectory, attach a monitor that extracts the quadrature component:

```julia
mon = Monitor(couple(x, I), (t, xq) -> xq[2][1] / max(t, eps()))
F(couple(x, I), (0.0, 100.0), mon)

# samples(mon)[end]  →  the same `avg` as above
```

---

## Custom adaptive time-step controller

A constant time-step works for Lorenz at moderate $\rho$, but for some PDE problems a CFL-like constraint changes with the state. The example below caps the step at a target CFL number computed from the state norm.

```julia
struct CFLHook <: AbstractTimeStepFromHook
    cmax::Float64
end

(h::CFLHook)(g, A, z) = h.cmax / (maximum(abs, z) + eps())

F = flow(Lorenz(10.0, 28.0, 8/3),
         RK4(zeros(3)),
         CFLHook(0.5))

x = rand(3)
F(x, (0.0, 50.0))
```

The hook returns whatever `Δt` is appropriate for the *next* step; the integrator clamps it so the requested endpoint is hit exactly.

---

## Sensitivity to initial conditions of the Lorenz system

The flow Jacobian $J^{\tau}(\mathbf{x}_0)$ maps an initial-condition perturbation $\mathbf{y}_0$ to its propagated form $\mathbf{y}(t_0+\tau)$. The coupled-forward path computes $J^{\tau}\mathbf{y}_0$ in a single pass.

```julia
# tangent right-hand-side: f_y' = J_x f · y
function lorenz_tan(t, x, dxdt, y, dydt)
    σ, ρ, β = 10.0, 28.0, 8/3
    dydt[1] = σ * (y[2] - y[1])
    dydt[2] = (ρ - x[3]) * y[1] - y[2] - x[1] * y[3]
    dydt[3] = x[2] * y[1] + x[1] * y[2] - β * y[3]
    return dydt
end

F = flow(couple(Lorenz(10.0, 28.0, 8/3), lorenz_tan),
         RK4(couple(zeros(3), zeros(3))),
         TimeStepConstant(1e-3))

x = rand(3)
y = [1.0, 0.0, 0.0]                    # perturb x[1] direction
F(couple(x, y), (0.0, 1.0))

y    # ≈ J^{1.0}(x₀) · [1, 0, 0]
```

To get the full $3\times 3$ Jacobian, run the integration three times with three basis vectors for $\mathbf{y}_0$ and stack the results.

---

## Lyapunov exponent of the Lorenz attractor

The maximum Lyapunov exponent $\lambda_1$ measures the average exponential growth rate of an infinitesimal perturbation along the attractor:

```math
\lambda_1 = \lim_{T\to\infty}\frac{1}{T}\int_0^T \frac{\langle\mathbf{y},\,\mathcal{J}\mathbf{y}\rangle}{\langle\mathbf{y},\mathbf{y}\rangle}\,\mathrm{d}t,
```

with $\mathbf{y}(t)$ the tangent perturbation re-normalised to unit length frequently enough that it does not overflow.

`Flows.jl` makes this three-line:

- a primal (Lorenz),
- a tangent (linearised Lorenz),
- a quadrature recording the instantaneous growth rate,

re-normalising the tangent periodically inside a monitor callback.

```julia
function growth_rate(t, x, dxdt, y, dydt, λ, dλdt)
    # advance the primal
    lorenz_rhs(t, x, dxdt)                            # fill dxdt
    # advance the tangent
    lorenz_tan(t, x, dxdt, y, dydt)                    # fill dydt
    # instantaneous logarithmic growth: ⟨y, J·y⟩ / ⟨y, y⟩
    dλdt[1] = dot(y, dydt) / dot(y, y)
    return dλdt
end

deps = CallDependency((1,), (1, 2), (1, 2, 3))

F = flow(couple(lorenz_rhs, lorenz_tan, growth_rate), deps,
         RK4(couple(zeros(3), zeros(3), zeros(1))),
         TimeStepConstant(1e-3))

# normalise tangent every 100 steps to avoid overflow
mon = Monitor(couple(zeros(3), zeros(3), zeros(1)),
              (t, z) -> (normalize!(z[2]); 0.0);
              oneevery=100)

x, y, λ = rand(3), normalize!(rand(3)), [0.0]
F(couple(x, y, λ), (0.0, 500.0), mon)

λ_1 = λ[1] / 500.0       # ≈ 0.906 for ρ=28
```

This uses every coupled-system feature: the call-dependency override, the per-component RHS, and the monitor-as-callback re-normalisation.

---

## Discrete adjoint sensitivity for an optimal-control problem

A textbook adjoint sensitivity computation, derived in detail in the *adjoint sensitivity* section of [Mathematical foundations](foundations.md). The objective is

```math
\mathcal{J}(\mathbf{x}_0) = \int_0^T \sin\bigl(x(t)\bigr)\,\mathrm{d}t,
```

for the toy scalar problem $\dot x = x^2 + u(t)$, with the goal of computing $\mathrm{d}\mathcal{J}/\mathrm{d}u(t)$.

The discrete adjoint path uses a stage cache:

```julia
# 1. primal: cache the stages
cache = RAMStageCache(nstages(RK4), zeros(1))
F     = flow(toy_rhs, RK4(zeros(1)), TimeStepConstant(0.01))
x     = [1.0]
F(x, (0.0, 1.0), cache)

# 2. adjoint: terminal condition q(T) = 0, march backwards
A = flow(toy_adj,
         RK4(zeros(1), DiscreteMode(true)),
         TimeStepFromCache())
q = [0.0]
A(q, cache)              # q now contains the discrete adjoint over the grid

# 3. gradient: dJ/du(t) = -q(t)
grad = -q[1]
```

The discrete-adjoint identity guarantees that `grad` is the **exact** derivative of the discretised objective with respect to the discretised control. This is the property a Newton optimiser will reward you for.

---

## Continuous adjoint for a stiff diffusion problem

The setup is a one-dimensional heat equation $\partial_t u = \nu\,\partial_x^2 u + s(t, u)$ discretised in Fourier space, where the linear part is stiff and the source $s$ is mild. We want the gradient of $\mathcal{J} = \int_0^T \|u(t)\|^2\,\mathrm{d}t$ with respect to a control parameter that enters $s$.

The CB3R2R3e scheme handles the IMEX primal cheaply; the storage records step-end snapshots, and a `ContinuousMode(true)` flow integrates the adjoint backwards against the interpolated primal.

```julia
# primal in Fourier space
ν, N = 0.01, 64
L    = Diagonal(-ν .* (0:N-1).^2)         # stiff diffusive operator
s    = (t, û, dûdt) -> ...                # the explicit part
F    = flow(s, L, CB3R2R3e(zeros(ComplexF64, N)), TimeStepConstant(1e-2))

store = RAMStorage(zeros(ComplexF64, N); degree=5)
û     = randn(ComplexF64, N)
F(û, (0.0, 1.0), store)

# continuous adjoint: ContinuousMode(true) with TimeStepFromStorage
adj_rhs = (t, û_interp, q, dqdt) -> ...

A = flow(adj_rhs, L', CB3R2R3e(zeros(ComplexF64, N), ContinuousMode(true)),
         TimeStepFromStorage(1e-2))

q = zeros(ComplexF64, N)              # terminal condition
A(q, store, (1.0, 0.0))               # backward span signals reverse time
```

The adjoint flow uses **its own** time step (here, the same as the primal but it could differ) and interpolates the primal at every internal stage time of every step.

---

## Stiff diffusion with CNRK2

A self-contained example of using an IMEX scheme on a finite-difference discretisation of the heat equation. The stiff linear part is a tridiagonal Laplacian; the rest is a small nonlinear source.

For brevity we use `Diagonal` to stand in for the tridiagonal, since `Flows.jl` ships an [`ImcA!`](@ref) for it out of the box:

```julia
using LinearAlgebra

N   = 64
ν   = 1.0
k   = (0:N-1)
L   = Diagonal(-ν .* k.^2)                       # spectral diffusion
src = (t, û, dûdt) -> (dûdt .= 0.01 .* sin.(û); dûdt)

F   = flow(src, L, CNRK2(zeros(ComplexF64, N)), TimeStepConstant(1e-2))

û   = randn(ComplexF64, N)
F(û, (0.0, 1.0))
```

`Diagonal` is the only out-of-the-box linear operator. For non-diagonal stiff parts (a real tridiagonal Laplacian, a banded matrix, …) the user provides a `struct`, a `LinearAlgebra.mul!` method, and an `ImcA!` method that solves $(I - c\mathcal{L})\,\mathbf{z} = \mathbf{y}$. See [States and vector fields](states.md).

---

## Further reading

- [Mathematical foundations](foundations.md) for the equations behind the examples.
- [Linearised dynamics](linearised.md) for the full discussion of the discrete-vs-continuous adjoint choice.
- [Architecture](architecture.md) and [Internals](internals.md) for the implementation details exposed by these examples.
