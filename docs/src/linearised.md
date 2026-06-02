# Linearised dynamics

This page is the most important conceptual chapter in the manual: it explains, in depth, the multiple ways `Flows.jl` lets you integrate the tangent and adjoint of a nonlinear dynamical system, when to prefer each, and why.

Linearisation problems are a recurring need in dynamical-systems theory and PDE-constrained optimisation. Concrete uses include

- **Sensitivity analysis** with respect to initial conditions or parameters;
- **Gradient computation** of trajectory-integrated objective functionals;
- **Stability analysis** (Floquet, Lyapunov exponents);
- **Newton–Krylov continuation** of periodic orbits and equilibria;
- **Optimal control** via adjoint-based gradient descent;
- **Variational data assimilation** (4D-Var).

What sets `Flows.jl` apart is that it exposes **four distinct integration paths** for these linearisations, each with a different trade-off between consistency, memory, and accuracy. This page maps the four paths to the underlying mathematics, derives the difference between them, and offers concrete guidance on which to pick for which problem.

## The two equations

We assume the reader has read [Mathematical foundations](@ref Mathematical-foundations); we briefly restate the equations for reference.

Let the primal nonlinear system be

```math
\dot{\mathbf{x}}(t) = \mathcal{L}\mathbf{x}(t) + \mathbf{f}(t,\mathbf{x}(t)), \qquad \mathbf{x}(t_0) = \mathbf{x}_0.
```

### Tangent equation

A first-order perturbation $\mathbf{y}(t)$ of the primal obeys the linear, non-autonomous equation

```math
\dot{\mathbf{y}}(t) = \mathcal{L}\mathbf{y}(t) + \mathbf{f}_{\mathbf{x}}(t,\mathbf{x}(t))\,\mathbf{y}(t), \qquad \mathbf{y}(t_0) = \mathbf{y}_0.
```

It runs **forward** in time and depends on the primal trajectory through the Jacobian $\mathbf{f}_{\mathbf{x}}$.

### Adjoint equation

The adjoint state $\mathbf{p}(t)$ obeys

```math
-\dot{\mathbf{p}}(t) = \mathcal{L}^{*}\mathbf{p}(t) + \bigl[\mathbf{f}_{\mathbf{x}}(t,\mathbf{x}(t))\bigr]^{*}\mathbf{p}(t), \qquad \mathbf{p}(T) = \mathbf{p}_T.
```

It runs **backward** in time, uses the **transpose** Jacobian, and again depends on the primal trajectory.

Both equations are **linear in the linearisation state** and **non-autonomous in time** through their dependence on the primal trajectory. The numerical difficulty is therefore entirely in *how the primal trajectory is communicated to the linearised integrator*. This is where `Flows.jl` offers a choice.

## The four integration paths

Every linearised integration path in `Flows.jl` is one of the four cells in the following 2×2:

|                          | **Tangent (forward)**                                | **Adjoint (backward)**                              |
|--------------------------|------------------------------------------------------|-----------------------------------------------------|
| **Discretely consistent** (replay stages) | `ContinuousMode(false)` → `DiscreteMode(false)` | `DiscreteMode(true)`                              |
| **Continuous**           | `ContinuousMode(false)`                              | `ContinuousMode(true)`                              |

There is also a **fifth** path: coupled forward integration of the primal and a tangent perturbation together as a `Coupled{2}` state. We discuss it in the section [Coupled (forward-mode tangent)](@ref) below because it has a different cost profile from the four above.

The five paths fall on a continuum of *what is communicated*:

```
   primal trajectory stored as          coupled forward             ← cheapest in passes
                                       integration of (x, y)
                                       (no stored trajectory)
       ─────────────────────────────────────────────────────────────
                                       continuous, RAMStorage
       ─────────────────────────────────────────────────────────────
                                       discrete, RAMStageCache
                                                                    ← most memory
```

The rest of the page discusses each in turn.

## Coupled (forward-mode tangent)

The simplest and most allocation-frugal path. Pair the primal $\mathbf{f}$ with the tangent right-hand-side $\mathbf{f}_{\mathbf{x}}\,\mathbf{y}$ and integrate the *augmented* state $(\mathbf{x},\mathbf{y})$ as one [`Coupled`](@ref):

```math
\frac{\mathrm{d}}{\mathrm{d}t}\begin{pmatrix}\mathbf{x}\\\mathbf{y}\end{pmatrix} =
\begin{pmatrix}\mathcal{L}\mathbf{x} + \mathbf{f}(t,\mathbf{x})\\\mathcal{L}\mathbf{y} + \mathbf{f}_{\mathbf{x}}(t,\mathbf{x})\,\mathbf{y}\end{pmatrix}.
```

In `Flows.jl`:

```julia
f_primal(t, x, dxdt)            = …
f_tangent(t, x, dxdt, y, dydt)  = …    # default call dep: sees (x, y)

F = flow(couple(f_primal, f_tangent),
         RK4(couple(zeros(3), zeros(3))),
         TimeStepConstant(0.01))

x = rand(3)
y = ones(3)
F(couple(x, y), (0.0, 1.0))
```

| Property                          | Value                                                       |
|-----------------------------------|-------------------------------------------------------------|
| Direction                         | Forward                                                     |
| Consistency with primal           | Exact (same scheme, same step, same time)                   |
| Memory cost                       | One extra full-state buffer per stage of the scheme         |
| Forward / backward passes         | One forward pass                                            |
| Adjoint capability                | **None.** This path only computes tangent perturbations.    |
| Use when                          | You want the action of $J^{\tau}(\mathbf{x}_0)$ on one (or a few) initial perturbations |

The coupled path is the right answer when:

- you have a forward sensitivity to compute,
- the number of perturbations you care about is small (one or two, say),
- you do not need an adjoint (i.e. you do not need the gradient of a scalar with respect to *all* components of $\mathbf{x}_0$).

It is **not** the right answer for gradient computation over high-dimensional initial conditions: each independent perturbation costs another tangent integration, while a single adjoint pass computes the full gradient.

### Coupled with quadrature

The coupled idiom extends naturally to trajectory integrals: couple in a quadrature component that integrates whatever observable you need. The cookbook's *Lyapunov exponent of the Lorenz attractor* example pairs a primal, a tangent, and a logarithmic quadrature, and reads the Lyapunov estimate out of the quadrature component.

## Discrete tangent and adjoint

The **discretely-consistent** path is the rigorous one. The idea is to run the primal once, record **every internal stage value of every step** into a [`RAMStageCache`](@ref), and then drive the linearised flow by *replaying* the exact same sequence of `(t, Δt, stages)` triples that the primal saw.

### Tangent: replay forwards

```julia
# 1. primal: cache the stages
cache = RAMStageCache(nstages(RK4), zeros(3))
F     = flow(f, RK4(zeros(3)), TimeStepConstant(0.01))
x     = rand(3); F(x, (0.0, 1.0), cache)

# 2. tangent: replay forwards
y = randn(3)
T = flow(f_tan,
         RK4(zeros(3), DiscreteMode(false)),
         TimeStepFromCache())
T(y, cache)              # marches y through cache.ts in order
```

The tangent right-hand-side receives the cached stage values explicitly:

```julia
f_tan(t, x_stage, y, dydt)   #  x_stage is one of the cached stages
```

i.e. the integrator hands the linearised step exactly the primal stage at which the Jacobian should be evaluated. This is what makes the scheme discretely consistent.

### Adjoint: replay backwards

```julia
q = zeros(3)              # terminal condition
A = flow(f_adj,
         RK4(zeros(3), DiscreteMode(true)),
         TimeStepFromCache())
A(q, cache)               # marches q through cache.ts in reverse
```

The adjoint right-hand-side has the same signature as the tangent one, but the schemes invert their internal stage order so that the discrete update is the exact transpose of the primal update at each step.

### Why this matters

The *discrete-adjoint identity* holds: for any inner product $\langle\cdot,\cdot\rangle$ on $\mathcal{X}$,

```math
\bigl\langle\mathbf{p}_0,\,\mathbf{y}_T\bigr\rangle = \bigl\langle\mathbf{q}_0,\,\mathbf{y}_0\bigr\rangle,
```

where $\mathbf{y}_T$ is the discrete-tangent state at $T$ starting from $\mathbf{y}_0$, and $\mathbf{q}_0$ is the discrete-adjoint state at $0$ starting from terminal $\mathbf{p}_0$ at $T$. The identity is exact at the level of floating-point: no $O(\Delta t^{p})$ slack.

This matters for two classes of algorithms:

1. **Newton–Krylov solvers** that use the action of the discrete Jacobian *and* its transpose interchangeably. A discrete inconsistency between the two breaks the symmetry of the bilinear form and degrades the Krylov convergence.
2. **Line-search optimisers** with strict Armijo conditions or trust-region updates. They demand that the gradient be the exact derivative of the *discretised* objective, not of the continuous objective evaluated on the discrete state.

### Cost

| Property                          | Value                                                       |
|-----------------------------------|-------------------------------------------------------------|
| Direction                         | Forward (tangent) or backward (adjoint)                     |
| Consistency with primal           | **Exact**, including floating-point round-off               |
| Memory cost                       | `NS × num_steps` full-state buffers in the cache            |
| Forward / backward passes         | One primal + one linearised pass                            |
| Use when                          | You need the adjoint identity to hold exactly               |

The memory cost is the dominant concern. For an integration of $N$ steps with a scheme of $\mathrm{NS}$ stages on a state of size $|\mathcal{X}|$, the cache stores roughly $N\cdot\mathrm{NS}\cdot|\mathcal{X}|$ floats. For DNS-scale problems this is impractical, which leads to the next path.

## Continuous tangent and adjoint

The **continuous** path approximates the primal trajectory by an interpolant of step-end snapshots, and integrates the linearised equation against that interpolant. It is the right choice when the state is too large to admit the stage cache.

### Setup

```julia
# 1. primal: record step-end snapshots
store = RAMStorage(zeros(3); degree=5)
F     = flow(f, RK4(zeros(3)), TimeStepConstant(0.01))
x     = rand(3); F(x, (0.0, 1.0), store)
```

The storage uses a Lagrange polynomial of the configured `degree` to interpolate between samples. Degree `5` is a good default; higher degree improves accuracy at the cost of stencil width near the boundaries (see [Trajectory data](@ref Trajectory-data)).

### Tangent: forward against the interpolant

```julia
y = randn(3)
T = flow(f_tan_cont,
         RK4(zeros(3), ContinuousMode(false)),
         TimeStepFromStorage(0.005))                  # may differ from primal Δt
T(y, store, (0.0, 1.0))
```

The continuous tangent right-hand-side receives the interpolated primal state as an extra argument:

```julia
f_tan_cont(t, x_interp, y, dydt)
#                ^^^^^^^^ interpolated from the storage by the scheme
```

The scheme calls `store(buf, t_stage)` internally to fill `x_interp` at each stage time before evaluating the user's right-hand-side.

### Adjoint: backwards against the interpolant

```julia
q = zeros(3)
A = flow(f_adj_cont,
         RK4(zeros(3), ContinuousMode(true)),
         TimeStepFromStorage(0.005))
A(q, store, (1.0, 0.0))                              # decreasing span
```

The integrator detects the decreasing span and flips the sign of `Δt` internally. The interpolation is the same as in the tangent case; only the marching direction and the user-supplied right-hand-side change.

### What "continuous" sacrifices

There is no discrete-adjoint identity for this path. Two errors enter:

1. **Interpolation error.** The Lagrange interpolant of degree $d$ has an $O(\Delta t^{d+1})$ error at non-storage times. The linearised scheme sees, in effect, a perturbed primal.
2. **Time-step mismatch.** The linearised flow has its own `Δt_{\text{lin}}`, which need not match `Δt_{\text{primal}}`. The tangent and adjoint paths therefore have independent error budgets.

In practice neither error is fatal for *sensitivity analysis* — both are of the same order as the discretisation error of the primal itself. They *are* fatal for the exact-discrete-adjoint use cases above.

### Cost

| Property                          | Value                                                       |
|-----------------------------------|-------------------------------------------------------------|
| Direction                         | Forward (tangent) or backward (adjoint)                     |
| Consistency with primal           | $O(\Delta t^{d+1})$ interpolation error                     |
| Memory cost                       | `num_steps` full-state snapshots in the storage             |
| Forward / backward passes         | One primal + one linearised pass                            |
| Use when                          | The state is too large for a stage cache                    |

The memory cost is independent of the scheme's stage count, which is the practical reason this path exists.

## Trade-offs in one table

|                                 | Coupled forward             | Discrete (cache)            | Continuous (storage)         |
|---------------------------------|------------------------------|------------------------------|------------------------------|
| Forward passes                  | 1 (combined)                 | 1                            | 1                            |
| Backward passes                  | n/a                          | 1                            | 1                            |
| Memory (per primal step)        | `(1 + 1) × scheme stage buf` | `NS × full state`            | `1 × full state`             |
| Discrete-adjoint identity       | n/a                          | **Exact**                    | $O(\Delta t^{d+1})$ slack    |
| Linearised `Δt`                 | = primal `Δt`                | = primal `Δt`                | independent                  |
| Suitable for adjoint?           | No                           | Yes                          | Yes                          |
| Suitable for many tangents?     | Yes (couple them)            | Yes (replay the cache N×)    | Yes                          |
| Typical use                     | Sensitivity to a few directions  | Newton–Krylov, optimisation | DNS-scale gradient computation |

## Practical guidance

A few decision rules that work in practice:

1. **Single tangent, no adjoint needed.** Use the coupled forward path. One integration, no extra memory, exact consistency by construction.
2. **Few tangents (≤ 5) over short trajectories.** Still the coupled path, with the additional perturbations as further coupled components.
3. **Single adjoint over a moderate trajectory.** Use the **discrete** (stage cache) path. The memory is bearable, and the discrete-adjoint identity may save you a Newton–Krylov debugging session later.
4. **Single adjoint over a very long trajectory or with a huge state.** Use the **continuous** (storage) path with a degree-5 interpolant. The interpolation error is dominated by the time-stepping error in any reasonable setup.
5. **Periodic-orbit shooting.** Continuous storage path with `period = T` on the storage; the periodic interpolator wraps endpoints cleanly.

When in doubt, prototype with the discrete path on a small problem (where memory is no concern), verify that the gradient passes a finite-difference sanity check, and then switch to the continuous path on the full problem if memory demands it.

## What you write and what the package does

Regardless of the path, you write **two** small pieces of code:

- a primal right-hand-side, and
- a linearised right-hand-side whose signature depends on the path.

| Path                  | Linearised RHS signature                                                                    | Primal-state argument                                |
|-----------------------|---------------------------------------------------------------------------------------------|------------------------------------------------------|
| Coupled forward       | `f_tan(t, x, dxdt, y, dydt)`                                                                | The current `x`, from the coupled call dependency.   |
| Discrete tangent/adj. | `f_lin(t, x_stage, y, dydt)`                                                                | The cached *stage value* at the relevant stage time.|
| Continuous tangent/adj. | `f_lin(t, x_interp, y, dydt)`                                                              | The interpolant of the storage at the stage time.    |

Inside, the package handles:

- Building the time grid (or replaying a cached one).
- Calling the primal interpolant or unpacking stage tuples.
- Inverting the time direction for adjoint modes.
- Maintaining allocation-free in-place updates.

You **never** write a Jacobian explicitly. You write the *action* of the Jacobian on a perturbation vector, which is what the linearised right-hand-side does.

## Where this is in the code

| File                         | What it implements                                                       |
|------------------------------|--------------------------------------------------------------------------|
| `src/steps/shared.jl`        | `NormalMode`, `ContinuousMode`, `DiscreteMode`; `isadjoint(MODE)`.       |
| `src/steps/rk4.jl`           | Three `step!` overloads for RK4 — normal, continuous, discrete.          |
| `src/steps/CNRK2.jl`         | Same three for CNRK2.                                                    |
| `src/steps/CB3R2R.jl`        | Three (or four) per scheme for CB3R2R\*.                                  |
| `src/integrator.jl`          | The three propagation loops driven by `TimeStepFromCache`, `TimeStepFromStorage`, and the constant-step normal loop. |
| `src/stagecache.jl`          | `RAMStageCache` and its push contract.                                   |
| `src/storage.jl`             | `RAMStorage` and its Lagrange interpolator.                              |

## Worked examples

The cookbook contains several worked examples that put these paths into practice:

- *Sensitivity to initial conditions of the Lorenz system* — coupled forward path.
- *Lyapunov exponent of the Lorenz attractor* — coupled forward + quadrature.
- *Discrete adjoint sensitivity for an optimal-control problem* — discrete path.
- *Continuous adjoint for a stiff diffusion problem* — continuous path.

See the [Cookbook](@ref Cookbook).

## Cross-references

- [Mathematical foundations](@ref Mathematical-foundations) — the equations and the adjoint identity.
- [Trajectory data](@ref Trajectory-data) — `RAMStorage` and `RAMStageCache` details.
- [Time stepping](@ref Time-stepping) — `TimeStepFromCache` and `TimeStepFromStorage` policies.
- [Coupled systems](@ref Coupled-systems) — the call-dependency machinery used by the coupled forward path.
