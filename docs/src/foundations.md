# Mathematical foundations

This page sets up the notation and the core mathematical objects that the rest of the manual relies on. Everything here is standard material; the goal is to make the connection between mathematical statements and concrete `Flows.jl` constructs explicit, so the rest of the manual can move quickly.

If you only want to use the package and not derive anything, you can skim this page and come back to specific sections when the manual references them.

## State space and dynamical systems

A state space $\mathcal{X}$ is a Banach space in which the solution of a differential equation lives. In practice $\mathcal{X}$ is one of:

- $\mathbb{R}^n$ for a finite-dimensional ODE,
- a finite-dimensional approximation of a function space (e.g. Fourier coefficients of a PDE solution),
- a Cartesian product of any of the above for coupled systems.

A dynamical system on $\mathcal{X}$ is an evolution equation of the form

```math
\dot{\mathbf{x}}(t) = \mathcal{L}\,\mathbf{x}(t) + \mathbf{f}(t,\mathbf{x}(t)),\qquad \mathbf{x}(t_0) = \mathbf{x}_0\in\mathcal{X}
```

where:

- $\mathcal{L} : \mathcal{X}\to\mathcal{X}$ is a **linear, time-invariant** operator. It carries the stiff part of the dynamics (e.g. diffusion, dissipation, hyperviscosity).
- $\mathbf{f}: \mathbb{R}\times\mathcal{X}\to\mathcal{X}$ is the **nonlinear, possibly time-dependent** remainder.

If the problem has no stiffness, take $\mathcal{L}\equiv 0$ and fold everything into $\mathbf{f}$. If the problem is purely linear, take $\mathbf{f}\equiv 0$. The splitting is purely operational: it tells the integrator which parts to advance implicitly and which explicitly.

## The flow operator

The **flow operator** $\Phi^{\,\tau} : \mathcal{X}\to\mathcal{X}$ maps an initial condition forward by an interval $\tau$:

```math
\mathbf{x}(t_0 + \tau) = \Phi^{\,\tau}\,\mathbf{x}(t_0).
```

The two-parameter family $\{\Phi^{\,\tau}\}_{\tau\ge 0}$ inherits the **semigroup property**

```math
\Phi^{\,\tau_1+\tau_2} = \Phi^{\,\tau_2}\circ\Phi^{\,\tau_1},
```

which is the structural reason every algorithm that walks a trajectory in pieces (shooting, time-marching, sub-stepping) is expressible in terms of $\Phi^{\,\tau}$ alone.

A [`Flows.Flow`](@ref Flows.Flow) is a **discrete approximation** of $\Phi^{\,\tau}$: it is a callable object built around a one-step scheme `step!` and a time-step iterator that subdivides $\tau$.

```julia
F = flow(f, RK4(zeros(3)), TimeStepConstant(0.01))
F(x, (t₀, T))    # produces an approximation of Φ^{T - t₀} x in place
```

The integration error is determined by the order of the scheme and the step size, *not* by the size of the interval $\tau$ — the iterator simply concatenates many one-step calls.

## Time discretisation: Butcher form

`Flows.jl` builds every integration scheme out of one or two **Butcher tableaux**. For an explicit Runge–Kutta method with $s$ stages the update reads

```math
\begin{aligned}
\mathbf{Y}_k &= \mathbf{x}_n + \Delta t\sum_{j=1}^{k-1} a_{kj}\,\mathbf{f}(t_n + c_j\Delta t,\,\mathbf{Y}_j),\quad k = 1,\dots,s,\\
\mathbf{x}_{n+1} &= \mathbf{x}_n + \Delta t\sum_{k=1}^{s} b_k\,\mathbf{f}(t_n + c_k\Delta t,\,\mathbf{Y}_k),
\end{aligned}
```

with coefficients packed in a matrix $A=(a_{kj})$ and vectors $\mathbf{b}, \mathbf{c}$. An **embedded** weight vector $\mathbf{e}$ is also stored alongside $\mathbf{b}$ so that an error-estimating sibling scheme can be evaluated at no extra stage cost — useful for adaptive variants, even though the schemes shipped today are not themselves adaptive.

For **IMEX** schemes, two tableaux travel together — one for the implicit part of $\mathbf{f}=\mathcal{L}\mathbf{x}+\mathbf{f}_{\text{ex}}$ and one for the explicit part. The implicit stages take the form

```math
\mathbf{Y}_k = \mathbf{x}_n + \Delta t\sum_{j=1}^{k-1} a_{kj}^{\mathrm I}\,\mathcal{L}\mathbf{Y}_j + \Delta t\,a_{kk}^{\mathrm I}\,\mathcal{L}\mathbf{Y}_k + \Delta t\sum_{j=1}^{k-1} a_{kj}^{\mathrm E}\,\mathbf{f}_{\text{ex}}(\cdots),
```

which, rearranged, requires solving the linear problem

```math
(I - \Delta t\,a_{kk}^{\mathrm I}\,\mathcal{L})\,\mathbf{Y}_k = \mathbf{x}_n + \Delta t\sum_{j=1}^{k-1} a_{kj}^{\mathrm I}\,\mathcal{L}\mathbf{Y}_j + \Delta t\sum_{j=1}^{k-1} a_{kj}^{\mathrm E}\,\mathbf{f}_{\text{ex}}(\cdots).
```

The user supplies $\mathcal{L}$ via its action `mul!(out, A, x)` and a method for [`ImcA!`](@ref) that solves $(I - c\mathcal{L})\mathbf{z} = \mathbf{y}$. The constant $c$ depends on the scheme and the stage; both the schemes and the solver are oblivious to its value, so the user writes a single $c$-parameterised routine that the integrator drives.

The implemented tableaux are stored in `src/tableaux.jl` as exact rationals and converted to `Float64` once at module load. See [Integration schemes](schemes.md) for the catalogue.

## Tangent equations

Suppose the trajectory $\mathbf{x}(t)$ is perturbed at $t=t_0$ by a small $\mathbf{y}_0\in\mathcal{X}$. To first order in $\|\mathbf{y}_0\|$, the perturbation $\mathbf{y}(t)$ propagates according to the **tangent equation**

```math
\dot{\mathbf{y}}(t) = \mathcal{L}\,\mathbf{y}(t) + \mathbf{f}_{\mathbf{x}}\!\bigl(t,\mathbf{x}(t)\bigr)\cdot\mathbf{y}(t),\qquad \mathbf{y}(t_0) = \mathbf{y}_0,
```

with $\mathbf{f}_{\mathbf{x}}$ the Jacobian of $\mathbf{f}$ along the nonlinear trajectory $\mathbf{x}(t)$. This is a **linear, non-autonomous** problem whose coefficients depend on the primal solution.

The corresponding **flow Jacobian** is

```math
J^{\,\tau}(\mathbf{x}_0) \;=\; \frac{\partial\,\Phi^{\,\tau}}{\partial\mathbf{x}_0}(\mathbf{x}_0),
```

so that $\mathbf{y}(t_0+\tau) = J^{\,\tau}\!(\mathbf{x}_0)\,\mathbf{y}_0$. `Flows.jl` does not build $J^{\,\tau}$ as a matrix; instead it constructs a tangent flow operator whose action on $\mathbf{y}_0$ is the matrix–vector product $J^{\,\tau}\mathbf{y}_0$.

Two tangent integration paths are provided. They are *discrete* (replay the primal stages stored in a [`RAMStageCache`](@ref)) or *continuous* (interpolate the primal trajectory from a [`RAMStorage`](@ref)); see [Linearised dynamics](linearised.md) for the trade-offs.

## Adjoint equations

The **adjoint** of the tangent operator $J^{\,\tau}$ is, with respect to a chosen inner product $\langle\cdot,\cdot\rangle$ on $\mathcal{X}$, the unique linear operator $(J^{\,\tau})^{*}$ satisfying

```math
\bigl\langle\mathbf{p},\,J^{\,\tau}\mathbf{y}\bigr\rangle = \bigl\langle (J^{\,\tau})^{*}\mathbf{p},\,\mathbf{y}\bigr\rangle\quad\forall\mathbf{p},\mathbf{y}\in\mathcal{X}.
```

The adjoint state $\mathbf{p}(t)$ obeys the **adjoint equation**

```math
-\dot{\mathbf{p}}(t) = \mathcal{L}^{*}\mathbf{p}(t) + \bigl[\mathbf{f}_{\mathbf{x}}\!\bigl(t,\mathbf{x}(t)\bigr)\bigr]^{*}\mathbf{p}(t),
```

which differs from the tangent equation in three ways: it runs **backwards in time** (initial condition specified at $t=T$), it uses the **transpose of the Jacobian**, and it depends on the same primal trajectory $\mathbf{x}(t)$.

Adjoint methods are the workhorse of gradient-based optimisation in dynamical systems: one backward sweep yields the gradient of an objective $\mathcal{J}(\mathbf{x}_0)$ with respect to **all** components of $\mathbf{x}_0$ at the cost of roughly one extra forward integration, regardless of the dimension of $\mathcal{X}$. See the Lorenz adjoint sensitivity entry in the [Cookbook](cookbook.md) for an end-to-end derivation.

### Discrete vs continuous adjoints

A continuous adjoint is the analytical derivation above, discretised on the integrator's grid *after* the math is done. A discrete adjoint is what you get by mechanically transposing the integrator's update formulas. The two agree as $\Delta t\to 0$ but can differ at finite $\Delta t$.

`Flows.jl` supports both:

- [`ContinuousMode`](@ref Flows.ContinuousMode)`(true)` selects the continuous adjoint, driven by an interpolant over a [`RAMStorage`](@ref).
- [`DiscreteMode`](@ref Flows.DiscreteMode)`(true)` selects the discrete adjoint, driven by stage values stored in a [`RAMStageCache`](@ref).

When the downstream optimisation algorithm relies on the gradient being an *exact* derivative of the **discretised** objective (e.g. line searches with strict Armijo conditions, second-order methods), the discrete adjoint is the safer choice. Otherwise the continuous adjoint is usually fine and is much cheaper in memory when the storage degree is low.

## Quadrature integrals along a trajectory

Many quantities of interest are not values of the state at a single time but integrals along a trajectory:

```math
I(T) = \int_{t_0}^{T} g\bigl(\mathbf{x}(t)\bigr)\,\mathrm{d}t.
```

`Flows.jl` supports two ways of computing such integrals:

1. **Couple the quadrature into the system.** Append a new component $I(t)\in\mathbb{R}^m$ whose dynamics are $\dot{I}(t) = g(\mathbf{x}(t))$, and integrate the augmented state $(\mathbf{x},\,I)$ as a [`Coupled`](@ref). The integral converges at the order of the time integrator. This is the preferred path.
2. **Post-process samples.** Record $g(\mathbf{x}(t))$ in a [`Monitor`](@ref) and apply [`trapz`](@ref Flows.trapz) or [`simps`](@ref Flows.simps) at the end. Convenient when $g$ is decided after the integration is done, but the order of accuracy is fixed by the quadrature rule, not the time integrator.

The coupled path is treated in [Quadrature equations](quadrature.md).

## Symmetry transformations

Some dynamical systems are equivariant under a continuous group action: if $g_s : \mathcal{X}\to\mathcal{X}$ is a one-parameter family of symmetries indexed by $s$, then

```math
\Phi^{\,\tau}\bigl(g_s\,\mathbf{x}\bigr) = g_s\,\Phi^{\,\tau}\mathbf{x}\qquad\forall\,\tau,\,s.
```

A common use case is reducing a translation- or rotation-invariant problem by post-composing the flow with a symmetry parameterised by a phase $s$ chosen by the user — e.g. to track a relative periodic orbit.

`Flows.jl` accepts an optional callable `sym(x, s)` at flow-construction time that is applied to the post-integration state. The wrapper is [`Flows.SymTransform`](@ref) for single states and [`Flows.CoupledTransform`](@ref) for coupled states. See [Symmetry transformations](symmetry.md) for the workflow.

## Notation used in the rest of the manual

| Symbol               | Meaning                                                                       | Code object                                              |
|----------------------|-------------------------------------------------------------------------------|----------------------------------------------------------|
| $\mathcal{X}$        | State space                                                                   | the type of `x`                                          |
| $\mathbf{x},\mathbf{f},\mathcal{L}$ | State, vector field, linear operator                           | `x`, `f`, `A`                                            |
| $\Phi^{\,\tau}$      | True flow operator                                                            | `F::Flow` is the discrete approximation                  |
| $\Delta t$           | Time step                                                                     | `ts.Δt` for `TimeStepConstant(Δt)`                       |
| $\mathbf{Y}_k$       | $k$-th internal stage                                                         | element of `method.store`                                |
| $a_{kj}, b_k, c_k$   | Tableau coefficients                                                          | `tab[:a, k, j]`, `tab[:b, k]`, `tab[:c, k]`              |
| $\mathbf{y},\mathbf{p}$ | Tangent and adjoint states                                                 | `y`, `q` in user code                                    |
| $J^{\,\tau}$         | Flow Jacobian                                                                 | not constructed; only its action is computed             |
| $I(T)$               | Trajectory integral                                                           | coupled component or post-hoc `trapz`/`simps`            |
| $g_s$                | Symmetry action with parameter $s$                                            | `sym(x, s)` callable wrapped in `SymTransform`           |

With the equations and the mapping to code in hand, the next page — [Architecture](architecture.md) — describes how these pieces fit together inside a single [`Flows.Flow`](@ref Flows.Flow) object.
