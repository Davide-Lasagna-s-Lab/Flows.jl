# Integration schemes

`Flows.jl` ships nine concrete schemes: two explicit Runge–Kutta methods ([`RK4`](@ref), [`RK8`](@ref)), five implicit-explicit (IMEX) methods, and two exponential time differencing methods ([`ETDRK4`](@ref), [`ETDRK2`](@ref)). They all share the same constructor shape and the same `step!` contract, so a flow is built by picking the scheme that matches the *stiffness* and the *accuracy* of the problem at hand. The schemes themselves are dumb data carriers — they hold preallocated stage buffers and implement `step!` — and they compose freely with any of the [Time stepping](time-stepping.md) policies.

This page catalogues the schemes, gives the trade-offs that should drive the choice between them, and documents the memory footprint of each.

## How to pick

| Question                                                                | Answer                                       |
|-------------------------------------------------------------------------|----------------------------------------------|
| Is there a stiff linear part?                                           | If yes, an IMEX or an exponential scheme; if no, [`RK4`](@ref) or [`RK8`](@ref). |
| Do you need the simplest possible IMEX scheme to validate against?      | [`CNRK2`](@ref) — predictor–corrector second order. |
| Do you need an IMEX scheme with **minimal memory footprint** for large states? | One of the **CB low-storage** family below.   |
| Do you need fourth-order accuracy?                                      | [`RK4`](@ref) (explicit), [`CB4R3R4`](@ref) (IMEX) or [`ETDRK4`](@ref) (exponential). |
| Do you need eighth-order accuracy on a non-stiff problem?               | [`RK8`](@ref). |
| Do you need to drive a discretely-consistent tangent/adjoint solve?     | Any of them — every scheme has `DiscreteMode(false)` and `DiscreteMode(true)`. |

If you are not sure, start with [`RK4`](@ref) (no stiff part) or [`CB3R2R3e`](@ref) (with a stiff part). Both are robust default choices.

## Common interface

Every scheme has the same constructor shape:

```julia
SCHEME(x::X, mode::AbstractMode = NormalMode()) -> SCHEME{X, MODE, NX}
```

| Argument | Role                                                                                                                                  |
|----------|---------------------------------------------------------------------------------------------------------------------------------------|
| `x`      | Template object used to preallocate all internal stage buffers as `similar(x)`. Locks the state type for the lifetime of the scheme.   |
| `mode`   | An [`Flows.AbstractMode`](@ref Flows.AbstractMode) tag selecting forward / tangent / adjoint behaviour. Default is [`NormalMode`](@ref Flows.NormalMode). |

The mode controls **only** how many buffers are allocated and which `step!` method is dispatched. Constructing the same scheme with different modes is the standard way to obtain a primal flow and its tangent/adjoint companions:

```julia
primal  = flow(f, RK4(zeros(3)),                            TimeStepConstant(0.01))
tangent = flow(f, RK4(zeros(3), DiscreteMode(false)),       TimeStepFromCache())
adjoint = flow(f, RK4(zeros(3), DiscreteMode(true)),        TimeStepFromCache())
```

The schemes are also available by name, in two dictionaries from a `Symbol` to the constructor: `imex_schemes`, the schemes of a system split into a linear part and the rest (`:CNRK2`, `:CB3R2R2`, `:CB3R2R3e`, `:CB3R2R3c`, `:CB4R3R4`, `:ETDRK2`, `:ETDRK4`), and `available_schemes`, those plus the explicit `:RK4` and `:RK8`. So `imex_schemes[:CB3R2R3e](x)` is `CB3R2R3e(x)`. A package that lets its user name the scheme checks the name against them.

Every scheme provides `NS` (stage count) via `nstages(method)`. This is the number a stage cache must be configured with, and is matched at integration time.

## RK4 — classical explicit

Classical four-stage, fourth-order Runge–Kutta. The canonical default for non-stiff problems.

```math
\begin{aligned}
\mathbf{k}_1 &= \mathbf{f}(t,\,\mathbf{x}_n),\\
\mathbf{k}_2 &= \mathbf{f}(t + \tfrac{\Delta t}{2},\,\mathbf{x}_n + \tfrac{\Delta t}{2}\mathbf{k}_1),\\
\mathbf{k}_3 &= \mathbf{f}(t + \tfrac{\Delta t}{2},\,\mathbf{x}_n + \tfrac{\Delta t}{2}\mathbf{k}_2),\\
\mathbf{k}_4 &= \mathbf{f}(t + \Delta t,\,\mathbf{x}_n + \Delta t\,\mathbf{k}_3),\\
\mathbf{x}_{n+1} &= \mathbf{x}_n + \tfrac{\Delta t}{6}\bigl(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4\bigr).
\end{aligned}
```

| | |
|---|---|
| Type        | Explicit Runge–Kutta            |
| Order       | 4                               |
| Stages      | 4                               |
| Stage buffers | 5 (`NormalMode` and `DiscreteMode`), 6 (`ContinuousMode`) |
| Stage cache `NS` | 4                          |

Use `RK4` whenever the linear part is absent or its contribution to the stability constraint is comparable to the nonlinear part. For convection-dominated problems, the explicit-CFL constraint is essentially the same as for any other classical RK scheme.

## RK8 — explicit, eighth order

[`RK8`](@ref) is the eleven-stage, eighth-order explicit Runge–Kutta method of Cooper and Verner (1972). It is a fixed-step formula, with no error estimator. It has the same modes as `RK4`, and the discrete adjoint accepts a forcing weighted consistently with the quadrature of the method.

| | |
|---|---|
| Type        | Explicit Runge–Kutta            |
| Order       | 8                               |
| Stages      | 11                              |
| Stage buffers | 12 (`NormalMode` and the discrete tangent), 13 (`ContinuousMode`), 14 (the discrete adjoint) |
| Stage cache `NS` | 11                         |


## CNRK2 — Crank–Nicolson / Heun predictor–corrector

A two-stage second-order IMEX scheme: predictor with explicit Euler on $\mathbf{f}_\mathrm{ex}$ and trapezoidal on $\mathcal{L}$, corrector with Heun on $\mathbf{f}_\mathrm{ex}$ and trapezoidal on $\mathcal{L}$. Algebraically:

```math
\begin{aligned}
\tilde{\mathbf{x}} &= \bigl(I - \tfrac{\Delta t}{2}\mathcal{L}\bigr)^{-1}\!\!\bigl[(I + \tfrac{\Delta t}{2}\mathcal{L})\mathbf{x}_n + \Delta t\,\mathbf{f}_\mathrm{ex}(t,\mathbf{x}_n)\bigr],\\
\mathbf{x}_{n+1} &= \bigl(I - \tfrac{\Delta t}{2}\mathcal{L}\bigr)^{-1}\!\!\bigl[(I + \tfrac{\Delta t}{2}\mathcal{L})\mathbf{x}_n + \tfrac{\Delta t}{2}\,\mathbf{f}_\mathrm{ex}(t,\mathbf{x}_n) + \tfrac{\Delta t}{2}\,\mathbf{f}_\mathrm{ex}(t+\Delta t,\,\tilde{\mathbf{x}})\bigr].
\end{aligned}
```

| | |
|---|---|
| Type        | IMEX, predictor–corrector       |
| Order       | 2                               |
| Implicit stages | 2 (one per [`ImcA!`](@ref) solve) |
| Stage buffers | 5                              |
| Stage cache `NS` | 2                          |

`CNRK2` is the right choice when you want the simplest IMEX scheme to validate a stiff problem before moving to one of the higher-order CB variants, or when third-order accuracy is not needed and you want the smallest stage count. It is also the cheapest IMEX option per step in terms of $\mathbf{f}_\mathrm{ex}$ evaluations.


## Cavaglieri–Bewley low-storage IMEX schemes

Four schemes (`CB3R2R2`, `CB3R2R3e`, `CB3R2R3c`, `CB4R3R4`) from Cavaglieri & Bewley (2015) [^CB]: a family of low-storage IMEX Runge–Kutta methods explicitly designed for spatially-discretised PDE solvers where the state is very large and memory bandwidth dominates the cost. The "low-storage" qualifier refers to the very small number of full-state buffers each scheme owns at runtime — three or four — independent of the stage count. The trade-off is that the schemes are derived to admit a specific buffer-recycling pattern; their tableaux are *not* arbitrary.

The constants `CB2`, `CB3e`, `CB3c`, `CB4` in `src/tableaux.jl` hold the tableaux in exact rational form; they are converted once to `Float64` at module load.

[^CB]: Cavaglieri, D. and Bewley, T., 2015. *Low-storage implicit/explicit Runge–Kutta schemes for the simulation of stiff high-dimensional ODE systems*. Journal of Computational Physics, 286, pp. 172–193.

The four schemes are summarised in the table below.

| Scheme                  | Order | Stages | Register pattern | Buffers (normal) | Buffers (continuous) | Buffers (discrete adjoint) |
|-------------------------|-------|--------|------------------|------------------|----------------------|---------------------------|
| [`CB3R2R2`](@ref)       | 2     | 3      | 3R, 2R           | 3                | 4                    | 6                         |
| [`CB3R2R3e`](@ref)      | 3     | 4      | 3R, 2R           | 3                | 4                    | 6                         |
| [`CB3R2R3c`](@ref)      | 3     | 4      | 3R, 2R           | 3                | 4                    | 6                         |
| [`CB4R3R4`](@ref)       | 4     | 6      | 4R, 3R           | 4                | 5                    | 9                         |

"3R, 2R" means three registers for the implicit part and two for the explicit part — the explanation of the register accounting is in the Cavaglieri–Bewley paper. From the user's point of view, the takeaway is that the per-step memory cost of CB3R2R* is **three state-sized buffers** plus the auxiliary vectors needed for the chosen mode, regardless of stage count.

### CB3R2R2 — second-order, three-stage

The cheapest of the family and the IMEX equivalent of `CNRK2` in terms of accuracy. Pick `CB3R2R2` over `CNRK2` when you want the low-storage register pattern and a slightly larger stability domain, at the price of one extra stage and an extra `ImcA!` solve per step.


### CB3R2R3e — third-order, four-stage ("e" variant)

The standard recommendation as a robust third-order IMEX scheme for spatially-discretised PDEs. The "e" subscript denotes one of two third-order tableaux in the original paper.


### CB3R2R3c — third-order, four-stage ("c" variant)

The companion to `CB3R2R3e`, with a slightly different coefficient distribution that may behave better on some stiff problems. The two are interchangeable for most purposes; if `CB3R2R3e` shows accuracy degradation on your problem at moderate $\Delta t$, try `CB3R2R3c`.


### CB4R3R4 — fourth-order, six-stage

The most accurate IMEX scheme in the package. Six stages, four state-sized buffers in `NormalMode` and in the discrete tangent, five in `ContinuousMode`. Worth the extra cost when fourth-order accuracy is required on a stiff problem or when long-time integrations need a low temporal error budget. Its discrete adjoint is the transpose of the step in the form of Butcher, back over the six adjoint stages, which it keeps, with three temporaries: nine buffers. A forcing in the adjoint system is weighted by the explicit weights of the scheme.


### ETDRK4 — exponential time differencing, fourth order

[`ETDRK4`](@ref) (Cox & Matthews, 2002) integrates `u' = A u + g(t, u)` with `A` a `Diagonal` treated exactly, with no step limit from the stiffness of `A`. The coefficients, the φ-functions of `Δt A`, are computed by the contour integrals of Kassam & Trefethen (2005) and cached until `Δt` changes. A component with `A = nothing` is classical RK4. Other operators need a method of [`etdrk4_coeffs`](@ref), whose six values must broadcast into a state (`c .= v`), so lazy broadcasts are allowed. `NormalMode`, `ContinuousMode` and `DiscreteMode` (tangent and adjoint) are supported. A step of zero length is the identity. A normal step keeps its four stages, the states at which it evaluates `g`, in a stage cache; the discrete tangent is the step with `g` linearised at them, and the discrete adjoint its transpose, read backwards, with the coefficients of the adjoint operator and a forcing weighted by the quadrature of RK4. Buffers: 6 stages and 6 coefficients (`NormalMode` and the discrete tangent), 7 stages and 6 coefficients (`ContinuousMode`), 8 stages and 6 coefficients (the discrete adjoint). Near the stiff limit the observed order is reduced until `|λΔt|` is moderate.

### ETDRK2 — exponential time differencing, second order

[`ETDRK2`](@ref) is the ETD2RK scheme of Cox & Matthews (2002), two stages, with the same operators, cache of the coefficients and modes as [`ETDRK4`](@ref); its stages are `x` and `a`, and a forcing of the discrete adjoint gets the weights of Heun's RK2. A component with `A = nothing` is Heun's RK2. Other operators need a method of [`etdrk2_coeffs`](@ref), under the same rule for its three values. A step of zero length is the identity. Buffers: 3 stages and 3 coefficients (`NormalMode` and the discrete tangent), 4 stages and 3 coefficients (`ContinuousMode` and the discrete adjoint).

## Order verification

The package's test suite exercises each scheme against a linear scalar problem with known exact solution, verifying that the leading error term scales as $\Delta t^{p}$ where $p$ is the nominal order. See `test/test_steps.jl` and `test/test_integrator.jl`. The discrete tangent and adjoint of every scheme are checked in `test/test_linearisation.jl` and `test/test_forced_adjoint.jl`.

## Stage caches

Every scheme integrates with the `NormalMode` `step!` overload by default. Supplying a [`RAMStageCache`](@ref) at integration time activates the caching branch: the scheme records the internal-stage values at every accepted step.

```julia
cache = RAMStageCache(nstages(RK4), zeros(3))
F(x, (0, 1), cache)
```

The stage counts `NS` are 4 for `RK4`, 11 for `RK8`, 2 for `CNRK2`, 3 for `CB3R2R2`, 4 for `CB3R2R3e` and `CB3R2R3c`, 6 for `CB4R3R4`, 4 for `ETDRK4` and 2 for `ETDRK2`.

The cached stages are exactly the inputs a [`DiscreteMode`](@ref Flows.DiscreteMode) replay of the same scheme requires; this is the mechanism by which `Flows.jl` produces *discretely consistent* tangent and adjoint integrations. See [Trajectory data](trajectories.md) and [Linearised dynamics](linearised.md).

## Memory usage

The dominant memory cost of any scheme is the `store::NTuple{N, X}` of stage buffers. The cookbook example *Stiff diffusion with CNRK2* in the [Cookbook](cookbook.md) walks through the full state-buffer accounting for a 1D PDE example.

If memory is a constraint, the CB family is the right starting point: their stage counts are dictated by the order, but their *buffer* counts are independent of the order. `CB3R2R3e` and `CB4R3R4` give you third- and fourth-order accuracy at the same buffer cost as a much cruder scheme.

## Cross-references

- [Mathematical foundations](foundations.md) — Butcher form, IMEX splitting.
- [Time stepping](time-stepping.md) — picks the policy that drives the chosen scheme.
- [Linearised dynamics](linearised.md) — full discussion of tangent/adjoint modes.
- [Internals](internals.md) — buffer accounting and `@generated` dispatch details.
