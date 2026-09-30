# Quadrature equations

A common task in dynamical systems is to compute integrals of observables along trajectories:

```math
I(T) = \int_{t_0}^{T} g\bigl(\mathbf{x}(t)\bigr)\,\mathrm{d}t.
```

`Flows.jl` offers two ways to compute such integrals. The first — **coupling the quadrature into the system** — gives an integral whose accuracy matches the time integrator's order. The second — **post-processing samples** with a stand-alone rule — gives an integral whose accuracy is determined by the chosen quadrature rule alone. The first is almost always what you want; the second is convenient when the observable is decided after the integration is done.

## Coupling the quadrature

The recipe is straightforward. Add a *quadrature component* $I(t)$ to the system whose dynamics is exactly the integrand, with initial condition zero:

```math
\left\{
\begin{aligned}
\dot{\mathbf{x}}(t) &= \mathbf{f}(t,\mathbf{x}(t)),\quad \mathbf{x}(t_0) = \mathbf{x}_0\\
\dot{I}(t)          &= g\bigl(\mathbf{x}(t)\bigr),\hspace{1.05em}\, I(t_0) = 0
\end{aligned}
\right.
```

Integrate the augmented system as a [`Coupled`](@ref) state from $t_0$ to $T$. The final value $I(T)$ is the requested integral, evaluated with an error that **inherits the order of accuracy of the integrator**. A fourth-order primal integrator gives a fourth-order accurate integral; there is no separate quadrature error to worry about.

A minimal example for a three-dimensional state:

```julia
g(t, x, dxdt, I, dIdt) = (dIdt[1] = x[1]^2; return dIdt)

x = Float64[1.0, 3.0, 4.0]
I = Float64[0.0]

F = flow(couple(f, g),
         RK4(couple(x, I)),
         TimeStepConstant(0.1))

F(couple(x, I), (0, 10))
I[1]    # ≈ ∫₀¹⁰ x₁(t)² dt
```

The quadrature component is a one-element `Vector`, because the package requires the state to be a mutable container. To compute several integrals simultaneously, widen the quadrature component to a multi-element vector and fill more slots in `dIdt`:

```julia
function quads(t, x, dxdt, I, dIdt)
    dIdt[1] = x[1]
    dIdt[2] = x[1]^2
    dIdt[3] = norm(x)
    return dIdt
end

I = zeros(3)
```

The default call dependency wires the second component to see the first, which is exactly the right pattern: $g$'s signature is `g(t, x, dxdt, I, dIdt)` and uses `x` to fill `dIdt`. No `CallDependency` argument is needed.

### IMEX systems with an explicit quadrature

When the primal is stiff and uses an IMEX scheme, the quadrature component is *almost always* advanced fully explicitly: $\dot{I}$ does not see the linear operator. The idiom is to couple the primal linear operator with `nothing` for the quadrature component:

```julia
L = Diagonal(...)            # the stiff primal linear operator

F = flow(couple(f_ex, g),
         couple(L, nothing),     # `nothing` means "no implicit part"
         CNRK2(couple(x, I)),
         TimeStepConstant(0.1))
```

The IMEX scheme detects the `nothing` and collapses the implicit subproblem for the quadrature component to the identity — no `mul!` call, no [`ImcA!`](@ref) call. The explicit half does the entire work for the quadrature row. The dispatch is resolved at compile time by `System`'s `@generated` machinery, so the `nothing` branch has zero runtime cost.

### Why couple, not post-process?

Coupling has three concrete advantages over a post-processing rule:

1. **Accuracy.** The integrator is, by construction, order-$p$. Coupling yields an order-$p$ integral. A post-hoc rule has its own (often lower) order.
2. **Same time grid.** The samples that feed the integrand are exactly the values the primal scheme computes anyway. There is no separate sampling decision.
3. **Memory.** A coupled integration carries one extra small component. A post-hoc rule requires storing every sampled value of the observable in a `Monitor` until the end.

The cookbook example *Lyapunov exponent of the Lorenz attractor* uses the coupling idiom to integrate a logarithmic growth rate — see the [Cookbook](cookbook.md).

## Monitoring the quadrature

It is sometimes useful to monitor the value of the quadrature variable along the integration — e.g. to plot the cumulative integral, or to compute a *running average*. The pattern is to monitor the coupled state and extract the quadrature component:

```julia
mon = Monitor(couple(x, I), (t, xq) -> xq[2])
F(couple(x, I), (0, T), mon)
```

The monitor receives the coupled state and returns whatever derived quantity you want — here, the *full* quadrature component (which itself may be a vector of integrals).

The *cumulative average* of $g(\mathbf{x}(t))$ is

```math
\bar g(\tau) = \frac{1}{\tau}\int_0^{\tau} g\bigl(\mathbf{x}(t)\bigr)\,\mathrm{d}t,
```

obtained from the recorded samples and times by

```julia
g_bar = samples(mon) ./ times(mon)
```

The first element is `0/0`; either skip it or initialise the monitor with `skipfirst=true`.

## Post-hoc quadrature rules

When the observable is decided *after* an integration has already happened, two stand-alone rules are exported: [`trapz`](@ref Flows.trapz) (composite trapezoidal) and [`simps`](@ref Flows.simps) (composite Simpson's). Both expect two same-length vectors `xs` (abscissae, sorted) and `ys` (integrand values), and return the integral as a scalar.

```julia
mon = Monitor(x, (t, x) -> g(x))
F(x, (0, T), mon)

I_trap = trapz(times(mon), samples(mon))
I_simp = simps(times(mon), samples(mon))
```

### `trapz`

The composite trapezoidal rule over (possibly non-uniform) intervals:

```math
\mathrm{trapz}(\mathbf{x},\mathbf{y}) = \sum_{k=1}^{N-1} \frac{y_{k+1}+y_k}{2}(x_{k+1}-x_k).
```

Exact for piecewise-linear integrands, $O(h^2)$ error for smooth ones. Robust and never produces a negative weight.


### `simps`

The composite Simpson rule with a quadratic correction for the last interval when the number of intervals is odd. Exact for cubics on a uniform grid, $O(h^4)$ for smooth integrands.


### Choosing between the two

For most workflows the choice is dictated by what is recorded:

| Situation                                                         | Use                  |
|-------------------------------------------------------------------|----------------------|
| Trajectory accurately monitored, smooth observable                | `simps`              |
| Trajectory monitored, sample timing is irregular                  | `trapz`              |
| Trajectory sparsely sampled and the integral is qualitative only  | `trapz`              |
| Trajectory monitored once, observable known at integration time   | **Couple it instead** |

If accuracy matters, prefer the coupled approach over either post-hoc rule.

## Cross-references

- [Coupled systems](coupled.md) — call-dependency rules for the coupled idiom.
- [Trajectory data](trajectories.md) — monitors and storages for sampling along a trajectory.
- [Cookbook](cookbook.md) — worked examples of the coupling pattern.
