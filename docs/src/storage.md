# Solution storages

Some integration paths need access to the *full* time history of a
trajectory rather than just its endpoint. The most common reason is to
solve a linearised equation (continuous tangent or adjoint) whose
right-hand side depends on the nonlinear state $\mathbf{x}(t)$ at
times in between the integration grid. To keep the integration code
unaware of the storage details, this package introduces
[`AbstractStorage`](@ref Flows.AbstractStorage) and ships one concrete
implementation, [`RAMStorage`](@ref).

## Storing the solution in RAM

The most natural place to keep a trajectory is plain RAM. A
[`RAMStorage`](@ref) is a thin wrapper around two parallel `Vector`s
(times and snapshots) together with three pieces of bookkeeping: the
polynomial degree used by the interpolator, an optional period if the
signal is periodic, and a flag controlling whether the propagation
loop should push the very last sample of an integration.

### Construction

A storage is constructed either from an example object or directly
from the element type:
```julia
using Flows

# from an example object
x = zeros(3)
store_from_x = RAMStorage(x)

# from the type only
store_from_T = RAMStorage(typeof(x))
```
The two forms are equivalent: the first just forwards to the second.
Both accept a small set of keyword arguments:

| Keyword            | Default      | Meaning |
|--------------------|--------------|---------|
| `ttype`            | `Float64`    | element type of the recorded times. |
| `degree`           | `3`          | odd (`≥ 3`) Lagrange polynomial degree used by the interpolator. |
| `period`           | `0.0`        | positive period if the data is periodic, zero otherwise. |
| `storelast`        | `true`       | whether the propagation loop should push the final sample of an integration. |

### Recording a trajectory

Pass the storage as an additional argument to a [`Flow`](@ref):
```julia
F     = flow(f, RK4(zeros(3)), TimeStepConstant(0.01))
store = RAMStorage(zeros(3))
F(rand(3), (0, 10), store)
```
After the call, [`times(store)`](@ref times) returns the recorded
times and [`samples(store)`](@ref samples) the snapshots. For a
non-periodic storage [`timespan(store)`](@ref timespan) is
`(first time, last time)`; for a periodic storage the second element
is `period(store)` instead.

### Interpolating between samples

A populated [`RAMStorage`](@ref) is callable. Given a buffer `out`
matching the snapshot type and a query time `t`, the call
```julia
store(out, t)
```
overwrites `out` with the Lagrange interpolant of the recorded
trajectory at `t`. Extrapolation is not allowed: `t` must lie inside
[`timespan(store)`](@ref timespan), otherwise an `ArgumentError` is
raised. The optional `Val(1)` argument requests a first-derivative
interpolant; the default is `Val(0)`, the function value.

This is the path used by the continuous tangent / adjoint integrators
to evaluate the nonlinear trajectory at the internal stages of an
implicit-explicit Runge–Kutta step.

### Periodic data

When the data represents one period of a periodic signal, set
`period` to the period at construction and *omit* the duplicated
endpoint when pushing samples. The interpolator then wraps around the
endpoints automatically. Combined with `storelast=false`, this is the
recommended setup for a "ring-buffer-like" representation of a
periodic orbit.

### Custom storage backends

[`AbstractStorage`](@ref Flows.AbstractStorage) is the supertype for
all storages. Concrete backends must implement `push!`, `reset!`,
[`times`](@ref), [`samples`](@ref), and [`timespan`](@ref), and may
optionally provide an interpolation callable if they are to be used
to drive a continuous tangent or adjoint integration. See
[`RAMStorage`](@ref) for the canonical example.
