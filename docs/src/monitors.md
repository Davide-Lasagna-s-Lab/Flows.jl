# Monitor objects
A [`Flow`](@ref) (see [Quick start](@ref) for an introduction) maps a state vector forward in time by some specified amount. It operates in place, and does not store or record anything during the trajectory. However, it is sometimes useful to record some quantity along a trajectory — for instance one of the degrees of freedom, or maybe some integral quantity. This is what a [`Monitor`](@ref) object is for.

## Basic usage
The constructor of the [`Monitor`](@ref) type has the signature
```julia
Monitor(x::X, f)
```
The first argument is an object of some user-defined type `X` — the same type used to represent the system's state. The second argument is the *observable*: a function or callable object that we use to "observe" the state along the simulation. It must accept the time and the state as arguments, i.e. have the signature
```julia
f(t::Real, x::X)
```
and may return anything. The return value is what is actually stored, so the observable may also be used to extract a derived quantity (a norm, a component, …) rather than the full state.

!!! example
    This example demonstrates how to define a monitor for the first component of a state with three degrees of freedom.
    ```julia
    mon = Monitor(zeros(3), (t, x)->x[1])
    ```

    Note how the observable is simply an anonymous function that ignores `t` and extracts the first element. If `t` is also of interest, just use it inside the observable.

In the constructor, the observable is called on the first argument with a dummy time `0.0`. The type of the output is inspected and storage to hold elements of the same type is allocated.

To record an observable during a trajectory, the monitor object can be passed as an additional argument to a [`Flow`](@ref) object. During the integration a sample of the observable is taken at the end of every accepted time step, including one sample at the beginning of the trajectory.

!!! example
    Assume `F` is a [`Flow`](@ref) for the Lorenz equations and we want to record the norm of the state vector over a short trajectory from $t=0$ to $t=1$. This can be achieved by
    ```julia
    mon = Monitor(zeros(3), (t, x)->norm(x))
    F(x, (0, 1), mon)
    ```

At the end of the integration, the content of the [`Monitor`](@ref) object `mon` can be accessed by two helper functions. The first
```julia
samples(mon)
```
returns a Julia `Vector` with samples of the observable, while
```julia
times(mon)
```
returns a `Vector` containing the times at which the samples were taken. These can be used, for instance, to plot the observable as a function of time.

!!! note
    The observable function can return anything. For instance, if we want to record the full state, we can wrap `copy`:
    ```julia
    mon = Monitor(zeros(3), (t, x)->copy(x))
    F(x, (0, 1), mon)
    ```

    If we want to record more quantities, we can return a `Tuple`:
    ```julia
    mon = Monitor(zeros(3), (t, x)->(x[1], x[2]^2))
    F(x, (0, 1), mon)
    ```
    so that `samples(mon)` returns a vector of `Tuple`s. Returning a `Tuple` also lets the optional [`Flows.Logger`](@ref) print each quantity in its own column.

## Monitors as callback functions
Despite its name, a [`Monitor`](@ref) can be used to *modify* the system state. The restriction is, of course, that the action can only fire at the end of every time step (or every `oneevery` time steps; see [`Monitor`](@ref) for usage of this keyword). For instance, one can define a monitor that normalises its input every 10 time steps by:
```julia
mon = Monitor(zeros(5), (t, x)->(x ./= norm(x)); oneevery=10)
```

## Advanced usage
The behaviour of [`Monitor`](@ref) objects can be customised more finely. Consult the [Monitor API](@ref) page for the full set of keyword arguments, including `savebetween`, `skipfirst`, `sizehint`, `io`, and `logevery`.
