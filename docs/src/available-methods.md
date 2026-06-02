# Available integration schemes
Currently the following integration schemes are supported:

  * [`RK4`](@ref) — classical fourth-order Runge–Kutta, for non-stiff problems.
  * A family of low-storage IMEX schemes developed by Cavaglieri and Bewley at UCSD [^1] for stiff problems, where stiffness arises from the linear term:
    * [`CB3R2R2`](@ref), a second-order three-register scheme;
    * [`CB3R2R3e`](@ref) and [`CB3R2R3c`](@ref), two third-order three-register schemes;
    * [`CB4R3R4`](@ref), a fourth-order four-register scheme.
  * [`CNRK2`](@ref) — classical second-order Crank–Nicolson / Runge–Kutta predictor–corrector for stiff problems.

## Usage
All schemes have constructors with the same signature.

### Standard problems and coupled systems
For standard problems, including coupled systems, the constructor accepts an object of the type used to represent the state (see [the quick start](@ref Quick-start)). For instance, to construct an [`RK4`](@ref) object for a system defined by a $4 \times 4$ Julia `Matrix`:
```julia
m = RK4(zeros(4, 4))
```

For [coupled systems](@ref Coupled-dynamical-systems), the object passed to the constructor should be a [`Coupled`](@ref) object whose components match the state type.

### Linearised equations
For linearised equations marched over an [`AbstractStorage`](@ref) the constructor accepts an additional argument specifying whether the equations correspond to a forward (tangent) or adjoint problem. An [`RK4`](@ref) method for the forward (tangent) problem on the same state type is constructed with
```julia
m = RK4(zeros(4, 4), ContinuousMode(false))
```
and for the adjoint problem with
```julia
m = RK4(zeros(4, 4), ContinuousMode(true))
```
[`ContinuousMode`](@ref) signals that we are solving a continuous approximation of the linearised equations. The discretely consistent variant uses a [`RAMStageCache`](@ref) instead of a storage, and is obtained with [`DiscreteMode`](@ref). See [`AbstractMode`](@ref Flows.AbstractMode) for the full set of mode tags.

## References
[^1]: Cavaglieri, D. and Bewley, T., 2015. Low-storage implicit/explicit Runge–Kutta schemes for the simulation of stiff high-dimensional ODE systems. *Journal of Computational Physics*, 286, pp. 172–193.
