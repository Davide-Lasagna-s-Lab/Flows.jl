# Quick start

## Fundamentals
Throughout this package we consider dynamical systems defined by an evolution equation of the type
```math
\dot{\mathbf{x}}(t) = \mathcal{L}\mathbf{x}(t) + \mathbf{f}(t, \mathbf{x}(t))
```
where $t$ is time and $\mathbf{x}$ is the state, an element of some space $\mathcal{X}$.

The operator $\mathcal{L}$ is linear and time invariant and contains stiff terms that are advanced implicitly in time. The function $\mathbf{f}(t, \mathbf{x}(t))$ is a nonlinear term that is advanced using an explicit scheme. This notation is purely operational: if the dynamical system does not have any stiff terms and can be advanced using an explicit scheme, all terms can be lumped into the function $\mathbf{f}(t, \mathbf{x}(t))$.

In many situations in dynamical systems theory, one requires the flow of the vector field $\mathbf{f}$, the nonlinear operator $\mathbf{\Phi}^t : \mathcal{X} \mapsto \mathcal{X}$ mapping the state $\mathbf{x}_0$ at $t=t_0$ to its forward-in-time image $\mathbf{x}(t)$. Mathematically, this operation is
```math
\mathbf{x}(t) = \mathbf{\Phi}^{t-t_0} \mathbf{x}_0.
```

The structure of this package, and the API it provides, is entirely based on satisfying this flow interface, where a [`Flow`](@ref) object is constructed that acts as the time discrete version of $\mathbf{\Phi}$.

## User interface
To construct a [`Flow`](@ref) object, the user needs to write Julia code for the state object $\mathbf{x}$ and for the right-hand side $\mathbf{f}$, in addition to specifying which integration scheme will be used to advance the state in time.

### Defining state objects
The package works transparently on state objects `x` of arbitrary Julia type, say objects of some user defined Julia type `X` that may loosely map to their mathematical equivalent $\mathcal{X}$ and which may provide some advanced API.

State objects represented by a standard Julia `AbstractArray` (of any dimension) work out-of-the-box. However, requiring all computations to be expressed in terms of objects of type `AbstractArray` (or even just `Vector`) can be restrictive in practice and would not leverage the multiple-dispatch capabilities of Julia. As an example, consider solving a time-dependent partial differential equation over a one-dimensional domain using a Fourier–Galerkin method. The user might wish to define a `SpectralField` type representing fields, with a custom API for, e.g., computing the derivative field:
```julia
struct SpectralField{T<:AbstractFloat}
    data::Vector{Complex{T}}
end

ddx(u::SpectralField) = # some definitions
```

To work with the package, user types should satisfy these two requirements:

  * Support dot-broadcasting in in-place algebraic expressions including scalars and objects of the same type. For instance, notation like `x .= 3.0.*y .+ 4.0.*z` for three elements `x`, `y` and `z` of the user type should be supported.
  * Provide methods for `Base.similar(x)` and `Base.copy(x)`.

Internally, the package makes extensive use of the dot notation when updating cache objects during a time step. This restriction prevents using immutable `StaticArrays` objects as state objects, which would probably lead to faster code on small problems. This is less important for the intended applications of this package — turbulent flow simulations where the state is generally given by multiple large three-dimensional arrays and having mutable objects is important for performance.

### Defining the right hand side
Given a state `x` as an object of type `X`, the steps required to specify the right-hand side depend on whether an explicit or semi-implicit integration method is used.

For explicit methods, a dynamical system is specified by a Julia function or some other callable object that has a method with signature
```julia
f(t::Real, x::X, dxdt::X) where {X}
```
This function is supposed to operate in place and should modify the content of the third argument `dxdt` with the time derivative at `x` and, if needed, time `t`. Modifying the content of the second argument in the body of this function is undefined behaviour.

!!! example
    Consider defining a custom type implementing the right-hand side of the Lorenz equations. A Julia type is defined with one field for the parameter $\rho$:
    ```julia
    struct Lorenz
        ρ::Float64
        Lorenz(ρ::Real = 28.0) = new(ρ)
    end
    ```

    Then the type is made callable by defining the following method:
    ```julia
    function (eq::Lorenz)(t, u, dudt)
        x, y, z = u
        dudt[1] =  10 * (y - x)
        dudt[2] =  eq.ρ *  x - y - x*z
        dudt[3] = -8/3 * z + x*y
        return dudt
    end
    ```

    Note how this function returns the argument that has been modified. This is not required by the package API, but can be useful in other situations.

The semi-implicit methods that are currently implemented assume that the stiff term is linear and given by a time-invariant operator $\mathcal{L}$. To define this term, the user must define a Julia type, say `A`, with a method for the `LinearAlgebra` function `mul!`, with signature
```julia
LinearAlgebra.mul!(out::X, a::A, x::X)
```
which computes the action of `a` on `x` and stores it in `out`. Modifying the content of `x` is undefined behaviour.

In addition, the type `A` should have a method for the function [`Flows.ImcA!`](@ref) provided by this package, with signature
```julia
Flows.ImcA!(a::A, c::Real, y::X, z::X)
```
which solves the linear system
```math
    (I - c\,a)\,z = y
```
for some right-hand side `y` of type `X` and a scalar `c`, and stores the result in `z` (also of type `X`). The function name should be read "I-minus-cee-A".


### Defining the integration scheme
Performing a time step typically requires temporary objects for intermediate computations, e.g. the internal stages of a Runge–Kutta method. In this package, these temporary objects are pre-allocated by creating a helper object — an instance of one of the concrete subtypes of [`Flows.AbstractMethod`](@ref) provided by the package (see [Available integration schemes](@ref)).

All constructors of such helper objects have the same signature and require a single argument of type `X`. This argument is used internally in the constructor to pre-allocate as many `similar` objects as needed for the intermediate computations in the time step.

!!! example
    Consider using the classical fourth-order Runge–Kutta method to integrate the Lorenz equations defined above. The state vector is implemented using a standard Julia `Vector` of three elements, and the helper object is constructed by
    ```julia
    method = RK4(zeros(3))
    ```

### Defining the time stepping method
Once the integration scheme is defined, the user needs to specify a time-stepping method. This is a specification of how time steps should be selected to advance the state forward in time. This package provides several methods, but for the purpose of this quick start document we only describe one of the most-used, [`TimeStepConstant`](@ref). As the name suggests, this method indicates that the time step should be constant and equal to a value specified at construction:
```julia
steps = TimeStepConstant(0.1)
```
which signals that a constant time step $\Delta t = 0.1$ will be used in the integration.

## Finally defining the flow operator
With state, dynamical system, integration method and time stepping defined, we are finally ready to define the flow operator. This is achieved by using the [`flow`](@ref) function, which constructs a [`Flow`](@ref) object. It has two basic methods, with signatures
```julia
flow(f, method::AbstractMethod, stepping::AbstractTimeStepping)
flow(f, L,  method::AbstractMethod, stepping::AbstractTimeStepping)
```
for problems that are integrated using explicit or semi-implicit methods, respectively.

A [`Flow`](@ref) object is callable with the signature
```julia
(::Flow)(x::X, span::Tuple{Real, Real})
```
and propagates the state object `x` from time `span[1]` to a later time `span[2]`, returning its first argument. The key observation is that for performance reasons the flow operator operates in place and overwrites its first argument.

!!! example
    This example demonstrates how to construct a [`Flow`](@ref) object approximating the flow of the Lorenz equations (defined above) using a classical fourth-order Runge–Kutta scheme with constant time step $\Delta t = 10^{-2}$.

    The flow object is constructed by
    ```julia
    F = flow(Lorenz(28), RK4(zeros(3)), TimeStepConstant(1e-2))
    ```

    A point on the attractor can be then obtained by propagating a random initial condition by some amount of time:
    ```julia
    x = F(rand(3), (0, 100))
    ```
