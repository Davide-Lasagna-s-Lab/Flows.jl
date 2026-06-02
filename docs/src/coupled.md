# Coupled dynamical systems

## Motivation
In some situations, the state is defined by two or more heterogeneous components, each obeying its own dynamics, but with some components that depend on each other. For instance, consider a problem where the state $\mathbf{z}$ is defined by two components, $\mathbf{x}\in\mathcal{X}$ and $\mathbf{y}\in\mathcal{Y}$, as
```math
\mathbf{z} = [\mathbf{x}, \mathbf{y}]
```
where $\mathbf{z} \in \mathcal{Z} \equiv \mathcal{X}\times \mathcal{Y}$. Assume the dynamics of $\mathbf{x}(t)$ to be independent of that of $\mathbf{y}(t)$, but that the latter depends on the former. This situation corresponds to the problem
```math
\left\{
\begin{aligned}
  \dot{\mathbf{x}}(t) &= \mathbf{f}(t, \mathbf{x}(t))\\
  \dot{\mathbf{y}}(t) &= \mathbf{g}(t, \mathbf{x}(t), \mathbf{y}(t))
\end{aligned}
\right.
```

A possible strategy to solve this problem is to first solve the first differential equation, store its solution, and then use it to solve the second, perhaps with some kind of interpolation if required. Storing a full solution is often not possible or desirable, and the two systems must be solved in a coupled manner.

!!! example
    A notable example is the solution of the linearised variational equations, governing the evolution of small perturbations to the initial conditions of an initial value problem. If $\mathbf{x}(t)$ denotes the system's state, governed by a nonlinear differential equation, and $\mathbf{y}(t)$ denotes the perturbation, the two components obey the coupled problem
    ```math
    \left\{
    \begin{aligned}
      \dot{\mathbf{x}}(t) &= \mathbf{f}(t, \mathbf{x}(t))\\
      \dot{\mathbf{y}}(t) &= \mathbf{f}_\mathbf{x}(t, \mathbf{x}(t))\cdot\mathbf{y}(t)
    \end{aligned}
    \right.
    ```
    where $\mathbf{f}_\mathbf{x}$ is the Jacobian of $\mathbf{f}$.

## Approach
The fundamental tool provided by this package to address this need is the [`couple`](@ref) function, which accepts two or more arguments and returns a [`Coupled`](@ref) object. This is a compact representation similar to a Julia `Tuple` (internally it is a thin wrapper around one) and behaves similarly to one. For instance:
```julia
julia> z = couple(Int[1, 2, 3], Float64[0, 4])
2-element Coupled{2, Tuple{Vector{Int64}, Vector{Float64}}}:
 [1, 2, 3]
 [0.0, 4.0]

julia> z[1]   # it is indexable
3-element Vector{Int64}:
 1
 2
 3

julia> length(z)   # it has a length
2
```
One feature of [`Coupled`](@ref) objects is that they support Julia's dot notation; operations are forwarded down to each of the internal components. For instance:
```julia
julia> z1 = couple(Int[1, 2, 3], Float64[0, 4]);

julia> z2 = couple(Int[2, 4, 5], Float64[1, 2]);

julia> z1 .= z2 .* 2
2-element Coupled{2, Tuple{Vector{Int64}, Vector{Float64}}}:
 [4, 8, 10]
 [2.0, 4.0]
```
Arithmetic operations are forwarded to each component. The utility of this behaviour is that the arithmetic operations arising in the computation of a time step — e.g. during the internal stages of a Runge–Kutta method — are forwarded transparently to all components.

!!! note
    Despite supporting the dot-broadcasting notation, its usage is considered an internal interface and should not be relied upon in user code.

## Example
Assume that we want to solve the problem
```math
\left\{
\begin{aligned}
  \dot{\mathbf{x}}(t) &= \mathbf{f}(t, \mathbf{x}(t))\\
  \dot{\mathbf{y}}(t) &= \mathbf{g}(t, \mathbf{x}(t), \mathbf{y}(t))
\end{aligned}
\right.
```
where the two components $\mathbf{x}(t) \in \mathbb{R}^3$ and $\mathbf{y}(t) \in \mathbb{R}^2$ are represented by standard Julia `Vector`s, for simplicity. Assume that two Julia functions `f` and `g` have been defined, with signatures
```julia
f(t, x, dxdt)
g(t, x, dxdt, y, dydt)
```
Note the definition of the second function and how it maps to the original problem.

Assume that an explicit fourth-order Runge–Kutta method, provided by this package's [`RK4`](@ref) type, is used to advance the system. Two steps are needed. First, we couple together the Julia functions representing the two components:
```julia
h = couple(f, g)
```
Second, we define a composite [`RK4`](@ref) method object with
```julia
m = RK4(couple(zeros(3), zeros(2)))
```
where we have passed an instance of the type representing the coupled state, obtained by coupling two temporary `Vector`s of the right size.

The flow operator can then finally be constructed as
```julia
F = flow(h, m, TimeStepConstant(0.1))
```
where we have declared that we will march the equations forward with time step $\Delta t = 0.1$.

The flow operator `F` can now be called on a composite state object. For instance, we define some initial conditions and march the problem in time:
```julia
x0 = Float64[0, 1, 0]
y0 = Float64[2, 3]

F(couple(x0, y0), (0, 1))
```
Because `F` modifies its first argument in place, the `Vector`s `x0` and `y0` are modified by the call to `F`.
