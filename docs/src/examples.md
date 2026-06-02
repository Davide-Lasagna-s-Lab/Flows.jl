# Examples

## Adjoint sensitivity
Variational techniques and adjoint methods can sometimes feel quite difficult to the novice, since sophisticated mathematical concepts (function spaces, functionals, etc.) are often introduced at an early stage. The use of such advanced concepts, however, provides a robust basis and can often clarify the meaning and value of adjoint equations [^1].

### Fundamentals
To show how adjoint sensitivity methods can be used in this package we will consider an optimal control problem on a toy model, given by the initial value problem
```math
\label{eq}\tag{1}
\left\{
\begin{aligned}
\dot{x}(t) &= x(t)^2 + u(t),\quad t \in [0, T]\\
x(0) &= x_0
\end{aligned}
\right.
```
where both the solution $x(t)$ and the control $u(t)$ are assumed to be elements of the function space $\mathcal{S}$, the space of square-integrable continuous functions mapping the interval $[0, T]$ to $\mathbb{R}$. We endow this space with the inner product
```math
\langle v(t), w(t)\rangle = \int_0^T v(t)w(t)\,\mathrm{d}t.
```

Now consider the objective functional
```math
\hat{\mathcal{J}}[x(t), u(t)] = \int_0^T \sin(x(t))\, \mathrm{d}t,
```
mapping two elements of $\mathcal{S}$ to $\mathbb{R}$. Because $x(t)$ and $u(t)$ are related by the initial value problem $(\ref{eq})$, we consider the reduced functional
```math
\mathcal{J}[u(t)] = \int_0^T \sin(x(t))\, \mathrm{d}t,
```
mapping one element of $\mathcal{S}$ to $\mathbb{R}$, where $x(t)$ is assumed to be the solution of $(\ref{eq})$. The goal is to find the gradient of the reduced functional with respect to the control $u(t)$, denoted by
```math
\mathrm{d}\mathcal{J} / \mathrm{d}u(t).
```
If gradient information is available, gradient descent can be used to optimise the controls and extremise the objective functional.

To this end, we proceed by first considering a small, arbitrary perturbation to the controls $v(t) \in \mathcal{S}$, producing a small change in the solution of the initial value problem, denoted by $y(t) \in \mathcal{S}$ and obeying the linear problem
```math
\label{eq:linear-eq}\tag{2}
\left\{
\begin{aligned}
\dot{y}(t) &= 2x(t)y(t) + v(t)\\
y(0) &= 0
\end{aligned}
\right.
```
where $y(t)$ is initially set to zero because the initial condition of problem ($\ref{eq}$) is assumed not to change when the controls are varied. This is an optimal control problem: we simply wish to modify the future evolution of a system by appropriately tuning the controls, from the same initial state.

Now, because of this perturbation $y(t)$, the reduced functional will change by a small amount. Linearising its definition shows that this change can be expressed by a new functional
```math
\mathcal{J}'_{u(t)}[v(t)] = \int_0^T \cos(x(t))\, y(t)\, \mathrm{d}t.
```
This notation should read: if the controls $u(t)$ are perturbed by a small amount $v(t)$, the change in the objective functional is expressed by the integral on the right, where $y(t)$ is obtained from the solution of the linearised equations ($\ref{eq:linear-eq}$). In fact, this functional defines the directional derivative, the change of some function along a specified direction.

To obtain the gradient, let's introduce a function $q(t) \in \mathcal{S}$ and consider the identity
```math
\langle q(t), \dot{y}(t) - 2x(t)y(t) - v(t) \rangle = 0,
```
where $y(t)$ is a solution of the linearised problem $(\ref{eq:linear-eq})$. We can add this term to the linear functional $\mathcal{J}'_{u(t)}[v(t)]$ obtaining
```math
\mathcal{J}'_{u(t)}[v(t)] = \int_0^T \cos(x(t))y(t) + q(t)[\dot{y}(t) - 2x(t)y(t) - v(t)]\, \mathrm{d}t,
```
without having changed anything. However, integrating by parts the term $q(t)\dot{y}(t)$ and collecting products of $y(t)$ leads to
```math
\mathcal{J}'_{u(t)}[v(t)] = - \int_0^T q(t)v(t)\, \mathrm{d}t + \int_0^T y(t)[\cos(x(t)) - \dot{q}(t) - 2q(t)x(t)]\,\mathrm{d}t + [q(T)y(T) - q(0)y(0)].
```
If we now specify that the adjoint function $q(t)$ satisfies the initial value problem
```math
\left\{
\begin{aligned}
\dot{q}(t) &= 2x(t)q(t) - \cos(x(t))\\
q(T) &= 0
\end{aligned}
\right.
```
the second and third terms vanish identically, since $y(0)$ is zero. The interesting thing about the adjoint problem (a linear problem) is that the initial condition is specified at the final time. This is not an issue per se — it just means that instead of marching the equations forward in time, as we would do to solve $(\ref{eq})$, we proceed backwards. However, we note that the adjoint equation depends on the solution $x(t)$, which is obtained by marching the equations forward in time. The practical solution is that we will need to store the entire solution $x(t)$ in memory (and possibly interpolate it) to solve the adjoint problem. This is not an issue in this toy problem, but it can be quite onerous for simulations of large-scale systems, e.g. direct simulation of turbulent flows.

With these steps we obtain an expression for the directional derivative
```math
\tag{3}\label{directional-derivative}
\mathcal{J}'_{u(t)}[v(t)] = \langle - q(t), v(t)\rangle = \int_0^T - q(t)v(t)\, \mathrm{d}t
```
that does not contain $y(t)$, only the perturbation to the controls. Now, there is a theorem, called the [Riesz Representation Theorem](https://en.wikipedia.org/wiki/Riesz_representation_theorem), that can help us read the gradient from this expression. In our setting, the theorem states that any linear functional can be represented as the inner product with an element of $\mathcal{S}$. If $\mathcal{L}$ is one of such functionals, there exists an element $v_\mathcal{L}(t)$ of $\mathcal{S}$ such that
```math
\mathcal{L}[v(t)] = \langle v_\mathcal{L}(t), v(t) \rangle = \int_0^T v_\mathcal{L}(t)v(t)\,\mathrm{d}t.
```
Comparing ($\ref{directional-derivative}$) with the basic definition of the directional derivative shows that the gradient is simply
```math
\tag{4}\label{gradient}
\mathrm{d}\mathcal{J}/\mathrm{d}u(t) = - q(t).
```

### Hands on
We now show how to set up and solve the adjoint problem derived in the previous section using this package [^2]. First, we discretise functions in $\mathcal{S}$ by introducing a uniform temporal mesh given by $N+1$ times $t_i = (i-1) \Delta t$, $i=1,\ldots,N+1$, with $\Delta t = T/N$. Without loss of generality, let us take $T=1$, so the mesh is defined by the Julia range
```julia
ti = 0:0.01:1
```
An element $u(t)$ of $\mathcal{S}$ is then discretised into a Julia `Vector` of $N+1$ elements, denoted by `u` with $u(t_i)$ = `u[i]`.

The first ingredient is code to solve the initial value problem ($\ref{eq}$) for any arbitrary control $u(t)$. This can be achieved by introducing a custom type `NonLinearSystem`:
```julia
struct NonLinearSystem{U}
    _u::U
    function NonLinearSystem(ti::AbstractVector, u::AbstractVector)
        # construct interpolator from ti and u
        _u = interp(ti, u)
        return new{typeof(_u)}(_u)
    end
end
```
Here, `interp` is some Julia function (not defined here) that interpolates the function given by the Julia `Vector`s `ti` and `u` at any arbitrary time, e.g. using a linear interpolation scheme. We need this interpolator since the time-stepping scheme may call the right-hand side in $(\ref{eq})$ at times between the available mesh points. Code that evaluates the right-hand side is, for instance,
```julia
function (sys::NonLinearSystem)(t, x, dxdt)
    dxdt[1] = x[1]^2 + sys._u(t)
    return nothing
end
```
Note how we call the interpolator `sys._u` at the input time `t`.

We now need code to solve the adjoint problem. This can be again achieved by introducing a custom Julia type, `AdjointSystem`:
```julia
struct AdjointSystem{X}
    _x::X
    function AdjointSystem(ti::AbstractVector, x::AbstractVector)
        # construct interpolator from ti and x
        _x = interp(ti, x)
        return new{typeof(_x)}(_x)
    end
end
```
The constructor requires a vector of times `ti` and a vector of solution points `x`. These are used in the constructor to build an interpolator `_x` so that we have access to the state $x(t)$ while marching the adjoint equation backwards.

The right-hand side of the adjoint problem can be coded up like this:
```julia
function (sys::AdjointSystem)(t, q, dqdt)
    x = sys._x(t)
    dqdt[1] = 2*x*q[1] - cos(x)
    return nothing
end
```
where we first interpolate the solution $x(t)$ and then evaluate the right-hand side.

So far we have only seen user code, and not really how to use this package. Hence, we first define the forward problem by constructing a right-hand side (with zero control):
```julia
f = NonLinearSystem(ti, zeros(length(ti)))
```
and then defining its associated flow operator
```julia
F = flow(f, RK4(zeros(1)), TimeStepConstant(0.01))
```
constructed using a Runge–Kutta integration method. Note how the state is represented by a one-element `Vector`, since state objects must be mutable in this package. To obtain the forward solution, we define a monitor extracting the first component
```julia
mon_F = Monitor(zeros(1), (t, x)->x[1])
```
and then solve the problem forward from some initial condition `x0 = Float64[1.0]` by
```julia
F(x0, extrema(ti), mon_F)
```
The [`Monitor`](@ref) object now contains samples of the state at the end of every time step. We can use these samples to define the adjoint system
```julia
g = AdjointSystem(ti, samples(mon_F))
```
and then the adjoint flow operator
```julia
G = flow(g, RK4(zeros(1), ContinuousMode(true)), TimeStepConstant(0.01))
```
Note how
  * the [`RK4`](@ref) constructor is passed `ContinuousMode(true)`, indicating that we have an adjoint problem to be marched backwards;
  * the time step, `0.01`, is constant and positive (the backwards direction is inferred from the `span` argument later).

With the adjoint flow operator defined, we can now solve the adjoint problem. We first define a [`Monitor`](@ref) to observe the adjoint solution, flipping its sign to be compatible with the definition of the gradient $(\ref{gradient})$:
```julia
mon_G = Monitor(zeros(1), (t, q)->-q[1])
```

To obtain the gradient, we march the equations backwards in time from a zero terminal condition
```julia
G(zeros(1), reverse(extrema(ti)), mon_G)
```
and get the gradient by `samples(mon_G)`.

[^1]: The exposition is meant to be illustrative and is mostly addressed to the author's students having to learn this material to complete their project.

[^2]: Note that the approach implemented in this package is the differentiate-then-discretise.
