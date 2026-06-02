# Advanced features

## Call dependencies
When solving several coupled systems, the default settings are such that the signatures of the first, second, third, etc. right-hand-side functions are
```julia
f1(t, x1, dx1dt)
f2(t, x1, dx1dt, x2, dx2dt)
f3(t, x1, dx1dt, x2, dx2dt, x3, dx3dt)
f4(t, x1, dx1dt, x2, dx2dt, x3, dx3dt, x4, dx4dt)
```
This is not always desirable. For instance, assume we need to solve the problem
```math
\tag{1}\label{eq}
\left\{
\begin{aligned}
  \dot{\mathbf{x}}(t)   &= \mathbf{f}(t, \mathbf{x}(t))\\
  \dot{\mathbf{y}}_1(t) &= \mathbf{g}_1(t, \mathbf{x}(t), \mathbf{y}_1(t))\\
  \dot{\mathbf{y}}_2(t) &= \mathbf{g}_2(t, \mathbf{x}(t), \mathbf{y}_2(t))\\
  \dot{\mathbf{y}}_3(t) &= \mathbf{g}_3(t, \mathbf{x}(t), \mathbf{y}_3(t))
\end{aligned}
\right.
```
where we wish to express the structure of the function calls explicitly.

This can be achieved with a [`CallDependency`](@ref) object. The constructor accepts as many tuples of integers as the number of coupled equations. Each tuple specifies which states participate in the corresponding function signature. For instance, for the example $(\ref{eq})$ the correct call dependency specification is
```julia
deps = CallDependency((1,), (1, 2), (1, 3), (1, 4))
```
which translates into
  * `f`  has signature `f(t, x, dxdt)`               — depends only on component `1`,
  * `g1` has signature `g1(t, x, dxdt, y1, dy1dt)`   — depends on components `1, 2`,
  * `g2` has signature `g2(t, x, dxdt, y2, dy2dt)`   — depends on components `1, 3`,
  * `g3` has signature `g3(t, x, dxdt, y3, dy3dt)`   — depends on components `1, 4`.

Each inner tuple must be sorted in increasing order and contain only indices in `1:N`.

The `deps` object is then passed as an additional argument to the constructor of the [`Flow`](@ref) object. For an explicit method:
```julia
F = flow(couple(f, g1, g2, g3), deps, RK4(couple(x, y1, y2, y3)), TimeStepConstant(0.1))
```

## Symmetry transformations

Every [`flow`](@ref) constructor accepts an optional trailing argument
`sym`, a callable `sym(x, s)` that applies a parameterised symmetry
transformation to the result of an integration. When a flow is
constructed without a `sym`, calling it with a symmetry parameter
raises an error; otherwise the parameter is forwarded to the wrapped
callable and the result of `_propagate!` is transformed in place
before being returned.

For non-coupled states the callable is wrapped in a
[`Flows.SymTransform`](@ref) and applied directly:
```julia
F = flow(f, RK4(zeros(1)), TimeStepConstant(0.01), (x, s) -> x[1] += s)
F(x, (0, 1), -1.0)   # last argument is `s`
```
For coupled states the callable is wrapped in a
[`Flows.CoupledTransform`](@ref) and applied component-wise:
```julia
F = flow(couple(f, g), RK4(couple(x, y)),
         TimeStepConstant(0.01),
         (xi, s) -> xi[1] += s)        # called for each component
F(couple(x, y), (0, 1), -1.0)
```
Both wrapper types collapse to `nothing` when constructed with
`nothing`, so passing `sym=nothing` (the default) is equivalent to
omitting the argument.
