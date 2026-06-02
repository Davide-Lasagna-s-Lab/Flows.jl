# Time stepping schemes

A [`Flow`](@ref) carries an integration *method* (the
Runge–Kutta-like scheme that performs one step) and a *time-stepping*
policy that decides *how* the steps cover the integration interval.
Method and policy are orthogonal: a single method (say [`RK4`](@ref))
can be paired with any of the policies described here, and the same
policy can drive any of the integration methods. The currently
supported policies are subtypes of [`Flows.AbstractTimeStepping`](@ref)
and are summarised below.

## Constant time step

The simplest policy is [`TimeStepConstant`](@ref Flows.TimeStepConstant),
which marches the system with a fixed $\Delta t$:
```julia
ts = TimeStepConstant(0.01)
F  = flow(f, RK4(zeros(3)), ts)
F(x, (0, 1))
```
The step size is validated at construction and stored as `Float64`.
Backward integration is requested by passing a decreasing `span` to
the flow; the sign of $\Delta t$ is flipped internally.

If the requested span is not an integer multiple of $\Delta t$, the
last step is shortened so that the endpoint is hit exactly. See the
internal [`Flows.Steps`](@ref Flows.Steps) iterator for the details.

## Adaptive time stepping

[`AbstractTimeStepFromHook`](@ref) is the supertype for policies whose
time step is selected at runtime by a user-supplied hook. A subtype
must be a callable with signature `hook(g, A, z)` returning a
positive scalar; the flow calls it once per step with the explicit
right-hand side, the linear operator, and the current state.

```julia
struct DiagHook <: AbstractTimeStepFromHook end
(::DiagHook)(g, A, z) = 0.1 * sqrt(max(z[1], eps()))

F = flow(g, RK4(zeros(1)), DiagHook())
```

When the hook would overshoot the requested endpoint, the flow
shrinks the step so the endpoint is reached exactly — there is no
loss of precision at the boundary.

## Discretely consistent linearisation

[`TimeStepFromCache`](@ref Flows.TimeStepFromCache) is a singleton
policy used to replay a previously recorded
[`RAMStageCache`](@ref) through a linearised system. There is no step
size to configure: each step is exactly the one cached during the
primal integration, so the linearised path is *discretely* consistent
with the primal.

```julia
# primal: cache the stages
cache = RAMStageCache(nstages(RK4), zeros(3))
F     = flow(f, RK4(zeros(3)), TimeStepConstant(0.01))
F(x, (0, 1), cache)

# linearised: replay the cache
L = flow(f_tangent, RK4(zeros(3), DiscreteMode()), TimeStepFromCache())
L(y, cache)
```

## Continuous linearisation

[`TimeStepFromStorage`](@ref Flows.TimeStepFromStorage) is the
*continuous* counterpart of `TimeStepFromCache`: instead of replaying
the recorded stages, the linearised system marches with its own
constant $\Delta t$ and queries the nonlinear trajectory through a
[`RAMStorage`](@ref) interpolator at the times required by its
internal stages. This is the appropriate choice when the trajectory
is too long to fit into a stage cache.

```julia
store = RAMStorage(zeros(3))
F     = flow(f, RK4(zeros(3)), TimeStepConstant(0.01))
F(x, (0, 1), store)

L = flow(f_adjoint, RK4(zeros(3), ContinuousMode(true)), TimeStepFromStorage(0.01))
L(q, store, (1, 0))
```

Note that the result is *not* discretely consistent: the linearised
path samples a continuous interpolant of the primal, not the exact
primal stage values.

For the corresponding API entries, see the
[Time Stepping API](@ref Time-Stepping-API) section of the API
reference.
