export flow, InvalidSpanError

# ============================================================================
# THE Flow TYPE
# ============================================================================
#
# `Flow` is the discrete approximation of the time-forward map associated
# with a vector field. The struct bundles four things that fully determine
# a time integration:
#
#   - `tstep` (TS) : an `AbstractTimeStepping` choosing how the step size
#                    is selected from one step to the next (constant,
#                    hook-driven, from a stage cache, …).
#   - `meth`  (M)  : an `AbstractMethod` carrying the integration scheme
#                    (RK4, CNRK2, the CB schemes) together with its
#                    preallocated stage buffers.
#   - `sys`   (S)  : the `System` wrapping the right-hand-side(s) and the
#                    optional linear/implicit operator(s).
#   - `sym`   (SO) : either `nothing` (no symmetry) or one of the symmetry
#                    transform wrappers from `couple.jl`. `Nothing`,
#                    `SymTransform`, and `CoupledTransform` are the three
#                    possible kinds; the absent case is always `nothing`
#                    so the `NoTransform = Nothing` alias is exhaustive.

mutable struct Flow{TS<:AbstractTimeStepping, M<:AbstractMethod, S<:System, SO}
    tstep::TS # the method used for time stepping
     meth::M  # the method, with storage, implementation and time stepping
      sys::S  # the system to be integrated
      sym::SO # symmetry operator applied to the flow map (or `nothing`)
    Flow(ts::TS, m::M, sys::S, sym::SO) where {TS, M, S, SO} = new{TS, M, S, SO}(ts, m, sys, sym)
end

"""
    flow(g, m::AbstractMethod, ts::AbstractTimeStepping, sym=nothing) -> Flow

Construct a [`Flow`](@ref) representing the numerical discretisation of
the time-forward operator associated with the vector field `g`. The
integration scheme is `m` and the time-stepping policy is `ts`.

This method is intended for use with explicit integration methods such
as [`RK4`](@ref). The optional `sym` argument is a callable
`sym(x, s)` that applies a symmetry transformation to the flow's result;
when omitted, no transformation is applied.
"""
flow(g, m::AbstractMethod, ts::AbstractTimeStepping, sym=nothing) =
    flow(g, nothing, m, ts, sym)

"""
    flow(g, A, m::AbstractMethod, ts::AbstractTimeStepping, sym=nothing) -> Flow

Construct a [`Flow`](@ref) for a system with linear part `A` and
nonlinear part `g`. An implicit-explicit integration method `m` such as
[`CNRK2`](@ref) or one of the `CB*` schemes should be supplied. As with
the explicit constructor, `sym(x, s)` is optional and applied via
[`SymTransform`](@ref) to the final state.
"""
flow(g, A, m::AbstractMethod, ts::AbstractTimeStepping, sym=nothing) =
    Flow(ts, m, System(g, A), SymTransform(sym))

"""
    flow(g::Coupled{N}, m::AbstractMethod, ts::AbstractTimeStepping, sym=nothing) where {N} -> Flow

Construct a [`Flow`](@ref) for a composite vector field `g` with the
default call-dependency pattern, i.e. the right-hand sides satisfy

    g[1](t, u[1], dudt[1])
    g[2](t, u[1], dudt[1], u[2], dudt[2])
    …
    g[N](t, u[1], dudt[1], …, u[N], dudt[N])

Use an explicit method such as [`RK4`](@ref). See
[`CallDependency`](@ref) for the syntax to express non-default dependency
patterns. The optional `sym(x, s)` is wrapped with
[`CoupledTransform`](@ref) and applied component-wise.
"""
flow(g::Coupled{N}, m::AbstractMethod, ts::AbstractTimeStepping, sym=nothing) where {N} =
     flow(g, default_dep(N), m, ts, sym)

"""
    flow(g::Coupled{N}, spec::CallDependency{N}, m::AbstractMethod, ts::AbstractTimeStepping, sym=nothing) where {N} -> Flow

Like the previous `flow` overload but with an explicit
[`CallDependency`](@ref) `spec` describing how each component's
right-hand-side depends on the other components.
"""
flow(g::Coupled{N}, spec::CallDependency{N}, m::AbstractMethod,
                                            ts::AbstractTimeStepping,
                                            sym=nothing) where {N} =
     flow(g, couple(ntuple(i->nothing, N)...), spec, m, ts, sym)

"""
    flow(g::Coupled{N}, A::Coupled{N}, m::AbstractMethod, ts::AbstractTimeStepping, sym=nothing) where {N} -> Flow

Construct an implicit-explicit [`Flow`](@ref) for a coupled system,
using the default call-dependency pattern. `A[i]` may be `nothing` for
components that should be advanced fully explicitly.
"""
flow(g::Coupled{N}, A::Coupled{N}, m::AbstractMethod,
                                  ts::AbstractTimeStepping,
                                  sym=nothing) where {N} =
    flow(g, A, default_dep(N), m, ts, sym)

"""
    flow(g::Coupled{N}, A::Coupled{N}, spec::CallDependency{N}, m::AbstractMethod, ts::AbstractTimeStepping, sym=nothing) where {N} -> Flow

Most general coupled constructor: explicit right-hand sides, linear
operators, call dependencies, integration method, time stepping, and
optional symmetry, all specified separately.
"""
flow(g::Coupled{N}, A::Coupled{N}, spec::CallDependency{N},
                                      m::AbstractMethod,
                                     ts::AbstractTimeStepping,
                                     sym=nothing) where {N} =
    Flow(ts, m, System(g, A, spec), CoupledTransform(sym))


# ============================================================================
# Flow CALLABLE INTERFACE
# ============================================================================
#
# All entry points propagate the state `x` in place. Optional positional
# arguments select an alternative kind of integration:
#
#   span      — explicit `(t0, t1)` integration via the time-stepping policy
#   m         — monitor that records observables along the way
#   c         — stage cache that saves internal stages for later replay
#   store     — full state storage, used to drive linearised equations later
#   s         — symmetry parameter, only valid when `Flow.sym` is non-trivial
#
# The methods come in pairs: a `NoTransform`-constrained method for the
# common "no symmetry" case, and a permissive variant that accepts a
# symmetry argument `s` and applies `I.sym` to the result before returning.

"""
    (I::Flow)(x, span::NTuple{2, Real}[, s])

Map `x` from time `span[1]` to time `span[2]`, optionally applying the
symmetry parameterised by `s` at the end.

`x` is modified in place. Its type must match the one used to construct
the integration method, since the method object holds preallocated
buffers of that same type. The optional symmetry parameter `s` may only
be passed when the flow was constructed with a non-trivial `sym`.
"""
(I::Flow{TS, M, S, SO})(x, span::NTuple{2, Real}) where {TS, M, S, SO<:NoTransform} =
    _propagate!(I.meth, I.tstep, I.sys, Float64.(span), x, nothing, nothing, nothing)

(I::Flow)(x, span::NTuple{2, Real}) =
    _propagate!(I.meth, I.tstep, I.sys, Float64.(span), x, nothing, nothing, nothing)

(I::Flow)(x, span::NTuple{2, Real}, s) =
    I.sym(_propagate!(I.meth, I.tstep, I.sys, Float64.(span), x, nothing, nothing, nothing), s)

"""
    (I::Flow)(x, span::NTuple{2, Real}[, s], m::AbstractMonitor)

Map `x` from `span[1]` to `span[2]`, recording samples into the monitor
`m` along the way. See the *Monitors* section of *Trajectory data* in the
manual for how to construct monitors. The optional symmetry parameter `s` is applied to
the final state, exactly as in the basic two-argument form.
"""
(I::Flow{TS, M, S, SO})(x, span::NTuple{2, Real}, m::AbstractMonitor) where {TS, M, S, SO<:NoTransform} =
    _propagate!(I.meth, I.tstep, I.sys, Float64.(span), x, nothing, nothing, m)

(I::Flow)(x, span::NTuple{2, Real}, m::AbstractMonitor) =
    _propagate!(I.meth, I.tstep, I.sys, Float64.(span), x, nothing, nothing, m)

(I::Flow)(x, span::NTuple{2, Real}, s, m::AbstractMonitor) =
    I.sym(_propagate!(I.meth, I.tstep, I.sys, Float64.(span), x, nothing, nothing, m), s)

"""
    (I::Flow)(x, span::NTuple{2, Real}[, s], c::AbstractStageCache)

Map `x` from `span[1]` to `span[2]`, pushing the internal stage values
of each time step into the stage cache `c`. The cache can later be used
to replay the same time grid through a linearised system in a discretely
consistent manner. See the *Stage caches* section of *Trajectory data*
in the manual for details.
"""
(I::Flow{TS, M, S, SO})(x, span::NTuple{2, Real}, c::AbstractStageCache) where {TS, M, S, SO<:NoTransform} =
    _propagate!(I.meth, I.tstep, I.sys, Float64.(span), x, c, nothing, nothing)

(I::Flow)(x, span::NTuple{2, Real}, c::AbstractStageCache) =
    _propagate!(I.meth, I.tstep, I.sys, Float64.(span), x, c, nothing, nothing)

(I::Flow)(x, span::NTuple{2, Real}, s, c::AbstractStageCache) =
    I.sym(_propagate!(I.meth, I.tstep, I.sys, Float64.(span), x, c, nothing, nothing), s)

"""
    (I::Flow)(x, span::NTuple{2, Real}[, s], store::AbstractStorage)

Map `x` from `span[1]` to `span[2]`, pushing snapshots of the state
into the storage `store` along the way. This is the typical way to
build the trajectory required by a continuous adjoint/tangent
integration: the storage acts as the source for the linear operator at
non-grid times. See the *Storages* section of *Trajectory data* in the
manual.
"""
(I::Flow{TS, M, S, SO})(x, span::NTuple{2, Real}, store::AbstractStorage) where {TS, M, S, SO<:NoTransform} =
    _propagate!(I.meth, I.tstep, I.sys, Float64.(span), x, nothing, store, nothing)

(I::Flow)(x, span::NTuple{2, Real}, store::AbstractStorage) =
    _propagate!(I.meth, I.tstep, I.sys, Float64.(span), x, nothing, store, nothing)

(I::Flow)(x, span::NTuple{2, Real}, s, store::AbstractStorage) =
    I.sym(_propagate!(I.meth, I.tstep, I.sys, Float64.(span), x, nothing, store, nothing), s)

"""
    (I::Flow{TimeStepFromCache})(x, c::AbstractStageCache[, s], m::Union{Nothing, <:AbstractMonitor}=nothing)

Replay the stage cache `c` through the linearised system held by `I`,
optionally recording samples into the monitor `m`. The time grid is
fully determined by `c`, so no `span` argument is required. This
produces a discretely consistent linearised/adjoint trajectory.
"""
(I::Flow{TimeStepFromCache, M, S, SO})(x, c::AbstractStageCache,
                                          m::Union{Nothing, <:AbstractMonitor}=nothing) where {M, S, SO<:NoTransform} =
    _propagate!(I.meth, I.sys, x, c, m)

(I::Flow{TimeStepFromCache})(x, c::AbstractStageCache,
                             m::Union{Nothing, <:AbstractMonitor}=nothing) =
    _propagate!(I.meth, I.sys, x, c, m)

(I::Flow{TimeStepFromCache})(x, c::AbstractStageCache, s,
                             m::Union{Nothing, <:AbstractMonitor}=nothing) =
    I.sym(_propagate!(I.meth, I.sys, x, c, m), s)

"""
    (I::Flow{TimeStepFromStorage})(x, store::AbstractStorage, span::NTuple{2, Real}[, s], m::Union{Nothing, <:AbstractMonitor}=nothing)

Map `x` from `span[1]` to `span[2]` using the trajectory in `store` to
evaluate the linear operator at the times required by the continuous
linearised/adjoint scheme. This is *not* a discretely consistent path —
see the `TimeStepFromCache` variant above for the consistent
alternative. Optional `m` records observables.
"""
(I::Flow{TimeStepFromStorage, M, S, SO})(x,
                                         store::AbstractStorage,
                                         span::NTuple{2, Real},
                                         m::Union{Nothing, <:AbstractMonitor}=nothing) where {M, S, SO<:NoTransform} =
    _propagate!(I.meth, I.tstep, I.sys, Float64.(span), x, store, m)

(I::Flow{TimeStepFromStorage})(x,
                               store::AbstractStorage,
                               span::NTuple{2, Real},
                               m::Union{Nothing, <:AbstractMonitor}=nothing) =
    _propagate!(I.meth, I.tstep, I.sys, Float64.(span), x, store, m)

(I::Flow{TimeStepFromStorage})(x,
                               store::AbstractStorage,
                               span::NTuple{2, Real},
                               s,
                               m::Union{Nothing, <:AbstractMonitor}=nothing) =
    I.sym(_propagate!(I.meth, I.tstep, I.sys, Float64.(span), x, store, m), s)


# ============================================================================
# PROPAGATION FUNCTIONS & UTILITIES
# ============================================================================

"""
    InvalidSpanError(span)

Exception raised by a [`Flow`](@ref) call when the requested integration
span is invalid. For the non-distributed paths this means the span is
strictly decreasing (`span[1] > span[2]`).
"""
struct InvalidSpanError{S} <: Exception
    span::S
end

Base.showerror(io::IO, e::InvalidSpanError) =
    print(io, "Invalid time span ", e.span, ". Time must be increasing.\n")

"""
    @_checkspan(span, z)

Validate an integration span and short-circuit pathological cases. If
`span[1] == span[2]` the macro causes the enclosing function to return
`z` immediately (no work to do); if `span[1] > span[2]` it throws an
[`InvalidSpanError`](@ref).
"""
macro _checkspan(span, z)
    quote
       $(esc(span))[1] ==$(esc(span))[2] && return $(esc(z))
       $(esc(span))[1]  >$(esc(span))[2] && throw(InvalidSpanError($(esc(span))))
    end
end

# ============================================================================
# CONSTANT TIME-STEP INTEGRATION FOR NONLINEAR EQUATIONS OR COUPLED SYSTEMS
# ============================================================================
function _propagate!(method::AbstractMethod{Z, NormalMode},
                   stepping::TimeStepConstant,
                     system::System,
                       span::NTuple{2, Real},
                          z::Z,
                      cache::C,
                      store::S,
                        mon::M) where {Z,
                                       S<:Union{Nothing, AbstractStorage},
                                       M<:Union{Nothing, AbstractMonitor},
                                       C<:Union{Nothing, AbstractStageCache}}
    # checks
    if cache isa AbstractStageCache
        nstages(method) == nstages(cache) ||
            throw(ArgumentError("incompatible method and stage cache"))
    end

    # check span is sane
    @_checkspan(span, z)

    # define integration times
    tdts = Steps(span[1], span[2], stepping.Δt)

    # the number of steps is used for the `StoreNFromLast` monitor
    nsteps = length(tdts)

    # always push initial state to monitor and storage
    M <: AbstractMonitor && push!(mon,   span[1], z, true)
    S <: AbstractStorage && push!(store, span[1], copy(z))

    # if we have a storage, we might need to skip pushing the last element, based
    # on the value of the boolean `storelast(store)`. If we need to skip it
    # we set the variable `j_skip` so that when `j == j_skip`, we do not push.
    # otherwise we set `j_skip` to zero, so we always push since `j = 1, 2, 3, ...`
    j_skip = (S <: AbstractStorage) && storelast(store) == true ? 0 : nsteps

    # start integration
    for (j, (t, dt)) in enumerate(tdts)
        step!(method, system, t, dt, z, cache)
        if  M <: AbstractMonitor
            # skip all pushes except N steps from the last one
            M <: StoreNFromLast && (j != nsteps - getN(mon) && continue)

            # we might need to force pushing the last element to the monitor
            force = j == nsteps ? true : false

            push!(mon, t + dt, z, force)
        end
        if S <: AbstractStorage
            if j != j_skip
                push!(store, t + dt, copy(z))
            end
        end
    end

    return z
end

# ============================================================================
# PROPAGATION BASED ON SYSTEMS HOOK: ONLY FOR STATE EQUATIONS
# ============================================================================
function _propagate!(method::AbstractMethod{Z, MODE},
                       hook::AbstractTimeStepFromHook,
                     system::System,
                       span::NTuple{2, Real},
                          z::Z,
                      cache::C,
                      store::S,
                        mon::M) where {Z,
                                       MODE<:NormalMode,
                                       S<:Union{Nothing, AbstractStorage},
                                       M<:Union{Nothing, AbstractMonitor},
                                       C<:Union{Nothing, AbstractStageCache}}
    # checks
    if cache isa AbstractStageCache
        nstages(method) == nstages(cache) ||
            throw(ArgumentError("incompatible method and stage cache "))
    end

    # check span is sane
    @_checkspan(span, z)

    # init and final times
    t, T = span

    # store initial state in monitors
    M <: AbstractMonitor && push!(mon,   t, z)
    S <: AbstractStorage && push!(store, t, copy(z))

    # run until condition
    while t != T
        # obtain time step from hook and what the next time is
        t_next, Δt = _next_Δt(t, T, hook(system.g, system.A, z))

        # advance
        step!(method, system, t, Δt, z, cache)

        # update
        t = t_next

        # store solution into monitor
        M <: AbstractMonitor && push!(mon, t, z)
        S <: AbstractStorage && push!(store, t, copy(z))
    end

    return z
end

# Return the next time and the time step that exactly reaches `T`, given the
# raw step size returned by the hook. We deliberately clamp at `T` so the
# integration terminates on the requested endpoint rather than overshooting.
function _next_Δt(t, T, Δt::S) where {S<:Real}
    @assert Δt > 0 "negative time step encountered"
    t_next = ifelse(t ≤ T, min(t+Δt, T), max(t-Δt, T))
    return t_next, S(t_next - t)
end

# ============================================================================
# TIME STEPPING BASED ON CACHED STAGES, ONLY FOR LINEARISED EQUATIONS
# ============================================================================
function _propagate!(method::AbstractMethod{Z, MODE},
                     system::System,
                          z::Z,
                      cache::AbstractStageCache,
                        mon::M) where {Z, MODE<:DiscreteMode,
                                       M<:Union{Nothing, AbstractMonitor}}
    # checks
    nstages(method) == nstages(cache) ||
        throw(ArgumentError("incompatible method and stage cache "))

    # TODO: fix this with proper iteration support for the stage cache
    ts  = cache.ts
    Δts = cache.Δts
    xs  = cache.xs

    # integrate forward or backward based on type of linear equation
    if isadjoint(MODE) == false
        # store final state in monitors. Note cache does not contain final T.
        M <: AbstractMonitor && push!(mon, ts[1], z)

        for i in 1:length(ts)
            # make step
            step!(method, system, ts[i], Δts[i], z, xs[i])

            # then save current state
            M <: AbstractMonitor && push!(mon, ts[i]+Δts[i], z)
        end
    else
        # store final state in monitors. Note cache does not contain final T.
        M <: AbstractMonitor && push!(mon, ts[end] + Δts[end], z)

        for i in reverse(1:length(ts))
            step!(method, system, ts[i], Δts[i], z, xs[i])
            M <: AbstractMonitor && push!(mon, ts[i], z)
        end
    end

    return z
end

# ============================================================================
# TIME STEPPING BASED ON STORAGE FOR CONTINUOUS ADJOINT/TANGENT EQUATIONS
# ============================================================================
function _propagate!(method::AbstractMethod{Z, MODE},
                   stepping::TimeStepFromStorage,
                     system::System,
                       span::NTuple{2, Any},
                          z::Z,
                      store::AbstractStorage,
                        mon::M) where {Z,
                                       MODE<:ContinuousMode,
                                       M<:Union{Nothing, AbstractMonitor}}

    # Define integration times and time steps. The adjoint case,
    # where span[1] > span[2], is handled automatically here by
    # `tdts`, where the `dt` is negative
    tdts = Steps(span[1], span[2], stepping.Δt)

    # count number of steps, because we might need to force pushing the
    # last element to the monitor, even if it has a `oneevery` parameter
    # that is not a integer divisor of `nsteps`
    nsteps = length(tdts)

    # store initial state in monitors (this could be the final adjoint state)
    M <: AbstractMonitor && push!(mon, span[1], z, true)

    # March in time. Note final value of`t` and `dt` is
    # such that `t + dt = span[2]`
    for (j, (t, dt)) in enumerate(tdts)
        # exec step
        step!(method, system, t, dt, z, store)

        # we might need to force pushing the last element to the monitor
        force = j == nsteps ? true : false

        # store
        M <: AbstractMonitor && push!(mon, t + dt, z, force)
    end

    return z
end
