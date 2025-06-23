export flow, InvalidSpanError

# ---------------------------------------------------------------------------- #
mutable struct Flow{TS<:AbstractTimeStepping, M<:AbstractMethod, S<:System}
    tstep::TS # the method used for time stepping
     meth::M  # the method, with storage, implementation and time stepping
      sys::S  # the system to be integrated
    Flow(ts::TS, m::M, sys::S) where {TS, M, S} = new{TS, M, S}(ts, m, sys)
end

"""
    flow(g, m::AbstractMethod, ts::AbstractTimeStepping)

Construct an object of type `Flow`, representing the numerical dicretisation
of the time-forward flow operator associated to the vector field `g`, using the
integration method `m`, with time stepping provided by `ts`. This method
should be used with an explicit integration method.
"""
flow(g, m::AbstractMethod, ts::AbstractTimeStepping) =
    flow(g, nothing, m, ts)

"""
    flow(g, A, m::AbstractMethod, ts::AbstractTimeStepping)

Construct a flow operator associated to the vector field defined by a linear
component `A` and a nonlinear part `g`. An implicit-explicit integration 
method `m` should be provided. 
"""
flow(g, A, m::AbstractMethod, ts::AbstractTimeStepping) =
     Flow(ts, m, System(g, A))

"""
    flow(g::Coupled{N}, m::AbstractMethod, ts::AbstractTimeStepping) where {N}

Construct a flow operator associated to the composite vector field `g`, using
a default call dependency structure, i.e. where the elements of `g` satisfy
the calling interface:

    g[1](t, u[1], dudt[1])
    g[2](t, u[1], dudt[1], u[2], dudt[2]) ...
    g[N](t, u[1], dudt[1], u[N], dudt[N])

See [Flows.jl Call Dependencies](@ref) for more details on how to specify
custom call dependencies. This method should be used with an explicit integrator.
"""
flow(g::Coupled{N}, m::AbstractMethod, ts::AbstractTimeStepping) where {N} =
     flow(g, default_dep(N), m, ts)

"""
    flow(g::Coupled{N}, spec::CallDependency{N}, m::AbstractMethod, ts::AbstractTimeStepping) where {N}

Similar to the method without `spec`, but specifying a custom call dependency structure.
"""
flow(g::Coupled{N}, spec::CallDependency{N}, m::AbstractMethod,
                                            ts::AbstractTimeStepping) where {N} =
     flow(g, couple(ntuple(i->nothing, N)...), spec, m, ts)

"""
    flow(g::Coupled{N}, A::Coupled{N}, m::AbstractMethod, ts::AbstractTimeStepping) where {N}

Similar to previous methods, but also provide the linear part of the dynamical
system. This method should be used with an implicit-explicit integrator.
"""
flow(g::Coupled{N}, A::Coupled{N}, m::AbstractMethod,
                                  ts::AbstractTimeStepping) where {N} =
    flow(g, A, default_dep(N), m, ts)

"""
    flow(g::Coupled{N}, A::Coupled{N}, spec::CallDependency{N}, m::AbstractMethod, ts::AbstractTimeStepping) where {N}

Similar to previous methods, but provide a custom call dependency structure.
This method should be used with an implicit-explicit integrator.
"""
flow(g::Coupled{N}, A::Coupled{N}, spec::CallDependency{N},
                                      m::AbstractMethod,
                                     ts::AbstractTimeStepping) where {N} =
    Flow(ts, m, System(g, A, spec))

# ---------------------------------------------------------------------------- #
# FLOWS ARE CALLABLE OBJECTS: THIS IS THE MAIN INTERFACE

"""
    (I::Flow)(x, span::NTuple{2, Real})

Map `x` at time `span[1]` to the later time `span[2]`. 

The object `x` is modified in place. The argument `x` shoule be of a type 
compatible to that used to create the integration method object for the `Flow`
object `I`, since the integration method contains preallocated elements used 
to perform the integration step.
"""
(I::Flow)(x, span::NTuple{2, Real}) =
    _propagate!(I.meth, I.tstep, I.sys, Float64.(span), x, nothing, nothing, nothing)

"""
    (I::Flow)(x, span::NTuple{2, Real}, m::AbstractMonitor)

Map `x` at time `span[1]` to the later time `span[2]`, filling the monitor
obejct `m` along the way. See [`Flow.jl Monitor objects`](@ref) for more details
on how to define and use `Monitor` objects.
"""
(I::Flow)(x, span::NTuple{2, Real}, m::AbstractMonitor) =
    _propagate!(I.meth, I.tstep, I.sys, Float64.(span), x, nothing, nothing, m)

"""
    (I::Flow)(x, span::NTuple{2, Real}, c::AbstractStageCache)

Map `x` at time `span[1]` to the later time `span[2]`, filling the stage cache
object `c` along the way. See [`Flow.jl Stage Caches`](@ref) for more details
on how to define and use `AbstractStageCache` objects.
"""
(I::Flow)(x, span::NTuple{2, Real}, c::AbstractStageCache) =
    _propagate!(I.meth, I.tstep, I.sys, Float64.(span), x, c, nothing, nothing)

"""
    (I::Flow)(x, span::NTuple{2, Real}, s::AbstractStorage)

Map `x` at time `span[1]` to the later time `span[2]`, filling the storage 
object `s` along the way. This method is used primarily to fill a storage 
object with the results of a nonlinear simulation, where the storage `s` can
be subsequently used for the linearised systems. See [`Flow.jl Storages`](@ref) 
for more details on how to define and use `AbstractStorage` objects.
"""
(I::Flow)(x, span::NTuple{2, Real}, s::AbstractStorage) =
    _propagate!(I.meth, I.tstep, I.sys, Float64.(span), x, nothing, s, nothing)

"""
    (I::Flow{TimeStepFromCache})(x, c::AbstractStageCache, m::Union{Nothing, <:AbstractMonitor}=nothing)

Map `x` forward/backward over a time span defined by the stage cache object `c`, 
filling the monitor object `m` along the way. This method is primarily used to 
integrate linearised equations forward/backward, so that the nonlinear and 
linearised methods are discretely consistent. See [`Flow.jl Stage Caches`](@ref) 
for more details on how to define and use `AbstractStorage` objects.
"""
(I::Flow{TimeStepFromCache})(x, c::AbstractStageCache,
                             m::Union{Nothing, <:AbstractMonitor}=nothing) =
    _propagate!(I.meth, I.sys, x, c, m)

"""
    (I::Flow{TimeStepFromStorage})(x, s::AbstractStorage, span::NTuple{2, Real}, m::Union{Nothing, <:AbstractMonitor}=nothing)

Map `x` forward/backward over a time span `(span[1], span[2])` using the nonlinear
trajectory stored in `s` to drive linearised equations. An additional `Monitor` object
`m` can be filled along the way. The monitor object is fed with elements that are 
similar to `x`. This method can be used to integrate forward or 
adjoint equations in a way that is not discretely consistent.
"""
(I::Flow{TimeStepFromStorage})(x,
                               s::AbstractStorage,
                               span::NTuple{2, Real},
                               m::Union{Nothing, <:AbstractMonitor}=nothing) =
    _propagate!(I.meth, I.tstep, I.sys, Float64.(span), x, s, m)

# ---------------------------------------------------------------------------- #
# PROPAGATION FUNCTIONS & UTILS

struct InvalidSpanError{S} <: Exception
    span::S
end

Base.showerror(io::IO, e::InvalidSpanError) =
    print(io, "Invalid time span ", e.span, ". Time must be increasing.\n")

"""
    @_checkspan(span, z)

Check `span[1] < span[2]`, throw an error `span[1] > span[2]` and return `z`
directly if `span[1] == span[2]`. 
"""
macro _checkspan(span, z)
    quote
       $(esc(span))[1] ==$(esc(span))[2] && return $(esc(z))
       $(esc(span))[1]  >$(esc(span))[2] && throw(InvalidSpanError($(esc(span))))
    end
end

# ---------------------------------------------------------------------------- #
# CONSTANT TIME-STEP INTEGRATION FOR NONLINEAR EQUATIONS OR COUPLED SYSTEMS
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

    # the number of steps is used for the `StoreOnlyLast` monitor
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
            # skip all pushes except the last but one
            M <: StoreOnlyLast && (j != nsteps && continue)

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

# ---------------------------------------------------------------------------- #
# PROPAGATION BASED ON SYSTEMS HOOK: ONLY FOR STATE EQUATIONS
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

# return next time and time step that allows hitting the end point exactly
function _next_Δt(t, T, Δt::S) where {S<:Real}
    @assert Δt > 0 "negative time step encountered"
    t_next = ifelse(t ≤ T, min(t+Δt, T), max(t-Δt, T))
    return t_next, S(t_next - t)
end

# ---------------------------------------------------------------------------- #
# TIME STEPPING BASED ON CACHED STAGES, ONLY FOR LINEARISED EQUATIONS
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

# ---------------------------------------------------------------------------- #
# TIME STEPPING BASED ON STORAGE FOR CONTINUOS ADJOINT/TANGENT EQUATIONS
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