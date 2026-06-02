export Monitor, reset!

# ============================================================================
# AbstractMonitor and helpers
# ============================================================================

"""
    AbstractMonitor{T, X}

Supertype for objects that observe a [`Flow`](@ref) trajectory. `T` is
the time type, `X` is the type of the *observed* sample (which may
differ from the system state if a non-identity observable is in use).

Concrete subtypes must implement `Base.push!(mon, t, x, force::Bool)`
to ingest one observation per time step. They may also implement
[`reset!`](@ref) and the accessors [`times`](@ref) / [`samples`](@ref)
when they back the observations with a storage.
"""
abstract type AbstractMonitor{T, X} end

# whether `t` falls in the closed interval `[low, high]`
isbetween(t::Real, low::Real, high::Real) = (t ≥ low && t ≤ high)

# trait used by `_propagate!` to dispatch differently on monitor presence
_ismonitor(::Type{<:AbstractMonitor}) = true
_ismonitor(::Any) = false


# ============================================================================
# Monitor — records one observable along a trajectory
# ============================================================================

"""
    Monitor{T, X, S, F, L}

Concrete monitor that records the value of a single observable along a
trajectory. The observable is computed by a user-supplied callable `f`
with signature `f(t, x)` and is stored in `store`, together with the
observation time.

The standard control knobs are:

  - `oneevery::Int`: keep one sample every `oneevery` calls.
  - `savebetween::Tuple{Float64, Float64}`: only store samples whose
    time falls in the closed interval `savebetween`.
  - `skipfirst::Bool`: drop the very first observation. Useful when the
    initial state is artificial.
  - `log::AbstractLogger`: a [`Logger`](@ref) used to print the running
    state of the monitor to a stream as samples come in.

Construct via the [`Monitor`](@ref) outer constructor below.
"""
mutable struct Monitor{T, X, S<:AbstractStorage{T, X}, F, L<:AbstractLogger} <: AbstractMonitor{T, X}
          store::S                       # (time, samples) backing storage
              f::F                       # observable f(t, x) → sample
       oneevery::Int                     # save every `oneevery` samples
    savebetween::Tuple{Float64, Float64} # save only between these two times
          count::Int                     # how many times push! has been called
      skipfirst::Bool                    # skip the very first observation?
            log::L                       # logger used to print samples
    Monitor(store::S,
                f::F,
                oneevery::Int,
                savebetween::Tuple{Real, Real},
                skipfirst::Bool,
                log::L) where {T, X, S<:AbstractStorage{T, X}, F, L<:AbstractLogger} =
        new{T, X, S, F, L}(store, f, oneevery, savebetween, 0, skipfirst, log)
end

"""
    Monitor(x, f=(t,x)->identity(x), store=RAMStorage(f(0.0, x)); kwargs...)

Construct a `Monitor` recording one observable along a trajectory.

`x` is an object of the same type used to represent the system state
and serves only as a template for type inference. `f` is the
observable: a callable with signature `f(t, x)` whose return value is
what will actually be stored. By default `f(t, x) = x` (just record
the state).

`store` defaults to a [`RAMStorage`](@ref) instantiated from the type
of `f(0.0, x)`. Supply a custom storage if the trajectory is too long
to keep entirely in RAM, or if a different element type is required.

# Keyword arguments
  - `oneevery::Int = 1`: keep one sample every `oneevery` calls.
  - `savebetween::Tuple{Real, Real} = (-Inf, Inf)`: only store samples
    whose time falls in this closed interval.
  - `skipfirst::Bool = false`: drop the very first observation.
  - `sizehint::Int = 0`: pre-allocate capacity in the storage. Useful
    to avoid reallocation when the number of samples is known.
  - `io::IO = devnull`: stream to which the [`Logger`](@ref) prints.
    Defaults to `devnull` (silent).
  - `logevery::Int = 1`: pass-through to [`Logger`](@ref); only every
    `logevery`-th sample is printed.

A `Monitor` instance is passed to a [`Flow`](@ref) as an additional
positional argument; the flow then drives `push!` at each accepted
sample point.

See also [`reset!`](@ref), [`times`](@ref) and [`samples`](@ref).
"""
Monitor(x,
        f::Base.Callable=(t,x)->identity(x),
        store::S=RAMStorage(f(0.0, x));
        oneevery::Int=1,
        savebetween::Tuple{Real, Real}=(-Inf, Inf),
        skipfirst::Bool=false,
        sizehint::Int=0,
        io::IO=devnull,
        logevery::Int=1) where {S<:AbstractStorage} =
    Monitor(reset!(store, sizehint), f, oneevery, savebetween, skipfirst, Logger(io, f(0.0, x), logevery))

"""
    push!(mon::Monitor, t::Real, x, force::Bool=false)

Push the observation `f(t, x)` into the monitor, subject to the
`oneevery`, `savebetween` and `skipfirst` policies. `force=true`
bypasses the `oneevery` decimation but still respects the time-window
filter, and is used by the propagation routines to guarantee that the
final state of an integration is always recorded.
"""
@inline function Base.push!(mon::Monitor, t::Real, x, force::Bool=false)
    if force == true || (mon.count % mon.oneevery == 0)
        if isbetween(t, mon.savebetween...)
            if !(mon.count == 0 && mon.skipfirst)
                push!(mon.store, t, mon.f(t, x))

                # output monitor state
                mon.count == 0 && mon.log()
                mon.log(mon.count, mon.store)
            end
        end
    end

    # update monitor call count
    mon.count += 1

    return nothing
end

"""
    reset!(mon::Monitor, sizehint::Int=0) -> mon

Clear the backing storage of `mon` and reset its sample counter to
zero. The optional `sizehint` is forwarded to the underlying storage
so that subsequent pushes can avoid reallocation. Returns `mon`.
"""
reset!(mon::Monitor, sizehint::Int=0) =
    (reset!(mon.store, sizehint); mon.count = 0; mon)

"""
    times(mon::Monitor) -> AbstractVector

Return the times at which the monitor's observable was sampled. Each
entry corresponds to one accepted `push!` call. For a
[`RAMStorage`](@ref) backing this is a standard `Vector`.
"""
times(mon::Monitor) = times(mon.store)

"""
    samples(mon::Monitor) -> AbstractVector

Return the recorded observable values, indexed compatibly with
[`times`](@ref). For a [`RAMStorage`](@ref) backing this is a standard
`Vector` whose element type is whatever the observable function
returns.
"""
samples(mon::Monitor) = samples(mon.store)
