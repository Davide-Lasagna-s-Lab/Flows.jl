export Monitor, reset!

# ///  Abstract type for all solution monitors ///
abstract type AbstractMonitor{T, X} end

# ///// UTILS //////
# whether t is between low and high
isbetween(t::Real, low::Real, high::Real) = (t ≥ low && t ≤ high)

_ismonitor(::Type{<:AbstractMonitor}) = true
_ismonitor(::Any) = false


# /// Monitor to save all time steps ///
mutable struct Monitor{T, X, S<:AbstractStorage{T, X}, F, L<:AbstractLogger} <: AbstractMonitor{T, X}
          store::S                       # (time, samples) tuples
              f::F                       # action on what is begin pushed
       oneevery::Int                     # save every ... time steps
    savebetween::Tuple{Float64, Float64} # save only between these two times 
          count::Int                     # how many items we have in the store
      skipfirst::Bool                    # skip the first sample?
       skiplast::Bool                    # skip each integration endpoint?
            log::L                       # logger to handle the print formatting
    Monitor(store::S,
                f::F,
                oneevery::Int, 
                savebetween::Tuple{Real, Real},
                skipfirst::Bool,
                skiplast::Bool,
                log::L) where {T, X, S<:AbstractStorage{T, X}, F, L<:AbstractLogger} =
        new{T, X, S, F, L}(store, f, oneevery, savebetween, 0, skipfirst, skiplast, log)
end

"""

    Monitor(x, f=(t, x)->x, store=RAMStorage(f(0.0, x));
            oneevery=1, savebetween=(-Inf, Inf), skipfirst=false, skiplast=false,
            sizehint=0, io=devnull, logevery=1)


Construct a `Monitor` object to record one observable quantity along a trajectory. 

The argument `x` is an object of the same type used to represent the system's state, 
while `f` is a callable object or function that calculates the observable from the state. 
In other words, the quantity `f(t, x)` is monitored along a trajectory, and stored in 
`store`, which defaults to a [`RAMStorage`](@ref) object. One sample every `onevery` 
samples is stored.

If required, only samples at times falling in the range specified by `savebetween` are 
stored. Specifying the number of samples stored with the `sizehint` keyword argument
may increase performance.

Set `skiplast=true` to omit the endpoint of each integration call, including
endpoints selected by `oneevery`. The observable and logger are not called for
that sample, but the sampling counter still advances within that integration call.
Each integration restarts the counter without clearing previously stored samples. `skipfirst=true` skips
the initial sample of each integration call. Both default to `false`.
Direct `push!` calls have no endpoint information and are unaffected by `skiplast`.

In addition, the monitor values can be output to `io`. Specifying `logevery` skips
the output of the monitor state for the given number of monitor counts. See
[`Logger`](@ref)

A `Monitor` object can then be passed as an additional argument to a [`Flows.Flow`](@ref)
object.

See also [`reset!`](@ref), [`times`](@ref) and [`samples`](@ref).
"""
Monitor(x,
        f::Base.Callable=(t,x)->identity(x),
        store::S=RAMStorage(f(0.0, x));
        oneevery::Int=1,
        savebetween::Tuple{Real, Real}=(-Inf, Inf),
        skipfirst::Bool=false,
        skiplast::Bool=false,
        sizehint::Int=0,
        io::IO=devnull,
        logevery::Int=1) where {S<:AbstractStorage} =
    Monitor(reset!(store, sizehint), f, oneevery, savebetween, skipfirst, skiplast, Logger(io, f(0.0, x), logevery))

"""
    push!(mon::Monitor, t, x, force=false, first=false, last=false)

Offer a sample to the monitor. `force` bypasses `oneevery`, but not the
endpoint flags or `savebetween`. Set `first=true` at an integration's initial
sample to restart the counter without clearing storage; set `last=true` at
its endpoint to apply `skiplast`. Skipped samples still advance the counter.
"""
# Integration boundaries control the cadence and endpoint filtering here,
# keeping those details out of the integrator. Ordinary pushes need no flags.
@inline function Base.push!(mon::Monitor, t::Real, x, force::Bool=false,
                           first::Bool=false, last::Bool=false)
    first && (mon.count = 0)
    if last && mon.skiplast
        mon.count += 1
        return nothing
    end
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
    reset!(mon::Monitor, sizehint::Int=0)

Reset the internal storage of a [`Monitor`](@ref) object `mon`.
"""
reset!(mon::Monitor, sizehint::Int=0) =
    (reset!(mon.store, sizehint); mon.count = 0; mon)

"""
    times(mon::Monitor)

Return the times at which samples of the observable have been stored. This is most 
typically after each time step, in addition to the initial condition. The type of the 
returned object depend on the internal storage. For [`RAMStorage`](@ref) storages, this
is a standard `Vector`.
"""
times(mon::Monitor) = times(mon.store)

"""
    samples(mon::Monitor)

Return samples of the observable that have been stored during a trajectory. This is most 
typically after each time step, in addition to the initial condition. The type of the 
returned object depend on the internal storage. For [`RAMStorage`](@ref) storages, this
is a standard `Vector`.
"""
samples(mon::Monitor) = samples(mon.store)
