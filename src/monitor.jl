export Monitor, reset!

# ///  Abstract type for all solution monitors ///
abstract type AbstractMonitor{T, X} end

# ///// UTILS //////
# whether t is between low and high
isbetween(t::Real, low::Real, high::Real) = (t ≥ low && t ≤ high)

_ismonitor(::Type{<:AbstractMonitor}) = true
_ismonitor(::Any) = false


# /// Monitor to save all time steps ///
mutable struct Monitor{T, X, S<:AbstractStorage{T, X}, F} <: AbstractMonitor{T, X}
          store::S                       # (time, samples) tuples
              f::F                       # action on what is begin pushed
       oneevery::Int                     # save every ... time steps
    savebetween::Tuple{Float64, Float64} # save only between these two times 
          count::Int                     # how many items we have in the store
      skipfirst::Bool                    # skip the first sample?
    Monitor(store::S,
                f::F,
                oneevery::Int, 
                savebetween::Tuple{Real, Real},
                skipfirst::Bool) where {T, X, S<:AbstractStorage{T, X}, F} =
        new{T, X, S, F}(store, f, oneevery, savebetween, 0, skipfirst)
end

"""

    Monitor(x, f::Base.Callable=identity, store::S=RAMStorage(f(x)); oneevery::Int=1, savebetween::Tuple{Real, Real}=(-Inf, Inf), sizehint::Int=0)


Construct a `Monitor` object to record one observable quantity along a trajectory. 

The argument `x` is an object of the same type used to represent the system's state, 
while `f` is a callable object or function that calculates the observable from the state. 
In other words, the quantity `f(x)` is monitored along a trajectory, and stored in 
`store`, which defaults to a [`RAMStorage`](@ref) object. One sample every `onevery` 
samples is stored.

If required, only samples at times falling in the range specified by `savebetween` are 
stored. Specifying the number of samples stored with the `sizehint` keyword argument
may increase performance.

A `Monitor` object can then be passed as an additional argument to a [`Flows.Flow`](@ref)
object.

See also [`reset!`](@ref), [`times`](@ref) and [`samples`](@ref).
"""
Monitor(x,
        f::Base.Callable=identity,
        store::S=RAMStorage(f(0.0, x));
        oneevery::Int=1,
        savebetween::Tuple{Real, Real}=(-Inf, Inf),
        skipfirst::Bool=false,
        sizehint::Int=0) where {S<:AbstractStorage} =
    Monitor(reset!(store, sizehint), f, oneevery, savebetween, skipfirst)

# Add sample and time to the storage
@inline function Base.push!(mon::Monitor, t::Real, x, force::Bool=false)
    if force == true || (mon.count % mon.oneevery == 0)
        if isbetween(t, mon.savebetween...)
            if !(mon.count == 0 && mon.skipfirst)
                push!(mon.store, t, mon.f(t, x))
            end
        end
    end
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