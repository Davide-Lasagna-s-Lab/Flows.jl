export StoreNFromLast

"""
    StoreNFromLast{N, X, F} <: AbstractMonitor{Float64, X}

A specialised monitor that records exactly one observation, taken `N`
time steps before the final step of an integration. The recorded
sample is `f(z)` where `f` defaults to `Base.copy`.

This is useful for collecting the terminal state of a trajectory and
its immediate predecessors — for instance, to seed a linearised
adjoint integration starting from a state slightly before the
nonlinear endpoint.

# Type parameters
  - `N`: the number of steps before the last to record. `N = 0`
    records the final state itself.
  - `X`: the type of the stored observation, inferred from `f(z)`.
  - `F`: the type of the observable callable.
"""
mutable struct StoreNFromLast{N, X, F} <: AbstractMonitor{Float64, X}
    x::X
    f::F
    t::Float64

    """
        StoreNFromLast{N}(z, f=Base.copy)

    Construct a [`StoreNFromLast`](@ref) that will record `f(z)`
    `N` steps before the last step of an integration. `z` is only
    used as a template to allocate the storage; its contents are not
    captured.
    """
    function StoreNFromLast{N}(z, f::F = Base.copy) where {N, F}
        x = f(z)
        new{N, typeof(x), F}(x, f, 0.0)
    end
end

"""
    getN(::StoreNFromLast{N}) -> Int

Return the offset `N` of a [`StoreNFromLast`](@ref) monitor: the
number of steps before the final step at which the observation is
recorded.
"""
getN(::StoreNFromLast{N}) where {N} = N

"""
    push!(mon::StoreNFromLast, t::Real, z, force::Bool=false)

Overwrite the single stored observation with `f(z)` and the recorded
time with `t`. The `force` flag is accepted for interface
compatibility with other monitors but is not used: the propagation
loop already chooses *when* to push by comparing the step index
against `getN(mon)`.
"""
Base.push!(mon::StoreNFromLast, t::Real, z, ::Bool) =
    (mon.x .= mon.f(z); mon.t = t; nothing)
