export AbstractStageCache, RAMStageCache

# ============================================================================
# AbstractStageCache
# ============================================================================

"""
    AbstractStageCache{NS, X}

Supertype for objects that buffer the internal stage values produced
by an integration scheme so a linearised counterpart can be replayed
on exactly the same time grid.

`NS` is the number of internal stages produced per time step (and must
match the integration scheme); `X` is the type of one stage value
(typically the state type). Concrete subtypes must implement
`Base.push!(cache, t, Δt, stages::NTuple{NS, X})` and the [`reset!`](@ref)
operation.
"""
abstract type AbstractStageCache{NS, X} end

"""
    nstages(::AbstractStageCache) -> Int

Return the number of internal stages stored per time step.
"""
nstages(::AbstractStageCache{NS}) where {NS} = NS

# trait used by step! methods to dispatch differently on stage caching
_iscache(::Type{<:AbstractStageCache}) = true
_iscache(::Any) = false


# ============================================================================
# RAMStageCache — concrete in-memory cache
# ============================================================================

"""
    RAMStageCache{NS, X} <: AbstractStageCache{NS, X}

Concrete in-memory implementation of [`AbstractStageCache`](@ref).
Three parallel vectors hold the start time, step size, and stage
values of each accepted step. Construct with one of the outer
constructors below.
"""
struct RAMStageCache{NS, X} <: AbstractStageCache{NS, X}
     ts::Vector{Float64}
    Δts::Vector{Float64}
     xs::Vector{NTuple{NS, X}}
end

"""
    RAMStageCache(NS::Int, x)
    RAMStageCache(NS::Int, ::Type{X})

Construct an empty [`RAMStageCache`](@ref). The first form infers the
stage element type from an example object `x`; the second form takes
the type directly. `NS` is the number of internal stages per step and
must match the integration scheme that will be used to populate the
cache.
"""
RAMStageCache(NS::Int, x) =
    RAMStageCache(NS, typeof(x))

RAMStageCache(NS::Int, ::Type{X}) where {X} =
    RAMStageCache{NS, X}(Float64[], Float64[], NTuple{NS, X}[])

"""
    push!(cache::RAMStageCache, t, Δt, stages) -> nothing

Append a `(t, Δt, stages)` triple to the cache. `stages` must be an
`NTuple{NS, X}` whose shape matches the type parameters of `cache`.
"""
@inline Base.push!(ss::RAMStageCache{NS, X},
                    t::Real,
                   Δt::Real,
                    x::NTuple{NS, X}) where {NS, X} =
    (push!(ss.ts, t); push!(ss.Δts, Δt); push!(ss.xs, x); nothing)

"""
    reset!(cache::RAMStageCache) -> cache

Drop all stored steps, leaving the cache empty and ready to be
populated again. Returns `cache`.
"""
reset!(ss::RAMStageCache) =
    (resize!(ss.ts, 0); resize!(ss.Δts, 0); resize!(ss.xs, 0); ss)

"""
    similar(cache::RAMStageCache) -> RAMStageCache

Return a fresh, empty cache of the same shape (`NS`, `X`) as `cache`.
"""
Base.similar(ss::RAMStageCache{NS, X}) where {X, NS} =
    RAMStageCache(NS, X)

"""
    copy(cache::RAMStageCache) -> RAMStageCache

Return a shallow copy of `cache`. The vectors backing `ts`, `Δts`, and
`xs` are independent; the stage tuples themselves are shared with the
original — copy each stage if independent mutation is required.
"""
Base.copy(ss::RAMStageCache) =
    RAMStageCache(copy(ss.ts), copy(ss.Δts), copy(ss.xs))
