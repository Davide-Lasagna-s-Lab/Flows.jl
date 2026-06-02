export AbstractTimeStepping,
       AbstractTimeStepFromHook,
       TimeStepConstant,
       TimeStepFromCache,
       TimeStepFromStorage

"""
    AbstractTimeStepping

Supertype for all time-stepping policies. A subtype determines how the
size of each step is chosen and how the integration domain is
traversed. Concrete subtypes include [`TimeStepConstant`](@ref),
[`TimeStepFromCache`](@ref), [`TimeStepFromStorage`](@ref), and
hook-driven schemes deriving from [`AbstractTimeStepFromHook`](@ref).
"""
abstract type AbstractTimeStepping end

"""
    AbstractTimeStepFromHook <: AbstractTimeStepping

Supertype for time-stepping policies that defer the choice of step
size to a user-supplied hook called at runtime. The hook is invoked
once per step with the current right-hand side, linear operator and
state, and must return a positive scalar Δt.

See the *Time stepping* page in the manual for the usage pattern.
"""
abstract type AbstractTimeStepFromHook <: AbstractTimeStepping end

"""
    TimeStepConstant(Δt::Real)

Specify that integration should proceed with a constant time step
`Δt`. The step is validated at construction (must be strictly
positive) and stored as a `Float64`.
"""
struct TimeStepConstant <: AbstractTimeStepping
    Δt::Float64
    function TimeStepConstant(Δt::Real)
        Δt > 0 || throw(ArgumentError("time step must be positive"))
        new(Float64(Δt))
    end
end

"""
    TimeStepFromCache <: AbstractTimeStepping

Singleton policy used to replay the time grid recorded in an
[`AbstractStageCache`](@ref) through a linearised system. There is no
parameter: every `(t, Δt, stages)` triple is taken from the cache in
order, so the linearised path is *discretely* consistent with the
nonlinear path that filled the cache.
"""
struct TimeStepFromCache <: AbstractTimeStepping end

"""
    TimeStepFromStorage(Δt::Real)

Specify that integration should proceed with constant time step `Δt`
while interpolating the linear operator's state from an
[`AbstractStorage`](@ref). This is the continuous (not discretely
consistent) counterpart of [`TimeStepFromCache`](@ref), and is the
appropriate choice when the linearised system is too long to fit into
a stage cache. Validates `Δt > 0` at construction.
"""
struct TimeStepFromStorage <: AbstractTimeStepping
    Δt::Float64
    function TimeStepFromStorage(Δt::Real)
        Δt > 0 || throw(ArgumentError("time step must be positive"))
        new(Float64(Δt))
    end
end
