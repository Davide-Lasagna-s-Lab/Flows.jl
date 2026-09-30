export ContinuousMode, DiscreteMode

# ============================================================================
# Integration mode tags
# ============================================================================
#
# Modes are zero-cost type tags that select a code path inside the
# `step!` methods of each integration scheme:
#
#   - `NormalMode`               : forward nonlinear integration of the
#                                  primal system, with optional caching
#                                  of internal stages.
#   - `ContinuousMode{ISADJOINT}`: continuous tangent (`ISADJOINT=false`)
#                                  or adjoint (`true`) integration using
#                                  a `RAMStorage`-based interpolator.
#   - `DiscreteMode{ISADJOINT}`  : discretely-consistent tangent or
#                                  adjoint integration driven by a
#                                  `RAMStageCache`.

"""
    AbstractMode

Supertype for integration-mode tags carried by [`AbstractMethod`](@ref Flows.AbstractMethod)
subtypes. Concrete subtypes are [`NormalMode`](@ref),
[`ContinuousMode`](@ref) and [`DiscreteMode`](@ref).
"""
abstract type AbstractMode end

"""
    NormalMode()

Mode tag for forward nonlinear integration. The integration scheme
allocates only the buffers it needs for the primal step and, if a
stage cache is supplied, pushes the internal stages so a linearised
sibling can later replay the same time grid.
"""
struct NormalMode <: AbstractMode end

"""
    ContinuousMode(isadjoint::Bool = false)

Mode tag for continuous linearised integration backed by an
[`AbstractStorage`](@ref). `isadjoint = true` selects the adjoint
variant, which marches backwards in time. The continuous path
interpolates the nonlinear trajectory at the times required by the
scheme and is *not* discretely consistent with the original primal
integration.
"""
struct ContinuousMode{ISADJOINT} <: AbstractMode
    ContinuousMode(isadjoint::Bool=false) = new{isadjoint}()
end

"""
    DiscreteMode(isadjoint::Bool = false)

Mode tag for discretely consistent linearised integration backed by a
[`RAMStageCache`](@ref). `isadjoint = true` selects the adjoint
variant. The discrete path replays the exact stage values recorded by
the primal run, so the linearised solution is the exact derivative
(or adjoint of the derivative) of the primal at the time-stepping
level.
"""
struct DiscreteMode{ISADJOINT} <: AbstractMode
    DiscreteMode(isadjoint::Bool=false) = new{isadjoint}()
end

"""
    isadjoint(mode_or_method) -> Bool

Return `true` when the mode (or the mode carried by the method) is the
adjoint variant. `NormalMode` always returns `false`.
"""
isadjoint(::Type{NormalMode}) = false
isadjoint(::Type{ContinuousMode{ISADJOINT}}) where {ISADJOINT} = ISADJOINT
isadjoint(::Type{DiscreteMode{ISADJOINT}})   where {ISADJOINT} = ISADJOINT


# ============================================================================
# AbstractMethod
# ============================================================================

"""
    AbstractMethod{X, MODE, NS}

Supertype for time-stepping schemes. Each concrete subtype owns the
preallocated buffers required to advance one step. Type parameters:

  - `X`    : the state type (matches the type used to construct the
             flow's working buffers).
  - `MODE` : the integration mode tag — [`NormalMode`](@ref),
             [`ContinuousMode`](@ref) or [`DiscreteMode`](@ref).
  - `NS`   : the number of internal stages the scheme uses. Stage
             caches must be configured with a matching `NS`.
"""
abstract type AbstractMethod{X, MODE, NS} end

isadjoint(::AbstractMethod{X, MODE}) where {X, MODE} = isadjoint(MODE)

"""
    nstages(method::AbstractMethod) -> Int

Return the number of internal stages the method uses per step. This
is compared against [`nstages`](@ref Flows.nstages(::AbstractStageCache))
when a stage cache is supplied, to catch shape mismatches early.
"""
nstages(::AbstractMethod{X, MODE, NS}) where {X, MODE, NS} = NS

"""
    mode(method::AbstractMethod) -> AbstractMode

Return the mode tag carried by `method`.
"""
mode(::AbstractMethod{X, MODE}) where {X, MODE} = MODE
