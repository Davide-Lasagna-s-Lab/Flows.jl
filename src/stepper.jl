"""
    Steps{S, R<:AbstractRange{S}} <: AbstractVector{Tuple{S, S}}

Lightweight iterator that enumerates `(t, Δt)` pairs spanning an
integration interval. Constructed from a triple `(t0, T, Δt)` and
internally backed by a Julia `Range`, so it has unit memory cost
regardless of the number of steps.

The convention is that the user always supplies a positive `Δt`; the
constructor flips the sign as needed when `t0 > T`, so backward
integration shares the same code path.

When `T` is not a whole number of steps `Δt` away from `t0`, the
iterator is *lossy*: the final pair has a smaller `dt`. A final step
shorter than a millionth of `Δt` is rounding, and is merged into the
step before. In all cases
the final `dt` is `T - t`, so that the last step ends precisely at
`T`, ensuring `t + dt ≤ T` (or `t + dt ≥ T` when integrating
backwards). The `isLossy` flag records this case so
`size` and the last `getindex` can adjust.
"""
struct Steps{S, R<:AbstractRange{S}} <: AbstractVector{Tuple{S, S}}
        rng::R    # underlying Julia range covering the regular steps
          T::S    # exact final time (may differ from `last(rng)`)
    isLossy::Bool # true when `last(rng) != T` and a short final step is needed
    Steps(rng::R, T::S, isLossy::Bool) where {S, R<:AbstractRange{S}} =
        new{S, R}(rng, T, isLossy)
end

"""
    Steps(t0::Real, T::Real, dt::Real) -> Steps

Build a [`Steps`](@ref) iterator covering `[t0, T]` (or `[T, t0]`
for backward integration) with nominal step `dt`. `dt` must be
strictly positive; its sign is flipped internally when `t0 > T`.
"""
function Steps(t0::Real, T::Real, dt::Real)
    # make sure we accept a positive dt and that internally we
    # change its sign depending on the order of t0 and T
    dt > 0 || throw(ArgumentError("Step must be positive. Got $dt."))
    dt = t0 > T ? -dt : dt
    rng = t0:dt:T

    # the range may stop a few ulps short of `T` when `T - t0` is a whole
    # number of steps, for instance with `dt = T/n`. A last step shorter than
    # a millionth of `dt` is such rounding, and is merged into the step
    # before, which then ends on `T`. A span shorter than that is one step
    isLossy = length(rng) == 1 || abs(T - last(rng)) > 1e-6 * abs(dt)

    return Steps(rng, oftype(first(rng), T), isLossy)
end

@inline Base.size(llr::Steps) =
    llr.isLossy ? (length(llr.rng), ) : (length(llr.rng) - 1, )

@inline function Base.getindex(llr::Steps, i::Integer)
    # The time `t` is the `i`-th element of the underlying range. The
    # step `dt` is normally `step(llr.rng)`, but the last one is `T - t`,
    # so that `t + dt` lands on `T` exactly. In the lossy case this is the
    # short final step. In the lossless case it differs from `step(llr.rng)`
    # by rounding only, but `t + step(llr.rng)` can overshoot `T` by one ulp,
    # which takes the stage times out of the span of a storage.
    t  = llr.rng[i]
    dt = i == length(llr) ? llr.T - t : step(llr.rng)
    return t, dt
end
