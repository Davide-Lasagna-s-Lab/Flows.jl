# A lightweight object to compute correct time steps
struct Steps{S, R<:AbstractRange{S}} <: AbstractVector{Tuple{S, S}}
        rng::R    # a Julia Range object
          T::S    # final time
    isLossy::Bool # if last(rng) != T
    Steps(rng::R, T::S, isLossy::Bool) where {S, R} =
        new{S, R}(rng, T, isLossy)
end

function Steps(t0::Real, T::Real, dt::Real)
    # make sure we accept a positive dt and that internally we 
    # change its sign depending on the order of t0 and T
    dt > 0 || throw(ArgumentError("Step must be positive. Got $dt."))
    dt = t0 > T ? -dt : dt
    rng = t0:dt:T
    last(rng) != T ? Steps(rng, oftype(first(rng), T), true) :
                     Steps(rng, oftype(first(rng), T), false)
end

@inline Base.size(llr::Steps) =
    llr.isLossy ? (length(llr.rng), ) : (length(llr.rng) - 1, )

@inline function Base.getindex(llr::Steps, i::Integer)
    # The time `t` is given by the range `rng` constructed when the 
    # Steps object is instantiated. This uses Julia's `Range`
    # type. The time dts depends on `i`, since the last dt can
    # occasionally be less than what specified initially. This should
    # ensure that `t + dt ≤ T`
    t  = llr.rng[i]
    dt = i == length(llr) && llr.isLossy ? (llr.T - last(llr.rng)) : step(llr.rng)
    return t, dt
end
