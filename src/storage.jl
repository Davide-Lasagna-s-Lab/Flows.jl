export AbstractStorage, RAMStorage, times, samples, degree, storelast, timespan, period, isperiodic

# ////// SOLUTION STORAGE //////
"""
    AbstractStorage{T, X, DEG}

Abstract type for storage of solution data. The type parameters specify:
`T<:Real`    : the type used to store times
`X`          : the type used to store snapshots
`DEG`        : the degree of the interpolating lagrange polynomial
"""
abstract type AbstractStorage{T<:Real, X, DEG} end

degree(::AbstractStorage{T, X, DEG}) where {T, X, DEG} = DEG

# checkers
_isstorage(::Type{<:AbstractStorage}) = true
_isstorage(::Any) = false

# interface for subtypes
reset!(store::AbstractStorage, sizehint::Int) = error("not implemented")
Base.push!(store::AbstractStorage{T, X}, t::T, x::X) where {T, X} =
    error("not implemented")

times(  store::AbstractStorage) = error("not implemented")
samples(store::AbstractStorage) = error("not implemented")
timespan(store::AbstractStorage) = error("not implemented")


# /// RAM storage ///
"""
    empty
"""
struct RAMStorage{T,
                  X,
                  DEG,
                  Vt<:AbstractVector{T},
                  Vx<:AbstractVector{X}} <: AbstractStorage{T, X, DEG}
           ts::Vt   # Vector of times
           xs::Vx   # Vector of snapshots
    storelast::Bool # this flag is used to avoid storing the last step of a simulation. it
                    # has no effect on the behaviour of this object alone but interacts
                    # when used in a call to a `Flow` object. This is useful for periodic
                    # problems, when we want don't to repeat storing the first/last element
                    # twice
       period::T    # The period in case data is periodic. Defaults to `0.0`, so non periodic.
                    # If the period is finite, it is assumed that the data is uniformly spaced
                    # and that the last/first element is not repeated in the sequence

    # constructor from type
    """
        some
    """
    function RAMStorage(::Type{X};
                   ttype::Type{T}=Float64,
                  degree::Int=3,
                  period::Real=0.0,
               storelast::Bool=true) where {T, X}
        degree ≥ 3 || throw(ArgumentError("polynomial degree must be ≥ 3, got $degree."))
        degree % 2 == 1 || throw(ArgumentError("polynomial degree must be odd, got $degree"))
        period ≥ 0 || throw(ArgumentError("period must be non-negative, got $period"))
        return new{T, X, degree, Vector{T}, Vector{X}}(T[], X[], storelast, period)
    end

    # constructor from object
    """
        Function
    """
    RAMStorage(::X; kwargs...) where {X} = RAMStorage(X; kwargs...)
end

@inline reset!(rs::RAMStorage, sizehint::Int=0) =
    (sizehint!(empty!(rs.ts), sizehint); sizehint!(empty!(rs.xs), sizehint); rs)

@inline Base.push!(rs::RAMStorage{T, X}, t::Real, x::X) where {T, X} =
    (push!(rs.ts, t); push!(rs.xs, x); nothing)

times(     rs::RAMStorage) = rs.ts
samples(   rs::RAMStorage) = rs.xs
storelast( rs::RAMStorage) = rs.storelast

"""
    period(rs::RAMStorage)

Return the period of the data, or `0` if the data in non periodic.
"""
period(rs::RAMStorage) = rs.period

"""
    isperiodic(rs::RAMStorage)

Return true if the data stored in `rs` represent a periodic signal.
"""
isperiodic(rs::RAMStorage) = period(rs) != 0

"""
    timespan(rs::RAMStorage)

Return a 2-tuple with the first and last times stored.
"""
timespan(rs::RAMStorage) = 
    isperiodic(rs) ? (first(times(rs)), period(rs)) : (first(times(rs)), last(times(rs))) 

# /// LAGRANGIAN INTERPOLATION ///

"""
    _lagr_weights(t::Real, ts::NTuple{N, Real}) where {N}

Compute a `N`-tuple for the interpolation weights to interpolate
data at position `t` using data points available at positions `ts`,
with a lagrange polynomial of degree `N-1`.
"""
@generated _lagr_weights(t::Real, ts::NTuple{N, Real}, ::Val{0}) where {N} =
    :(Base.Cartesian.@ntuple $N j->_prod(t, ts, Val(j))/_prod(ts[j], ts, Val(j)))

# FIXME: implement proper rule for differentiation!!!
@generated function _lagr_weights(t::Real, ts::NTuple{N, Real}, ::Val{1}) where {N}
    quote
        a = _lagr_weights(t-2e-6, ts, Val(0))
        b = _lagr_weights(t-1e-6, ts, Val(0))
        c = _lagr_weights(t+1e-6, ts, Val(0))
        d = _lagr_weights(t+2e-6, ts, Val(0)) 
        return  (a .- 8.0.*b .+ 8.0.*c .- d)./12e-6
    end
end

"""
    _prod(t::T, ts::NTuple{N, T}, ::Val{SKIP}) where {N, T, SKIP}

Compute the product `(t - ts[1])*(t - ts[2])...(t - ts[N])` excluding the
factor `(t - ts[SKIP])`.
"""
@generated _prod(t::Real, ts::NTuple{N, Real}, ::Val{SKIP}) where {N, SKIP} =
    :(return *($([:(t - ts[$k]) for k in 1:N if k != SKIP]...)))

"""
    _lagr_interp(out::X, t::Real, ts::NTuple{N, Real},
                 xs::AbstractVector{X}, rng::NTuple{N, Int}) where {X, N}

Interpolate data points `(ts, xs[rng])` at location `t`, using a
lagrange polynomial of degree `N-1`, and overwrite the first argument `out`.
"""
function _lagr_interp(out::X,
                       t::Real,
                      ts::NTuple{N, Real},
                      xs::AbstractVector{X},
                     rng::NTuple{N, Int},
                        ::Val{ORD}) where {X, N, ORD}

    # get weights
    ws = _lagr_weights(t, ts, Val(ORD))

    # compute linear combination and return output
    out .= ws[1].*xs[rng[1]]
    for i in 2:N
        out .+= ws[i].*xs[rng[i]]
    end

    return out
end

"""
    _wrap_around_point(idxs::NTuple{N, Int}) where {N}

Helper function to determine if the stencil for the interpolation 
wraps around. This happens for periodic data. This function is 
defined by the following test cases:

    _wrap_around_point((  1,   2,   3, 4)) -> 5 # no wrap around
    _wrap_around_point((100,   1,   2, 3)) -> 1
    _wrap_around_point(( 99, 100,   1, 2)) -> 2
    _wrap_around_point(( 98,  99, 100, 1)) -> 3
"""
function _wrap_around_point(idxs::NTuple{N, Int}) where {N}
    for i in 1:N-1
        if idxs[i] > idxs[i+1]
            return i
        end
    end
    return N+1
end

"""
    _make_tuple_of_times(t::Real, 
                        ts::AbstractVector{<:Real},
                      idxs::NTuple{N, Int},
                    period::Real) where {N}

Construct a tuple of times `_ts` that is in principle equivalent to 
`_ts = ts[idxs]`, but takes cares of situations where the interpolation 
stencil wraps around, e.g. when interpolating periodic data near `t=0.0` 
or near `t=period`. This returns a tuple of strictly increasing times, 
and an adjusted time `_t` that sits in between the extrema of `_ts`.
"""
@generated function _make_tuple_of_times(t::Real, 
                                        ts::AbstractVector{<:Real},
                                      idxs::NTuple{N, Int},
                                    period::Real) where {N}
    # julia struggles with inference here, so we make this a generated function
    quote 
        # determine the wrapping point
        p = _wrap_around_point(idxs)

        # construct a tuple of increasing times adding `period` if needed
        _ts = Base.Cartesian.@ntuple $N j -> begin
            j > p ? ts[idxs[j]] + period : ts[idxs[j]]
        end

        # also adjust time if needed
        _t = isbetween(t, extrema(_ts)...) ? t : t + period

        return _ts, _t
    end
end

"""
    _interp_indices(t::Real, ts::AbstractVector{<:Real},
                    ::Val{N}, isperiodic::Bool) where {N}

Return an `N`-tuple of integer indices for the elements of `ts`
that participate in the interpolation at some point `t`. It is
assumed that `ts` is sorted, that `t ≥ ts[1]` and that the 
length of `ts` is larger than `N`.
"""
@generated function _interp_indices(t::Real,
                                   ts::AbstractVector{<:Real},
                                     ::Val{N},
                           isperiodic::Bool) where {N}
    # julia struggles with inference here, so we make this a generated function
    quote
        # search last index `idx` in `ts` for which `t ≥ ts[idx]`
        idx = searchsortedlast(ts, t)

        # we 'll use this quite a bit
        M = length(ts)

        # boundary conditions in the non-periodic case might need shifting of the stencil
        if isperiodic  == false
            Δ = idx == 1     ?  $(N>>1) - 1 :
                idx == M     ? -$(N>>1)     :
                idx == M - 1 ? -$(N>>1) + 1 : 0
                idx += Δ
        end
        return Base.Cartesian.@ntuple $N j -> mod(idx - $(N>>1) + j - 1 + M, M) + 1
    end
end


#    (store::RAMStorage{T, X, DEG})(out::X, t::Real) where {T, X, DEG}
# 
# Interpolate the storage data at time `t` and overwrite the first argument
# `out`, using lagrangian interpolation of order `DEG`.
function (store::RAMStorage{T, X, DEG})(out::X,
                                          t::Real,
                                           ::Val{ORD}=Val(0)) where {T,
                                                                     X,
                                                                     DEG,
                                                                     ORD}
    # Aliases. These should be lazy objects
    ts, xs = times(store), samples(store)

    #  we must have enough data
    length(ts) ≥ DEG+1 ||
            throw(ArgumentError("input array length must be greater than DEG+1"))

    # check if `t` is in bounds and disallow extrapolation. Code that calls
    # this interpolator must make sure that no extrapolation is requested
    isbetween(t, timespan(store)...) ||
        throw(ArgumentError("time $t is out of range $timespan(store)"))

    # Obtain the indices of the elements that participate in the interpolation
    idxs = _interp_indices(t, ts, Val(DEG+1), isperiodic(store))

    # define the abscissa of the interpolation data
    _ts, _t = _make_tuple_of_times(t, ts, idxs, period(store))
    
    return _lagr_interp(out, _t, _ts, xs, idxs, Val(ORD))
end

@generated function (store::RAMStorage)(out::Coupled, t::Real, ::Val{ORD}=Val(0)) where {ORD}
    return quote
        store(out[1], t, Val(ORD))
        return out[1]
    end
end