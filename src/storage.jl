export AbstractStorage, RAMStorage, times, samples, degree, storelast, timespan, period, isperiodic

# ============================================================================
# AbstractStorage
# ============================================================================

"""
    AbstractStorage{T<:Real, X, DEG}

Supertype for objects that store a time series of states along a
trajectory. The type parameters are:

  - `T<:Real` — the type used to represent times.
  - `X`       — the type used to represent state snapshots.
  - `DEG`     — the degree of the Lagrange polynomial used by the
                interpolator built on top of the storage.

Concrete subtypes must implement `reset!`, `Base.push!`, [`times`](@ref),
[`samples`](@ref), and [`timespan`](@ref). Concrete subtypes that need
to interpolate between samples must also be callable with the signature
`(store)(out, t, ::Val{ORD})`.
"""
abstract type AbstractStorage{T<:Real, X, DEG} end

"""
    degree(store::AbstractStorage) -> Int

Return the polynomial degree `DEG` used by the storage's Lagrange
interpolator.
"""
degree(::AbstractStorage{T, X, DEG}) where {T, X, DEG} = DEG

# trait used by `_propagate!` to dispatch differently on storage presence
_isstorage(::Type{<:AbstractStorage}) = true
_isstorage(::Any) = false

# Abstract interface for subtypes. Concrete subtypes must override these.
reset!(store::AbstractStorage, sizehint::Int) = error("not implemented")
Base.push!(store::AbstractStorage{T, X}, t::T, x::X) where {T, X} =
    error("not implemented")

times(  store::AbstractStorage) = error("not implemented")
samples(store::AbstractStorage) = error("not implemented")
timespan(store::AbstractStorage) = error("not implemented")


# ============================================================================
# RAMStorage — concrete in-memory storage
# ============================================================================

"""
    RAMStorage{T, X, DEG, Vt, Vx} <: AbstractStorage{T, X, DEG}

Concrete in-memory implementation of [`AbstractStorage`](@ref). Times
and samples are kept in two parallel `Vector`s, indexed identically.

The internal flag `storelast` controls whether the very last sample of
an integration is pushed: it is left to the caller to decide because
some workflows (notably periodic problems where the period is known)
prefer to omit the repeated endpoint. The `period` field is set to a
non-zero value when the data should be treated as one period of a
periodic signal; the interpolator then performs modular wrap-around
when the requested time is near an endpoint.

Construct via the [`RAMStorage`](@ref) outer constructors.
"""
struct RAMStorage{T,
                  X,
                  DEG,
                  Vt<:AbstractVector{T},
                  Vx<:AbstractVector{X}} <: AbstractStorage{T, X, DEG}
           ts::Vt   # vector of sample times
           xs::Vx   # vector of state snapshots
    storelast::Bool # whether to store the last sample of an integration
       period::T    # signal period; `0` means the data is non-periodic

    """
        RAMStorage(::Type{X}; ttype=Float64, degree=3, period=0.0, storelast=true)

    Construct a [`RAMStorage`](@ref) for snapshots of type `X`. The
    keyword arguments are:

      - `ttype::Type{T} = Float64`: time element type.
      - `degree::Int = 3`: odd polynomial degree (≥ 3) used by the
        Lagrange interpolator built on top of the storage.
      - `period::Real = 0.0`: positive period if the stored signal is
        periodic; zero (the default) means non-periodic.
      - `storelast::Bool = true`: whether the propagation loop should
        push the final state of an integration. Set to `false` when
        building a one-period buffer that should not contain the
        duplicated endpoint.
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

    """
        RAMStorage(x; kwargs...)

    Convenience constructor that infers the snapshot type from `x`.
    Equivalent to `RAMStorage(typeof(x); kwargs...)`. The same keyword
    arguments documented on the type-based constructor apply.
    """
    RAMStorage(::X; kwargs...) where {X} = RAMStorage(X; kwargs...)
end

"""
    reset!(rs::RAMStorage, sizehint::Int=0) -> rs

Empty the time and sample buffers of `rs`. The optional `sizehint`
preallocates capacity in both buffers so that subsequent `push!`
calls can avoid reallocation. Returns `rs`.
"""
@inline reset!(rs::RAMStorage, sizehint::Int=0) =
    (sizehint!(empty!(rs.ts), sizehint); sizehint!(empty!(rs.xs), sizehint); rs)

"""
    push!(rs::RAMStorage, t::Real, x) -> nothing

Append the pair `(t, x)` to the storage. `x` must already match the
snapshot type the storage was constructed with; in particular, callers
that pass a mutable buffer should `copy` it first if they intend to
keep mutating it after the push.
"""
@inline Base.push!(rs::RAMStorage{T, X}, t::Real, x::X) where {T, X} =
    (push!(rs.ts, t); push!(rs.xs, x); nothing)

"""
    times(rs::RAMStorage) -> Vector

Return the vector of times stored in `rs`, in push order.
"""
times(rs::RAMStorage) = rs.ts

"""
    samples(rs::RAMStorage) -> Vector

Return the vector of state snapshots stored in `rs`, in push order.
"""
samples(rs::RAMStorage) = rs.xs

"""
    storelast(rs::RAMStorage) -> Bool

Return the `storelast` flag of `rs`. When `false`, the propagation
loop is expected to omit the final sample of an integration — this is
how the no-duplicated-endpoint convention for periodic storage is
expressed.
"""
storelast(rs::RAMStorage) = rs.storelast

"""
    period(rs::RAMStorage) -> Real

Return the period of the data stored in `rs`, or zero if the data is
not periodic.
"""
period(rs::RAMStorage) = rs.period

"""
    isperiodic(rs::RAMStorage) -> Bool

Return `true` when the storage represents a periodic signal (i.e.
`period(rs) > 0`).
"""
isperiodic(rs::RAMStorage) = period(rs) != 0

"""
    timespan(rs::RAMStorage) -> Tuple{Real, Real}

Return the `(first, last)` time of the stored data. For periodic
storages the second element is `period(rs)` rather than the last
sample time, because the storage convention is to omit the repeated
endpoint.
"""
timespan(rs::RAMStorage) =
    isperiodic(rs) ? (first(times(rs)), period(rs)) : (first(times(rs)), last(times(rs)))


# ============================================================================
# LAGRANGIAN INTERPOLATION
# ============================================================================
# These helpers implement Lagrange interpolation on top of an
# `AbstractStorage`, with optional first-derivative support via central
# finite differences (a stop-gap until proper analytic derivatives are
# wired in). They are not exported.

"""
    _lagr_weights(t::Real, ts::NTuple{N, Real}, ::Val{0}) -> NTuple{N, Real}

Return the `N` Lagrange interpolation weights for interpolating at `t`
using the nodes `ts`. The result is an `N`-tuple of weights that sum
to one. The `Val{0}` distinguishes the function-value variant from the
first-derivative variant below.
"""
@generated _lagr_weights(t::Real, ts::NTuple{N, Real}, ::Val{0}) where {N} =
    :(Base.Cartesian.@ntuple $N j->_prod(t, ts, Val(j))/_prod(ts[j], ts, Val(j)))

"""
    _lagr_weights(t::Real, ts::NTuple{N, Real}, ::Val{1}) -> NTuple{N, Real}

Approximate the first-derivative Lagrange weights at `t` using a
fourth-order central finite-difference stencil applied to the
function-value weights. This is a placeholder for a proper analytic
differentiation rule and trades a small amount of accuracy for a very
simple implementation.
"""
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
    _prod(t::Real, ts::NTuple{N, Real}, ::Val{SKIP}) -> Real

Compute `(t - ts[1])(t - ts[2])…(t - ts[N])` while skipping the factor
`(t - ts[SKIP])`. Used to build the Lagrange basis polynomials at
generation time.
"""
@generated _prod(t::Real, ts::NTuple{N, Real}, ::Val{SKIP}) where {N, SKIP} =
    :(return *($([:(t - ts[$k]) for k in 1:N if k != SKIP]...)))

"""
    _lagr_interp(out, t, ts, xs, rng, ::Val{ORD}) -> out

Interpolate the samples `xs[rng]` at the nodes `ts` to obtain the value
at time `t`, writing the result in place into `out`. `ORD` selects the
function-value (`0`) or first-derivative (`1`) interpolant.
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
    _wrap_around_point(idxs::NTuple{N, Int}) -> Int

Return the first index `i` such that `idxs[i] > idxs[i+1]`, or `N+1` if
the tuple is already strictly increasing. This is the wrap point of a
periodic interpolation stencil; positions in `idxs` that fall after the
wrap point should have one period added to them before interpolation.

# Examples
```julia
_wrap_around_point((  1,   2,   3, 4)) == 5  # no wrap
_wrap_around_point((100,   1,   2, 3)) == 1
_wrap_around_point(( 99, 100,   1, 2)) == 2
_wrap_around_point(( 98,  99, 100, 1)) == 3
```
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
    _make_tuple_of_times(t, ts, idxs, period) -> (NTuple{N, Real}, Real)

Build a strictly increasing tuple of times equivalent to `ts[idxs]`,
adding `period` to entries that follow the wrap point so the result is
monotone. The query time `t` is shifted by `period` when needed so
that it falls inside the resulting interval, which is required for
Lagrange interpolation to evaluate inside the stencil.
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
    _interp_indices(t, ts, ::Val{N}, isperiodic::Bool) -> NTuple{N, Int}

Return the `N` indices into the sorted `ts` whose corresponding samples
participate in an interpolation at `t`. `ts` is assumed to be sorted
and at least `N` elements long. When `isperiodic` is true, the stencil
wraps around the endpoints modulo the period.
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


"""
    (store::RAMStorage)(out, t::Real, ::Val{ORD}=Val(0)) -> out

Interpolate the storage data at time `t`, writing the result into the
preallocated buffer `out` in place. `ORD` selects the function-value
(`0`, default) or first-derivative (`1`) interpolant. The storage
must contain at least `DEG+1` samples, and `t` must lie within
[`timespan`](@ref); extrapolation is not allowed and raises an
`ArgumentError`.
"""
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

"""
    (store::RAMStorage)(out::Coupled, t::Real, ::Val{ORD}=Val(0)) -> out[1]

Coupled-state convenience: interpolate only the first component of
`out` from the storage. This is used by the IMEX integration paths
where the storage holds the nonlinear-state trajectory and the
linearised state is the remaining coupled component.
"""
@generated function (store::RAMStorage)(out::Coupled, t::Real, ::Val{ORD}=Val(0)) where {ORD}
    return quote
        store(out[1], t, Val(ORD))
        return out[1]
    end
end
