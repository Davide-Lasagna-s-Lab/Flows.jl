export Coupled, couple, couplecopy

"""
    Coupled{N, ARGS<:NTuple{N, Any}} <: AbstractVector{Any}

Couple `N` heterogeneous objects together into a single composite state.

The components are stored internally in a Julia `NTuple` of size `N`. A
`Coupled` object is itself immutable, but its individual elements are
expected to be mutable so that in-place updates from a time-stepping
scheme can propagate to the underlying storage. Elements are accessed
with the standard indexing notation (`x[i]`), and `setindex!` is
deliberately not defined: the *contents* of an element may be modified,
but the element itself cannot be replaced.

# Examples
```julia
z = couple(Float64[1, 2, 3], Float64[0, 4])
z[1]       # → [1.0, 2.0, 3.0]
length(z)  # → 2
```
"""
struct Coupled{N, ARGS<:NTuple{N, Any}} <: AbstractVector{Any}
    args::ARGS
    Coupled(args::ARGS) where {N, ARGS<:NTuple{N, Any}} = new{N, ARGS}(args)
end

"""
    couple(args...) -> Coupled

Construct a [`Coupled`](@ref) object from the supplied positional
arguments. The order of the arguments fixes the indexing order of the
resulting composite state.
"""
couple(args...) = Coupled(args)

"""
    couplecopy(N::Int, x) -> Coupled

Couple `N` independent deep copies of `x` together. Useful when an
algorithm needs `N` working buffers of the same structure as `x` but
without aliasing them through the same underlying memory.
"""
couplecopy(N::Int, x) = couple(ntuple(i->deepcopy(x), N)...)

"""
    getindex(x::Coupled, i::Int) -> Any
    getindex(x::Coupled, ::Val{I}) -> Any

Return the `i`-th component of a [`Coupled`](@ref) object. The `Val`
variant is type-stable and is used inside the generated code paths of
[`Flows.System`](@ref) for coupled right-hand-side evaluation.
"""
Base.@propagate_inbounds function Base.getindex(x::Coupled{N}, i::Int) where {N}
    @boundscheck 1 ≤ i ≤ N || throw(BoundsError())
    @inbounds val = x.args[i]
    return val
end

@inline function Base.getindex(x::Coupled{N}, ::Val{I}) where {N, I}
    @inbounds val = x.args[I]
    return val
end

"""
    similar(x::Coupled) -> Coupled

Call `Base.similar` on each component of `x` and recouple the results
into a fresh `Coupled` object. The components are independent buffers.
"""
Base.similar(x::Coupled{N}) where {N} = couple(ntuple(i->similar(x[i]), N)...)

"""
    copy(x::Coupled) -> Coupled

Return a fresh `Coupled` whose components are independent copies of the
components of `x`.
"""
Base.copy(x::Coupled{N}) where {N} = couple(ntuple(i->copy(x[i]), N)...)

"""
    size(x::Coupled{N}) -> (N,)

Return `(N,)`, where `N` is the number of components coupled together.
"""
@inline Base.size(::Coupled{N}) where {N} = (N, )


# ============================================================================
# BROADCASTING
# ============================================================================
#
# This block lets `Coupled` participate in Julia's dot-broadcasting machinery
# by forwarding the broadcast operation down to each of the wrapped
# components. The approach is the one used by Chris Rackauckas' MultiScaleArrays
# package: we walk the `Broadcasted` tree, peel off one component index at a
# time, and apply the fused function to the matching components.
#
# Algebraic broadcasts are intended to work when every operand is either a
# `Coupled` or a `Number`. Mixing in other array types is unsupported.

const CoupledStyle = Broadcast.ArrayStyle{Coupled}
Base.BroadcastStyle(::Type{<:Coupled}) = CoupledStyle()

@generated function Base.copyto!(dest::Coupled{N},
                                   bc::Broadcast.Broadcasted{CoupledStyle}) where {N}
    quote
        $(Expr(:meta, :inline))
        Base.Cartesian.@nexprs $N i -> begin
            @inbounds copyto!(getfield(dest.args, i), unpack(bc, Val(i)))
        end
        return dest
    end
end

# Create a new `Broadcasted` whose arguments are the `i`-th element of each
# `Coupled` operand. Numbers pass through unchanged.
@inline unpack(bc::Broadcast.Broadcasted, v::Val) =
    Broadcast.Broadcasted(bc.f, _unpack(bc.args, v))

# `Val` is required here for type stability when unpacking the tuple.
@inline unpack(x::Coupled, ::Val{i}) where {i} = getfield(x.args, i)
@inline unpack(x::Number, i) = x

# Use `unpack` at the leaves; recurse with `_unpack` on the tail to handle
# nested `Broadcasted` arguments such as `a .+ b .* c`.
@inline _unpack(args::Tuple, i)        = (unpack(args[1], i), _unpack(Base.tail(args), i)...)
@inline _unpack(args::Tuple{Any}, i)   = (unpack(args[1], i),)
@inline _unpack(args::Tuple{}, i)      = ()


# ============================================================================
# SYMMETRY TRANSFORMS
# ============================================================================
#
# A `Flow` operator may optionally apply a symmetry transformation `sym(x, s)`
# to its result. To keep the dispatch in `integrator.jl` uniform across the
# coupled and non-coupled cases, the wrapped transform is always one of
#   - `nothing`                (no transformation)
#   - `SymTransform(f)`        (applied to a single state)
#   - `CoupledTransform(f)`    (applied component-wise to a `Coupled`)
# The two wrapper constructors collapse to `nothing` when called with
# `nothing`, so the `nothing` case fully describes "no symmetry".

"""
    SymTransform(sym) <: Function

Wrap a single-state symmetry transformation so that a [`Flow`](@ref) object
carries a uniform `sym` field type. `sym` is expected to be a callable
implementing `sym(x, s)` that applies a symmetry parameterised by `s` to a
non-`Coupled` state `x` in place.

When constructed with `nothing`, the constructor collapses to `nothing` so
that flows built without a symmetry have an empty `sym` field.
"""
struct SymTransform{SO}
    sym::SO
end
SymTransform(::Nothing) = nothing

# Apply the wrapped symmetry to a single state.
(f::SymTransform)(x, s) = f.sym(x, s)


"""
    CoupledTransform(sym) <: Function

Wrap a symmetry transformation that should be applied component-wise to a
[`Coupled`](@ref) state. `sym(x[i], s)` is called for every component
`i ∈ 1:N`. The constructor collapses to `nothing` when given `nothing`, so
coupled flows built without a symmetry have an empty `sym` field.
"""
struct CoupledTransform{SO}
    sym::SO
end
CoupledTransform(::Nothing) = nothing

# Apply the wrapped symmetry to each component of the coupled state.
function (f::CoupledTransform)(x::Coupled{N}, s) where {N}
    for i in 1:N
        f.sym(x[i], s)
    end
    return x
end

"""
    NoTransform

Type alias used by [`Flow`](@ref) dispatch to identify the "no symmetry"
case. It is exactly `Nothing` — both `SymTransform(nothing)` and
`CoupledTransform(nothing)` collapse to `nothing`, so a single `Nothing`
check is sufficient to detect an absent symmetry on either device.
"""
const NoTransform = Nothing
