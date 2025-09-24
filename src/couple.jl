export Coupled, couple, couplecopy

"""
    Coupled{N, ARGS<:NTuple{N, Any}} <: AbstractVector{Any}

Couple together `N` objects and store them internally in a Julia `ntuple` of size `N`.

Coupled` objects are immutable, but their elements must be mutable, allowing the 
content to be modified. The individual elements of a `Coupled` object can be 
accessed using the standard indexing notation, but the elements
cannot be changed, i.e. there is no `Base.setindex!` defined for `Coupled` objects.
"""
struct Coupled{N, ARGS<:NTuple{N, Any}} <: AbstractVector{Any}
    args::ARGS
    Coupled(args::ARGS) where {N, ARGS<:NTuple{N, Any}} = new{N, ARGS}(args)
end

"""
    couple(args...)

Create a `Coupled` object from the sequence of arguments `args`. 
"""
couple(args...) = Coupled(args)

"""
    couplecopy(N::Int, x)
    
Couple `N` copies of `x` together, created using `Base.deepcopy`.
"""
couplecopy(N::Int, x) = couple(ntuple(i->deepcopy(x), N)...)

"""
    Base.getindex(x::Coupled, i::Int)

Return the `i`-th element of a `Coupled` object. 
"""
Base.@propagate_inbounds function Base.getindex(x::Coupled{N}, i::Int) where {N}
    @boundscheck 1 ≤ i ≤ N || throw(BoundsError())
    @inbounds val = x.args[i]
    return val
end

""" Type stable indexing. Mostly used in `System`. """
@inline function Base.getindex(x::Coupled{N}, ::Val{I}) where {N, I}
    @inbounds val = x.args[I]
    return val
end

"""
    Base.similar(x::Coupled)

Call `Base.similar` on the elements of `x` and couple them together in a new object.
"""
Base.similar(x::Coupled{N}) where {N} = couple(ntuple(i->similar(x[i]), N)...)

"""
    Base.copy(x::Coupled)

Create a copy of `x` by coupling copies of its elements in a new object.
"""
Base.copy(x::Coupled{N})    where {N} = couple(ntuple(i->copy(x[i]), N)...)

"""
    Base.size(::Coupled{N}) where {N}

Return `(N, )`, the size of a `Coupled{N}` object, coupling `N` elements together.
"""
@inline Base.size(::Coupled{N}) where {N} = (N, )


# ~ BROADCASTING ~

# DEV NOTES: this code makes coupled objects support the dot broadcasting notation
# by forwarding the broadcasting to each of the objects held together. Broadcasting
# of algebraic functions is meant to work only for operands that are either numbers
# or coupled objects. 
const CoupledStyle = Broadcast.ArrayStyle{Coupled}
Base.BroadcastStyle(::Type{<:Coupled}) = CoupledStyle()

# The approach used here is taken from Chris Rackaukas' MultiScaleArrays package. It
# works directly with Julia's `Broadcast.Broadcasted` objects and extracts recursively
# the components of a `Coupled` object, applying the fused function to each of them.
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

# This creates a new `Broadcasted` object with the fused function and the `i`-th 
# elements of each `Coupled` object involved in the expression, as well any numbers.
@inline unpack(bc::Broadcast.Broadcasted, v::Val) =
    Broadcast.Broadcasted(bc.f, _unpack(bc.args, v))

# Here we need Val for type stability
@inline unpack(x::Coupled, ::Val{i}) where {i} = getfield(x.args, i)
@inline unpack(x::Number, i) = x

# The first item is `unpack` and not `_unpack` when we have nested `Broadcasted` objects.
@inline _unpack(args::Tuple, i) = (unpack(args[1], i), _unpack(Base.tail(args), i)...)
@inline _unpack(args::Tuple{Any}, i) = (unpack(args[1], i),)
@inline _unpack(args::Tuple{}, i) = ()



# Symmetry transformation for coupled flow variables
struct CoupledTransform{SO}
    sym::SO
end
CoupledTransform(::Nothing) = nothing

# unified type for when no transform is specified
const NoTransform = Union{Nothing, CoupledTransform{Nothing}}

# transform each component of the coupled variables
function (f::CoupledTransform)(x::Coupled{N}, s) where {N}
    for i in 1:N
        f.sym(x[i], s)
    end
    return x
end
