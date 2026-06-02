import LinearAlgebra: mul!

export CallDependency

# ============================================================================
# CallDependency
# ============================================================================

"""
    CallDependency{N, INFO}

Encode the call-dependency pattern for a composite right-hand-side of
`N` coupled equations. `INFO` is a tuple of `N` tuples of positive
integers, where the `i`-th tuple lists the components that participate
in the call signature of `g[i]`.

For example, the default pattern for `N = 3`,

    CallDependency((1,), (1, 2), (1, 2, 3))

means that `g[1]` is called as `g[1](t, u[1], dudt[1])`, `g[2]` as
`g[2](t, u[1], dudt[1], u[2], dudt[2])`, and so on — each subsequent
component depends on all the preceding ones.

The type parameter is used to dispatch the generated `System` call.
"""
struct CallDependency{N, INFO} end

# `INFO` is a tuple-of-tuples; indexing returns the inner tuple for component i.
Base.getindex(::CallDependency{N, INFO}, i::Int) where {N, INFO} = INFO[i]

"""
    CallDependency(spec::Tuple...) -> CallDependency

Construct a [`CallDependency`](@ref) describing the call signature of
each component of a coupled right-hand-side. Each `spec` argument is
a tuple of component indices and must:

  - contain only indices in `1:N` (where `N` is the number of tuples),
  - be sorted in increasing order.

# Examples
```julia
deps = CallDependency((1,), (1, 2), (1, 3))
```
"""
# TODO: accept only a tuple of tuples of integers
function CallDependency(spec::Tuple...)
    N = length(spec)
    for el in spec
        m, M = extrema(el)
        ((m > 0 && M <= N && issorted(el)) ||
            throw(ArgumentError("invalid call dependency specification")))
    end
    return CallDependency{N, spec}()
end

"""
    default_dep(N::Int) -> CallDependency

Return the default [`CallDependency`](@ref) for `N` coupled equations,
in which each component depends on all the preceding ones. Defined
explicitly for `N ∈ 1:3` because these are the cases that the package
exercises in practice.
"""
default_dep(N::Int) = _default_dep(Val(N))
_default_dep(::Val{1}) = CallDependency((1,))
_default_dep(::Val{2}) = CallDependency((1,), (1, 2))
_default_dep(::Val{3}) = CallDependency((1,), (1, 2), (1, 2, 3))

# ============================================================================
# System — N coupled problems with linear + nonlinear parts
# ============================================================================

"""
    System{N, DEPS, GT, AT}

Internal representation of a dynamical system as the sum of an
explicit nonlinear term `g` and an optional linear/implicit term `A`.

`N` is the number of coupled equations and `DEPS` is the
[`CallDependency`](@ref) used to dispatch the generated call into the
component right-hand-sides. Constructed by [`flow`](@ref); end users
should not need to construct it directly.
"""
struct System{N, DEPS, GT, AT}
    g::GT # explicit term
    A::AT # linear implicit term
end

# Single-component constructors (scalar problem).
System(g, A)                          = System(g, A, default_dep(1))
System(g, A, DEPS::CallDependency{1}) = System{1, DEPS, typeof(g), typeof(A)}(g, A)

# Coupled constructors.
System(g::Coupled{N}, A::Coupled{N}) where {N} = System(g, A, default_dep(N))
System(g::Coupled{N}, A::Coupled{N}, DEPS::CallDependency{N}) where {N} =
    System{N, DEPS, typeof(g), typeof(A)}(g, A)

# ----------------------------------------------------------------------------
# Explicit part: evaluate g
# ----------------------------------------------------------------------------
#
# The generated method walks the dependency tuple at compile time and
# expands a call to each component's right-hand-side with exactly the
# state/derivative pairs declared by `DEPS[i]`. This avoids any runtime
# tuple manipulation in the hot path.

@generated function (sys::System{N, DEPS})(t::Real, z::Coupled{N}, dzdt::Coupled{N}) where {N, DEPS}
    expr = quote end
    for i = 1:N
        tup = Expr(:tuple)
        for d in DEPS[i]
            append!(tup.args, (:(z[$d]), :(dzdt[$d]), ))
        end
        push!(expr.args, :(sys.g[$(Val(i))](t, $(tup)...)))
    end
    return expr
end

# Single-component fallback: no coupling means call straight through.
(sys::System{1})(t::Real, z, dzdt) = sys.g(t, z, dzdt)

# Continuous-mode IMEX schemes pass an extra `u` argument (the
# nonlinear-trajectory state from a storage) before the working state.
(sys::System{1})(t::Real, u, z, dzdt) = sys.g(t, u, z, dzdt)

# Generated multi-component variational call: same dispatch idea as
# above, but each component receives the auxiliary `u` first.
@generated function (sys::System{N})(t::Real,
                                     u,
                                     z::Coupled{N},
                                     dzdt::Coupled{N}) where {N}
    expr = quote return dzdt end
    for i = 1:N
        pushfirst!(expr.args, :(sys.g[$i](t, u, z[$i], dzdt[$i])))
    end
    return expr
end

# ----------------------------------------------------------------------------
# Implicit part I: action of the linear operator A on z
# ----------------------------------------------------------------------------
mul!(out, sys::System{1, DEPS, GT, AT}, z) where {GT, AT, DEPS} =
    ((AT isa Nothing ? (out .= 0) : mul!(out, sys.A, z)); out)

@generated function mul!(out::Coupled{N},
                         sys::System{N, DEPS, Coupled{N, GT}, Coupled{N, AT}},
                           z::Coupled{N}) where {N, DEPS, GT, AT}
    return quote
        Base.Cartesian.@nexprs $N i->($(AT.parameters)[i] == Nothing ?
                (out[i] .= 0) : mul!(out[i], sys.A[i], z[i]))
        return out
    end
end

# ----------------------------------------------------------------------------
# Implicit part II: solve (I - cA) z = y for z
# ----------------------------------------------------------------------------
#
# When a coupled component has `A == nothing` (typically the
# quadrature variable), the implicit solve degenerates to a copy:
# the dynamics is fully explicit and `(I - 0)z = y` ⇒ z = y.

ImcA!(sys::System{1, DEPS, GT, AT}, c::Real, y, z) where {DEPS, GT, AT} =
    ((AT isa Nothing ? (z .= y) : ImcA!(sys.A, c, y, z)); z)

@generated function ImcA!(sys::System{N, DEPS, Coupled{N, GT}, Coupled{N, AT}},
                            c::Real,
                            y::Coupled{N},
                            z::Coupled{N}) where {N, DEPS, GT, AT}
    return quote
        Base.Cartesian.@nexprs $N i->($(AT.parameters)[i] == Nothing ?
                (z[i] .= y[i]) : ImcA!(sys.A[i], c, y[i], z[i]))
        return z
    end
end

# ----------------------------------------------------------------------------
# Implicit part III: compute z = (I - cA) y (no solve, just the action)
# ----------------------------------------------------------------------------
#
# Same shortcut as ImcA! for `A == nothing`: when the component has no
# linear term, the action of `(I - cA)` is the identity.

ImcA_mul!(sys::System{1, DEPS, GT, AT}, c::Real, y, z) where {DEPS, GT, AT} =
    ((AT isa Nothing ? (z .= y) : ImcA_mul!(sys.A, c, y, z)); z)

@generated function ImcA_mul!(sys::System{N, DEPS, Coupled{N, GT}, Coupled{N, AT}},
                                c::Real,
                                y::Coupled{N},
                                z::Coupled{N}) where {N, DEPS, GT, AT}
    return quote
        Base.Cartesian.@nexprs $N i->($(AT.parameters)[i] == Nothing ?
                (z[i] .= y[i]) : ImcA_mul!(sys.A[i], c, y[i], z[i]))
        return z
    end
end
