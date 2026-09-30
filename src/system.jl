import LinearAlgebra: mul!

export CallDependency

# structure to provide dependency specification for composite dynamical systems
struct CallDependency{N, INFO} end

# read-only data structure
Base.getindex(::CallDependency{N, INFO}, i::Int) where {N, INFO} = INFO[i]

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

# provide defaults up to N = 3 (most likely to occur)
default_dep(N::Int) = _default_dep(Val(N))
_default_dep(::Val{1}) = CallDependency((1,))
_default_dep(::Val{2}) = CallDependency((1,), (1, 2))
_default_dep(::Val{3}) = CallDependency((1,), (1, 2), (1, 2, 3))

# ~~~ TYPE FOR INTEGRATING N COUPLED PROBLEMS ~~~
struct System{N, DEPS, GT, AT}
    g::GT # explicit term
    A::AT # linear implicit term
end

# constructors
System(g, A)                          = System(g, A, default_dep(1))
System(g, A, DEPS::CallDependency{1}) = System{1, DEPS, typeof(g), typeof(A)}(g, A)

System(g::Coupled{N}, A::Coupled{N}) where {N} = System(g, A, default_dep(N))
System(g::Coupled{N}, A::Coupled{N}, DEPS::CallDependency{N}) where {N} =
    System{N, DEPS, typeof(g), typeof(A)}(g, A)

# ~ Explicit part ~
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

# no coupling means no deps
(sys::System{1})(t::Real, z, dzdt) = sys.g(t, z, dzdt)

# this is for the continuous schemes
(sys::System{1})(t::Real, u, z, dzdt) = sys.g(t, u, z, dzdt)

# this is useful for integrating a set of variational equations
@generated function (sys::System{N})(t::Real,
                                     u,
                                     z::Coupled{N},
                                     dzdt::Coupled{N}) where {N}
    expr = quote return dzdt end
    for i = 1:N
        pushfirst!(expr.args, :(sys.g[$(Val(i))](t, u, z[$i], dzdt[$i])))
    end
    return expr
end

@generated function (sys::System{N})(t::Real,
                                     u::Coupled{N},
                                     z::Coupled{N},
                                     dzdt::Coupled{N}) where {N}
    expr = quote return dzdt end
    for i = 1:N
        pushfirst!(expr.args, :(sys.g[$(Val(i))](t, u[$i], z[$i], dzdt[$i])))
    end
    return expr
end

# ~ Implicit part I ~
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

# ~ Implicit part II ~
# Since we treat the quadrature fully explicitly, the solution of
# (I-cA)z = y for the quadrature part is simply z = y, because the
# component of A associated to this part is zero and the state
# and quadrature parts are decoupled.
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

# ~ Implicit part III ~
# Since we treat the quadrature fully explicitly, the solution of
# (I-cA)z = y for the quadrature part is simply z = y, because the
# component of A associated to this part is zero and the state
# and quadrature parts are decoupled.
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