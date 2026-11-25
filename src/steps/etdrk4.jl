import LinearAlgebra

export ETDRK4, etdrk4_coeffs

# ---------------------------------------------------------------------------- #
# Exponential time differencing RK4 of Cox & Matthews (2002), with the
# coefficients computed by the contour integrals of Kassam & Trefethen (2005).
# The equation is  u' = A u + g(t, u), with A treated exactly.
struct ETDRK4{X, MODE, NX, NC} <: AbstractMethod{X, MODE, 4}
    store::NX                      # stage buffers
     coef::NC                      # coefficients E, E2, Q, f1, f2, f3, as states
       Δt::Base.RefValue{Float64}  # step of the coefficients
        A::Base.RefValue{Any}      # linear operator of the coefficients

    ETDRK4{X, MODE}(store::NX,
                     coef::NC) where {X, MODE, NX<:Tuple{Vararg{X}}, NC<:NTuple{6, X}} =
        new{X, MODE, NX, NC}(store, coef, Ref(NaN), Ref{Any}(nothing))
end

"""
    ETDRK4(x::X, mode::AbstractMode = NormalMode()) -> ETDRK4

Construct the fourth-order exponential time differencing Runge–Kutta
scheme of Cox & Matthews (2002) for `u' = A u + g(t, u)`. The linear part
`A` is integrated exactly, so the scheme is not limited in `Δt` by the
stiffness of `A`. `x` is a template for the buffers: `6` stages and `6`
coefficients for [`NormalMode`](@ref) and the tangent of [`DiscreteMode`](@ref),
`7` stages for [`ContinuousMode`](@ref), `8` for the adjoint of
[`DiscreteMode`](@ref).

`A` is a `LinearAlgebra.Diagonal`, or `nothing` (plain RK4 for that
component), also as components of a [`Coupled`](@ref) system. Other
operators need a method of [`etdrk4_coeffs`](@ref).

The coefficients depend on the step `Δt` and are cached in the method. They
are recomputed when `Δt` changes (also on the shorter last step of an
integration) or when `A` is another object; they are not recomputed if the
entries of `A` are modified in place.

[`ContinuousMode`](@ref) is supported, forward and adjoint. The state on the
stored orbit is interpolated at `t`, `t + Δt/2` and `t + Δt`. In adjoint mode
the step is `-Δt` and `A` must be the adjoint linear operator.

[`DiscreteMode`](@ref) is supported, forward and adjoint, over the four stages
that a normal step keeps in a stage cache, the states at which it evaluates
`g`: `x`, `a`, `b` and `c`. The tangent is the step with `g` linearised there,
and the adjoint its transpose, back over the stages, with `A` the adjoint
linear operator, whose coefficients are those of the transposes. A forcing in
the adjoint system gets the weights of the quadrature of the scheme, those of
RK4, `Δt/6`, `Δt/3`, `Δt/3` and `Δt/6`.
"""
function ETDRK4(x::X, mode::MODE=NormalMode()) where {X, MODE<:AbstractMode}
    # ---- buffers ----
    N = mode isa ContinuousMode     ? 7 :
        mode isa DiscreteMode{true} ? 8 : 6

    return ETDRK4{X, MODE}(ntuple(i->similar(x), N),
                           ntuple(i->similar(x), 6))
end

"""
    etdrk4_coeffs(A, Δt::Real) -> (E, E2, Q, f1, f2, f3)

Coefficients of [`ETDRK4`](@ref) for the linear operator `A` and the step
`Δt > 0`, with `L = Δt A` and `φ₁, φ₂, φ₃` the φ-functions of exponential integrators:

    E  = exp(L)            E2 = exp(L/2)          Q  = Δt φ₁(L/2) / 2
    f1 = Δt (φ₁ - 3φ₂ + 4φ₃)(L)
    f2 = Δt (φ₂ - 2φ₃)(L)
    f3 = Δt (-φ₂ + 4φ₃)(L)

These are the coefficients of Cox & Matthews (2002). Each of the six values must be
an object that multiplies a state by broadcasting, `c .* x`; for a diagonal
operator these are vectors of the diagonal entries. Users add a method for
their operator type; the fallback raises an error. A method for
`LinearAlgebra.Diagonal` is provided, with the contour integrals of
Kassam & Trefethen (2005) with 32 points of radius 1 around each `Δt A[i,i]`.
"""
etdrk4_coeffs(A, Δt::Real) =
    error("etdrk4_coeffs missing implementation for operator `A` of type $(typeof(A))")

function etdrk4_coeffs(A::LinearAlgebra.Diagonal, Δt::Real)
    # ---- coefficients ----
    L  = Δt .* A.diag
    E  = exp.(L)
    E2 = exp.(L ./ 2)
    Q  = Δt .* _contour(z -> (exp(z/2) - 1) / z,                          L)
    f1 = Δt .* _contour(z -> (-4 - z + exp(z)*(4 - 3z + z^2)) / z^3,      L)
    f2 = Δt .* _contour(z -> ( 2 + z + exp(z)*(-2 + z)) / z^3,            L)
    f3 = Δt .* _contour(z -> (-4 - 3z - z^2 + exp(z)*(4 - z)) / z^3,      L)

    return (E, E2, Q, f1, f2, f3)
end

# Mean of the scalar function `f` on the circle of radius 1 around each entry
# of `L` (Kassam & Trefethen, 2005): 32 points, and the real part of the upper
# half circle is enough for real `L`. Removes the cancellation of φ-functions
# near zero. Shared by the ETD schemes.
function _contour(f, L)
    M = 32
    r = exp.(im .* π .* ((1:M) .- 0.5) ./ M)
    z = L .+ transpose(r)
    return vec(real.(sum(f.(z); dims=2)) ./ M)
end

# The hook and the limits at L = 0 of the coefficients of each scheme
_etd_hook(::ETDRK4, A, Δt)   = etdrk4_coeffs(A, Δt)
_etd_limits(::ETDRK4, Δt)    = (1, 1, Δt/2, Δt/6, Δt/6, Δt/6)   # those of RK4

# ---------------------------------------------------------------------------- #
# Update the coefficients when the step or the operator change. Each is a state
# like x, so that the stages are plain broadcasts. A component without linear
# part has the limits of the coefficients at L = 0, those of RK4 (or RK2).
function _etd_update!(method, A, Δt)
    if method.Δt[] != Δt || method.A[] !== A
        comps = A isa Coupled ? A.args : (A,)
        for (i, Ai) in enumerate(comps)
            cs = Ai === nothing ? _etd_limits(method, Δt) : _etd_hook(method, Ai, Δt)
            for (c, v) in zip(method.coef, cs)
                ci = c isa Coupled ? c[i] : c
                ci .= v
            end
        end
        method.Δt[] = Δt
        method.A[] = A
    end
    return nothing
end

# ---------------------------------------------------------------------------- #
# Normal time stepping, with the four stages kept in a stage cache if one is
# given: the states x, a, b and c at which g is evaluated, which c holds in y
function step!(method::ETDRK4{X, NormalMode},
                  sys::System,
                    t::Real,
                   Δt::Real,
                    x::X,
                    c::C) where {X, C<:Union{Nothing, AbstractStageCache{4, X}}}
    # ---- coefficients and buffers ----
    _etd_update!(method, sys.A, Δt)
    E, E2, Q, f1, f2, f3 = method.coef
    Na, Nb, Nc, a, b, y  = method.store

    # ---- stages ----
    sys(t, x, Na)
    a .= E2 .* x .+ Q .* Na

    sys(t + Δt/2, a, Nb)
    b .= E2 .* x .+ Q .* Nb

    sys(t + Δt/2, b, Nc)
    y .= E2 .* a .+ Q .* (2 .* Nc .- Na)

    # ---- the stages kept, before b holds Nd and x the step ----
    _iscache(C) && push!(c, t, Δt, (copy(x), copy(a), copy(b), copy(y)))

    sys(t + Δt, y, b)                           # Nd, in b

    # ---- update ----
    x .= E .* x .+ f1 .* Na .+ 2 .* f2 .* (Nb .+ Nc) .+ f3 .* b

    return nothing
end

# ---------------------------------------------------------------------------- #
# Continuous tangent or adjoint along an orbit in a storage, interpolated at the
# times t, t + Δt/2 and t + Δt. In adjoint mode Δt < 0 and the equation is
# marched in the reversed time, with step -Δt.
function step!(method::ETDRK4{X, MODE},
                  sys::System,
                    t::Real,
                   Δt::Real,
                    x::X,
                store::AbstractStorage) where {X, MODE<:ContinuousMode}
    # ---- coefficients and buffers ----
    _m_ = isadjoint(MODE) ? -1 : 1
    _etd_update!(method, sys.A, _m_*Δt)
    E, E2, Q, f1, f2, f3 = method.coef
    Na, Nb, Nc, a, b, y, u = method.store

    # ---- stages. For coupled vectors the storage fills the first component of `u`
    # and returns it, and that one state is passed to all of them ----
    ū = store(u, t)
    sys(t, ū, x, Na)
    a .= E2 .* x .+ Q .* Na

    ū = store(u, t + Δt/2)
    sys(t + Δt/2, ū, a, Nb)
    b .= E2 .* x .+ Q .* Nb

    sys(t + Δt/2, ū, b, Nc)
    y .= E2 .* a .+ Q .* (2 .* Nc .- Na)

    ū = store(u, t + Δt)
    sys(t + Δt, ū, y, b)                        # Nd, in b

    # ---- update ----
    x .= E .* x .+ f1 .* Na .+ 2 .* f2 .* (Nb .+ Nc) .+ f3 .* b

    return nothing
end


# ---------------------------------------------------------------------------- #
# Forward linearised method over the stages x, a, b and c of a step, which takes
# x_n and overwrites it with x_{n+1}: the normal step with g linearised at them
function step!(method::ETDRK4{X, MODE},
                  sys::System,
                    t::Real,
                   Δt::Real,
                    x::X,
               stages::NTuple{4, X}) where {X, MODE<:DiscreteMode{false}}
    # ---- coefficients and buffers ----
    _etd_update!(method, sys.A, Δt)
    E, E2, Q, f1, f2, f3 = method.coef
    Na, Nb, Nc, a, b, y  = method.store

    # ---- stages, each linearised at its stage ----
    sys(t, stages[1], x, Na)
    a .= E2 .* x .+ Q .* Na

    sys(t + Δt/2, stages[2], a, Nb)
    b .= E2 .* x .+ Q .* Nb

    sys(t + Δt/2, stages[3], b, Nc)
    y .= E2 .* a .+ Q .* (2 .* Nc .- Na)

    sys(t + Δt, stages[4], y, b)                # Nd, in b

    # ---- update ----
    x .= E .* x .+ f1 .* Na .+ 2 .* f2 .* (Nb .+ Nc) .+ f3 .* b

    return nothing
end

# ---------------------------------------------------------------------------- #
# Adjoint linearised method over the stages, which takes x_{n+1} and overwrites
# it with x_n: the transpose of the step, its lines read backwards, from the
# update to the first stage, with the transposes of the coefficients, those of
# the adjoint operator sys.A, and of g at the stages,
#
#     x̄ = Eᵀλ,  N̄a = f1ᵀλ,  N̄b = N̄c = 2f2ᵀλ,  N̄d = f3ᵀλ
#     c̄ = J₄ᵀ N̄d
#     ā = E2ᵀ c̄,   N̄c += 2Qᵀ c̄,   N̄a -= Qᵀ c̄
#     b̄ = J₃ᵀ N̄c
#     x̄ += E2ᵀ b̄,  N̄b += Qᵀ b̄
#     ā += J₂ᵀ N̄b
#     x̄ += E2ᵀ ā,  N̄a += Qᵀ ā
#     x̄ += J₁ᵀ N̄a
#
# with λ = x_{n+1} and x_n = x̄. A step of zero length is the identity, and so is
# its transpose.
function step!(method::ETDRK4{X, MODE},
                  sys::System,
                    t::Real,
                   Δt::Real,
                    x::X,
               stages::NTuple{4, X}) where {X, MODE<:DiscreteMode{true}}
    Δt == 0 && return nothing

    # ---- coefficients and buffers ----
    _etd_update!(method, sys.A, Δt)
    E, E2, Q, f1, f2, f3             = method.coef
    N̄a, N̄b, N̄c, N̄d, ā, b̄, c̄, v = method.store

    # ---- the transpose of the update, λ in x ----
    N̄a .= f1 .* x
    N̄b .= 2 .* f2 .* x
    N̄c .= N̄b
    N̄d .= f3 .* x

    # ---- the fourth stage, at c, and the third, at b ----
    _transposed!(sys, t + Δt, stages[4], N̄d, c̄, Δt/6)
    ā   .= E2 .* c̄
    N̄c .+= 2 .* Q .* c̄
    N̄a .-= Q .* c̄

    _transposed!(sys, t + Δt/2, stages[3], N̄c, b̄, Δt/3)
    x   .= E .* x .+ E2 .* b̄
    N̄b .+= Q .* b̄

    # ---- the second, at a, and the first, at x ----
    _transposed!(sys, t + Δt/2, stages[2], N̄b, v, Δt/3)
    ā  .+= v
    x  .+= E2 .* ā
    N̄a .+= Q .* ā

    _transposed!(sys, t, stages[1], N̄a, v, Δt/6)
    x .+= v

    return nothing
end

# The transpose of g at a stage applied to y, written into out, as the adjoint
# system gives it, with its forcing weighted by w, the weight of the stage in the
# quadrature of the scheme: `sys` on y/w, its output times w. Overwrites y.
# Shared by the exponential schemes.
function _transposed!(sys::System, t::Real, stage, y, out, w::Real)
    y   .= y ./ w
    sys(t, stage, y, out)
    out .= w .* out

    return out
end
