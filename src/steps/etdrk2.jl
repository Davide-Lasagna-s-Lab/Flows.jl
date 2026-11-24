export ETDRK2, etdrk2_coeffs

# ---------------------------------------------------------------------------- #
# Exponential time differencing RK2 (ETD2RK) of Cox & Matthews (2002), with the
# coefficients computed by the contour integrals of Kassam & Trefethen (2005).
# The equation is  u' = A u + g(t, u), with A treated exactly.
struct ETDRK2{X, MODE, NX, NC} <: AbstractMethod{X, MODE, 2}
    store::NX                      # stage buffers
     coef::NC                      # coefficients E, f1, f2, as states
       Δt::Base.RefValue{Float64}  # step of the coefficients
        A::Base.RefValue{Any}      # linear operator of the coefficients

    ETDRK2{X, MODE}(store::NX,
                     coef::NC) where {X, MODE, NX<:Tuple{Vararg{X}}, NC<:NTuple{3, X}} =
        new{X, MODE, NX, NC}(store, coef, Ref(NaN), Ref{Any}(nothing))
end

"""
    ETDRK2(x::X, mode::AbstractMode = NormalMode()) -> ETDRK2

Construct the second-order exponential time differencing Runge–Kutta
scheme (ETD2RK) of Cox & Matthews (2002) for `u' = A u + g(t, u)`:

    a  = E x + f1 g(t, x)
    x₊ = a + f2 (g(t + Δt, a) - g(t, x))

The linear part `A` is integrated exactly, so the scheme is not limited in
`Δt` by the stiffness of `A`. `x` is a template for the buffers: `3` stages
and `3` coefficients for [`NormalMode`](@ref), `4` stages for
[`ContinuousMode`](@ref).

`A` is a `LinearAlgebra.Diagonal`, or `nothing` (Heun's RK2 for that
component), also as components of a [`Coupled`](@ref) system. Other
operators need a method of [`etdrk2_coeffs`](@ref). The cache of the
coefficients, [`ContinuousMode`](@ref) (forward and adjoint, the orbit
interpolated at `t` and `t + Δt`) and the lack of [`DiscreteMode`](@ref)
and of stage caches are as for [`ETDRK4`](@ref).
"""
function ETDRK2(x::X, mode::MODE=NormalMode()) where {X, MODE<:AbstractMode}
    # ---- input checks ----
    mode isa DiscreteMode &&
        throw(ArgumentError("ETDRK2 has no DiscreteMode: use NormalMode or ContinuousMode"))

    # ---- buffers ----
    N = mode isa ContinuousMode ? 4 : 3

    return ETDRK2{X, MODE}(ntuple(i->similar(x), N),
                           ntuple(i->similar(x), 3))
end

"""
    etdrk2_coeffs(A, Δt::Real) -> (E, f1, f2)

Coefficients of [`ETDRK2`](@ref) for the linear operator `A` and the step
`Δt > 0`, with `L = Δt A`, `φ₁ = (eᶻ - 1)/z` and `φ₂ = (eᶻ - 1 - z)/z²`:

    E = exp(L)       f1 = Δt φ₁(L)       f2 = Δt φ₂(L)

Each of the three values must be an object that multiplies a state by
broadcasting, `c .* x`; for a diagonal operator these are vectors of the
diagonal entries. Users add a method for their operator type; the fallback
raises an error. A method for `LinearAlgebra.Diagonal` is provided, with the
contour integrals of Kassam & Trefethen (2005).
"""
etdrk2_coeffs(A, Δt::Real) =
    error("etdrk2_coeffs missing implementation for operator `A` of type $(typeof(A))")

function etdrk2_coeffs(A::LinearAlgebra.Diagonal, Δt::Real)
    # ---- coefficients ----
    L  = Δt .* A.diag
    E  = exp.(L)
    f1 = Δt .* _contour(z -> (exp(z) - 1) / z,        L)
    f2 = Δt .* _contour(z -> (exp(z) - 1 - z) / z^2,  L)

    return (E, f1, f2)
end

# The hook and the limits at L = 0 of the coefficients, those of Heun's RK2
_etd_hook(::ETDRK2, A, Δt)   = etdrk2_coeffs(A, Δt)
_etd_limits(::ETDRK2, Δt)    = (1, Δt, Δt/2)

# ---------------------------------------------------------------------------- #
# Normal time stepping. No stage cache
function step!(method::ETDRK2{X, NormalMode},
                  sys::System,
                    t::Real,
                   Δt::Real,
                    x::X,
                    c::Nothing) where {X}
    # ---- coefficients and buffers ----
    _etd_update!(method, sys.A, Δt)
    E, f1, f2  = method.coef
    Na, Nb, a  = method.store

    # ---- stages ----
    sys(t, x, Na)
    a .= E .* x .+ f1 .* Na

    sys(t + Δt, a, Nb)

    # ---- update ----
    x .= a .+ f2 .* (Nb .- Na)

    return nothing
end

# ---------------------------------------------------------------------------- #
# Continuous tangent or adjoint along an orbit in a storage, interpolated at the
# times t and t + Δt. In adjoint mode Δt < 0 and the equation is marched in the
# reversed time, with step -Δt.
function step!(method::ETDRK2{X, MODE},
                  sys::System,
                    t::Real,
                   Δt::Real,
                    x::X,
                store::AbstractStorage) where {X, MODE<:ContinuousMode}
    # ---- coefficients and buffers ----
    _m_ = isadjoint(MODE) ? -1 : 1
    _etd_update!(method, sys.A, _m_*Δt)
    E, f1, f2     = method.coef
    Na, Nb, a, u  = method.store

    # ---- stages. For coupled vectors the storage fills the first component of `u`
    # and returns it, and that one state is passed to all of them ----
    ū = store(u, t)
    sys(t, ū, x, Na)
    a .= E .* x .+ f1 .* Na

    ū = store(u, t + Δt)
    sys(t + Δt, ū, a, Nb)

    # ---- update ----
    x .= a .+ f2 .* (Nb .- Na)

    return nothing
end
