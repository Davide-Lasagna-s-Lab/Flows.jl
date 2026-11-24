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
coefficients for [`NormalMode`](@ref), `7` stages for [`ContinuousMode`](@ref).

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
[`DiscreteMode`](@ref) is not supported, and the constructor throws an error.
No stage cache can be used.
"""
function ETDRK4(x::X, mode::MODE=NormalMode()) where {X, MODE<:AbstractMode}
    # ---- input checks ----
    mode isa DiscreteMode &&
        throw(ArgumentError("ETDRK4 has no DiscreteMode: use NormalMode or ContinuousMode"))

    # ---- buffers ----
    N = mode isa ContinuousMode ? 7 : 6

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
    # ---- contour points: z[i] + r[j], r on the unit circle ----
    M = 32
    r = exp.(im .* π .* ((1:M) .- 0.5) ./ M)
    L = Δt .* A.diag
    z = L .+ transpose(r)
    e = exp.(z)

    # ---- mean over the contour, real part (the upper half circle is enough) ----
    avg(f) = vec(real.(sum(f; dims=2)) ./ M)

    # ---- coefficients ----
    E  = exp.(L)
    E2 = exp.(L ./ 2)
    Q  = Δt .* avg((exp.(z ./ 2) .- 1) ./ z)
    f1 = Δt .* avg((-4 .- z .+ e .* (4 .- 3 .* z .+ z.^2)) ./ z.^3)
    f2 = Δt .* avg(( 2 .+ z .+ e .* (-2 .+ z))             ./ z.^3)
    f3 = Δt .* avg((-4 .- 3 .* z .- z.^2 .+ e .* (4 .- z)) ./ z.^3)

    return (E, E2, Q, f1, f2, f3)
end

# ---------------------------------------------------------------------------- #
# Update the coefficients when the step or the operator change. Each is a state
# like x, so that the stages are plain broadcasts. A component without linear
# part has the limits of the coefficients at L = 0, those of RK4.
function _etd_update!(method::ETDRK4, A, Δt)
    if method.Δt[] != Δt || method.A[] !== A
        comps = A isa Coupled ? A.args : (A,)
        for (i, Ai) in enumerate(comps)
            cs = Ai === nothing ? (1, 1, Δt/2, Δt/6, Δt/6, Δt/6) : etdrk4_coeffs(Ai, Δt)
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
# Normal time stepping. No stage cache
function step!(method::ETDRK4{X, NormalMode},
                  sys::System,
                    t::Real,
                   Δt::Real,
                    x::X,
                    c::Nothing) where {X}
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
