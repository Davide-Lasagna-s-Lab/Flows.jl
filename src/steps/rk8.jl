export RK8

# ---------------------------------------------------------------------------- #
# Cooper–Verner eleven-stage, eighth-order explicit Runge–Kutta method.
# G. J. Cooper and J. H. Verner, Some Explicit Runge-Kutta Methods of High
# Order, SIAM Journal on Numerical Analysis 9(3), 389–405 (1972).
# https://doi.org/10.1137/0709037
# Coefficient reference: rkcv8 in rklib (standard A, b, c notation here):
# https://github.com/jacobwilliams/rklib/blob/master/src/rklib_fixed_steps.f90
#
# The coefficients involve sqrt(21) and are evaluated once in Float64,
# consistent with the other schemes in this package. There is no embedded
# formula: only the eleven stages required for the order-eight update.

const _RK8_C = let s = sqrt(21.0)
    (0.0, 1/2, 1/2, 1/2 - s/14, 1/2 - s/14, 1/2,
     1/2 + s/14, 1/2 + s/14, 1/2, 1/2 - s/14, 1.0)
end

const _RK8_B = (
    1/20, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    49/180, 16/45, 49/180, 1/20,
)

const _RK8_A = let a = zeros(11, 11), s = sqrt(21.0)
    a[2, 1] = 1/2

    a[3, 1] = 1/4
    a[3, 2] = 1/4

    a[4, 1] = 1/7
    a[4, 2] = -1/14 + 3*s/98
    a[4, 3] = 3/7 - 5*s/49

    a[5, 1] = 11/84 - s/84
    a[5, 3] = 2/7 - 4*s/63
    a[5, 4] = 1/12 + s/252

    a[6, 1] = 5/48 - s/48
    a[6, 3] = 1/4 - s/36
    a[6, 4] = -77/120 - 7*s/180
    a[6, 5] = 63/80 + 7*s/80

    a[7, 1] = 5/21 + s/42
    a[7, 3] = -48/35 - 92*s/315
    a[7, 4] = 211/30 + 29*s/18
    a[7, 5] = -36/5 - 23*s/14
    a[7, 6] = 9/5 + 13*s/35

    a[8, 1] = 1/14
    a[8, 5] = 1/9 + s/42
    a[8, 6] = 13/63 + s/21
    a[8, 7] = 1/9

    a[9, 1] = 1/32
    a[9, 5] = 91/576 + 7*s/192
    a[9, 6] = 11/72
    a[9, 7] = -385/1152 + 25*s/384
    a[9, 8] = 63/128 - 13*s/128

    a[10, 1] = 1/14
    a[10, 5] = 1/9
    a[10, 6] = -733/2205 + s/15
    a[10, 7] = 515/504 - 37*s/168
    a[10, 8] = -51/56 + 11*s/56
    a[10, 9] = 132/245 - 4*s/35

    a[11, 5] = -7/3 - 7*s/18
    a[11, 6] = -2/5 - 28*s/45
    a[11, 7] = -91/24 + 53*s/72
    a[11, 8] = 301/72 - 53*s/72
    a[11, 9] = 28/45 + 28*s/45
    a[11, 10] = 49/18 + 7*s/18
    a
end

# ---------------------------------------------------------------------------- #
struct RK8{X, MODE, NX} <: AbstractMethod{X, MODE, 11}
    store::NX
    RK8{X, MODE}(store::NX) where {X, MODE, N, NX<:NTuple{N, X}} =
        new{X, MODE, NX}(store)
end

"""
    RK8(x::X, mode::AbstractMode = NormalMode()) -> RK8

Construct the eleven-stage, eighth-order explicit Runge–Kutta method
of Cooper and Verner (1972). `x` is the template for the
preallocated working buffers. Supply the full vector field through
the explicit `flow(g, method, stepping)` interface, as for [`RK4`](@ref).

Supports nonlinear integration with optional caching of the eleven
stage states, continuous tangent/adjoint integration using an
[`AbstractStorage`](@ref), and discrete tangent/adjoint replay of a
[`RAMStageCache`](@ref). Allocate a cache with `RAMStageCache(11, x)`.
The discrete adjoint also supports an additive forcing, weighted
consistently with the Runge–Kutta quadrature.

This is a fixed-step formula with no embedded error estimator or
automatic step-size control. Step sizes are supplied by the existing
time-stepping policies. Continuous modes use the interpolation
provided by the storage.
There are 12 state buffers in normal and discrete tangent modes,
13 in continuous modes, and 14 in discrete adjoint mode.

# Example
```julia
method = RK8(x)
F = flow(g!, method, TimeStepConstant(0.01))
```
"""
function RK8(x::X, mode::MODE = NormalMode()) where {X, MODE<:AbstractMode}
    N = mode isa ContinuousMode ? 13 :
        mode isa DiscreteMode{true} ? 14 : 12
    return RK8{X, MODE}(ntuple(i -> similar(x), N))
end

# ---------------------------------------------------------------------------- #
# Nonlinear step, optionally recording the input state of each RHS evaluation.
function step!(method::RK8{X, NormalMode},
                  sys::System,
                    t::Real,
                   Δt::Real,
                    x::X,
                cache::C) where {X, C<:Union{Nothing, AbstractStageCache{11, X}}}
    # ---- aliases ----
    k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, y = method.store
    A, b, c = _RK8_A, _RK8_B, _RK8_C

    # ---- stages ----
    # Stage i calls the RHS at y = x + Δt Σ_j A[i, j] k_j and time t + c[i] Δt.
    # Only the nonzero A[i, j] appear. A copy of y is kept for the cache.
    y .= x
    sys(t + c[1]*Δt,  y, k1);  _iscache(C) && (s1  = copy(y))

    y .= x .+ Δt.*(A[2, 1].*k1)
    sys(t + c[2]*Δt,  y, k2);  _iscache(C) && (s2  = copy(y))

    y .= x .+ Δt.*(A[3, 1].*k1 .+ A[3, 2].*k2)
    sys(t + c[3]*Δt,  y, k3);  _iscache(C) && (s3  = copy(y))

    y .= x .+ Δt.*(A[4, 1].*k1 .+ A[4, 2].*k2 .+ A[4, 3].*k3)
    sys(t + c[4]*Δt,  y, k4);  _iscache(C) && (s4  = copy(y))

    y .= x .+ Δt.*(A[5, 1].*k1 .+ A[5, 3].*k3 .+ A[5, 4].*k4)
    sys(t + c[5]*Δt,  y, k5);  _iscache(C) && (s5  = copy(y))

    y .= x .+ Δt.*(A[6, 1].*k1 .+ A[6, 3].*k3 .+ A[6, 4].*k4 .+ A[6, 5].*k5)
    sys(t + c[6]*Δt,  y, k6);  _iscache(C) && (s6  = copy(y))

    y .= x .+ Δt.*(A[7, 1].*k1 .+ A[7, 3].*k3 .+ A[7, 4].*k4 .+ A[7, 5].*k5 .+ A[7, 6].*k6)
    sys(t + c[7]*Δt,  y, k7);  _iscache(C) && (s7  = copy(y))

    y .= x .+ Δt.*(A[8, 1].*k1 .+ A[8, 5].*k5 .+ A[8, 6].*k6 .+ A[8, 7].*k7)
    sys(t + c[8]*Δt,  y, k8);  _iscache(C) && (s8  = copy(y))

    y .= x .+ Δt.*(A[9, 1].*k1 .+ A[9, 5].*k5 .+ A[9, 6].*k6 .+ A[9, 7].*k7 .+ A[9, 8].*k8)
    sys(t + c[9]*Δt,  y, k9);  _iscache(C) && (s9  = copy(y))

    y .= x .+ Δt.*(A[10, 1].*k1 .+ A[10, 5].*k5 .+ A[10, 6].*k6 .+ A[10, 7].*k7 .+ A[10, 8].*k8 .+ A[10, 9].*k9)
    sys(t + c[10]*Δt, y, k10); _iscache(C) && (s10 = copy(y))

    y .= x .+ Δt.*(A[11, 5].*k5 .+ A[11, 6].*k6 .+ A[11, 7].*k7 .+ A[11, 8].*k8 .+ A[11, 9].*k9 .+ A[11, 10].*k10)
    sys(t + c[11]*Δt, y, k11); _iscache(C) && (s11 = copy(y))

    # ---- store the stages, if requested ----
    _iscache(C) && push!(cache, t, Δt, (s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11))

    # ---- wrap up ----
    x .= x .+ Δt.*(b[1].*k1 .+ b[8].*k8 .+ b[9].*k9 .+ b[10].*k10 .+ b[11].*k11)

    return nothing
end

# ---------------------------------------------------------------------------- #
# Continuous tangent/adjoint step. Stage times use the signed Δt supplied by
# the integrator. Only the RHS contribution changes sign for the adjoint.
function step!(method::RK8{X, MODE},
                  sys::System,
                    t::Real,
                   Δt::Real,
                    x::X,
                store::AbstractStorage) where {X, MODE<:ContinuousMode}
    # ---- aliases ----
    k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, y, u = method.store
    A, b, c = _RK8_A, _RK8_B, _RK8_C

    # ---- sign of the RHS contribution ----
    _m_ = isadjoint(MODE) ? -1 : 1

    # ---- stages ----
    # As in the nonlinear step, with the trajectory interpolated into u at
    # each stage time.
    y .= x
    sys(t + c[1]*Δt,  store(u, t + c[1]*Δt),  y, k1)

    y .= x .+ _m_.*Δt.*(A[2, 1].*k1)
    sys(t + c[2]*Δt,  store(u, t + c[2]*Δt),  y, k2)

    y .= x .+ _m_.*Δt.*(A[3, 1].*k1 .+ A[3, 2].*k2)
    sys(t + c[3]*Δt,  store(u, t + c[3]*Δt),  y, k3)

    y .= x .+ _m_.*Δt.*(A[4, 1].*k1 .+ A[4, 2].*k2 .+ A[4, 3].*k3)
    sys(t + c[4]*Δt,  store(u, t + c[4]*Δt),  y, k4)

    y .= x .+ _m_.*Δt.*(A[5, 1].*k1 .+ A[5, 3].*k3 .+ A[5, 4].*k4)
    sys(t + c[5]*Δt,  store(u, t + c[5]*Δt),  y, k5)

    y .= x .+ _m_.*Δt.*(A[6, 1].*k1 .+ A[6, 3].*k3 .+ A[6, 4].*k4 .+ A[6, 5].*k5)
    sys(t + c[6]*Δt,  store(u, t + c[6]*Δt),  y, k6)

    y .= x .+ _m_.*Δt.*(A[7, 1].*k1 .+ A[7, 3].*k3 .+ A[7, 4].*k4 .+ A[7, 5].*k5 .+ A[7, 6].*k6)
    sys(t + c[7]*Δt,  store(u, t + c[7]*Δt),  y, k7)

    y .= x .+ _m_.*Δt.*(A[8, 1].*k1 .+ A[8, 5].*k5 .+ A[8, 6].*k6 .+ A[8, 7].*k7)
    sys(t + c[8]*Δt,  store(u, t + c[8]*Δt),  y, k8)

    y .= x .+ _m_.*Δt.*(A[9, 1].*k1 .+ A[9, 5].*k5 .+ A[9, 6].*k6 .+ A[9, 7].*k7 .+ A[9, 8].*k8)
    sys(t + c[9]*Δt,  store(u, t + c[9]*Δt),  y, k9)

    y .= x .+ _m_.*Δt.*(A[10, 1].*k1 .+ A[10, 5].*k5 .+ A[10, 6].*k6 .+ A[10, 7].*k7 .+ A[10, 8].*k8 .+ A[10, 9].*k9)
    sys(t + c[10]*Δt, store(u, t + c[10]*Δt), y, k10)

    y .= x .+ _m_.*Δt.*(A[11, 5].*k5 .+ A[11, 6].*k6 .+ A[11, 7].*k7 .+ A[11, 8].*k8 .+ A[11, 9].*k9 .+ A[11, 10].*k10)
    sys(t + c[11]*Δt, store(u, t + c[11]*Δt), y, k11)

    # ---- wrap up ----
    x .= x .+ _m_.*Δt.*(b[1].*k1 .+ b[8].*k8 .+ b[9].*k9 .+ b[10].*k10 .+ b[11].*k11)

    return nothing
end

# ---------------------------------------------------------------------------- #
# Tangent of the discrete nonlinear step, evaluated at its cached stage states.
function step!(method::RK8{X, DiscreteMode{false}},
                  sys::System,
                    t::Real,
                   Δt::Real,
                    x::X,
               stages::NTuple{11, X}) where {X}
    # ---- aliases ----
    k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, y = method.store
    A, b, c = _RK8_A, _RK8_B, _RK8_C

    # ---- stages, linearised about the cached ones ----
    y .= x
    sys(t + c[1]*Δt,  stages[1],  y, k1)

    y .= x .+ Δt.*(A[2, 1].*k1)
    sys(t + c[2]*Δt,  stages[2],  y, k2)

    y .= x .+ Δt.*(A[3, 1].*k1 .+ A[3, 2].*k2)
    sys(t + c[3]*Δt,  stages[3],  y, k3)

    y .= x .+ Δt.*(A[4, 1].*k1 .+ A[4, 2].*k2 .+ A[4, 3].*k3)
    sys(t + c[4]*Δt,  stages[4],  y, k4)

    y .= x .+ Δt.*(A[5, 1].*k1 .+ A[5, 3].*k3 .+ A[5, 4].*k4)
    sys(t + c[5]*Δt,  stages[5],  y, k5)

    y .= x .+ Δt.*(A[6, 1].*k1 .+ A[6, 3].*k3 .+ A[6, 4].*k4 .+ A[6, 5].*k5)
    sys(t + c[6]*Δt,  stages[6],  y, k6)

    y .= x .+ Δt.*(A[7, 1].*k1 .+ A[7, 3].*k3 .+ A[7, 4].*k4 .+ A[7, 5].*k5 .+ A[7, 6].*k6)
    sys(t + c[7]*Δt,  stages[7],  y, k7)

    y .= x .+ Δt.*(A[8, 1].*k1 .+ A[8, 5].*k5 .+ A[8, 6].*k6 .+ A[8, 7].*k7)
    sys(t + c[8]*Δt,  stages[8],  y, k8)

    y .= x .+ Δt.*(A[9, 1].*k1 .+ A[9, 5].*k5 .+ A[9, 6].*k6 .+ A[9, 7].*k7 .+ A[9, 8].*k8)
    sys(t + c[9]*Δt,  stages[9],  y, k9)

    y .= x .+ Δt.*(A[10, 1].*k1 .+ A[10, 5].*k5 .+ A[10, 6].*k6 .+ A[10, 7].*k7 .+ A[10, 8].*k8 .+ A[10, 9].*k9)
    sys(t + c[10]*Δt, stages[10], y, k10)

    y .= x .+ Δt.*(A[11, 5].*k5 .+ A[11, 6].*k6 .+ A[11, 7].*k7 .+ A[11, 8].*k8 .+ A[11, 9].*k9 .+ A[11, 10].*k10)
    sys(t + c[11]*Δt, stages[11], y, k11)

    # ---- wrap up ----
    x .= x .+ Δt.*(b[1].*k1 .+ b[8].*k8 .+ b[9].*k9 .+ b[10].*k10 .+ b[11].*k11)

    return nothing
end

# ---------------------------------------------------------------------------- #
# Reverse accumulation through the explicit RK stages. On entry x = λ_{n+1},
# and on return x = λ_n. The RHS supplied by the caller is J_i' * y + r_i.
function step!(method::RK8{X, DiscreteMode{true}},
                  sys::System,
                    t::Real,
                   Δt::Real,
                    x::X,
               stages::NTuple{11, X}) where {X}
    # ---- a step of zero length leaves the adjoint unchanged ----
    # The stages with b_i ≠ 0 below divide by Δt b_i, so return before them.
    iszero(Δt) && return nothing

    # ---- aliases ----
    k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, y, v, r = method.store
    A, b, c = _RK8_A, _RK8_B, _RK8_C

    # ---- adjoint of the update ----
    # k_i is the adjoint of the i-th stage derivative. The update
    # x + Δt Σ_i b_i k_i gives it Δt b_i x, before the stages add their own
    # contributions. The k_i with b_i = 0 start from zero.
    k1  .= (Δt*b[1]).*x
    k2  .= (Δt*b[2]).*x
    k3  .= (Δt*b[3]).*x
    k4  .= (Δt*b[4]).*x
    k5  .= (Δt*b[5]).*x
    k6  .= (Δt*b[6]).*x
    k7  .= (Δt*b[7]).*x
    k8  .= (Δt*b[8]).*x
    k9  .= (Δt*b[9]).*x
    k10 .= (Δt*b[10]).*x
    k11 .= (Δt*b[11]).*x

    # ---- stages, in reverse order ----
    # When stage i is reached, k_i is complete. With w = Δt b_i, the stage forms
    # v = J_i' k_i + w r_i, so that the forcing gets its weight in the
    # quadrature. The input of stage i, x + Δt Σ_j A[i, j] k_j, then sends v
    # back to x and to the k_j with A[i, j] ≠ 0. The weights b are fixed, so
    # each stage takes its branch directly, with no test on w at run time.

    # Stages 11 to 8 have b_i ≠ 0. The RHS acts on k_i/w and its output is
    # scaled by w.
    y .= k11./(Δt*b[11]); sys(t + c[11]*Δt, stages[11], y, v); v .= (Δt*b[11]).*v
    x   .= x   .+ v
    k5  .= k5  .+ (Δt*A[11, 5]).*v
    k6  .= k6  .+ (Δt*A[11, 6]).*v
    k7  .= k7  .+ (Δt*A[11, 7]).*v
    k8  .= k8  .+ (Δt*A[11, 8]).*v
    k9  .= k9  .+ (Δt*A[11, 9]).*v
    k10 .= k10 .+ (Δt*A[11, 10]).*v

    y .= k10./(Δt*b[10]); sys(t + c[10]*Δt, stages[10], y, v); v .= (Δt*b[10]).*v
    x  .= x  .+ v
    k1 .= k1 .+ (Δt*A[10, 1]).*v
    k5 .= k5 .+ (Δt*A[10, 5]).*v
    k6 .= k6 .+ (Δt*A[10, 6]).*v
    k7 .= k7 .+ (Δt*A[10, 7]).*v
    k8 .= k8 .+ (Δt*A[10, 8]).*v
    k9 .= k9 .+ (Δt*A[10, 9]).*v

    y .= k9./(Δt*b[9]); sys(t + c[9]*Δt, stages[9], y, v); v .= (Δt*b[9]).*v
    x  .= x  .+ v
    k1 .= k1 .+ (Δt*A[9, 1]).*v
    k5 .= k5 .+ (Δt*A[9, 5]).*v
    k6 .= k6 .+ (Δt*A[9, 6]).*v
    k7 .= k7 .+ (Δt*A[9, 7]).*v
    k8 .= k8 .+ (Δt*A[9, 8]).*v

    y .= k8./(Δt*b[8]); sys(t + c[8]*Δt, stages[8], y, v); v .= (Δt*b[8]).*v
    x  .= x  .+ v
    k1 .= k1 .+ (Δt*A[8, 1]).*v
    k5 .= k5 .+ (Δt*A[8, 5]).*v
    k6 .= k6 .+ (Δt*A[8, 6]).*v
    k7 .= k7 .+ (Δt*A[8, 7]).*v

    # Stages 7 to 2 have b_i = 0, so the forcing has no weight there. The RHS
    # on k_i gives J_i' k_i + r_i, and the RHS on zero, r_i, is taken away.
    sys(t + c[7]*Δt, stages[7], k7, v); y .= zero.(y)
    sys(t + c[7]*Δt, stages[7], y,  r); v .= v .- r
    x  .= x  .+ v
    k1 .= k1 .+ (Δt*A[7, 1]).*v
    k3 .= k3 .+ (Δt*A[7, 3]).*v
    k4 .= k4 .+ (Δt*A[7, 4]).*v
    k5 .= k5 .+ (Δt*A[7, 5]).*v
    k6 .= k6 .+ (Δt*A[7, 6]).*v

    sys(t + c[6]*Δt, stages[6], k6, v); y .= zero.(y)
    sys(t + c[6]*Δt, stages[6], y,  r); v .= v .- r
    x  .= x  .+ v
    k1 .= k1 .+ (Δt*A[6, 1]).*v
    k3 .= k3 .+ (Δt*A[6, 3]).*v
    k4 .= k4 .+ (Δt*A[6, 4]).*v
    k5 .= k5 .+ (Δt*A[6, 5]).*v

    sys(t + c[5]*Δt, stages[5], k5, v); y .= zero.(y)
    sys(t + c[5]*Δt, stages[5], y,  r); v .= v .- r
    x  .= x  .+ v
    k1 .= k1 .+ (Δt*A[5, 1]).*v
    k3 .= k3 .+ (Δt*A[5, 3]).*v
    k4 .= k4 .+ (Δt*A[5, 4]).*v

    sys(t + c[4]*Δt, stages[4], k4, v); y .= zero.(y)
    sys(t + c[4]*Δt, stages[4], y,  r); v .= v .- r
    x  .= x  .+ v
    k1 .= k1 .+ (Δt*A[4, 1]).*v
    k2 .= k2 .+ (Δt*A[4, 2]).*v
    k3 .= k3 .+ (Δt*A[4, 3]).*v

    sys(t + c[3]*Δt, stages[3], k3, v); y .= zero.(y)
    sys(t + c[3]*Δt, stages[3], y,  r); v .= v .- r
    x  .= x  .+ v
    k1 .= k1 .+ (Δt*A[3, 1]).*v
    k2 .= k2 .+ (Δt*A[3, 2]).*v

    sys(t + c[2]*Δt, stages[2], k2, v); y .= zero.(y)
    sys(t + c[2]*Δt, stages[2], y,  r); v .= v .- r
    x  .= x  .+ v
    k1 .= k1 .+ (Δt*A[2, 1]).*v

    # Stage 1 has b_1 ≠ 0, as stages 11 to 8. It sends v back to x alone.
    y .= k1./(Δt*b[1]); sys(t + c[1]*Δt, stages[1], y, v); v .= (Δt*b[1]).*v
    x .= x .+ v

    return nothing
end
