export CNRK2

# ---------------------------------------------------------------------------- #
# Crank-Nicolson/Heun scheme
struct CNRK2{X, MODE, NX} <: AbstractMethod{X, MODE, 2}
    store::NX
    CNRK2{X, MODE}(store::NX) where {X, MODE, N, NX<:NTuple{N, X}} = 
        new{X, MODE, NX}(store)
end

"""
    CNRK2(x::X, mode::AbstractMode=NormalMode())

Constructs a `CNRK2` integration scheme object for integration with mode `mode`.
"""
CNRK2(x::X, ::MODE = NormalMode()) where {X, MODE<:AbstractMode} = 
    CNRK2{X, MODE}(ntuple(i->similar(x), 5))

# ---------------------------------------------------------------------------- #
# Normal time stepping with optional stage caching
function step!(method::CNRK2{X, NormalMode},
                  sys::System,
                    t::Real,
                   Δt::Real,
                    x::X,
                    c::C) where {X, C<:Union{Nothing, AbstractStageCache{2, X}}}
    # aliases
    k1, k2, k3, k4, k5 = method.store

    # predictor step
    ImcA_mul!(sys, -0.5*Δt, x, k1)
    sys(t, x, k2); _iscache(C) && (s1 = copy(x))
    k3 .= k1 .+ Δt.*k2
    ImcA!(sys, 0.5*Δt, k3, k4)

    # corrector
    sys(t+Δt, k4, k5); _iscache(C) && (s2 = copy(k4))
    k3 .= k1 .+ 0.5.*Δt.*(k2 .+ k5)
    ImcA!(sys, 0.5*Δt, k3, x)

    # store stages if requested
    _iscache(C) && push!(c, t, Δt, (s1, s2))

    return nothing
end

# ---------------------------------------------------------------------------- #
# Continuous time stepping for linearised/adjoint equations with interpolation
# from an `AbstractStorage` object for the evaluation of the linear operator.
function step!(method::CNRK2{X, MODE},
                  sys::System,
                    t::Real,
                   Δt::Real,
                    x::X,
                store::AbstractStorage) where {X, MODE<:ContinuousMode}

    # modifier for the location of the interpolation
    _m_ = isadjoint(MODE) == true ? -1 : 1

    # aliases
    k1, k2, k3, k4, k5 = method.store

    ImcA_mul!(sys, -0.5*_m_*Δt, x, k1)
    sys(t, store(k5, t), x, k2)
    k3 .= k1 .+ _m_.*Δt.*k2
    ImcA!(sys, 0.5*_m_*Δt, k3, k4)
    sys(t + Δt, store(k3, t + Δt), k4, k5)
    k3 .= k1 .+ 0.5.*_m_.*Δt.*(k2 .+ k5)
    ImcA!(sys, 0.5*_m_*Δt, k3, x)

    return nothing
end

# ---------------------------------------------------------------------------- #
# Forward linearised method takes x_{n} and overwrites it with x_{n+1}
# Adjoint linearised method takes x_{n+1} and overwrites it with x_{n}
function step!(method::CNRK2{X, MODE},
                  sys::System,
                    t::Real,
                   Δt::Real,
                    x::X,
               stages::NTuple{2, X}) where {X, MODE<:DiscreteMode}
    # aliases
    k1, k2, k3, k4, k5 = method.store

    if isadjoint(MODE)
        ImcA!(sys, 0.5*Δt, x, k1)
        sys(t + Δt, stages[2], k1, k2)
        k2 .= k2.*Δt./2
        ImcA!(sys, 0.5*Δt, k2, k3)
        k4 .= Δt./2.0.*k1 .+ Δt.*k3
        sys(t, stages[1], k4, k5)
        k2 .= k1 .+ k3
        ImcA_mul!(sys, -0.5*Δt, k2, k3)
        x .= k3 .+ k5
    else
        ImcA_mul!(sys, -0.5*Δt, x, k1)
        sys(t, stages[1], x, k2)
        k3 .= k1 .+ Δt.*k2
        ImcA!(sys, 0.5*Δt, k3, k4)
        sys(t+Δt, stages[2], k4, k5)
        k3 .= k1 .+ 0.5.*Δt.*(k2 .+ k5)
        ImcA!(sys, 0.5*Δt, k3, x)
    end

    return nothing
end
