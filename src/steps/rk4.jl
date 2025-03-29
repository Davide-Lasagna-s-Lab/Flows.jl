export RK4

# ---------------------------------------------------------------------------- #
# Classical fourth order Runge-Kutta
struct RK4{X, MODE, NX} <: AbstractMethod{X, MODE, 4}
    store::NX
    RK4{X, MODE}(store::NX) where {X, MODE, N, NX<:NTuple{N, X}} = 
        new{X, MODE, NX}(store)
end

"""
    RK4(x::X, mode::AbstractMode=NormalMode())

Constructs a `RK4` integration scheme object for integration with mode `mode`.
"""
function RK4(x::X, mode::MODE = NormalMode()) where {X, MODE<:AbstractMode}
    N = mode isa ContinuousMode ? 6 : 5
    return RK4{X, MODE}(ntuple(i->similar(x), N))
end

# ---------------------------------------------------------------------------- #
# Perform time step and optionally push the four internal stages to a stage cache
function step!(method::RK4{X, NormalMode},
                  sys::System,
                    t::Real,
                   Δt::Real,
                    x::X,
                    c::C) where {X, 
                                 C<:Union{Nothing, AbstractStageCache{4, X}}}
    # aliases
    k1, k2, k3, k4, y = method.store

    # stages
    # TODO: seems like there could be a better way to push to the cache than copy at each RK step
    y .= x;              sys(t,        y, k1); _iscache(C) && (s1 = copy(y))
    y .= x .+ Δt.*k1./2; sys(t + Δt/2, y, k2); _iscache(C) && (s2 = copy(y))
    y .= x .+ Δt.*k2./2; sys(t + Δt/2, y, k3); _iscache(C) && (s3 = copy(y))
    y .= x .+ Δt.*k3   ; sys(t + Δt,   y, k4); _iscache(C) && (s4 = copy(y))

    # store stages if requested
    _iscache(C) && push!(c, t, Δt, (s1, s2, s3, s4))

    # wrap up
    x .= x .+ Δt./6 .* (k1 .+ 2.0.*k2 .+ 2.0.*k3 .+ k4)

    return nothing
end

# ---------------------------------------------------------------------------- #
# Continuous time stepping for linearised/adjoint equations with interpolation
# from an `AbstractStorage` object for the evaluation of the linear operator.
function step!(method::RK4{X, MODE},
                  sys::System,
                    t::Real,
                   Δt::Real,
                    x::X,
                store::AbstractStorage) where {X, MODE<:ContinuousMode}
    # aliases
    k1, k2, k3, k4, k5, y = method.store

    # modifier for the location of the interpolation
    _m_ = isadjoint(MODE) == true ? -1 : 1

    # stages
    y .= x                    ; sys(t,        store(k5, t       ), y, k1)
    y .= x .+ 0.5.*_m_.*Δt.*k1; sys(t + Δt/2, store(k5, t + Δt/2), y, k2)
    y .= x .+ 0.5.*_m_.*Δt.*k2; sys(t + Δt/2, store(k5, t + Δt/2), y, k3)
    y .= x .+      _m_.*Δt.*k3; sys(t + Δt,   store(k5, t + Δt  ), y, k4)

    # wrap up
    x .= x .+ _m_.*Δt./6 .* (k1 .+ 2.0.*k2 .+ 2.0.*k3 .+ k4)

    return nothing
end

# ---------------------------------------------------------------------------- #
# Forward linearised method takes x_{n} and overwrites it with x_{n+1}
# Adjoint linearised method takes x_{n+1} and overwrites it with x_{n}
function step!(method::RK4{X, MODE},
                  sys::System,
                    t::Real, # the time corresponding to x_{n}
                   Δt::Real,
                    x::X,
               stages::NTuple{4, X}) where {X, MODE<:DiscreteMode}
    # aliases
    k1, k2, k3, k4, y = method.store

    # stages
    if isadjoint(MODE)
        y .= x               ; sys(t + Δt  , stages[4], y, k4)
        y .= x .+ 0.5.*Δt.*k4; sys(t + Δt/2, stages[3], y, k3)
        y .= x .+ 0.5.*Δt.*k3; sys(t + Δt/2, stages[2], y, k2)
        y .= x .+      Δt.*k2; sys(t       , stages[1], y, k1)
    else
        y .= x               ; sys(t,        stages[1], y, k1)
        y .= x .+ 0.5.*Δt.*k1; sys(t + Δt/2, stages[2], y, k2)
        y .= x .+ 0.5.*Δt.*k2; sys(t + Δt/2, stages[3], y, k3)
        y .= x .+      Δt.*k3; sys(t + Δt,   stages[4], y, k4)
    end

    # wrap up
    x .= x .+ Δt./6 .* (k1 .+ 2.0.*k2 .+ 2.0.*k3 .+ k4)

    return nothing
end