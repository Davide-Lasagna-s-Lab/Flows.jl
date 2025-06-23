export StoreOneButLast, StoreOnlyLast

# A special monitor that only store the time and 
# state before the last step is made. The state should
# allow broadcasting assignment
mutable struct StoreOneButLast{X, F} <: AbstractMonitor{Float64, X}
    x::X
    f::F
    t::Float64
    function StoreOneButLast(z, f::F = Base.copy) where {F}
        x = f(z)
        new{typeof(x), F}(x, f, 0.0)
    end
end

Base.push!(mon::StoreOneButLast, t::Real, z, ::Bool) =
    (mon.x .= mon.f(z); mon.t = t; nothing)

# A special monitor that only store the time and 
# state for the last step only. The state should
# allow broadcasting assignment
mutable struct StoreOnlyLast{X, F} <: AbstractMonitor{Float64, X}
    x::X
    f::F
    t::Float64
    function StoreOnlyLast(z, f::F = Base.copy) where {F}
        x = f(z)
        new{typeof(x), F}(x, f, 0.0)
    end
end

Base.push!(mon::StoreOnlyLast, t::Real, z, ::Bool) =
    (mon.x .= mon.f(z); mon.t = t; nothing)