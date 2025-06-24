export StoreNFromLast

# A special monitor that only store the time and 
# state N time steps before the last step is made.
# The state should allow broadcasting assignment
mutable struct StoreNFromLast{N, X, F} <: AbstractMonitor{Float64, X}
    x::X
    f::F
    t::Float64
    function StoreNFromLast{N}(z, f::F = Base.copy) where {N, F}
        x = f(z)
        new{N, typeof(x), F}(x, f, 0.0)
    end
end

getN(::StoreNFromLast{N}) where {N} = N

Base.push!(mon::StoreNFromLast, t::Real, z, ::Bool) =
    (mon.x .= mon.f(z); mon.t = t; nothing)
