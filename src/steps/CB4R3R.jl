export CB4R3R4

# type definition
struct CB4R3R4{X, MODE, NX} <: AbstractMethod{X, MODE, 6}
    store::NX
    CB4R3R4{X, MODE}(store::NX) where {X, MODE, N, NX<:NTuple{N, X}} = 
        new{X, MODE, NX}(store)
end

"""
    CB4R3R4(x::X, mode::AbstractMode=NormalMode())

Constructs a `CB4R3R4` integration scheme object for integration with mode `mode`.
"""
function CB4R3R4(x::X, mode::MODE = NormalMode()) where {X, MODE<:AbstractMode}
    N = mode isa NormalMode ? 4 : 5
    return CB4R3R4{X, MODE}(ntuple(i->similar(x), N))
end

# ---------------------------------------------------------------------------- #
# Normal time stepping with optional stage caching
function step!(method::CB4R3R4{X},
                  sys::System,
                    t::Real,
                   Δt::Real,
                    x::X,
                    c::C) where {X, C<:Union{Nothing, AbstractStageCache{6, X}}}

    # hoist temporaries out
    y, zᴵ, zᴱ, w   = method.store
    tab = CB4

    # temporary vector for storing stages
    _iscache(C) && (stages = sizehint!(X[], 6))

    # loop over stages
    @inbounds for k = 1:6
        if k == 1
             y .= x
            zᴵ .= x
            zᴱ .= x
        else
            zᴱ .= y .+ tab[:aᴱ, k, k-1].*Δt.*zᴱ
            if k < 6
                y .= x .+ (tab[:aᴵ, k+1, k-1] .- tab[:bᴵ, k-1]).*Δt.*zᴵ .+
                          (tab[:aᴱ, k+1, k-1] .- tab[:bᴱ, k-1])./tab[:aᴱ, k, k-1].*(zᴱ .- y)
            end
            zᴱ .= zᴱ .+ tab[:aᴵ, k, k-1]*Δt.*zᴵ
        end
        mul!(w, sys, zᴱ)                     # compute w = A*zᴱ then
        ImcA!(sys, tab[:aᴵ, k, k]*Δt, w, zᴵ) # compute z = (I-cA)⁻¹*w in place
        w .= zᴱ .+ tab[:aᴵ, k, k]*Δt.*zᴵ     # w is the temp input, output is zᴱ
        sys(t + tab[:cᴱ, k]*Δt, w, zᴱ); _iscache(C) && (push!(stages, copy(w)))
        x .= x .+ tab[:bᴵ, k].*Δt.*zᴵ .+ tab[:bᴱ, k].*Δt.*zᴱ
    end

    # cache stages if requested
    _iscache(C) && push!(c, t, Δt, tuple(stages...))

    return nothing
end

# ---------------------------------------------------------------------------- #
# Continuous time stepping for linearised/adjoint equations with interpolation
# from an `AbstractStorage` object for the evaluation of the linear operator.
function step!(method::CB4R3R4{X, MODE},
                  sys::System,
                    t::Real,
                   Δt::Real,
                    x::X,
                store::AbstractStorage) where {X, MODE<:ContinuousMode}

    # hoist temporaries out
    y, zᴵ, zᴱ, w, u   = method.store
    tab = CB4

    # modifier for the location of the interpolation
    _m_ = isadjoint(MODE) == true ? -1 : 1

    # loop over stages
    @inbounds for k = 1:6
        if k == 1
             y .= x
            zᴵ .= x
            zᴱ .= x
        else
            zᴱ .= y .+ tab[:aᴱ, k, k-1].*_m_.*Δt.*zᴱ
            if k < 6
                y .= x .+ (tab[:aᴵ, k+1, k-1] .- tab[:bᴵ, k-1]).*_m_.*Δt.*zᴵ .+
                          (tab[:aᴱ, k+1, k-1] .- tab[:bᴱ, k-1])./tab[:aᴱ, k, k-1].*(zᴱ .- y)
            end
            zᴱ .= zᴱ .+ tab[:aᴵ, k, k-1].*_m_.*Δt.*zᴵ
        end
        mul!(w, sys, zᴱ)                         # compute w = A*zᴱ then
        ImcA!(sys, _m_*tab[:aᴵ, k, k]*Δt, w, zᴵ) # compute z = (I-cA)⁻¹*w in place
        w .= zᴱ .+ tab[:aᴵ, k, k].*_m_.*Δt.*zᴵ   # w is the temp input, output is zᴱ
        sys(t + tab[:cᴱ, k]*Δt, store(u, t + tab[:cᴱ, k]*Δt), w, zᴱ)
        x .= x .+ tab[:bᴵ, k].*_m_.*Δt.*zᴵ .+ tab[:bᴱ, k].*_m_.*Δt.*zᴱ
    end

    return nothing
end