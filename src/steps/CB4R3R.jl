# type definition
struct CB4R3R4{X, MODE, NX} <: AbstractMethod{X, MODE, 6}
    store::NX
    CB4R3R4{X, MODE}(store::NX) where {X, MODE, N, NX<:NTuple{N, X}} = 
        new{X, MODE, NX}(store)
end

"""
    CB4R3R4(x::X, mode::AbstractMode = NormalMode()) -> CB4R3R4

Construct the four-register, six-stage, fourth-order IMEX Runge–Kutta
scheme of Cavaglieri & Bewley (2015). `x` is a template used to
preallocate the internal stage buffers; the count depends on `mode`:
`4` for [`NormalMode`](@ref) and the tangent of [`DiscreteMode`](@ref),
`5` for [`ContinuousMode`](@ref), `9` for the adjoint of
[`DiscreteMode`](@ref), its six adjoint stages and three temporaries.
"""
function CB4R3R4(x::X, mode::MODE = NormalMode()) where {X, MODE<:AbstractMode}
    N = mode isa ContinuousMode      ? 5 :
        mode isa DiscreteMode{true}  ? 9 : 4
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

# ---------------------------------------------------------------------------- #
# Forward linearised method takes x_{n} and overwrites it with x_{n+1}: the
# normal step with the explicit term linearised about the cached stages, the
# tangent of the discrete flow
function step!(method::CB4R3R4{X, MODE},
                  sys::System,
                    t::Real,
                   Δt::Real,
                    x::X,
               stages::NTuple{6, X}) where {X, MODE<:DiscreteMode{false}}

    # hoist temporaries out
    y, zᴵ, zᴱ, w = method.store
    tab = CB4

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
        sys(t + tab[:cᴱ, k]*Δt, stages[k], w, zᴱ)
        x .= x .+ tab[:bᴵ, k].*Δt.*zᴵ .+ tab[:bᴱ, k].*Δt.*zᴱ
    end

    return nothing
end


# ---------------------------------------------------------------------------- #
# Adjoint linearised method takes x_{n+1} and overwrites it with x_{n}.
#
# The step in the form of Butcher, with A the linear part and f the rest,
#
#     Yᵢ = x + Δt Σ_{j<i} aᴱᵢⱼ f(Yⱼ) + Δt Σ_{j≤i} aᴵᵢⱼ A Yⱼ,   i = 1, …, 6
#     x⁺ = x + Δt Σᵢ (bᴱᵢ f(Yᵢ) + bᴵᵢ A Yᵢ),
#
# which the low-storage loop of the normal step computes. Its transpose, from
# λ⁺ = x and with Jᵢ the Jacobian of f at the stage Yᵢ, goes back over the
# adjoint stages θᵢ, i = 6, …, 1,
#
#     (I - Δt aᴵᵢᵢ Aᵀ) θᵢ = Δt Jᵢᵀ eᵢ + Δt Aᵀ gᵢ,
#     eᵢ = bᴱᵢ λ⁺ + Σ_{l>i} aᴱₗᵢ θₗ,   gᵢ = bᴵᵢ λ⁺ + Σ_{l>i} aᴵₗᵢ θₗ,
#
# and λ = λ⁺ + Σᵢ θᵢ, so that ⟨λ⁺, δx⁺⟩ = ⟨λ, δx⟩ for the tangent δx⁺ of δx.
# `sys` holds Aᵀ and Jᵢᵀ.
function step!(method::CB4R3R4{X, MODE},
                  sys::System,
                    t::Real,
                   Δt::Real,
                    x::X,
               stages::NTuple{6, X}) where {X, MODE<:DiscreteMode{true}}

    # ---- the adjoint stages θ₁, …, θ₆, and the temporaries ----
    θ       = method.store
    e, g, v = method.store[7], method.store[8], method.store[9]
    tab     = CB4

    # ---- back over the stages ----
    @inbounds for i = 6:-1:1
        # ---- eᵢ and gᵢ, from λ⁺ and the later adjoint stages ----
        e .= tab[:bᴱ, i].*x
        g .= tab[:bᴵ, i].*x
        for l = (i + 1):6
            e .= e .+ tab[:aᴱ, l, i].*θ[l]
            g .= g .+ tab[:aᴵ, l, i].*θ[l]
        end

        # ---- Δt Jᵢᵀ eᵢ: `sys` on eᵢ/bᴱᵢ and its output scaled back by bᴱᵢ Δt,
        #      so that a forcing in `sys` gets the weight bᴱᵢ Δt of the
        #      quadrature of the scheme. Every bᴱᵢ of CB4 is nonzero ----
        e .= e ./ tab[:bᴱ, i]
        sys(t + tab[:cᴱ, i]*Δt, stages[i], e, v)
        e .= (tab[:bᴱ, i]*Δt).*v

        # ---- plus Δt Aᵀ gᵢ ----
        mul!(v, sys, g)
        e .= e .+ Δt.*v

        # ---- θᵢ = (I - Δt aᴵᵢᵢ Aᵀ)⁻¹ (Δt Jᵢᵀ eᵢ + Δt Aᵀ gᵢ) ----
        ImcA!(sys, tab[:aᴵ, i, i]*Δt, e, θ[i])
    end

    # ---- λ = λ⁺ + Σᵢ θᵢ ----
    for i = 1:6
        x .= x .+ θ[i]
    end

    return nothing
end
