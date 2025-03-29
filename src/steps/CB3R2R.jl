export CB3R2R2, CB3R2R3e, CB3R2R3c

# ---------------------------------------------------------------------------- #
# Three-register version of [2R] IMEXRK schemes from section 1.2.1 of CB 2015

for (name, tab, NS) in zip((:CB3R2R2, :CB3R2R3e, :CB3R2R3c),
                           ( CB2,      CB3e,      CB3c),
                           ( 3,        4,         4))
    @eval begin

        # type
        struct $name{X, MODE, NX} <: AbstractMethod{X, MODE, $NS}
            store::NX
            $name{X, MODE}(store::NX) where {X, MODE, N, NX<:NTuple{N, X}} = 
                new{X, MODE, NX}(store)
        end

        function $name(x::X, mode::MODE = NormalMode()) where {X, MODE<:AbstractMode}
            N = mode isa NormalMode     ? 3 :
                mode isa ContinuousMode ? 4 : 6
            return $name{X, MODE}(ntuple(i->similar(x), N))
        end

        # required to cope with buggy julia deepcopy implementation
        # TODO: test if deep copy method is necessary
        function Base.deepcopy_internal(x::$name, dict::IdDict)
            if !( haskey(dict, x) )
                dict[x] = $name(x.store[1], mode(x)())
            end
            return dict[x]
        end

        # ---------------------------------------------------------------------------- #
        # Normal time stepping with optional stage caching
        function step!(method::$name{X, NormalMode},
                          sys::System,
                            t::Real,
                           Δt::Real,
                            x::X,
                            c::C) where {X, C<:Union{Nothing, AbstractStageCache{$NS, X}}}

            # hoist temporaries out
            y, z, w  = method.store
            tab = $tab

            # temporary vector for storing stages
            _iscache(C) && (stages = sizehint!(X[], $NS))

            # loop over stages
            @inbounds for k = 1:$NS
                if k == 1
                    y .= x
                else
                    y .= x .+ (tab[:aᴵ, k, k-1] .- tab[:bᴵ, k-1]).*Δt.*z .+
                              (tab[:aᴱ, k, k-1] .- tab[:bᴱ, k-1]).*Δt.*y
                end
                mul!(z, sys, y)                     # compute z = A*y then
                ImcA!(sys, tab[:aᴵ, k, k]*Δt, z, w); z .= w # get w = (I-cA)⁻¹*z
                w .= y .+ tab[:aᴵ, k, k].*Δt.*z     # w is the temp input, output is y
                sys(t + tab[:cᴱ, k]*Δt, w, y); _iscache(C) && (push!(stages, copy(w)))
                x .= x .+ tab[:bᴵ, k].*Δt.*z .+ tab[:bᴱ, k].*Δt.*y
            end

            # cache stages if requested
            _iscache(C) && push!(c, t, Δt, tuple(stages...))

            return nothing
        end

        # ---------------------------------------------------------------------------- #
        # Continuous time stepping for linearised/adjoint equations with interpolation
        # from an `AbstractStorage` object for the evaluation of the linear operator.
        function step!(method::$name{X, MODE},
                          sys::System,
                            t::Real,
                           Δt::Real,
                            x::X,
                       store::AbstractStorage) where {X, MODE<:ContinuousMode}
            # hoist temporaries out
            y, z, w, u  = method.store
            tab = $tab

            # modifier for the location of the interpolation
            _m_ = isadjoint(MODE) == true ? -1 : 1

            # loop over stages
            @inbounds for k = 1:$NS
                if k == 1
                    y .= x
                else
                    y .= x .+ _m_.*(tab[:aᴵ, k, k-1] .- tab[:bᴵ, k-1]).*Δt.*z .+
                              _m_.*(tab[:aᴱ, k, k-1] .- tab[:bᴱ, k-1]).*Δt.*y
                end
                mul!(z, sys, y)                          # compute z = A*y then
                ImcA!(sys, _m_*tab[:aᴵ, k, k]*Δt, z, z)  # get z = (I-cA)⁻¹*(A*y) in place
                w .= y .+ _m_.*tab[:aᴵ, k, k].*Δt.*z     # w is the temp input, output is y
                if k > 1 && tab[:cᴱ, k-1] != tab[:cᴱ, k] # avoid double interpolation
                    store(u, t + tab[:cᴱ, k]*Δt)
                end
                sys(t + tab[:cᴱ, k]*Δt, u, w, y)
                x .= x .+ _m_.*tab[:bᴵ, k].*Δt.*z .+ _m_.*tab[:bᴱ, k].*Δt.*y
            end

            return nothing
        end


        # ---------------------------------------------------------------------------- #
        # Forward linearised method takes x_{n} and overwrites it with x_{n+1}
        function step!(method::$name{X, MODE},
                          sys::System,
                            t::Real,
                           Δt::Real,
                            x::X,
                       stages::NTuple{$NS, X}) where {X, MODE<:DiscreteMode{false}}
            # hoist temporaries out
            y, z, w, v, s, u  = method.store
            tab = $tab

            # loop over stages
            @inbounds for k = 1:$NS
                if k == 1
                    y .= x
                else
                    y .= x .+ (tab[:aᴵ, k, k-1] .- tab[:bᴵ, k-1]).*Δt.*z .+
                              (tab[:aᴱ, k, k-1] .- tab[:bᴱ, k-1]).*Δt.*y
                end
                mul!(z, sys, y)                     # compute z = A*y then
                ImcA!(sys, tab[:aᴵ, k, k]*Δt, z, z) # get z = (I-cA)⁻¹*(A*y) in place
                w .= y .+ tab[:aᴵ, k, k].*Δt.*z     # w is the temp input, output is y
                sys(t + tab[:cᴱ, k]*Δt, stages[k], w, y)
                x .= x .+ tab[:bᴵ, k].*Δt.*z .+ tab[:bᴱ, k].*Δt.*y
            end

            return nothing
        end


        # ---------------------------------------------------------------------------- #
        # Adjoint linearised method takes x_{n+1} and overwrites it with x_{n}
        function step!(method::$name{X, MODE},
                          sys::System,
                            t::Real,
                           Δt::Real,
                            x::X,
                       stages::NTuple{$NS, X}) where {X, MODE<:DiscreteMode{true}}

            # hoist temporaries out
            y, z, w, v, s, u  = method.store
            y .= 0; z .= 0; w .= 0; v .= 0; s .= 0; u .= 0;
            tab = $tab

            @inbounds for k = reverse(1:$NS)
                z .= z .+ tab[:bᴵ, k].*Δt.*x
                y .= y .+ tab[:bᴱ, k].*Δt.*x
                v .= w
                sys(t + tab[:cᴱ, k]*Δt, stages[k], y, v)
                w .= w .+ v
                y .= 0
                z .= z .+ tab[:aᴵ, k, k].*Δt.*w
                y .= y .+ w
                w .= 0
                ImcA!(sys, tab[:aᴵ, k, k]*Δt, z, v)
                s .= v .+ s
                z .= 0
                mul!(v, sys, s)
                y .= v .+ y
                s .= 0
                if k == 1
                    x .= x .+ y
                    y .= 0
                else
                    x .= y .+ x
                    z .= (tab[:aᴵ, k, k-1] .- tab[:bᴵ, k-1]).*Δt.*y .+ z
                    y .= (tab[:aᴱ, k, k-1] .- tab[:bᴱ, k-1]).*Δt.*y
                end
            end

            return nothing
        end
    end
end

@doc """
    CB3R2R2(x::X, mode::AbstractMode=NormalMode())

Constructs a `CB3R2R2` integration scheme object for integration with mode `mode`.
""" CB3R2R2

@doc """
    CB3R2R3e(x::X, mode::AbstractMode=NormalMode())

Constructs a `CB3R2R3e` integration scheme object for integration with mode `mode`.
""" CB3R2R3e

@doc """
    CB3R2R3c(x::X, mode::AbstractMode=NormalMode())

Constructs a `CB3R2R3c` integration scheme object for integration with mode `mode`.
""" CB3R2R3c