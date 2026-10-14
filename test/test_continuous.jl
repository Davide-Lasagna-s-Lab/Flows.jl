# Tangent and adjoint runs in continuous mode, on coupled vectors, along an orbit held in a
# storage. The orbit is u(t) = (cos t, sin t) and every vector obeys y' = -y + cos(t) y, with the
# term -y implicit, so that y(t₁) = exp(-(t₁ - t₀) + sin t₁ - sin t₀) y(t₀) componentwise. The
# buffers of the methods are filled with NaN first, so that a stage that reads a buffer before
# writing it shows up as NaN. The spans start at t₀ = 1, away from the start of the orbit.

@testset "continuous mode, coupled vectors       " begin

    # ---- the orbit, one period in a periodic storage ----
    T     = 2π
    n     = 400
    store = RAMStorage(zeros(2); period=T, storelast=false, degree=5)
    rot   = flow((t, x, dxdt) -> (dxdt[1] = -x[2]; dxdt[2] = x[1]; dxdt),
                 RK4(zeros(2)),
                 TimeStepConstant(T / n))
    rot([1.0, 0.0], (0, T), store)

    # ---- the linear equation, the explicit part along the orbit ----
    Lex(t, u, y, dydt) = (dydt .= u[1] .* y; dydt)
    Aim = Diagonal([-1.0, -1.0])

    # ---- the span and the exact growth ----
    t₀ = 1.0
    t₁ = 1.0 + 2.3
    g  = exp(-(t₁ - t₀) + sin(t₁) - sin(t₀))

    for (METHOD, order) in ((CNRK2,    2),
                            (CB3R2R2,  2),
                            (CB3R2R3e, 3),
                            (CB3R2R3c, 3),
                            (CB4R3R4,  4))
        for adjoint in (false, true)

            # ---- the flow, with its buffers poisoned ----
            method = METHOD(couple(zeros(2), zeros(2)), ContinuousMode(adjoint))
            for b in method.store, k in 1:2
                b[k] .= NaN
            end
            ψ = flow(couple(Lex, Lex),
                     couple(Aim, Aim),
                     CallDependency((1,), (2,)),
                     method,
                     TimeStepFromStorage(0.01))

            # ---- forward from t₀ to t₁, or backward from t₁ to t₀ ----
            y = couple([1.0, 2.0], [-3.0, 0.5])
            span = adjoint ? (t₁, t₀) : (t₀, t₁)
            ψ(y, store, span)

            # ---- every component of every vector grows by g ----
            @test all(isfinite, y[1])
            @test all(isfinite, y[2])
            @test y[1] ≈ g .* [1.0, 2.0]  rtol = 10.0^(-order - 1)
            @test y[2] ≈ g .* [-3.0, 0.5] rtol = 10.0^(-order - 1)
        end
    end
end

# A coupled system of one component used to be ambiguous between the methods
# for single-component and for coupled systems. A single tangent vector along
# the same orbit, and the forward problem y' = -y + cos(t) y of one component.
@testset "coupled system of one component        " begin

    # ---- the orbit ----
    T     = 2π
    store = RAMStorage(zeros(2); period=T, storelast=false, degree=5)
    rot   = flow((t, x, dxdt) -> (dxdt[1] = -x[2]; dxdt[2] = x[1]; dxdt),
                 RK4(zeros(2)),
                 TimeStepConstant(T / 400))
    rot([1.0, 0.0], (0, T), store)

    # ---- the equations, tangent and forward ----
    Lex(t, u, y, dydt) = (dydt .= u[1] .* y; dydt)
    fex(t, y, dydt)    = (dydt .= cos(t) .* y; dydt)
    Aim = Diagonal([-1.0, -1.0])

    t₀ = 1.0
    t₁ = 3.3
    g  = exp(-(t₁ - t₀) + sin(t₁) - sin(t₀))

    for METHOD in (CNRK2, CB3R2R2, CB3R2R3e, CB3R2R3c, CB4R3R4)

        # ---- tangent, along the stored orbit ----
        ψ = flow(couple(Lex), couple(Aim), CallDependency((1,)),
                 METHOD(couple(zeros(2)), ContinuousMode(false)),
                 TimeStepFromStorage(0.01))
        y = couple([1.0, 2.0])
        ψ(y, store, (t₀, t₁))
        @test y[1] ≈ g .* [1.0, 2.0] rtol = 1e-3

        # ---- forward ----
        ϕ = flow(couple(fex), couple(Aim), METHOD(couple(zeros(2))), TimeStepConstant(0.01))
        y = couple([1.0, 2.0])
        ϕ(y, (t₀, t₁))
        @test y[1] ≈ g .* [1.0, 2.0] rtol = 1e-3
    end

    # ---- an explicit scheme ----
    ϕ = flow(couple((t, y, dydt) -> (dydt .= (cos(t) - 1) .* y; dydt)), RK4(couple(zeros(2))), TimeStepConstant(0.01))
    y = couple([1.0, 2.0])
    ϕ(y, (t₀, t₁))
    @test y[1] ≈ g .* [1.0, 2.0] rtol = 1e-8
end

# An IMEX scheme on a single-component system without a linear part, A = nothing.
@testset "IMEX scheme without linear part        " begin
    for METHOD in (CNRK2, CB3R2R3e, CB4R3R4)
        ϕ = flow((t, y, dydt) -> (dydt .= .-y; dydt), nothing, METHOD(zeros(1)), TimeStepConstant(0.01))
        y = [1.0]
        ϕ(y, (0, 1))
        @test y[1] ≈ exp(-1) rtol = 1e-4
    end
end
