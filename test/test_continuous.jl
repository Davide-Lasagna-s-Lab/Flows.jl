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
