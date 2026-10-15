# Discrete adjoint runs with a forcing, from a stage cache. The objective is J = ∫₀ᵀ z dt + z(T) on
# the Lorenz equations, integrated with the scheme itself as a coupled quadrature, so that the adjoint
# with forcing e_z and q(T) = e_z gives its exact gradient q(0) = ∂J/∂x(0), up to the error of the
# central finite differences that check it. A forcing that does not get the weights of the quadrature
# of the scheme shows up here, and not in the dot-product tests, which have no forcing.

# the objective, integrated by the scheme: dJ/dt = z
zquad(t, x, dxdt, q, dqdt) = (dqdt[1] = x[3]; dqdt)

@testset "forced discrete adjoint                " begin

    # ---- the initial condition, near the attractor, the span and the step ----
    x0 = [9.1419853, 1.648665, 35.21793]
    T  = 2.0
    Δt = 1e-2

    for (METHOD, NS, flag, Aop) in ((RK4,      4, 0, nothing),
                                    (CNRK2,    2, 1, A),
                                    (CB3R2R2,  3, 1, A),
                                    (CB3R2R3e, 4, 1, A),
                                    (CB3R2R3c, 4, 1, A))

        # ---- the objective J(x(0)), with the quadrature coupled to the state ----
        ϕJ = flow(couple(Lorenz(flag), zquad),
                  couple(Aop, nothing),
                  METHOD(couple(copy(x0), [0.0])),
                  TimeStepConstant(Δt))
        J(x) = (z = ϕJ(couple(copy(x), [0.0]), (0, T)); z[2][1] + z[1][3])

        # ---- its gradient by central finite differences ----
        ε  = 1e-5
        ∇J = [(J(x0 + ε*e) - J(x0 - ε*e))/(2ε) for e in ([1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0])]

        # ---- the forward run into the stage cache, then the forced adjoint back from q(T) = e_z ----
        cache = RAMStageCache(NS, x0)
        flow(Lorenz(flag), Aop, METHOD(x0), TimeStepConstant(Δt))(copy(x0), (0, T), cache)
        q = [0.0, 0.0, 1.0]
        flow(LorenzAdj(flag, 1.0), Aop, METHOD(x0, DiscreteMode(true)), TimeStepFromCache())(q, cache)

        @test norm(q - ∇J)/norm(∇J) < 1e-7
    end
end
