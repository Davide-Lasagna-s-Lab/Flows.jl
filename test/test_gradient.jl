@testset "tests continuous RK4                   " begin
    T = 1
    flag = 0
        
    for dt = [1e-2, 1e-3, 1e-4]
        x0 = Float64[9.1419853, 1.648665, 35.21793]
        y0 = zeros(3)
        w0 = zeros(3)

        # finite difference
        method = RK4(x0)
        alpha = 1e-4
        ϕp = flow(Lorenz(flag, +alpha), method, TimeStepConstant(dt))
        ϕm = flow(Lorenz(flag, -alpha), method, TimeStepConstant(dt))

        mon = Monitor(x0, (t, x)->x[3])
        ϕp(copy(x0), (0, T), reset!(mon))
        vp = simps(times(mon), samples(mon))
        ϕm(copy(x0), (0, T), reset!(mon))
        vm = simps(times(mon), samples(mon))
        Jp_FD = (vp-vm)/2/alpha

        # tangent
        # fill storage first
        ϕ = flow(Lorenz(flag, +alpha), method, TimeStepConstant(dt))
        storage = RAMStorage(x0)
        ϕ(copy(x0), (0, T), storage)

        # define linear propagator and monitor
        ψ = flow(LorenzTan(flag, 1), RK4(y0, ContinuousMode(false)), TimeStepFromStorage(dt))
        mon = Monitor(y0, (t, x)->x[3])
        ψ(copy(y0), storage, (0, T), reset!(mon))
        Jp_TAN = simps(times(mon), samples(mon))
        # println(times(mon))

        # adjoint
        ψ_A = flow(LorenzAdj(flag, 1), RK4(w0, ContinuousMode(true)), TimeStepFromStorage(dt))
        mon = Monitor(w0, (t, x)->x[1])
        ψ_A(copy(w0), storage, (T, 0), reset!(mon))
        Jp_ADJ = simps(reverse(times(mon)), reverse(samples(mon)))

        # we check that the difference between the gradient obtained
        # from the adjoint and the tangent code decays with the order
        # of integration of the method. The simpson rule is fourth
        # order accurate and thus the leading error term will always
        # be the integration method.
        @test abs(Jp_ADJ - Jp_TAN)/abs(Jp_TAN)/dt^4 < 2006
        # println(abs(Jp_ADJ - Jp_TAN)/abs(Jp_TAN)/dt^4)

        # the error between linearised and finite difference decays
        # println(abs(Jp_TAN - Jp_FD)/abs(Jp_TAN)/alpha)
        @test (abs(Jp_TAN - Jp_FD)/abs(Jp_TAN)/alpha) < 0.47
        @test (abs(Jp_TAN - Jp_FD)/abs(Jp_TAN)/alpha) > 0.34
    end
end

@testset "test continuous imex                   " begin
    T = 1
    flag = 1
    IMPL = flag*A

    # FIXME: sometimes these tests fail with NaN values being output
    for (METHOD, bnd, (bnd_low, bnd_upp), order) in [(CB3R2R3c, 1170, (0.34,  8.00), 3),
                                                     (CB3R2R3e,  275, (0.34,  0.55), 3),
                                                     (CB3R2R2,    44, (0.34, 21.50), 2),
                                                     (CNRK2,     130, (0.34, 52.00), 2)]

        for dt = [1e-2, 1e-3, 1e-4]
            x0 = Float64[9.1419853, 1.648665, 35.21793]
            y0 = zeros(3)
            w0 = zeros(3)

            # finite difference
            method = METHOD(x0)
            alpha = 1e-4
            ϕp = flow(Lorenz(flag, +alpha), IMPL, method, TimeStepConstant(dt))
            ϕm = flow(Lorenz(flag, -alpha), IMPL, method, TimeStepConstant(dt))

            mon = Monitor(x0, (t, x)->x[3])
            ϕp(copy(x0), (0, T), reset!(mon))
            vp = simps(times(mon), samples(mon))
            ϕm(copy(x0), (0, T), reset!(mon))
            vm = simps(times(mon), samples(mon))
            Jp_FD = (vp-vm)/2/alpha

            # tangent
            # fill storage first
            ϕ = flow(Lorenz(flag, +alpha), IMPL, method, TimeStepConstant(dt))
            storage = RAMStorage(x0)
            ϕ(copy(x0), (0, T), storage)

            # define linear propagator and monitor
            ψ = flow(LorenzTan(flag, 1), IMPL, METHOD(y0, ContinuousMode()), TimeStepFromStorage(dt))
            mon = Monitor(y0, (t, x)->x[3])
            ψ(copy(y0), storage, (0, T), reset!(mon))
            Jp_TAN = simps(times(mon), samples(mon))

            # adjoint
            ψ_A = flow(LorenzAdj(flag, 1), IMPL, METHOD(w0, ContinuousMode(true)), TimeStepFromStorage(dt))
            mon = Monitor(w0, (t, x)->x[1])
            ψ_A(copy(w0), storage, (T, 0), reset!(mon))
            Jp_ADJ = simps(reverse(times(mon)), reverse(samples(mon)))

            # we check that the difference between the gradient obtained
            # from the adjoint and the tangent code decays with the order
            # of integration of the method. The simpson rule is fourth
            # order accurate and thus the leading error term will always
            # be the integration method.
            # println(abs(Jp_ADJ - Jp_TAN)/abs(Jp_TAN)/dt^order)
            @test abs(Jp_ADJ - Jp_TAN)/abs(Jp_TAN)/dt^order < bnd

            # the error between linearised and finite difference decays
            # but approaches a constant which is the truncation involved
            # in the finite difference
            # println(abs(Jp_TAN - Jp_FD)/abs(Jp_TAN)/alpha)
            @test (abs(Jp_TAN - Jp_FD)/abs(Jp_TAN)/alpha) < bnd_upp
            @test (abs(Jp_TAN - Jp_FD)/abs(Jp_TAN)/alpha) > bnd_low
        end
    end
end

# @testset "test discrete CNRK2                    " begin
#     T = 1
#     flag = 0
#     IMPL = flag*B
    
#     for dt = [1e-2, 1e-3, 1e-4]
#         x0 = Float64[9.1419853, 1.648665, 35.21793]
#         y0 = zeros(3)
#         w0 = zeros(3)

#         METHOD = CNRK2

#         # finite difference
#         method = METHOD(x0)
#         alpha = 1e-6
#         ϕp = flow(Lorenz(flag, +alpha), IMPL, method, TimeStepConstant(dt))
#         ϕm = flow(Lorenz(flag, -alpha), IMPL, method, TimeStepConstant(dt))

#         mon = Monitor(x0, x->x[3])
#         ϕp(copy(x0), (0, T), reset!(mon))
#         vp = simps(times(mon), samples(mon))
#         ϕm(copy(x0), (0, T), reset!(mon))
#         vm = simps(times(mon), samples(mon))
#         Jp_FD = (vp-vm)/2/alpha

#         # tangent
#         method = METHOD(couple(x0, y0))
#         ψ = flow(couple(Lorenz(flag, 0), LorenzTan(flag)), couple(IMPL, IMPL), method, TimeStepConstant(dt))
#         mon = Monitor(couple(x0, y0), x->x[2][3])
#         ψ(couple(copy(x0), y0), (0, T), reset!(mon))
#         Jp_TAN = simps(times(mon), samples(mon))
#         # println(times(mon))

#         # adjoint
#         method = METHOD(x0)
#         ϕ = flow(Lorenz(flag, 0), IMPL, method, TimeStepConstant(dt))
#         cache = RAMStageCache(2, similar(x0))
#         ϕ(copy(x0), (0, T), cache)
        
#         method = METHOD(w0, true)
#         ψ_A = flow(LorenzAdj(flag), IMPL, method, TimeStepFromCache())
#         mon = Monitor(w0, x->x[1])
#         ψ_A(w0, cache, reset!(mon))
#         Jp_ADJ = -simps(times(mon), samples(mon))

#         println(Jp_FD)
#         println(Jp_TAN)
#         println(Jp_ADJ) 
#         # println( abs(Jp_ADJ - Jp_TAN)/abs(Jp_TAN)/dt^2 )
#         println()
#     end
# end
