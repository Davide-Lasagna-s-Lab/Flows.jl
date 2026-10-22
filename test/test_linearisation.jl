# TEST DISCRETE ADJOINT
@testset "discrete adjoint ($METHOD)" for (METHOD, NS) in ((RK4, 4), (RK8, 11))

    # initial conditions
    x0 = Float64[1, 1, 2]

    # methods
    nl    = METHOD(x0)
    l_t   = METHOD(x0, DiscreteMode(false))
    l_adj = METHOD(x0, DiscreteMode(true))

    # stage cache
    scache = RAMStageCache(NS, x0)

    # system (without diagonal)
    sys_nl    = Flows.System(Lorenz(0),    nothing)
    sys_l_tan = Flows.System(LorenzTan(0), nothing)
    sys_l_adj = Flows.System(LorenzAdj(0), nothing)

    # execute step
    N = 5
    for i = 1:N
        Flows.step!(nl, sys_nl,    0, 1e-2, x0, scache)
    end

    y0 = Float64[1, 2, 3]
    for i = 1:N
        Flows.step!(l_t, sys_l_tan, 0, 1e-2, y0, scache.xs[i])
    end

    q1 = Float64[4, 5, 7]
    for i = N:-1:1
        Flows.step!(l_adj, sys_l_adj, 0, 1e-2, q1, scache.xs[i])
    end

    a = dot(y0, [4, 5, 7])
    b = dot(q1, [1, 2, 3])
    @test abs(a-b)/abs(a) < 1e-14

    # a step of zero length leaves the adjoint unchanged
    q2 = Float64[4, 5, 7]
    Flows.step!(l_adj, sys_l_adj, 0, 0.0, q2, scache.xs[1])
    @test q2 == [4, 5, 7]

    # these take the same time
    # ta = @belapsed Flows.step!($l_adj, $sys_l_adj, 0, 1e-2, $q1, $(scache.xs[1]))
    # tb = @belapsed Flows.step!($l_t, $sys_l_tan, 0, 1e-2, $q1, $(scache.xs[1]))
    # println(ta/tb)
end

@testset "CB3R2R                                 " begin 
    # initial conditions
    x0 = Float64[15, 16, 20]

    for (nl, l_t, l_adj, NS) in [(CB3R2R2( x0),  CB3R2R2(x0, DiscreteMode(false)),  CB3R2R2(x0, DiscreteMode(true)),  3),
                                 (CB3R2R3e(x0), CB3R2R3e(x0, DiscreteMode(false)), CB3R2R3e(x0, DiscreteMode(true)), 4),
                                 (CB3R2R3c(x0), CB3R2R3c(x0, DiscreteMode(false)), CB3R2R3c(x0, DiscreteMode(true)), 4)]

        # stage cache
        scache = RAMStageCache(NS, x0)

        # system
        sys_nl    = Flows.System(Lorenz(1),    A)
        sys_l_tan = Flows.System(LorenzTan(1), A)
        sys_l_adj = Flows.System(LorenzAdj(1), A)

        # execute step
        N = 50
        for i = 1:N
            Flows.step!(nl, sys_nl,    0, 1e-2, x0, scache)
        end

        y0 = Float64[1, 2, 3]
        for i = 1:N
            Flows.step!(l_t, sys_l_tan, 0, 1e-2, y0, scache.xs[i])
        end

        q1 = Float64[4, 5, 7]
        for i = N:-1:1
            Flows.step!(l_adj, sys_l_adj, 0, 1e-2, q1, scache.xs[i])
        end

        a = dot(y0, [4, 5, 7])
        b = dot(q1, [1, 2, 3])
        @test abs(a-b)/abs(a) < 1e-14

        # the adjoint code is ~30% slower here
        # ta = @belapsed Flows.step!($l_adj, $sys_l_adj, 0, 1e-2, $q1, $(scache.xs[1]))
        # tb = @belapsed Flows.step!($l_t, $sys_l_tan, 0, 1e-2, $q1, $(scache.xs[1]))
        # println(ta/tb)
    end
end

@testset "CNRK2                                  " begin 

    # initial conditions
    x0 = Float64[1, 1, 2]

    # methods
    nl    = CNRK2(x0)
    l_t   = CNRK2(x0, DiscreteMode(false))
    l_adj = CNRK2(x0, DiscreteMode(true))

    # stage cache
    scache = RAMStageCache(2, x0)

    # system (without diagonal)
    sys_nl    = Flows.System(Lorenz(1),    A)
    sys_l_tan = Flows.System(LorenzTan(1), A)
    sys_l_adj = Flows.System(LorenzAdj(1), A)

    # execute step
    N = 5
    for i = 1:N
        Flows.step!(nl, sys_nl,    0, 1e-2, x0, scache)
    end

    y0 = Float64[1, 2, 3]
    for i = 1:N
        Flows.step!(l_t, sys_l_tan, 0, 1e-2, y0, scache.xs[i])
    end

    q1 = Float64[4, 5, 7]
    for i = N:-1:1
        Flows.step!(l_adj, sys_l_adj, 0, 1e-2, q1, scache.xs[i])
    end

    a = dot(y0, [4, 5, 7])
    b = dot(q1, [1, 2, 3])
    @test abs(a-b)/abs(a) < 1e-14

    # these take the same time
    # ta = @belapsed Flows.step!($l_adj, $sys_l_adj, 0, 1e-2, $q1, $(scache.xs[1]))
    # tb = @belapsed Flows.step!($l_t, $sys_l_tan, 0, 1e-2, $q1, $(scache.xs[1]))
    # println(ta/tb)
end

# ---------------------------------------------------------------------------- #
# TEST LINEARISED STEP IS REALLY THE LINEARISATION OF THE NONLINEAR STEP
@testset "Complex step derivative                " begin

    # initial conditions using complex numbers
    x0 = zeros(ComplexF64, 3)

    # complex step
    ϵ = 1e-100

    for (nl, l_t, NS, _g_nl, _g_t, _A) in [(     RK4(x0),      RK4(real.(x0), DiscreteMode(false)), 4, Lorenz(0), LorenzTan(0), nothing),
                                           (   CNRK2(x0),    CNRK2(real.(x0), DiscreteMode(false)), 2, Lorenz(1), LorenzTan(1), A),
                                           ( CB3R2R2(x0),  CB3R2R2(real.(x0), DiscreteMode(false)), 3, Lorenz(1), LorenzTan(1), A),
                                           (CB3R2R3e(x0), CB3R2R3e(real.(x0), DiscreteMode(false)), 4, Lorenz(1), LorenzTan(1), A),
                                           (CB3R2R3c(x0), CB3R2R3c(real.(x0), DiscreteMode(false)), 4, Lorenz(1), LorenzTan(1), A)]

        for i = 1:3
            x0 = [9.1419853, 1.648665, 35.21793] + im*[0.0, 0.0, 0.0]

            # stage cache
            scache = RAMStageCache(NS, x0)

            # system (without diagonal)
            sys_nl    = Flows.System(_g_nl, _A)
            sys_l_tan = Flows.System(_g_t,  _A)

            # go to attractor
            for j = 1:100
                Flows.step!(nl, sys_nl, 0, 1e-2, x0, nothing)
            end

            # reset perturbation
            x0 .= real.(x0)
            x0[i] += ϵ*im

            # number of steps
            N = 1000

            for j = 1:N
                Flows.step!(nl, sys_nl, 0, 1e-2, x0, scache)
            end

            # perturb z component only
            y0 = Float64[0, 0, 0]
            y0[i] += 1
            for j = 1:N
                Flows.step!(l_t, sys_l_tan, 0, 1e-2, y0, real.(scache.xs[j]))
            end

            @test norm(imag.(x0)./ϵ - y0)/norm(y0) < 5e-14
        end
    end
end

# ---------------------------------------------------------------------------- #
# TEST LINEARISED EQUATIONS API
@testset "linear api                             " begin

    # initial conditions
    x0 = Float64[9.1419853, 1.648665, 35.21793]

    for (nl, l_t, l_a, NS, _g_nl, _g_t, _g_a, _A) in [(     RK4(x0),      RK4(x0, DiscreteMode(false)),      RK4(x0, DiscreteMode(true)), 4, Lorenz(0), LorenzTan(0), LorenzAdj(0), nothing),
                                                      (   CNRK2(x0),    CNRK2(x0, DiscreteMode(false)),    CNRK2(x0, DiscreteMode(true)), 2, Lorenz(1), LorenzTan(1), LorenzAdj(1), A),
                                                      ( CB3R2R2(x0),  CB3R2R2(x0, DiscreteMode(false)),  CB3R2R2(x0, DiscreteMode(true)), 3, Lorenz(1), LorenzTan(1), LorenzAdj(1), A),
                                                      (CB3R2R3e(x0), CB3R2R3e(x0, DiscreteMode(false)), CB3R2R3e(x0, DiscreteMode(true)), 4, Lorenz(1), LorenzTan(1), LorenzAdj(1), A),
                                                      (CB3R2R3c(x0), CB3R2R3c(x0, DiscreteMode(false)), CB3R2R3c(x0, DiscreteMode(true)), 4, Lorenz(1), LorenzTan(1), LorenzAdj(1), A)]


        # stage cache
        scache = RAMStageCache(NS, x0)

        # non linear flow map
        ϕ = flow(_g_nl, _A, nl, TimeStepConstant(1e-2))

        # linearised propagator and adjoint
        ψ  = flow(_g_t, _A, l_t, TimeStepFromCache())
        ψ⁺ = flow(_g_a, _A, l_a, TimeStepFromCache())

        # propagate nonlinear operator
        ϕ(copy(x0), (0, 5), reset!(scache))

        # also make sure monitor are good
        mon_tan = Monitor(x0, (t, x)->copy(x))
        mon_adj = Monitor(x0, (t, x)->copy(x))

        # propagate linear operators forward/backward
        y0 = Float64[1, 2, 3]
        ψ(y0, scache, mon_tan)

        qT = Float64[4, 5, 6]
        ψ⁺(qT, scache, mon_adj)

        # test 
        a = dot(y0, [4, 5, 6])
        b = dot(qT, [1, 2, 3])
        @test abs(a - b)/a < 2e-14

        # test monitors
        @test times(mon_tan)[1]   == 0
        @test times(mon_tan)[end] == 5
        @test times(mon_adj)[1]   == 5
        @test times(mon_adj)[end] == 0
    end
end

# ---------------------------------------------------------------------------- #
# COUPLED INTEGRATION OF THE GOVERNING EQUATIONS PLUS LINEARISED EQUATIONS
@testset "coupled integration                    " begin
    # initial conditions
    x0 = Float64[9.1419853, 1.648665, 35.21793]

    # INTEGRATE USING COUPLED INTEGRATION
    # explicit integrator
    method = RK4(couple(x0, x0))

    # linear flow map note we first pass the nonlinear equations
    ψ = flow(couple(Lorenz(0), LorenzTan(0)), method, TimeStepConstant(1e-2))

    # initial condition for the linearised equations
    y0 = Float64[1.0, 0.0, 0.0]

    # propagate
    ψ(couple(copy(x0), y0), (0, 10))

    # get norm
    n_coupled = norm(y0)

    # INTEGRATE USING CACHE
    # stage cache
    scache = RAMStageCache(4, x0)

    # non linear flow map
    ϕ = flow(Lorenz(0), RK4(x0), TimeStepConstant(1e-2))

    # linearised propagator and adjoint
    ψ  = flow(LorenzTan(0), RK4(x0, DiscreteMode(false)), TimeStepFromCache())

    # propagate nonlinear operator
    ϕ(copy(x0), (0, 10), reset!(scache))

    # propagate linear operators forward/backward
    y0 = Float64[1.0, 0.0, 0.0]
    ψ(y0, scache)
    
    # get norm
    n_cache = norm(y0)

    # the two methods should really be the same
    @test n_coupled == n_cache
end


# ---------------------------------------------------------------------------- #
# COUPLED INTEGRATION OF THE GOVERNING EQUATIONS PLUS LINEARISED EQUATIONS

# example quadrature function
function quadfun(t, x, dxdt, y, dydt, q, dqdt)
    dqdt[1] = x[1]*y[2]
    return dqdt
end

@testset "coupled integration                    " begin
    # initial conditions
    x0 = Float64[9.1419853, 1.648665, 35.21793]

    # initial condition for the linearised equations
    y0 = Float64[1.0, 0.0, 0.0]

    # initial condition for the quadrature
    q0 = Float64[0.0]

    # explicit integrator
    method = RK4(couple(x0, y0, q0))

    # linear flow map note we first pass the nonlinear equations
    ψ = flow(couple(Lorenz(0), LorenzTan(0), quadfun),
             method,
             TimeStepConstant(1e-3))

    # propagate
    ψ(couple(copy(x0), y0, q0), (0, 1))

    # get norm
    val_a = q0[1]

    # now compute the same by storing the entire solution
    monϕ = Monitor(x0, (t, x)->x[1])
    ϕ = flow(Lorenz(0), RK4(x0), TimeStepConstant(1e-3))

    # stage cache
    scache = RAMStageCache(4, x0)

    # linearised propagator and adjoint
    monψ = Monitor(x0, (t, x)->x[2])
    ψ  = flow(LorenzTan(0), RK4(x0, DiscreteMode(false)), TimeStepFromCache())

    # propagate nonlinear operator and fill cache
    ϕ(copy(x0), (0, 1), reset!(scache))

    # then fill monitor
    ϕ(copy(x0), (0, 1), reset!(monϕ))

    # propagate linear operators forward/backward
    y0 = Float64[1.0, 0.0, 0.0]
    ψ(y0, scache, reset!(monψ))

    # naive rectangle integration
    val_b = sum(samples(monψ).*samples(monϕ)) * 1e-3

    @test abs(val_a - val_b)/abs(val_a) < 1e-4
end