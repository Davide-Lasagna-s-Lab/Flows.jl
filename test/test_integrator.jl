@testset "integrator                             " begin
    # make system
    g(t, x, ẋ) = (ẋ .= .-0.5.*x; ẋ)
    D = Diagonal([-0.5])

    # also define full explicit term for RK4
    gfull(t, x, ẋ) = (ẋ .= .-x; ẋ)

    # try on standard vector and on custom type
    for x in [Float64[1.0]]

        # integration scheme
        for (method, err, _g, _D) in [(RK4(     x), 1e-9, gfull, nothing),
                                      (CNRK2(   x), 2e-5, g,     D),
                                      (CB3R2R2( x), 2e-5, g,     D),
                                      (CB3R2R3e(x), 5e-8, g,     D),
                                      (CB3R2R3c(x), 6e-8, g,     D),
                                      (CB4R3R4( x), 3e-12,g,     D)]

            # forward map
            ϕ = flow(_g, _D, method, TimeStepConstant(0.01123))

            # check relative error, for a few repetitions of the integration
            @test abs(ϕ(copy(x), (0, 1))[1] - exp(-1))/exp(-1) < err
            @test abs(ϕ(copy(x), (0, 2))[1] - exp(-2))/exp(-2) < err
            @test abs(ϕ(copy(x), (0, 3))[1] - exp(-3))/exp(-3) < err
            @test abs(ϕ(copy(x), (0, 4))[1] - exp(-4))/exp(-4) < err
            @test abs(ϕ(copy(x), (0, 5))[1] - exp(-5))/exp(-5) < err

            # # try giving a different input
            @test_throws MethodError ϕ([1], (0, 1))

            # test allocation
            fun(ϕ, x₀, span) = @allocated ϕ(x₀, span)
            @test fun(ϕ, copy(x), (0, 100)) == 0
            @test fun(ϕ, copy(x), (0.0, 100.0)) == 0
            @test fun(ϕ, copy(x), (0, 100.0)) == 0
        end
    end
end

@testset "integrator with shift                  " begin
    # make system
    g(t, x, ẋ) = (ẋ .= .-x; ẋ)

    # try on standard vector
    x = Float64[1.0]

    # forward map
    ϕ = flow(g, RK4(x), TimeStepConstant(0.01123), (x, s)->x[1] += s)

    # check relative error, for a few repetitions of the integration
    @test abs(ϕ(copy(x), (0, 1), -exp(-1)))/exp(-1) < 1e-9
end
