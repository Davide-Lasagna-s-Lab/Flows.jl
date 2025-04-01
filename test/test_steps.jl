import LinearAlgebra: Diagonal

@testset "linear system                          " begin
    # define linear system ẋ = x, but splitting the right hand side
    # into the explicit and implicit parts, to check both.
    g(t, x, ẋ) = (ẋ .= 0.5.*x; ẋ)
    L = Diagonal([0.5])

    # also define full explicit term for RK4
    gfull(t, x, ẋ) = (ẋ .= x; ẋ)

    # the type of the solution space
    x0 = Float64[1.0]

    #                                     method        -    ord  bounds
    for (scheme, order, bnd, _g, _A) in [(RK4(     x0), 4, (0.008300, 0.008700), gfull, nothing),
                                         (CB3R2R2( x0), 2, (0.025000, 0.027100), g,     L),
                                         (CNRK2(   x0), 2, (0.021000, 0.025500), g,     L),
                                         (CB3R2R3e(x0), 3, (0.006900, 0.008100), g,     L),
                                         (CB3R2R3c(x0), 3, (0.007300, 0.008600), g,     L)
                                        ]

        # ensure that the error decays with expected rate
        for Δt = [5^(-i) for i in range(1, stop=3, length=5)]

            # step forward
            x0 = Float64[1.0]
            Flows.step!(scheme, Flows.System(_g, _A), 0., Δt, x0, nothing)

            # check error decays with expected power. The bounds bnd are used
            # to check whether the error decays at the expected rate, up
            # to a small variation, within the bounds.
            err = abs(x0[1] - exp(Δt))
            # @show err/Δt^(order+1)
            @test err/Δt^(order + 1) > bnd[1]
            @test err/Δt^(order + 1) < bnd[2]
        end

        # test allocation
        function fun(sys, scheme, Δt, x0)
            @allocated Flows.step!(scheme, sys, 0.0, Δt, x0, nothing)
        end
        sys = Flows.System(g, L)
        # @code_warntype fun(sys, scheme, 0.1, x0)
        @test fun(sys, scheme, 0.1, x0) == 0
    end
end