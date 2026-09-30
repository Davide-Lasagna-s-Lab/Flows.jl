@testset "coupled state broadcast                " begin
    x = [1,   2,   3]
    q = [3.0, 4.0, 5.0]

    # define two coupled states
    z = Flows.couple(x, q)
    y = Flows.couple(copy(x).+1, copy(q).+1)

    # define some operation
    fun!(z, y) = (z .= 2 .*z .+ 1 .*y; z)

    # apply
    fun!(z, y)

    # value
    @test z[1] == [4, 7, 10]
    @test z[2] == [10, 13, 16]

    # allocation
    @test (@allocated fun!(z, y)) == 0
end

# ----------------------------------------------------------------------------
# Standalone quadrature rules — trapz and simps applied directly to
# (xs, ys) vectors, independently of the flow-based pipeline below.
# ----------------------------------------------------------------------------

@testset "trapz on a uniform grid                " begin
    # ∫₀¹ x dx = 1/2
    xs = collect(range(0.0, stop=1.0, length=11))
    ys = xs
    @test trapz(xs, ys) ≈ 0.5

    # ∫₀¹ x² dx = 1/3, but trapezoidal rule has finite error on a coarse grid
    xs = collect(range(0.0, stop=1.0, length=1001))
    ys = xs.^2
    @test isapprox(trapz(xs, ys), 1/3; atol=1e-6)
end

@testset "trapz on a non-uniform grid            " begin
    # exact for linear integrands regardless of spacing
    xs = [0.0, 0.1, 0.4, 0.7, 1.0]
    ys = 2 .* xs .+ 1                       # ∫₀¹ (2x + 1) dx = 2
    @test trapz(xs, ys) ≈ 2.0
end

@testset "trapz mismatched lengths errors        " begin
    @test_throws ArgumentError trapz([1.0, 2.0], [1.0])
end

@testset "simps on a uniform grid                " begin
    # Simpson is exact for cubic polynomials on a uniform grid
    xs = collect(range(0.0, stop=1.0, length=101))
    ys = xs.^3                              # ∫₀¹ x³ dx = 1/4
    @test simps(xs, ys) ≈ 1/4 atol=1e-12

    # smooth non-polynomial integrand: tight tolerance with enough points
    xs = collect(range(0.0, stop=π, length=1001))
    ys = sin.(xs)                           # ∫₀^π sin x dx = 2
    @test isapprox(simps(xs, ys), 2.0; atol=1e-8)

    # last-interval correction kicks in for an odd number of intervals
    xs = collect(range(0.0, stop=1.0, length=12))  # 11 intervals
    ys = xs.^2
    @test isapprox(simps(xs, ys), 1/3; atol=1e-3)
end

@testset "quadrature                             " begin

    # define linear system
    g(t, x, ẋ) = (ẋ .= 0.5.*x; ẋ)
    A = Diagonal([0.5])

    # also define full explicit term for RK4
    gfull(t, x, ẋ) = (ẋ .= x; ẋ)

    # define example quadrature functions
    @inline function quad(t, x, dxdt, q, dqdt)
        dqdt[1] = 1
        dqdt[2] = x[1]
        dqdt[3] = t
        return dqdt
    end

    # state and quadrature
    x, q = Float64[0.0], Float64[0.0, 0.0, 0.0]

    # monitor first quadrature equation
    mon = Monitor(couple(x, q), (t, x)->copy(x))

    # integration method
    for (method, order, value, _g, _A) in [(RK4(     couple(x, q)), 4, 6.2,  gfull, nothing),
                                           (CB3R2R2( couple(x, q)), 2, 19 ,  g,     A),
                                           (CB3R2R3e(couple(x, q)), 3, 5.5,  g,     A),
                                           (CB3R2R3c(couple(x, q)), 3, 5.8,  g,     A),
                                           (CB4R3R4( couple(x, q)), 4, 0.16, g,    A)]

        # exact values of integral
        exact = [5, exp(5) - exp(0), 5^2/2]

        # error should decrease at certain rate
        for Δt = 10 .^ range(0, stop=-2.5, length=5)

            # forward map
            f = flow(couple(_g, quad), couple(_A, nothing), method, TimeStepConstant(Δt))

            # initial conditions
            x₀ = Float64[1.0]
            q₀ = Float64[0.0, 0.0, 0.0]

            # call
            f(couple(x₀, q₀), (0, 5), mon)

            # test 
            @test eltype(samples(mon)) == Coupled{2, Tuple{Vector{Float64}, Vector{Float64}}}

            # integrals
            @test norm(abs.(q₀ - exact)) / Δt^order < value

            # test allocations
            fun(f, xq, span) = @allocated f(xq, span)
            @test fun(f, couple(x₀, q₀), (0.0, 1000.0)) <= 32
        end
    end
end