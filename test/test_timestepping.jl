using Flows

# example time stepping from hook
struct TestFromHook <: AbstractTimeStepFromHook end
(::TestFromHook)(g, A, z) = 0.1*sqrt(z[1])

# system
g(t, x, ẋ) = (ẋ .= -1; ẋ)

@testset "from hook ($METHOD)" for METHOD in (RK4, RK8)
    # integration scheme
    scheme = METHOD(zeros(1))

    # forward map
    ϕ = flow(g, scheme, TestFromHook())

    # monitor
    mon = Monitor(zeros(1), (t, x)->x[1])

    # FORWARD INTEGRATION
    # run
    ϕ([1.0], (0, 1), reset!(mon))

    # test we hit the end point
    @test times(mon)[1]   == 0
    @test times(mon)[end] == 1

    # and that the solution is OK
    @test abs(samples(mon)[end]) < 2e-16

    # backward integration fails
    @test_throws InvalidSpanError ϕ([1.0], (1, 0), reset!(mon))
end

@testset "constant time step ($METHOD)" for METHOD in (RK4, RK8)
    # integration scheme
    scheme = METHOD(zeros(1))

    # forward map
    ϕ = flow(g, scheme, TimeStepConstant(0.5))

    # monitor
    mon = Monitor(zeros(1), (t, x)->x[1])

    # FORWARD INTEGRATION
    # run
    ϕ([1.0], (0, 1), reset!(mon))

    # test we hit the end point
    @test times(mon)[1] == 0.0
    @test times(mon)[2] == 0.5
    @test times(mon)[3] == 1.0

    # and that the solution is OK
    @test abs(samples(mon)[end]) < 2e-16

    # backward integration fails
    @test_throws InvalidSpanError ϕ([1.0], (1, 0), reset!(mon))
end
