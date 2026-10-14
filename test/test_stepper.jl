@testset "lossy                                  " begin
    rng = Flows.Steps(0, 1, 0.4)

    expected = [(0.0, 0.4), (0.4, 0.4), (0.8, 0.2)]
    for (i, (t, dt)) in enumerate(rng)
        @test  t ≈ expected[i][1]
        @test dt ≈ expected[i][2]
    end

    # test interface used in the integration routines
    for rep = 1:100
        T  = 100 + rand()
        dt =       rand()
        rng = Flows.Steps(0, T, dt)
        for (i, (t, dt)) in enumerate(rng)
            @test t + dt ≤ T
        end
    end
end

@testset "lossless                               " begin
    rng = Flows.Steps(0, 0.8, 0.4)

    expected = [(0.0, 0.4), (0.4, 0.4)]
    for (i, (t, dt)) in enumerate(rng)
        @test  t ≈ expected[i][1]
        @test dt ≈ expected[i][2]
    end

    # test interface used in the integration routines
    T  = 10.0
    dt = 1e-3
    rng = Flows.Steps(0, T, dt)
    for (i, (t, dt)) in enumerate(rng)
        @test t + dt ≤ T
    end
end

@testset "backwards checks                       " begin
    @test_throws ArgumentError Flows.Steps(1, 0, -0.1)
end

@testset "backwards lossy                        " begin
    rng = Flows.Steps(1, 0, 0.4)
    expected = [(1.0, -0.4), (0.6, -0.4), (0.2, -0.2)]
    for (i, (t, dt)) in enumerate(rng)
        @test  t ≈ expected[i][1]
        @test dt ≈ expected[i][2]
    end

    # test interface used in the integration routines
    for rep = 1:100
        T  = 100 + rand()
        dt =       rand()
        rng = Flows.Steps(T, 0, dt)
        for (i, (t, dt)) in enumerate(rng)
            @test t + dt ≥ 0
        end
    end

end

@testset "backwards lossless                     " begin
    rng = Flows.Steps(0.8, 0, 0.4)
    
    expected = [(0.8, -0.4), (0.4, -0.4)]
    for (i, (t, dt)) in enumerate(rng)
        @test  t ≈ expected[i][1]
        @test dt ≈ expected[i][2]
    end
    
    # test interface used in the integration routines
    T  = 10.0
    dt = 1e-3
    rng = Flows.Steps(T, 0, dt)
    for (i, (t, dt)) in enumerate(rng)
        @test t + dt  ≥ 0
    end
end

@testset "last step ends on T                    " begin
    # with T an integer number of steps, t + dt of the last step used to
    # overshoot T by one ulp for some T and n, for instance 2π and 111,
    # which took the stage times out of the span of a storage
    for T in (2π, 0.7, 1.5586522107, 12.345678), n in 1:2000
        for (t0, t1) in ((0.0, T), (T, 0.0), (0.3, 0.3 + T))
            t, dt = Flows.Steps(t0, t1, abs(t1 - t0) / n)[end]
            @test t + dt == t1
        end
    end
end

@testset "whole number of steps                  " begin
    # with dt = T/n the range can stop a few ulps short of T, which used to
    # add an extra step of a few ulps, and an extra sample just before T in a
    # storage. There must be exactly n steps, the last one ending on T
    for T in (2π, 0.7, 1.5586522107, 2.3059072, 12.345678), n in 1:3000
        for (t0, t1) in ((0.0, T), (T, 0.0))
            s = Flows.Steps(t0, t1, T / n)
            @test length(s) == n
            @test sum(last, s) ≈ t1 - t0
        end
    end

    # a span much shorter than the step is a single step
    @test length(Flows.Steps(0, 1e-9, 0.1)) == 1
    @test Flows.Steps(0, 1e-9, 0.1)[1] == (0.0, 1e-9)
end
