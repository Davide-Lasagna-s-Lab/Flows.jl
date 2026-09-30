import LinearAlgebra: Diagonal, norm, dot
using Statistics
using Flows
using Test

@testset "test monitor type                      " begin
    m = Monitor(0, (t, x)->string(x))
    push!(m, 0.0, 0)
    @test times(m)   == [0.0]
    @test samples(m) == ["0"]
    @test eltype(samples(m)) == String
end

@testset "test oneevery                          " begin
    @testset "TimeStepConstant" begin
        # make system
        g(t, x, ẋ) = (ẋ .= .-0.5.*x; ẋ)

        # try on standard vector
        x = Float64[1.0]
        
        # forward map
        ϕ = flow(g, RK4(x), TimeStepConstant(0.1))

        # define monitor
        mon = Monitor(x, (t, x)->x[1]; oneevery=3)

        # make sure we save the last step too
        ϕ(x, (0, 1), mon)
        @test last(times(mon)) == 1.0
    end
    @testset "TimeStepFromStorage" begin
        # make system
        g(t, u, x, ẋ) = (ẋ .= .-0.5.*x; ẋ)

        # try on standard vector
        x = Float64[1.0]

        # forward map
        ϕ = flow(g, RK4(x, ContinuousMode(false)), TimeStepFromStorage(0.1))

        # define a dummy store
        store = RAMStorage(x)
        for t in 0:0.1:1
            push!(store, t, [1.0])
        end

        # define monitor
        mon = Monitor(x, (t, x)->x[1]; oneevery=3)

        # make sure we save the last step too
        ϕ(x, store, (0, 1), mon)
        @test last(times(mon)) == 1.0
    end
end

@testset "test monitor content                   " begin
	# store every push
    m = Monitor([1.0], (t, x)->x[1]^2)

    push!(m, 0.0, [0.0])
    push!(m, 0.1, [1.0])
    push!(m, 0.2, [2.0])
    push!(m, 0.3, [3.0])

    @test times(m)   == [0.0, 0.1, 0.2, 0.3]
    @test samples(m) == [0.0, 1.0, 4.0, 9.0]

    # skip one sample
    m = Monitor([1.0], (t, x)->x[1]^2; oneevery=2)

    push!(m, 0.0, [0.0])
    push!(m, 0.1, [1.0])
    push!(m, 0.2, [2.0])
    push!(m, 0.3, [3.0])

    @test times(m)   == [0.0, 0.2]
    @test samples(m) == [0.0, 4.0]
end

@testset "storenfromlast                         " begin
    # integral of t in dt
    g(t, x, dxdt) = (dxdt[1] = t; dxdt)
    L = Diagonal([0.0])

    # integration scheme
    scheme = CB3R2R3e(Float64[0.0])

    # monitors
    ms = [StoreNFromLast{N}(zeros(1)) for N in 0:2]

    # forward map
    ϕ = flow(g, L, scheme, TimeStepConstant(0.1))

    for (m, tf) in zip(ms, [1.0, 0.9, 0.8])
        # test end point is calculated correctly
        ϕ([0.0], (0, 1), m)

        @test m.t ≈ tf
        @test m.x ≈ [0.5*tf^2]
    end
end

@testset "allocation                             " begin
    # integral of t in dt
    g(t, x, ẋ) = (ẋ[1] = t; ẋ)
    L = Diagonal([0.0])

    # integration scheme
    scheme = CB3R2R3e(Float64[0.0])

    # monitors
    m = Monitor([1.0], (t, x)->x[1]^2; sizehint=10000)

    # forward map
    ϕ = flow(g, L, scheme, TimeStepConstant(0.01))

    # initial condition
    x₀ = [0.0]

    # test end point is calculated correctly
    ϕ(x₀, (0, 1.005), m)

    @test times(m)[end  ] == 1.005
    @test times(m)[end-1] == 1.000
    @test times(m)[end-2] == 0.990

    # warm up
    fun(ϕ, x₀, span, m) = @allocated ϕ(x₀, span, m)

    # does not allocate because we do not grow the arrays in the monitor
    @test_broken fun(ϕ, x₀, (0, 1), m) == 0 # ! allocation tests fail due to logger doing some fancy formatting

    # try resetting and see we still do not allocate
    reset!(m, 101) # ! this sizehint value is just enough to ensure the monitor doesn't have to allocate any new memory
    @test (@allocated reset!(m, 101)) == 0

    @test_broken fun(ϕ, x₀, (0, 1), m) == 0
end

@testset "reset monitors                         " begin
    m = Monitor([1.0], (t, x)->x[1]^2)

    push!(m, 0.0, [0.0])
    push!(m, 0.1, [1.0])
    push!(m, 0.2, [2.0])
    push!(m, 0.3, [3.0])

    @test times(m)   == [0.0, 0.1, 0.2, 0.3]
    @test samples(m) == [0.0, 1.0, 4.0, 9.0]

    reset!(m)
    @test times(m)   == []
    @test samples(m) == []
end

@testset "savebetween                            " begin
    m = Monitor([0.0], (t, x)->copy(x); savebetween=(1, 2))
    push!(m, 0, [0.0])
    push!(m, 1, [0.0])
    push!(m, 2, [0.0])
    push!(m, 3, [0.0])
    @test length(times(m)) == 2
end

@testset "skipfirst                              " begin
    m = Monitor([0.0], (t, x)->copy(x); skipfirst=true)
    push!(m, 0, [0.0])
    push!(m, 1, [0.0])
    push!(m, 2, [0.0])
    push!(m, 3, [0.0])
    @test times(m) == [1, 2, 3]
end

@testset "monitor logging                        " begin
    io = IOBuffer()
    # io = stdout
    m = Monitor(0.0, (t, x)->(x[1], rand(1:100000), randn(ComplexF64), nothing, rand(20, 10, 2)), io=io)
    x = randn(4)

    # easiest way to test this is to simply look if the out is correct
    push!(m, 0, x[1])
    push!(m, 1, x[2])
    push!(m, 2, x[3])
    push!(m, 3, x[4])

    # put test here for output to io
    # @test String(take!(io)) == "some complicated string"
end

# A deterministic hook exercises endpoint detection without relying on the
# constant-step integration path.
struct MonitorTestHook <: AbstractTimeStepFromHook end
(::MonitorTestHook)(g, A, x) = 0.25

@testset "skiplast" begin
    g(t, x, dx) = (dx .= 1; dx)
    x = [0.0]
    for stepping in (TimeStepConstant(0.25), MonitorTestHook())
        F = flow(g, RK4(x), stepping)
        # Cover endpoints both on and off the ordinary sampling schedule.
        for oneevery in (1, 3), stop in (1.0, 1.125)
            baseline = Monitor(x, (t, x)->x[1]; oneevery)
            calls = Float64[]
            mon = Monitor(x, (t, x)->(push!(calls, t); x[1]); oneevery, skiplast=true)
            empty!(calls) # Constructor probes the observable to infer its type.
            a, b = copy(x), copy(x)
            F(a, (0.0, stop), baseline)
            F(b, (0.0, stop), mon)
            keep = findall(!=(stop), times(baseline))
            @test times(mon) == times(baseline)[keep]
            @test samples(mon) == samples(baseline)[keep]
            @test calls == times(mon) # No callback for the discarded endpoint.
            @test a == b             # Skipping storage does not skip integration.
            @test mon.count == baseline.count
        end
    end

    # Reusing a monitor omits every integration endpoint, but records the next
    # initial state. reset! clears samples while preserving the configuration.
    F = flow(g, RK4(x), TimeStepConstant(0.25))
    mon = Monitor(x; skiplast=true)
    F(copy(x), (0.0, 0.5), mon)
    F(copy(x), (0.5, 1.0), mon)
    @test times(mon) == [0.0, 0.25, 0.5, 0.75]
    reset!(mon)
    @test mon.skiplast && mon.count == 0 && isempty(times(mon))
    mon = Monitor(x; skipfirst=true, skiplast=true)
    F(copy(x), (0.0, 0.25), mon)
    @test isempty(times(mon)) # A single step has no interior samples.
    reset!(mon)
    F(copy(x), (0.0, 0.75), mon)
    @test times(mon) == [0.25, 0.5]

    # savebetween can exclude the endpoint already; no previous sample should
    # be removed. Direct pushes are not implicitly treated as final samples.
    mon = Monitor(x; skiplast=true, savebetween=(0.25, 0.5))
    F(copy(x), (0.0, 1.0), mon)
    @test times(mon) == [0.25, 0.5]
    mon = Monitor(x; skiplast=true)
    push!(mon, 1.0, x, true)
    @test times(mon) == [1.0]

    # Stored trajectories support both forward and backward propagation.
    gl(t, u, x, dx) = (dx .= 0; dx)
    store = RAMStorage(x)
    for t in 0:0.25:1
        push!(store, t, copy(x))
    end
    for backward in (false, true)
        span = backward ? (1.0, 0.0) : (0.0, 1.0)
        expected = backward ? [1.0, 0.75, 0.5, 0.25] : [0.0, 0.25, 0.5, 0.75]
        mon = Monitor(x; skiplast=true)
        F = flow(gl, RK4(x, ContinuousMode(backward)), TimeStepFromStorage(0.25))
        F(copy(x), store, span, mon)
        @test times(mon) == expected
    end

    # Cached-stage tangent/adjoint propagation reaches opposite endpoints.
    cache = RAMStageCache(4, x)
    flow(g, RK4(x), TimeStepConstant(0.25))(copy(x), (0.0, 1.0), cache)
    for backward in (false, true)
        mon = Monitor(x; skiplast=true)
        F = flow(gl, RK4(x, DiscreteMode(backward)), TimeStepFromCache())
        F(copy(x), cache, mon)
        expected = backward ? [1.0, 0.75, 0.5, 0.25] : [0.0, 0.25, 0.5, 0.75]
        @test times(mon) == expected
    end
end

@testset "skiplast sampling cadence" begin
    g(t, x, dx) = (dx .= 1; dx)
    F = flow(g, RK4([0.0]), TimeStepConstant(0.25))

    # The endpoint must be absent whether it lands on the regular cadence or
    # would otherwise be forced. Use explicit expected times, not another monitor.
    for (oneevery, expected) in ((2, [0.0, 0.5]), (3, [0.0, 0.75]), (10, [0.0]))
        mon = Monitor([0.0]; oneevery, skiplast=true)
        F([0.0], (0.0, 1.0), mon)
        @test times(mon) == expected
        @test mon.count == 5
    end

    # Both boundary flags leave interior samples at the same counter positions.
    mon = Monitor([0.0]; oneevery=2, skipfirst=true, skiplast=true)
    F([0.0], (0.0, 1.0), mon)
    @test times(mon) == [0.5]

    # Filtering by time must not move the cadence to the first eligible time.
    mon = Monitor([0.0]; oneevery=2, skiplast=true, savebetween=(0.25, 1.0))
    F([0.0], (0.0, 1.0), mon)
    @test times(mon) == [0.5]

    # Each integration restarts its cadence but appends to existing samples.
    mon = Monitor([0.0]; oneevery=3, skiplast=true)
    F([0.0], (0.0, 1.0), mon)
    F([1.0], (1.0, 2.0), mon)
    @test times(mon) == [0.0, 0.75, 1.0, 1.75]
    @test mon.count == 5
    reset!(mon)
    F([0.0], (0.0, 1.0), mon)
    @test times(mon) == [0.0, 0.75]
end


@testset "sampling restarts on each integration" begin
    g(t, x, dx) = (dx .= 1; dx)
    # Repeating the same span must append the same sampling times. Include
    # skipfirst to check that both endpoint flags apply independently per call.
    for stepping in (TimeStepConstant(1.0), MonitorTestHook()),
        skiplast in (false, true), skipfirst in (false, true)
        dt = stepping isa TimeStepConstant ? 1.0 : 0.25
        F = flow(g, RK4([0.0]), stepping)
        mon = Monitor([0.0]; oneevery=2, skiplast, skipfirst)
        expected = collect(0:2:10) .* dt
        skiplast && pop!(expected)
        skipfirst && popfirst!(expected)
        F([0.0], (0.0, 10dt), mon)
        F([0.0], (0.0, 10dt), mon)
        @test times(mon) == vcat(expected, expected)
        @test mon.count == 11
    end
end
