import LinearAlgebra: Diagonal, norm
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

@testset "storeonebutlast                        " begin
    # integral of t in dt
    g(t, x, dxdt) = (dxdt[1] = t; dxdt)
    L = Diagonal([0.0])

    # integration scheme
    scheme = CB3R2R3e(Float64[0.0])

    # monitors
    m = StoreOneButLast(zeros(1))

    # forward map
    ϕ = flow(g, L, scheme, TimeStepConstant(0.1))

    # initial condition
    x₀ = [0.0]

    # test end point is calculated correctly
    ϕ(x₀, (0, 1), m)

    @test m.t == 0.9
    @test m.x ≈ [0.5*(0.9)^2]
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
    @test fun(ϕ, x₀, (0, 1), m) == 0

    # try resetting and see we still do not allocate
    reset!(m)
    @test (@allocated reset!(m)) == 0

    # FIXME: this is most likely a result of allocation in the step! method, although the call above doesn't seem to have an issue
    # ! the allocation here only happens after the monitor is reset, so maybe initialising the monitor for the first time adds an overhead in memory allocation
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