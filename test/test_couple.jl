# ----------------------------------------------------------------------------
# Construction and basic accessors
# ----------------------------------------------------------------------------

@testset "couple construction                    " begin
    a = couple([1.0, 2.0], [3, 4, 5])
    @test a isa Coupled{2}
    @test a[1] == [1.0, 2.0]
    @test a[2] == [3, 4, 5]
    @test length(a) == 2
    @test size(a) == (2,)
end

@testset "couplecopy independence                " begin
    src = [1.0, 2.0]
    a = couplecopy(3, src)
    @test length(a) == 3
    for i in 1:3
        @test a[i] == src
    end

    # mutating the original or one element must not affect the others
    src[1] = -99
    @test a[1][1] == 1.0   # was deep-copied
    a[1][2] = -7
    @test a[2][2] == 2.0   # other components are independent
end

@testset "couple getindex bounds                 " begin
    a = couple([1.0], [2.0])
    @test_throws BoundsError a[3]
    # Val-indexed accessor is type-stable and used in System dispatch
    @test a[Val(1)] == [1.0]
    @test a[Val(2)] == [2.0]
end

@testset "couple copy / similar                  " begin
    a = couple([1.0], [2.0, 3.0])

    # similar: same shape, independent storage
    b = similar(a)
    @test length(b) == 2
    @test size(b[1]) == size(a[1])
    @test size(b[2]) == size(a[2])
    b[1][1] = -1.0
    @test a[1][1] == 1.0

    # copy: same contents, independent storage
    c = copy(a)
    @test c[1] == a[1]
    @test c[2] == a[2]
    c[1][1] = -1.0
    @test a[1][1] == 1.0
end

# ----------------------------------------------------------------------------
# Broadcasting
# ----------------------------------------------------------------------------

@testset "couple broadcasting                    " begin
    x = couple([1.0, 2.0], [3.0])
    y = couple([4.0, 5.0], [6.0])

    # fused expression with a scalar
    z = similar(x)
    z .= 2 .* x .+ y
    @test z[1] == 2 .* [1.0, 2.0] .+ [4.0, 5.0]
    @test z[2] == 2 .* [3.0]      .+ [6.0]

    # in-place broadcast must not allocate
    fun!(z, x, y) = (z .= 2 .* x .+ y; z)
    fun!(z, x, y)
    @test (@allocated fun!(z, x, y)) == 0
end

# ----------------------------------------------------------------------------
# Symmetry transforms
# ----------------------------------------------------------------------------

@testset "CoupledTransform applies per component " begin
    a = couple([1.0], [2.0])
    t = Flows.CoupledTransform((x, s) -> x[1] += s)

    @test nothing isa Flows.NoTransform
    @test Flows.CoupledTransform(nothing) isa Flows.NoTransform

    # applies sym to each component, in place, with the same s
    t(a, 1.0)
    @test a == couple([2.0], [3.0])
end

@testset "SymTransform wraps a single state      " begin
    x = [10.0, 20.0]

    # constructor collapses `nothing` to `nothing`
    @test Flows.SymTransform(nothing) isa Flows.NoTransform

    # wrapped callable is forwarded to the underlying function
    f = Flows.SymTransform((x, s) -> (x .+= s; x))
    @test f(x, 1.0) == [11.0, 21.0]
    @test x         == [11.0, 21.0]
end

# Flow-level integration with a non-trivial sym on a single state goes
# through SymTransform; verify that the result of `_propagate!` is actually
# transformed by the wrapper.
@testset "flow sym on single state                " begin
    g(t, x, ẋ) = (ẋ .= 0; ẋ)
    F = flow(g, RK4([0.0]), TimeStepConstant(0.1), (x, s) -> (x .+= s; x))

    # constant ODE: state stays at 0 over the interval, then sym adds s
    x = [0.0]
    F(x, (0.0, 1.0), 5.0)
    @test x == [5.0]

    # without a sym parameter the wrapper is not applied
    x = [0.0]
    F(x, (0.0, 1.0))
    @test x == [0.0]
end
