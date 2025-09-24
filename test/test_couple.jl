@testset "couple                                 " begin
    # couplecopy
    a = couplecopy(2, [1.0])
    @test a[1] == [1.0]
    @test a[2] == [1.0]
    @test length(a) == 2
end

@testset "couple transform                       " begin
    # transform couple
    a = couple([1.0], [2.0])
    t = Flows.CoupledTransform((x, s)->x[1] += s)
    @test nothing isa Flows.NoTransform
    @test Flows.CoupledTransform(nothing) isa Flows.NoTransform
    @test t(a, 1.0) == couple([2.0], [3.0])
end
