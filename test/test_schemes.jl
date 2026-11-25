@testset "schemes by name                         " begin
    x = zeros(3)

    # ---- every name gives its scheme, those of a split system among all of them ----
    @test imex_schemes[:CB3R2R3e](x) isa CB3R2R3e
    @test imex_schemes[:ETDRK4](x) isa ETDRK4
    @test available_schemes[:RK4](x) isa RK4
    @test keys(imex_schemes) ⊆ keys(available_schemes)
    @test sort(collect(keys(available_schemes))) ==
          [:CB3R2R2, :CB3R2R3c, :CB3R2R3e, :CB4R3R4, :CNRK2, :ETDRK2, :ETDRK4, :RK4, :RK8]
    @test !haskey(imex_schemes, :RK4)
end
