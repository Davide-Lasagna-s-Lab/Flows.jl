import LinearAlgebra: Diagonal

@testset "verify interface                       " begin
    L = Diagonal([0.5])
    @test ImcA!(L, 0.1, [1.0], [0.0]) == [1/0.95]
end