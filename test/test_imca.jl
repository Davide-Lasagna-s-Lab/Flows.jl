import LinearAlgebra: Diagonal, I

# ----------------------------------------------------------------------------
# Diagonal fallback: (I - c·A) z = y ⇒ z[i] = y[i] / (1 - c·A.diag[i])
# ----------------------------------------------------------------------------

@testset "ImcA! Diagonal solve                   " begin
    # scalar diagonal: (I - 0.1·0.5) z = 1 ⇒ z = 1 / 0.95
    L = Diagonal([0.5])
    @test ImcA!(L, 0.1, [1.0], [0.0]) == [1 / 0.95]

    # multi-element diagonal: each element solved independently
    L = Diagonal([0.5, -2.0, 3.0])
    y = [1.0, 2.0, 3.0]
    z = similar(y)
    @test ImcA!(L, 0.1, y, z) ≈ [1/0.95, 2/1.2, 3/0.7]
    @test z === ImcA!(L, 0.1, y, z)   # returns the output buffer

    # `c = 0` collapses to the identity solve
    @test ImcA!(L, 0.0, y, z) ≈ y
end

# ----------------------------------------------------------------------------
# Generic ImcA_mul!: z = (I - c·A) y, using the Diagonal mul! implementation
# ----------------------------------------------------------------------------

@testset "ImcA_mul! generic action                " begin
    L = Diagonal([0.5, -2.0, 3.0])
    y = [1.0, 2.0, 3.0]
    z = similar(y)

    # (I - c·A) y = y - c·A·y
    @test Flows.ImcA_mul!(L, 0.1, y, z) ≈ y .- 0.1 .* (L.diag .* y)

    # ImcA! and ImcA_mul! should be inverses
    w  = similar(y)
    z  = similar(y)
    z .= ImcA!(L, 0.2, y, z)
    @test Flows.ImcA_mul!(L, 0.2, z, w) ≈ y
end

# ----------------------------------------------------------------------------
# Round-trip with a custom operator type — exercise the dispatch path that
# user code is expected to take.
# ----------------------------------------------------------------------------

struct ScaledIdentity
    α::Float64
end
LinearAlgebra.mul!(out::V, A::ScaledIdentity, x::V) where {V<:AbstractVector} =
    (out .= A.α .* x; out)
Flows.ImcA!(A::ScaledIdentity, c::Real, y::V, z::V) where {V<:AbstractVector} =
    (z .= y ./ (1 - c * A.α); z)

@testset "ImcA! custom operator                   " begin
    A  = ScaledIdentity(2.5)
    y  = [1.0, -3.0, 7.0]
    z  = similar(y)
    @test ImcA!(A, 0.1, y, z) ≈ y ./ (1 - 0.1 * 2.5)

    # ImcA_mul! exercises the generic fallback that uses our mul!
    w = similar(y)
    @test Flows.ImcA_mul!(A, 0.1, y, w) ≈ y .- 0.1 .* (2.5 .* y)
end

# ----------------------------------------------------------------------------
# Missing implementation: the abstract fallback should error informatively
# rather than silently returning.
# ----------------------------------------------------------------------------

struct UnimplementedOp end

@testset "ImcA! abstract fallback errors          " begin
    @test_throws ErrorException ImcA!(UnimplementedOp(), 0.1, [1.0], [0.0])
end
