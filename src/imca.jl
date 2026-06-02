import LinearAlgebra

export ImcA!

"""
    ImcA!(A, c::Real, y, z) -> z

Solve the linear problem `(I - c·A) z = y` for `z`, where `A` is a
linear operator, `c` is a scalar, and `I` is the identity. The result
is written in place into `z` and returned.

This is one of the two operator-level primitives the IMEX time
schemes rely on (the other being [`ImcA_mul!`](@ref)). Users with a
custom `A` type should add a specialised method to `ImcA!` that
exploits any structure available. A fallback implementation is
provided for `LinearAlgebra.Diagonal`; everything else raises an
informative error.

# Notes
The function name should be read "I-minus-cee-A".
"""
ImcA!(A, c::Real, y, z) =
    error("ImcA! missing implementation for operator `A` of type $(typeof(A))")

"""
    ImcA_mul!(A, c::Real, y, z) -> z

Compute `z = (I - c·A) y` for `z`, where `A` is a linear operator, `c`
is a scalar, and `I` is the identity. The result is written in place
into `z` and returned.

This pairs with [`ImcA!`](@ref): the IMEX schemes use `ImcA_mul!` to
form the right-hand-side of the linear system and `ImcA!` to solve
it. The default implementation calls `mul!(z, A, y)` and then performs
the axpby in place; specialise it only when a faster `(I - c·A) y`
evaluation is available.
"""
ImcA_mul!(A, c::Real, y, z) = (mul!(z, A, y); z .= y .- c.*z; z)

# ----------------------------------------------------------------------------
# Diagonal fallback
# ----------------------------------------------------------------------------
#
# Provide a built-in implementation for the very common case where the
# stiff operator is a diagonal matrix. `(I - cA) z = y` becomes
# `z[i] = y[i] / (1 - c·A.diag[i])`, an embarrassingly parallel
# elementwise operation.

ImcA!(A::LinearAlgebra.Diagonal, c::Real, y::V, z::V) where {V<:AbstractVector} =
    (z .= y./(1 .- c.*A.diag); z)

LinearAlgebra.mul!(out::V, A::LinearAlgebra.Diagonal, in::V) where {V<:AbstractVector} =
    (out .= A.diag.*in; out)
