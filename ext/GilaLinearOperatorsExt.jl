# Interface to use Gila operators with `LinearOperators.jl`.
module GilaLinearOperatorsExt

using GilaElectromagnetics
using GilaElectromagnetics: arrTyp
using LinearAlgebra: mul!
import LinearOperators
import LinearOperators: LinearOperator

function LinearOperators.LinearOperator(opr::AbstractGlaOpr)
    T = eltype(opr)
    m, n = size(opr)
    fwd!(w, v) = mul!(w, opr, v, one(T), zero(T)) # Matrix-vector product
    adj!(w, v) = begin # Adjoint matrix-vector product
        adjOpr = adjoint!(opr) # Compute with the adjoint operator
        out = mul!(w, adjOpr, v, one(T), zero(T))
        adjoint!(opr) # Restore the original operator
        return out
    end
    is_symmetric = false
    is_hermitian = false
    return LinearOperator(T, m, n, is_symmetric, is_hermitian, fwd!, nothing, adj!; S=arrTyp(opr))
end

end
