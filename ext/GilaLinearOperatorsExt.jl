# Interface to use Gila operators with `LinearOperators.jl`.
module GilaLinearOperatorsExt

using GilaElectromagnetics
using GilaElectromagnetics.GilaOperators: _witAdj
using LinearAlgebra: mul!
import LinearOperators
import LinearOperators: LinearOperator

function LinearOperators.LinearOperator(opr::AbstractGlaOpr)
    T = eltype(opr)
    m, n = size(opr)
    fwd!(w, v) = mul!(w, opr, v, one(T), zero(T)) # Matrix-vector product
    adj!(w, v) = _witAdj(adjOpr -> mul!(w, adjOpr, v, one(T), zero(T)), opr) # Adjoint matrix-vector product
    is_symmetric = false
    is_hermitian = false
    return LinearOperator(T, m, n, is_symmetric, is_hermitian, fwd!, nothing, adj!; S=typeof(similar(opr, (0,))))
end

end
