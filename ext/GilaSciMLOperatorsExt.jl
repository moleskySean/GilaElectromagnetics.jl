# Interface to use Gila operators with `SciMLOperators.jl`.
module GilaSciMLOperatorsExt

using GilaElectromagnetics
using GilaElectromagnetics: arrTyp
using GilaElectromagnetics.GilaOperators: invMul!, invMulAdj!
using LinearAlgebra: mul!
import SciMLOperators: FunctionOperator

function FunctionOperator(opr::AbstractGlaOpr)
    T = eltype(opr)
    m, n = size(opr)
    inp = fill!(arrTyp(opr)(undef, n), zero(T)) # Input prototype
    out = fill!(arrTyp(opr)(undef, m), zero(T)) # Output prototype
    fwd!(w, v, _u, _p, _t) = mul!(w, opr, v, one(T), zero(T)) # Matrix-vector product
    adj!(w, v, _u, _p, _t) = begin # Adjoint matrix-vector product
        adjOpr = adjoint!(opr) # Compute with the adjoint operator
        out = mul!(w, adjOpr, v, one(T), zero(T))
        adjoint!(opr) # Restore the original operator
        return out
    end
    invFwd!(w, v, _u, _p, _t) = invMul!(w, opr, v, one(T), zero(T)) # Inverse matrix-vector product
    invAdj!(w, v, _u, _p, _t) = invMulAdj!(w, opr, v, one(T), zero(T)) # Inverse adjoint matrix-vector product
    return FunctionOperator(fwd!, inp, out; op_adjoint=adj!, op_inverse=invFwd!, op_adjoint_inverse=invAdj!, isconstant=true, islinear=true)
end

end
