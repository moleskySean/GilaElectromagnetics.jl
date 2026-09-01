# Interface to use Gila operators with `LinearMaps.jl`.
module GilaLinearMapsExt

using GilaElectromagnetics
using LinearAlgebra: mul!
import LinearMaps
import LinearMaps: LinearMap

function LinearMaps.LinearMap(opr::AbstractGlaOpr)
    T = eltype(opr)
    m, n = size(opr)
    fwd!(w, v) = mul!(w, opr, v, one(T), zero(T)) # Matrix-vector product
    adj!(w, v) = begin # Adjoint matrix-vector product
        adjOpr = adjoint!(opr) # Compute with the adjoint operator
        out = mul!(w, adjOpr, v, one(T), zero(T))
        adjoint!(opr) # Restore the original operator
        return out
    end
    return LinearMap{T}(fwd!, adj!, m, n; ismutating=true)
end

end
