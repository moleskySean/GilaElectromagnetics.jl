# Interface to use Gila operators with `LinearMaps.jl`.
module GilaLinearMapsExt

using GilaElectromagnetics
using GilaElectromagnetics: _witAdj
using LinearAlgebra: mul!
import LinearMaps
import LinearMaps: LinearMap

function LinearMaps.LinearMap(opr::AbstractGlaOpr)
    T = eltype(opr)
    m, n = size(opr)
    fwd!(w, v) = mul!(w, opr, v, one(T), zero(T)) # Matrix-vector product
    adj!(w, v) = _witAdj(adjOpr -> mul!(w, adjOpr, v, one(T), zero(T)), opr) # Adjoint matrix-vector product
    return LinearMap{T}(fwd!, adj!, m, n; ismutating=true)
end

end
