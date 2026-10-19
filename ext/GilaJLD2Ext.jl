#= Archival storage of Gila operators through `JLD2.jl`. Write with
`jldsave("g0.jld2"; opr)` and read with `load("g0.jld2", "opr")`; the extension
adds no names to Gila's interface. Only two types need a hook: `GlaVacOprMem`,
whose FFTW plans are process-local C pointers and are rebuilt from the stored
Fourier data on load, and `SusOpr`, whose susceptibility may live on a device.
Everything wrapping them is plain data that JLD2's own struct handling covers.
An operator written from a GPU reads back on the CPU---`adapt` it to put it
back on the device. =#
module GilaJLD2Ext

using GilaElectromagnetics
using GilaElectromagnetics: GlaVacOprMem, _memDsk, _memLod
import JLD2

# The stored form of _memDsk
struct MemDsk{T<:AbstractFloat}
    egoFur::Vector{Array{Complex{T}, 6}}
    frqPhz::ComplexF64
    qssApx::Bool
    adjMod::Bool
    hrmPrt::Union{Nothing, Symbol}
    blc::Union{Nothing, NamedTuple}
    trgVol::GlaVol
    srcVol::GlaVol
end

struct SusDsk{T<:AbstractFloat, N}
    sus::Array{Complex{T}, N}
    cvol::GlaCmpVol
    adjMod::Bool
end

JLD2.writeas(::Type{<:GlaVacOprMem{T}}) where T<:AbstractFloat = MemDsk{T}

JLD2.wconvert(::Type{MemDsk{T}}, mem::GlaVacOprMem{T}) where T<:AbstractFloat =
    MemDsk{T}(_memDsk(mem)...)
JLD2.rconvert(::Type{<:GlaVacOprMem{T}}, dsk::MemDsk{T}) where T<:AbstractFloat = _memLod(dsk)

JLD2.writeas(::Type{<:SusOpr{T, A}}) where {T<:AbstractFloat, A} = SusDsk{T, ndims(A)}

JLD2.wconvert(::Type{SusDsk{T, N}}, opr::SusOpr{T}) where {T<:AbstractFloat, N} =
    SusDsk{T, N}(Array(opr.sus), opr.cvol, opr.adjMod)

JLD2.rconvert(::Type{<:SusOpr}, dsk::SusDsk) = SusOpr(dsk.sus, dsk.cvol, dsk.adjMod)

end
