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
using GilaElectromagnetics.GilaVacuum: glaOprPrp
using GilaElectromagnetics.GilaVolumes: genEveExtInf
using DoubleFloats: Double64
using KernelAbstractions: CPU
import JLD2

# The Fourier data, and what glaOprPrp needs to rebuild the plans around it
struct MemDsk{T<:AbstractFloat}
    egoFur::Vector{Array{Complex{T}, 6}}
    frqPhz::ComplexF64
    genPrc::Symbol
    qssApx::Bool
    adjMod::Bool
    blc::Union{Nothing, NamedTuple}
    trgVol::GlaVol
    srcVol::GlaVol
end

struct SusDsk{T<:AbstractFloat, N}
    sus::Array{Complex{T}, N}
    cvol::GlaCmpVol
    adjMod::Bool
end

#= genPrc is a type-valued field; a stored DataType is the cross-version
reconstruction hazard JLD2 is here to avoid, so the two admissible precisions
are named explicitly. Float32 never reaches here, chkGenPrc rejects it. =#
_prcSym(genPrc::Type{<:AbstractFloat}) = genPrc === Float64 ? :Float64 :
    genPrc === Double64 ? :Double64 :
    throw(ArgumentError("Cannot write an operator generated at $genPrc: add that precision to GilaJLD2Ext's genPrc table first."))
_symPrc(genPrc::Symbol) = genPrc === :Float64 ? Float64 :
    genPrc === :Double64 ? Double64 :
    throw(ArgumentError("The file names generation precision $genPrc, which this version of Gila does not know; read it with a version whose GilaJLD2Ext lists $genPrc."))

JLD2.writeas(::Type{<:GlaVacOprMem{T}}) where T<:AbstractFloat = MemDsk{T}

function JLD2.wconvert(::Type{MemDsk{T}}, mem::GlaVacOprMem{T}) where T<:AbstractFloat
    return MemDsk{T}(collect(map(Array, mem.egoFur)), mem.cmpInf.frqPhz,
        _prcSym(mem.cmpInf.genPrc), mem.cmpInf.qssApx, mem.cmpInf.adjMod, mem.cmpInf.blc, mem.trgVol, mem.srcVol)
end

#= mixInf is a pure function of the two volumes---GlaVacOprMem's own constructor
computes it as genEveExtInf(trgVol, srcVol)---so it never reaches the disk, and
neither do the CartesianIndices it holds. =#
JLD2.rconvert(::Type{<:GlaVacOprMem{T}}, dsk::MemDsk{T}) where T<:AbstractFloat =
    glaOprPrp(dsk.egoFur, dsk.trgVol, dsk.srcVol, genEveExtInf(dsk.trgVol, dsk.srcVol),
        GlaKerOpt{T}(dsk.frqPhz, _symPrc(dsk.genPrc), dsk.qssApx, dsk.adjMod, CPU(), dsk.blc))

JLD2.writeas(::Type{<:SusOpr{T, A}}) where {T<:AbstractFloat, A} = SusDsk{T, ndims(A)}

JLD2.wconvert(::Type{SusDsk{T, N}}, opr::SusOpr{T}) where {T<:AbstractFloat, N} =
    SusDsk{T, N}(Array(opr.sus), opr.cvol, opr.adjMod)

JLD2.rconvert(::Type{<:SusOpr}, dsk::SusDsk) = SusOpr(dsk.sus, dsk.cvol, dsk.adjMod)

end
