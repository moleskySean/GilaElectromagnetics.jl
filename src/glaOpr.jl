"""
    GilaOperators

This module provides the core operator types for the Gila package, including the vacuum Green function operator,
scattering operators, and their compositions.

# Types
- `AbstractGlaOpr`: Abstract base type for all operators
- `GlaOprVac`: Vacuum Green function operator G₀
- `InvSctOpr`: Inverse scattering operator (I - XG₀)⁻¹
- `SctOpr`: Scattering operator with solver
- `GlaOpr`: Full Green function operator G₀(I - XG₀)⁻¹

# Type Aliases
- `VacuumGreenOperator`: Alias for `GlaOprVac`
- `InverseScatteringOperator`: Alias for `InvSctOpr`
- `ScatteringOperator`: Alias for `SctOpr`
- `GreenOperator`: Alias for `GlaOpr`
"""
module GilaOperators

using ..GilaVolumes
using ..GilaFields
using ..GilaVacuum
using ..GilaTypes
using ..GilaSolvers
using CUDA
using Serialization

import LinearAlgebra: adjoint!

import ..GilaVacuum: useCpu!, useGpu!, egoCmpPos
import ..GilaVolumes: _lwrEdg, _uprEdg, _ovrLap

export GlaOprVac, AsyGlaOprVac, SymGlaOprVac, MulRegGlaOprVac, GlaCmpOprVac, AsyGlaCmpOprVac, SymGlaCmpOprVac, InvSctOpr, SctOpr, GlaOpr
export VacuumGreenOperator, AsymVacuumGreenOperator, SymVacuumGreenOperator, MultiRegionVacuumGreenOperator, CompositeVacuumGreenOperator, AsymCompositeVacuumGreenOperator, SymCompositeVacuumGreenOperator, InverseScatteringOperator, ScatteringOperator, GreenOperator
export isadjoint, isselfoperator, isexternaloperator, isoverlappingoperator, isgpu, adjoint!, glaSze, slv, asym

"""
    GlaOprVac

Represents the vacuum Green function operator G₀, which describes electromagnetic
interactions in free space.

# Fields
- `mem::GlaVacOprMem`: Memory structure containing the operator's data, including
  volume information and Fourier coefficients
- `srcMsk::NTuple{StepRange{Int64, Int64}, 3}`: Tuple of ranges defining the mask for
  the input volume (for overlapping operators only)
- `trgMsk::NTuple{StepRage{Int64, Int64}, 3}`: Tuple of ranges defining the mask for
  the output volume (for overlapping operators only)
"""
struct GlaOprVac <: AbstractGlaVacOpr
    mem::GlaVacOprMem
    srcMsk::NTuple{3, OrdinalRange{Int64, Int64}}
    trgMsk::NTuple{3, OrdinalRange{Int64, Int64}}
end

"""
    MulRegGlaOprVac

Represents the vacuum Green function operator G₀ for multiple disjoint domains, i.e., the source and/or target volumes consist of multiple non-overlapping regions (where the gaps are *not* computed).

# Fields
- oprMat::Matrix{GlaOprVac}: Matrix of vacuum Green function operators for each disjoint region pair
"""
struct MulRegGlaOprVac <: AbstractGlaVacOpr
    oprMat::Matrix{GlaOprVac}
end

"""
    AsyGlaOprVac

Represents the anti-Hermitian part of the vacuum Green function operator.

# Fields
- `mem::GlaVacOprMem`: Memory structure containing the operator's data, including
  volume information and Fourier coefficients
"""
struct AsyGlaOprVac <: AbstractGlaVacOpr
    mem::GlaVacOprMem
end

"""
    SymGlaOprVac

Represents the Hermitian part of the vacuum Green function operator.

# Fields
- `mem::GlaVacOprMem`: Memory structure containing the operator's data, including
  volume information and Fourier coefficients
"""
struct SymGlaOprVac <: AbstractGlaVacOpr
    mem::GlaVacOprMem
end

"""
    InvSctOpr

Represents the inverse scattering operator (I - XG₀), where X is the susceptibility
tensor. This operator describes how electromagnetic fields interact with a material
medium.

# Fields
- `oprVac::AbstractGlaVacOpr`: The vacuum Green function operator
- `sus::AbstractArray{ComplexF64}`: The susceptibility (isotropic medium)
  representing the material response. Over a single volume it is a 3-tensor of
  cell values, over a composite volume a flat vector with one entry per degree of
  freedom
"""
mutable struct InvSctOpr <: AbstractGlaOpr
    oprVac::AbstractGlaVacOpr
    sus::AbstractArray{ComplexF64}

    #= A composite operator takes the susceptibility in any of the forms _cmpSus
    accepts and stores it in the flat degree of freedom layout of GlaFld. =#
    function InvSctOpr(oprVac::AbstractGlaVacOpr, sus)
        oprVac isa GlaCmpOprVac || return new(oprVac, sus)
        if !isselfoperator(oprVac)
            throw(ArgumentError("An inverse scattering operator needs a self operator, and this composite operator maps between two different tilings."))
        end
        return new(oprVac, _cmpSus(oprVac.srcCvl, sus, isgpu(oprVac)))
    end
end

"""
    SctOpr

Represents the scattering operator (I - XG₀)⁻¹, which includes a solver for
computing the action of the inverse scattering operator.

# Fields
- `invSctOpr::InvSctOpr`: The inverse scattering operator
- `slv::GlaSlv`: The solver to use for solving the linear system
"""
mutable struct SctOpr <: AbstractGlaOpr
    invSctOpr::InvSctOpr
    slv::GlaSlv
end

"""
    GlaOpr

Represents the full Green function operator G₀(I - XG₀)⁻¹, which combines the
vacuum Green function with the scattering operator to describe electromagnetic
interactions in a material medium.

# Fields
- `sctOpr::SctOpr`: The scattering operator
"""
mutable struct GlaOpr <: AbstractGlaOpr
    sctOpr::SctOpr
end

# Type aliases for convenience
const VacuumGreenOperator = GlaOprVac
const AsymVacuumGreenOperator = AsyGlaOprVac
const SymVacuumGreenOperator = SymGlaOprVac
const MultiRegionVacuumGreenOperator = MulRegGlaOprVac
const InverseScatteringOperator = InvSctOpr
const ScatteringOperator = SctOpr
const GreenOperator = GlaOpr

# Returns true if the volumes share interior. Face, edge, and corner contact is
# not overlap: the external construction has contact corrections for it, and the
# union path cannot represent contact between volumes at different cell scales.
function ovrChk(vol1::GlaVol, vol2::GlaVol)
    # Upper/lower edges of the volumes
    lwrEdg1 = first.(vol1.grd) .- (vol1.scl .// 2)
    uprEdg1 = last.(vol1.grd) .+ (vol1.scl .//2)
    lwrEdg2 = first.(vol2.grd) .- (vol2.scl .// 2)
    uprEdg2 = last.(vol2.grd) .+ (vol2.scl .//2)

    # Volume overlap check
    return all(max.(lwrEdg1, lwrEdg2) .< min.(uprEdg1, uprEdg2)) # < not <= to exclude contact
end

#= Six cells of the coarser interface scale is where the relative Frobenius
error between two separated volumes drops under 1e-8, measured across mesh
scales, same and mixed scale pairs, and body sizes. =#
const prxCelMin = 6

#= Separation of two volumes in cells of the coarser interface scale, paired
with that scale, or nothing when the volumes touch or overlap. A touching pair
runs the contact quadrature, which is exact, so it carries no proximity cost. =#
function _prxGap(volA::GlaVol, volB::GlaVol)
    sep = max.(_lwrEdg(volB) .- _uprEdg(volA), _lwrEdg(volA) .- _uprEdg(volB))
    all(sep .<= 0) && return nothing
    sclCrs = max.(volA.scl, volB.scl)
    dir = argmax(idx -> max(sep[idx], 0//1) // sclCrs[idx], 1:3)
    return (sep[dir] // sclCrs[dir], sclCrs[dir])
end

# Warn about a separated pair sitting closer than the measured accuracy bound
function prxChk(trgVol::GlaVol, srcVol::GlaVol)
    gap = _prxGap(trgVol, srcVol)
    (isnothing(gap) || gap[1] >= prxCelMin) && return nothing

    celStr = (isone(denominator(gap[1])) ? string(numerator(gap[1])) : string(round(Float64(gap[1]); digits=2))) * (isone(gap[1]) ? " cell" : " cells")
    @warn "The target and source are separated by $celStr of the coarser $(gap[2]) interface scale. Operator entries this close are quadrature limited, and the relative error only reaches about 1e-8 at a gap of $prxCelMin cells. It is strongly recommended to refine the facing surfaces with `refine` to increase the accuracy of the operator."
    return nothing
end

# Returns the region where subVol is contained within vol
function mskRng(subVol::GlaVol, vol::GlaVol)
    stpVol = Rational.(step.(vol.grd))
    stpSub = Rational.(step.(subVol.grd))
    stpRat = stpSub .// stpVol
    @assert all(isinteger.(stpRat)) "Volumes must share a common scale grid for masking"
    stpRat = Tuple(numerator.(stpRat))

    off = (first.(subVol.grd) .- first.(vol.grd)) .// stpVol # Offset between volumes
    @assert all(isinteger.(off)) "Sub-volume must align with volume grid for masking"
    off = Tuple(numerator.(off))
    idxBeg = 1 .+ off # 1-based indexing
    idxEnd = idxBeg .+ ((subVol.cel .- 1) .* stpRat)
    return (idxBeg[1]:stpRat[1]:idxEnd[1],
            idxBeg[2]:stpRat[2]:idxEnd[2],
            idxBeg[3]:stpRat[3]:idxEnd[3])
end

"""
    GlaOprVac(trgVol::GlaVol, srcVol::GlaVol; useGpu::Bool=false, prxWrn::Bool=true)

Construct a vacuum Green function operator for external interactions between different volumes.

This constructor creates an external Green function operator that describes electromagnetic interactions between distinct regions in free space. The operator maps sources in the source volume to fields in the target volume, enabling the modeling of coupling effects between different parts of an electromagnetic system. For the computation to work correctly, the source and target volumes must share a common scale grid.

Two volumes separated by fewer than six cells of the coarser interface scale are quadrature limited, so the constructor warns. The six cell bound is measured, and holds across mesh scales and body sizes. Touching volumes are handled by the contact quadrature and never warn.

# Arguments
- `trgVol::GlaVol`: The target volume where the field will be computed
- `srcVol::GlaVol`: The source volume containing the sources
- `useGpu::Bool=false`: Whether to use GPU computation. If true, uses GPU acceleration, otherwise uses CPU
- `prxWrn::Bool=true`: Whether to warn about a separation under the accuracy bound

# Returns
- `GlaOprVac`: The vacuum Green function operator

"""
function GlaOprVac(trgVol::GlaVol, srcVol::GlaVol; useGpu::Bool=false,
    prxWrn::Bool=true)
    prxWrn && prxChk(trgVol, srcVol)
    innMsk = ntuple(_ -> 0:0, 3)
    outMsk = ntuple(_ -> 0:0, 3)
    if trgVol != srcVol && ovrChk(trgVol, srcVol)
        # Overlap: create the union volume and mask out the input/output regions
        vol = uniVol(trgVol, srcVol)
        innMsk = mskRng(srcVol, vol)
        outMsk = mskRng(trgVol, vol)
        trgVol, srcVol = vol, vol
    end

    # Create the memory structure with appropriate GPU/CPU options
    mem = GlaVacOprMem(useGpu ? GPUKerOpt() : CPUKerOpt(), trgVol, srcVol)
    return GlaOprVac(mem, innMsk, outMsk)
end

"""
    GlaOprVac(mem::GlaVacOprMem)

Construct a vacuum Green function operator from a memory structure.

# Arguments
- `mem::GlaVacOprMem`: The memory structure containing the operator's data

# Returns
- `GlaOprVac`: The vacuum Green function operator
"""
function GlaOprVac(mem::GlaVacOprMem)
    innMsk = ntuple(_ -> 0:0, 3)
    outMsk = ntuple(_ -> 0:0, 3)
    trgVol, srcVol = mem.trgVol, mem.srcVol
    if trgVol != srcVol && ovrChk(trgVol, srcVol)
        # Overlap: create the union volume and mask out the input/output regions
        vol = uniVol(trgVol, srcVol)
        innMsk = mskRng(srcVol, vol)
        outMsk = mskRng(trgVol, vol)
        trgVol, srcVol = vol, vol
    end
    return GlaOprVac(mem, innMsk, outMsk)
end

"""
    GlaOprVac(vol::GlaVol; useGpu::Bool=false)

Construct a vacuum Green function operator for self-interactions on a single volume.

This constructor creates a self-interaction Green function operator where the source and target volumes are identical. The operator describes electromagnetic interactions within a single volume in free space, making it suitable for modeling self-coupling effects in electromagnetic systems.

# Arguments
- `vol::GlaVol`: The volume to compute the self Green function for
- `useGpu::Bool=false`: Whether to use GPU computation. If true, uses GPU acceleration, otherwise uses CPU

# Returns
- `GlaOprVac`: The vacuum Green function operator

"""
GlaOprVac(vol::GlaVol; useGpu::Bool=false) = GlaOprVac(vol, vol; useGpu=useGpu)

"""
    GlaOprVac(opr::InvSctOpr)

Construct a vacuum Green function operator from an inverse scattering operator.

# Arguments
- `opr::InvSctOpr`: The inverse scattering operator to convert into a vacuum Green function operator

# Returns
- `GlaOprVac`: The vacuum Green function operator
"""
GlaOprVac(opr::InvSctOpr) = opr.oprVac

"""
    GlaOprVac(opr::SctOpr)

Construct a vacuum Green function operator from a scattering operator.

# Arguments
- `opr::SctOpr`: The scattering operator to convert into a vacuum Green function operator

# Returns
- `GlaOprVac`: The vacuum Green function operator
"""
GlaOprVac(opr::SctOpr) = GlaOprVac(opr.invSctOpr)

"""
    GlaOprVac(opr::GlaOpr)

Construct a vacuum Green function operator from a full Green function operator.

# Arguments
- `opr::GlaOpr`: The full Green function operator to convert into a vacuum Green function operator

# Returns
- `GlaOprVac`: The vacuum Green function operator
"""
GlaOprVac(opr::GlaOpr) = GlaOprVac(opr.sctOpr)

"""
    AsyGlaOprVac(vol::GlaVol; useGpu::Bool=false)

Construct the anti-Hermitian part of the vacuum Green function operator for self-interactions on a single volume.

This constructor creates the anti-Hermitian part of the vacuum Green function operator, which describes the radiated components of electromagnetic interactions within a single volume in free space. This operator shows up in the fluctuation-dissipation theorem and is thus related to certain losses in the system.

# Arguments
- `vol::GlaVol`: The volume to compute the anti-Hermitian part of the vacuum Green function for
- `useGpu::Bool=false`: Whether to use GPU computation. If true, uses GPU acceleration, otherwise uses CPU

# Returns
- `AsyGlaOprVac`: The anti-Hermitian part of the vacuum Green function operator
"""
function AsyGlaOprVac(vol::GlaVol; useGpu::Bool=false)
    kerOpt = useGpu ? GPUKerOpt() : CPUKerOpt()
    mem = GlaVacOprMem(kerOpt, vol, vol)
    map!(fur -> complex.(imag.(fur)), mem.egoFur) # Take the imaginary part of the Fourier coefficients since Asym commutes with the FFT (to machine epsilon)
    return AsyGlaOprVac(mem)
end

"""
    AsyGlaOprVac(opr::GlaOprVac)

Construct the anti-Hermitian part of the vacuum Green function operator from a vacuum Green function operator.

# Arguments
- `opr::GlaOprVac`: The vacuum Green function operator to convert into its anti-Hermitian part

# Returns
- `AsyGlaOprVac`: The anti-Hermitian part of the vacuum Green function operator
"""
function AsyGlaOprVac(opr::GlaOprVac)
    srcVol, trgVol = opr.mem.srcVol, opr.mem.trgVol
    if srcVol != trgVol
        throw(ArgumentError("AsyGlaOprVac can only be constructed from a GlaOprVac with identical source and target volumes"))
    end
    mem = deepcopy(opr.mem)
    map!(fur -> complex.(imag.(fur)), mem.egoFur) # Take the imaginary part of the Fourier coefficients since Asym commutes with the FFT (to machine epsilon)
    return AsyGlaOprVac(mem)
end

"""
    SymGlaOprVac(vol::GlaVol; useGpu::Bool=false)

Construct the Hermitian part of the vacuum Green function operator for self-interactions on a single volume.

This constructor creates the Hermitian part of the vacuum Green function operator

# Arguments
- `vol::GlaVol`: The volume to compute the anti-Hermitian part of the vacuum Green function for
- `useGpu::Bool=false`: Whether to use GPU computation. If true, uses GPU acceleration, otherwise uses CPU

# Returns
- `SymGlaOprVac`: The Hermitian part of the vacuum Green function operator
"""
function SymGlaOprVac(vol::GlaVol; useGpu::Bool=false)
    kerOpt = useGpu ? GPUKerOpt() : CPUKerOpt()
    mem = GlaVacOprMem(kerOpt, vol, vol)
    map!(fur -> complex.(real.(fur)), mem.egoFur) # Take the real part of the Fourier coefficients since Sym commutes with the FFT (to machine epsilon)
    return SymGlaOprVac(mem)
end

"""
    SymGlaOprVac(opr::GlaOprVac)

Construct the Hermitian part of the vacuum Green function operator from a vacuum Green function operator.

# Arguments
- `opr::GlaOprVac`: The vacuum Green function operator to convert into its Hermitian part

# Returns
- `SymGlaOprVac`: The Hermitian part of the vacuum Green function operator
"""
function SymGlaOprVac(opr::GlaOprVac)
    srcVol, trgVol = opr.mem.srcVol, opr.mem.trgVol
    if srcVol != trgVol
        throw(ArgumentError("SymGlaOprVac can only be constructed from a GlaOprVac with identical source and target volumes"))
    end
    mem = deepcopy(opr.mem)
    map!(fur -> complex.(real.(fur)), mem.egoFur) # Take the real part of the Fourier coefficients since Asym commutes with the FFT (to machine epsilon)
    return SymGlaOprVac(mem)
end

"""
    MulRegGlaOprVac(trgVols::VT{GlaVol}, srcVols::VT{GlaVol}; useGpu::Bool=false) where VT <: AbstractVector

Construct a vacuum Green function operator for multiple target and source volumes.

This constructor creates a vacuum Green function operator that describes interactions between multiple target and source volumes.

# Arguments
- `trgVols::VT{GlaVol}`: A vector of target volumes
- `srcVols::VT{GlaVol}`: A vector of source volumes
- `useGpu::Bool=false`: Whether to use GPU computation. If true, uses GPU acceleration, otherwise uses CPU

# Returns
- `MulRegGlaOprVac`: The vacuum Green function operator for multiple target and source volumes
"""
function MulRegGlaOprVac(trgVols::VT, srcVols::VT; useGpu::Bool=false) where VT <: AbstractVector{GlaVol}
    ops = [GlaOprVac(trgVol, srcVol; useGpu=useGpu) for trgVol in trgVols, srcVol in srcVols]
    return MulRegGlaOprVac(ops)
end

"""
    InvSctOpr(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{ComplexF64}; useGpu::Bool=isa(sus, CuArray))

Construct an inverse scattering operator for external interactions between different volumes.

# Arguments
- `trgVol::GlaVol`: The target volume where the field will be computed
- `srcVol::GlaVol`: The source volume containing the sources
- `sus::AbstractArray{ComplexF64}`: The susceptibility tensor, either as a flat vector or a 3-tensor
- `useGpu::Bool=false`: Whether to use GPU computation. If true, uses GPU acceleration, otherwise uses CPU

# Returns
- `InvSctOpr`: The inverse scattering operator

This constructor creates an external inverse scattering operator that describes how electromagnetic fields interact with a material medium between distinct regions. The susceptibility tensor can be provided either as a flat vector (which will be reshaped to match the source volume dimensions) or as a 3-tensor directly. The tensor must match the dimensions of the source volume.
"""
function InvSctOpr(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{ComplexF64}; useGpu::Bool=isa(sus, CuArray))
    # Create the vacuum operator
    oprVac = GlaOprVac(trgVol, srcVol; useGpu=useGpu)
    
    # Reshape susceptibility if needed and validate size
    if size(sus) != srcVol.cel
        throw(ArgumentError("Susceptibility tensor dimensions $(size(sus)) do not match volume dimensions $(srcVol.cel)"))
    end
    susTen = rszSus(sus, srcVol.cel)
    
    return InvSctOpr(oprVac, susTen)
end

"""
    InvSctOpr(vol::GlaVol, sus::AbstractArray{ComplexF64}; useGpu::Bool=isa(sus, CuArray))

Construct an inverse scattering operator for self-interactions on a single volume.

# Arguments
- `vol::GlaVol`: The volume to compute the self-interaction for
- `sus::AbstractArray{ComplexF64}`: The susceptibility tensor, either as a flat vector or a 3-tensor
- `useGpu::Bool=isa(sus, CuArray)`: Whether to use GPU computation. If true, uses GPU acceleration, otherwise uses CPU

# Returns
- `InvSctOpr`: The inverse scattering operator

This constructor creates a self-interaction inverse scattering operator that describes how electromagnetic fields interact with a material medium within a single volume. The susceptibility tensor can be provided either as a flat vector (which will be reshaped to match the volume dimensions) or as a 3-tensor directly. The tensor must match the dimensions of the volume.
"""
InvSctOpr(vol::GlaVol, sus::AbstractArray{ComplexF64}; useGpu::Bool=isa(sus, CuArray)) = InvSctOpr(vol, vol, sus; useGpu=useGpu)

"""

    InvSctOpr(sctOpr::SctOpr)

Construct an inverse scattering operator from a scattering operator.

# Arguments
- `sctOpr::SctOpr`: The scattering operator to convert into an inverse scattering operator

# Returns
- `InvSctOpr`: The inverse scattering operator
"""
InvSctOpr(opr::SctOpr) = opr.invSctOpr

"""
    InvSctOpr(opr::GlaOpr)

Construct an inverse scattering operator from a full Green function operator.

# Arguments
- `opr::GlaOpr`: The full Green function operator to convert into an inverse scattering operator

# Returns
- `InvSctOpr`: The inverse scattering operator
"""
InvSctOpr(opr::GlaOpr) = InvSctOpr(opr.sctOpr)

# Reshape a flat susceptibility vector into a 3-tensor matching the volume dimensions.
function rszSus(sus::AbstractArray{ComplexF64}, cel::NTuple{3,Integer})
    if ndims(sus) == 3
        return sus
    elseif ndims(sus) == 1
        if length(sus) != prod(cel)
            throw(ArgumentError("Flat susceptibility vector length ($(length(sus))) does not match volume size ($(prod(cel)))"))
        end
        return reshape(sus, cel)
    else
        throw(ArgumentError("Susceptibility must be either a flat vector or a 3-tensor"))
    end
end

"""
    SctOpr(vol::GlaVol, sus::AbstractArray{ComplexF64}; useGpu::Bool=isa(sus, CuArray), slv::GlaSlv=BiCGStabSolver())

Construct a scattering operator for self-interactions on a single volume.

This constructor creates a self-interaction scattering operator that describes how electromagnetic fields interact with a material medium within a single volume. The susceptibility tensor can be provided either as a flat vector (which will be reshaped to match the volume dimensions) or as a 3-tensor directly. The tensor must match the dimensions of the volume.

# Arguments
- `vol::GlaVol`: The volume to compute the self-interaction for
- `sus::AbstractArray{ComplexF64}`: The susceptibility tensor, either as a flat vector or a 3-tensor
- `useGpu::Bool=isa(sus, CuArray)`: Whether to use GPU computation. If true, uses GPU acceleration, otherwise uses CPU
- `slv::GlaSlv=BiCGStabSolver()`: The solver to use for solving the linear system

# Returns
- `SctOpr`: The scattering operator
"""
SctOpr(vol::GlaVol, sus::AbstractArray{ComplexF64}; useGpu::Bool=isa(sus, CuArray), slv::GlaSlv=BiCGStabSolver()) = 
    SctOpr(vol, vol, sus; useGpu=useGpu, slv=slv)

"""
    SctOpr(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{ComplexF64}; useGpu::Bool=isa(sus, CuArray), slv::GlaSlv=BiCGStabSolver())

Construct a scattering operator for external interactions between different volumes.

This constructor creates an external scattering operator that describes how electromagnetic fields interact with a material medium between distinct regions. The susceptibility tensor can be provided either as a flat vector (which will be reshaped to match the source volume dimensions) or as a 3-tensor directly. The tensor must match the dimensions of the source volume.

# Arguments
- `trgVol::GlaVol`: The target volume where the field will be computed
- `srcVol::GlaVol`: The source volume containing the sources
- `sus::AbstractArray{ComplexF64}`: The susceptibility tensor, either as a flat vector or a 3-tensor
- `useGpu::Bool=isa(sus, CuArray)`: Whether to use GPU computation. If true, uses GPU acceleration, otherwise uses CPU
- `slv::GlaSlv=BiCGStabSolver()`: The solver to use for solving the linear system

# Returns
- `SctOpr`: The scattering operator
"""
function SctOpr(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{ComplexF64}; useGpu::Bool=isa(sus, CuArray), slv::GlaSlv=BiCGStabSolver())
    invSctOpr = InvSctOpr(trgVol, srcVol, sus; useGpu=useGpu)
    return SctOpr(invSctOpr, slv)
end

"""
    SctOpr(opr::GlaOprVac, sus::AbstractArray{ComplexF64}; useGpu::Bool=isa(sus, CuArray), slv::GlaSlv=BiCGStabSolver())

Construct a scattering operator from a vacuum Green function operator.

This constructor creates a scattering operator that describes how electromagnetic fields interact with a material medium based on a given vacuum Green function operator. The susceptibility tensor can be provided either as a flat vector (which will be reshaped to match the source volume dimensions) or as a 3-tensor directly. The tensor must match the dimensions of the source volume.

# Arguments
- `opr::GlaOprVac`: The vacuum Green function operator to convert into a scattering operator
- `sus::AbstractArray{ComplexF64}`: The susceptibility tensor, either as a flat vector or a 3-tensor
- `useGpu::Bool=isa(sus, CuArray)`: Whether to use GPU computation. If true, uses GPU acceleration, otherwise uses CPU
- `slv::GlaSlv=BiCGStabSolver()`: The solver to use for solving the linear system

# Returns
- `SctOpr`: The scattering operator
"""
function SctOpr(opr::GlaOprVac, sus::AbstractArray{ComplexF64}; slv::GlaSlv=BiCGStabSolver())
    invSctOpr = InvSctOpr(opr, sus)
    return SctOpr(invSctOpr, slv)
end

"""
    SctOpr(opr::GlaOpr)

Construct a scattering operator from a full Green function operator.

# Arguments
- `opr::GlaOpr`: The full Green function operator to convert into a scattering operator

# Returns
- `SctOpr`: The scattering operator
"""
SctOpr(opr::GlaOpr) = opr.sctOpr

"""
    GlaOpr(vol::GlaVol, sus::AbstractArray{ComplexF64}; useGpu::Bool=isa(sus, CuArray), slv::GlaSlv=BiCGStabSolver())

Construct a full Green function operator for self-interactions on a single volume.

This constructor creates a self-interaction full Green function operator that combines the vacuum Green function with the scattering operator to describe electromagnetic interactions in a material medium within a single volume. The susceptibility tensor can be provided either as a flat vector (which will be reshaped to match the volume dimensions) or as a 3-tensor directly. The tensor must match the dimensions of the volume.

# Arguments
- `vol::GlaVol`: The volume to compute the self-interaction for
- `sus::AbstractArray{ComplexF64}`: The susceptibility tensor, either as a flat vector or a 3-tensor
- `useGpu::Bool=isa(sus, CuArray)`: Whether to use GPU computation. If true, uses GPU acceleration, otherwise uses CPU
- `slv::GlaSlv=BiCGStabSolver()`: The solver to use for solving the linear system

# Returns
- `GlaOpr`: The full Green function operator
"""
GlaOpr(vol::GlaVol, sus::AbstractArray{ComplexF64}; useGpu::Bool=isa(sus, CuArray), slv::GlaSlv=BiCGStabSolver()) = 
    GlaOpr(vol, vol, sus; useGpu=useGpu, slv=slv)

"""
    GlaOpr(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{ComplexF64}; useGpu::Bool=isa(sus, CuArray), slv::GlaSlv=BiCGStabSolver())

Construct a full Green function operator for external interactions between different volumes.

This constructor creates an external full Green function operator that combines the vacuum Green function with the scattering operator to describe electromagnetic interactions in a material medium between distinct regions. The susceptibility tensor can be provided either as a flat vector (which will be reshaped to match the source volume dimensions) or as a 3-tensor directly. The tensor must match the dimensions of the source volume.

# Arguments
- `trgVol::GlaVol`: The target volume where the field will be computed
- `srcVol::GlaVol`: The source volume containing the sources
- `sus::AbstractArray{ComplexF64}`: The susceptibility tensor, either as a flat vector or a 3-tensor
- `useGpu::Bool=isa(sus, CuArray)`: Whether to use GPU computation. If true, uses GPU acceleration, otherwise uses CPU
- `slv::GlaSlv=BiCGStabSolver()`: The solver to use for solving the linear system

# Returns
- `GlaOpr`: The full Green function operator
"""
function GlaOpr(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{ComplexF64}; useGpu::Bool=isa(sus, CuArray), slv::GlaSlv=BiCGStabSolver())
    sctOpr = SctOpr(trgVol, srcVol, sus; useGpu=useGpu, slv=slv)
    return GlaOpr(sctOpr)
end

"""
    GlaOpr(opr::GlaOprVac, sus::AbstractArray{ComplexF64}; useGpu::Bool=isa(sus, CuArray), slv::GlaSlv=BiCGStabSolver())

Construct a full Green function operator from a vacuum Green function operator.

This constructor creates a full Green function operator that combines the vacuum Green function with the scattering operator to describe electromagnetic interactions in a material medium based on a given vacuum Green function operator. The susceptibility tensor can be provided either as a flat vector (which will be reshaped to match the source volume dimensions) or as a 3-tensor directly. The tensor must match the dimensions of the source volume.

# Arguments
- `opr::GlaOprVac`: The vacuum Green function operator to convert into a full Green function operator
- `sus::AbstractArray{ComplexF64}`: The susceptibility tensor, either as a flat vector or a 3-tensor
- `useGpu::Bool=isa(sus, CuArray)`: Whether to use GPU computation. If true, uses GPU acceleration, otherwise uses CPU
- `slv::GlaSlv=BiCGStabSolver()`: The solver to use for solving the linear system

# Returns
- `GlaOpr`: The full Green function operator
"""
GlaOpr(opr::GlaOprVac, sus::AbstractArray{ComplexF64}; slv::GlaSlv=BiCGStabSolver()) = GlaOpr(SctOpr(opr, sus; slv=slv))

"""
    GlaOpr(opr::InvSctOpr)

Construct a full Green function operator from an inverse scattering operator.

# Arguments
- `opr::InvSctOpr`: The inverse scattering operator to convert into a full Green function operator

# Returns
- `GlaOpr`: The full Green function operator
"""
GlaOpr(opr::InvSctOpr) = GlaOpr(SctOpr(opr))

function useCpu!(opr::Union{GlaOprVac, AsyGlaOprVac, SymGlaOprVac})
    useCpu!(opr.mem)
    return opr
end

function useGpu!(opr::Union{GlaOprVac, AsyGlaOprVac, SymGlaOprVac})
    useGpu!(opr.mem)
    return opr
end

function useCpu!(opr::MulRegGlaOprVac)
    useCpu!.(opr.oprMat)
    return opr
end

function useGpu!(opr::MulRegGlaOprVac)
    useGpu!.(opr.oprMat)
    return opr
end

function useCpu!(opr::InvSctOpr)
    useCpu!(opr.oprVac)
    opr.sus = Array(opr.sus)
    return opr
end

function useGpu!(opr::InvSctOpr)
    useGpu!(opr.oprVac)
    opr.sus = CuArray(opr.sus)
    return opr
end

function useCpu!(opr::SctOpr)
    useCpu!(opr.invSctOpr)
    return opr
end

function useGpu!(opr::SctOpr)
    useGpu!(opr.invSctOpr)
    return opr
end

function useCpu!(opr::GlaOpr)
    useCpu!(opr.sctOpr)
    return opr
end

function useGpu!(opr::GlaOpr)
    useGpu!(opr.sctOpr)
    return opr
end

GilaVacuum.arrTyp(opr::Union{GlaOprVac, AsyGlaOprVac, SymGlaOprVac}) = arrTyp(opr.mem.cmpInf)
GilaVacuum.arrTyp(opr::MulRegGlaOprVac) = arrTyp(first(opr.oprMat))
GilaVacuum.arrTyp(opr::InvSctOpr) = arrTyp(opr.oprVac)
GilaVacuum.arrTyp(opr::SctOpr) = arrTyp(opr.invSctOpr)
GilaVacuum.arrTyp(opr::GlaOpr) = arrTyp(opr.sctOpr)

"""
    asym(opr::GlaOprVac)

Construct the anti-Hermitian part of a vacuum Green function operator.

# Arguments
- `opr::GlaOprVac`: The vacuum Green function operator to convert into its anti-Hermitian part

# Returns
- `AsyGlaOprVac`: The anti-Hermitian part of the vacuum Green function operator
"""
asym(opr::GlaOprVac) = AsyGlaOprVac(opr)

"""
    sym(opr::GlaOprVac)

Construct the -ermitian part of a vacuum Green function operator.

# Arguments
- `opr::GlaOprVac`: The vacuum Green function operator to convert into its Hermitian part

# Returns
- `SymGlaOprVac`: The Hermitian part of the vacuum Green function operator
"""
sym(opr::GlaOprVac) = SymGlaOprVac(opr)

"""
    isadjoint(opr::AbstractGlaOpr)

Checks if the operator is the adjoint of the Green operator.

# Arguments
- `opr::AbstractGlaOpr`: The operator to check.

# Returns
- `true` if the operator is the adjoint, `false` otherwise.
"""
isadjoint(opr::GlaOprVac) = opr.mem.cmpInf.adjMod
isadjoint(::Union{AsyGlaOprVac, SymGlaOprVac}) = false
isadjoint(opr::MulRegGlaOprVac) = all(isadjoint.(opr.oprMat))
isadjoint(opr::InvSctOpr) = isadjoint(opr.oprVac)
isadjoint(opr::SctOpr) = isadjoint(opr.invSctOpr)
isadjoint(opr::GlaOpr) = isadjoint(opr.sctOpr)

"""
    isselfoperator(opr::AbstractGlaOpr)

Checks if the operator is a self Green operator.

# Arguments
- `opr::AbstractGlaOpr`: The operator to check.

# Returns
- `true` if the operator is a self Green operator, `false` otherwise.
"""
isselfoperator(opr::GlaOprVac) = (opr.mem.srcVol == opr.mem.trgVol) && all(==(0:0), opr.srcMsk) && all(==(0:0), opr.trgMsk)
isselfoperator(::Union{AsyGlaOprVac, SymGlaOprVac}) = true # Both are always self operators
isselfoperator(opr::MulRegGlaOprVac) = all(isselfoperator.(opr.oprMat))
isselfoperator(opr::InvSctOpr) = isselfoperator(opr.oprVac)
isselfoperator(opr::SctOpr) = isselfoperator(opr.invSctOpr)
isselfoperator(opr::GlaOpr) = isselfoperator(opr.sctOpr)

"""
    isexternaloperator(opr::GlaOprVac)

Checks if the operator is an external Green operator.

# Arguments
- `opr::GlaOprVac`: The operator to check.

# Returns
- `true` if the operator is an external Green operator, `false` otherwise.
"""
isexternaloperator(opr::GlaOprVac) = opr.mem.srcVol != opr.mem.trgVol && all(==(0:0), opr.srcMsk) && all(==(0:0), opr.trgMsk)
isexternaloperator(::Union{AsyGlaOprVac, SymGlaOprVac}) = false # Both are always self operators
isexternaloperator(opr::MulRegGlaOprVac) = any(isexternaloperator.(opr.oprMat))
isexternaloperator(opr::InvSctOpr) = isexternaloperator(opr.oprVac)
isexternaloperator(opr::SctOpr) = isexternaloperator(opr.invSctOpr)
isexternaloperator(opr::GlaOpr) = isexternaloperator(opr.sctOpr)

"""
    isoverlappingoperator(opr::GlaOprVac)

Checks if the operator is an overlapping Green operator.

# Arguments
- `opr::GlaOprVac`: The operator to check.

# Returns
- `true` if the operator is an overlapping Green operator, `false` otherwise.
"""
isoverlappingoperator(::AbstractGlaVacOpr) = false # Only the masked routes overlap
isoverlappingoperator(opr::GlaOprVac) = !(isselfoperator(opr) || isexternaloperator(opr))
isoverlappingoperator(opr::MulRegGlaOprVac) = any(isoverlappingoperator.(opr.oprMat))
isoverlappingoperator(opr::InvSctOpr) = isoverlappingoperator(opr.oprVac)
isoverlappingoperator(opr::SctOpr) = isoverlappingoperator(opr.invSctOpr)
isoverlappingoperator(opr::GlaOpr) = isoverlappingoperator(opr.sctOpr)

"""
    isgpu(opr::AbstractGlaOpr)

Checks if the operator is using GPU computation.

# Arguments
- `opr::AbstractGlaOpr`: The operator to check.

# Returns
- `true` if the operator is using GPU computation, `false` otherwise.
"""
# cmpInf itself is the GlaKerOpt; bckEnd(cmpInf) is a KernelAbstractions backend,
# which is never a GlaKerOpt
isgpu(opr::Union{GlaOprVac, AsyGlaOprVac, SymGlaOprVac}) = opr.mem.cmpInf isa GPUKerOpt
isgpu(opr::MulRegGlaOprVac) = all(isgpu.(opr.oprMat))
isgpu(opr::InvSctOpr) = isgpu(opr.oprVac)
isgpu(opr::SctOpr) = isgpu(opr.invSctOpr)
isgpu(opr::GlaOpr) = isgpu(opr.sctOpr)

"""
    setSus!(opr::InvSctOpr, sus::AbstractArray{ComplexF64})

Sets the susceptibility tensor for the inverse scattering operator.

# Arguments
- `opr::InvSctOpr`: The inverse scattering operator to modify.
- `sus::AbstractArray{ComplexF64}`: The new susceptibility tensor, either as a flat vector or a 3-tensor.

# Returns
- The modified operator with the new susceptibility tensor set.
"""
function setSus!(opr::InvSctOpr, sus)
    if opr.oprVac isa GlaCmpOprVac
        opr.sus = _cmpSus(opr.oprVac.srcCvl, sus, isgpu(opr.oprVac))
        return opr
    end
    # Reshape susceptibility if needed and validate size
    if size(sus) != opr.oprVac.mem.srcVol.cel
        throw(ArgumentError("Susceptibility tensor dimensions $(size(sus)) do not match source volume dimensions $(opr.oprVac.mem.srcVol.cel)"))
    end
    opr.sus = rszSus(sus, opr.oprVac.mem.srcVol.cel)
    return opr
end

"""
    setSus!(opr::SctOpr, sus::AbstractArray{ComplexF64})

Sets the susceptibility tensor for the scattering operator.

# Arguments
- `opr::SctOpr`: The scattering operator to modify.
- `sus::AbstractArray{ComplexF64}`: The new susceptibility tensor, either as a flat vector or a 3-tensor.

# Returns
- The modified operator with the new susceptibility tensor set.
"""
function setSus!(opr::SctOpr, sus)
    setSus!(opr.invSctOpr, sus)
    return opr
end

"""
    setSus!(opr::GlaOpr, sus::AbstractArray{ComplexF64})

Sets the susceptibility tensor for the full Green function operator.

# Arguments
- `opr::GlaOpr`: The full Green function operator to modify.
- `sus::AbstractArray{ComplexF64}`: The new susceptibility tensor, either as a flat vector or a 3-tensor.

# Returns
- The modified operator with the new susceptibility tensor set.
"""
function setSus!(opr::GlaOpr, sus)
    setSus!(opr.sctOpr, sus)
    return opr
end

"""
    slv(opr::AbstractGlaOpr)

Returns the solver associated with the operator.

# Arguments
- `opr::AbstractGlaOpr`: The operator for which to get the solver.

# Returns
- The solver used by the operator, which is always a `GlaSlv` instance.
"""
slv(::AbstractGlaVacOpr) = GilaSolvers.BiCGStabSolver() # Default solver
slv(opr::InvSctOpr) = slv(opr.oprVac)
slv(opr::SctOpr) = opr.slv
slv(opr::GlaOpr) = opr.sctOpr.slv

_strKnd(opr::GlaOprVac) = "G₀"
_strKnd(opr::AsyGlaOprVac) = "Asym(G₀)"
_strKnd(opr::SymGlaOprVac) = "Sym(G₀)"
_strKnd(opr::MulRegGlaOprVac) = "multi-region G₀"
_strKnd(opr::InvSctOpr) = "(I - XG₀)"
_strKnd(opr::SctOpr) = "(I - XG₀)⁻¹"
_strKnd(opr::GlaOpr) = "G₀(I - XG₀)⁻¹"

_srcVol(opr::Union{GlaOprVac, AsyGlaOprVac, SymGlaOprVac}) = opr.mem.srcVol
_srcVol(opr::InvSctOpr) = _srcVol(opr.oprVac)
_srcVol(opr::SctOpr) = _srcVol(opr.invSctOpr)
_srcVol(opr::GlaOpr) = _srcVol(opr.sctOpr)
_trgVol(opr::Union{GlaOprVac, AsyGlaOprVac, SymGlaOprVac}) = opr.mem.trgVol
_trgVol(opr::InvSctOpr) = _trgVol(opr.oprVac)
_trgVol(opr::SctOpr) = _trgVol(opr.invSctOpr)
_trgVol(opr::GlaOpr) = _trgVol(opr.sctOpr)

Base.show(io::IO, opr::AbstractGlaOpr) = _shwOpr(io, opr)

#= A scattering operator over a composite volume has no single source volume to
print, so it borrows the layout of the composite vacuum operator instead. =#
function Base.show(io::IO, opr::Union{InvSctOpr, SctOpr, GlaOpr})
    oprVac = GlaOprVac(opr)
    oprVac isa GlaCmpOprVac || return _shwOpr(io, opr)
    isadjoint(opr) && print(io, "Adjoint ")
    print(io, isgpu(opr) ? "GPU " : "CPU ")
    print(io, "composite ", _strKnd(opr))
    print(io, "\n  $(size(opr, 1)) × $(size(opr, 2)) degrees of freedom")
    print(io, "\n  ", oprVac.srcCvl)
end

function _shwOpr(io::IO, opr::AbstractGlaOpr)
    if isadjoint(opr)
        print(io, "Adjoint ")
    end
    if isselfoperator(opr)
        print(io, "Self ")
    elseif isexternaloperator(opr)
        print(io, "External ")
    else
        print(io, "Overlapping ")
    end
    if isgpu(opr)
        print(io, "GPU ")
    else
        print(io, "CPU ")
    end
    print(io, _strKnd(opr))
    print(io, " for ")
    if isselfoperator(opr)
        print(io, "a $(eltype(opr)) (" * join(_srcVol(opr).cel, "×") * ") volume ")
        print(io, "of size (" * join(_srcVol(opr).scl, "×") * ")λ³")
    else
        print(io, "$(eltype(opr)) (" * join(glaSze(opr)[2][1:3], "×") * ") -> (" * join(glaSze(opr)[1][1:3], "×") * ") volumes ")
        print(io, "of sizes (" * join(_srcVol(opr).scl, "×") * ")λ³ -> (" * join(_trgVol(opr).scl, "×") * ")λ³")
        if isexternaloperator(opr)
            print(io, " with center separation (" * join(_trgVol(opr).org .- _srcVol(opr).org, ", ") * ")λ")
        end
    end
end
Base.show(io::IO, ::MIME"text/plain", opr::AbstractGlaOpr) = show(io, opr)
function Base.show(io::IO, opr::MulRegGlaOprVac)
    m, n = size(opr.oprMat)
    isadjoint(opr) && print(io, "Adjoint ")
    print(io, isgpu(opr) ? "GPU " : "CPU ")
    print(io, "multi-region G₀ ")
    print(io, "($m target", m == 1 ? "" : "s", " × $n source", n == 1 ? "" : "s", ")")

    trgVols = [_trgVol(opr.oprMat[i, 1]) for i in 1:m]
    srcVols = [_srcVol(opr.oprMat[1, j]) for j in 1:n]

    _fmtVol(v) = "(" * join(v.cel, "×") * ") cells, (" * join(v.scl, "×") * ")λ³"

    println(io)
    print(io, "  targets:")
    for (i, v) in enumerate(trgVols)
        print(io, "\n    [$i] ", _fmtVol(v))
    end
    println(io)
    print(io, "  sources:")
    for (j, v) in enumerate(srcVols)
        print(io, "\n    [$j] ", _fmtVol(v))
    end
end
Base.show(io::IO, ::MIME"text/plain", opr::MulRegGlaOprVac) = show(io, opr)

include("glaLinAlg.jl")
include("glaCmpOpr.jl")

#= An overlapping operator holds the union volume in its memory, so the masks are
the only record of the two sub-volumes and have to be written alongside it. =#
function Serialization.serialize(io::IO, opr::GlaOprVac)
    serialize(io, opr.mem)
    serialize(io, opr.srcMsk)
    serialize(io, opr.trgMsk)
end
function Serialization.deserialize(io::IO, ::Type{GlaOprVac})
    mem = deserialize(io, GlaVacOprMem)
    srcMsk = deserialize(io)
    trgMsk = deserialize(io)
    return GlaOprVac(mem, srcMsk, trgMsk)
end
Serialization.serialize(io::IO, opr::AsyGlaOprVac) = serialize(io, opr.mem)
Serialization.deserialize(io::IO, ::Type{AsyGlaOprVac}) = AsyGlaOprVac(deserialize(io, GlaVacOprMem))
Serialization.serialize(io::IO, opr::SymGlaOprVac) = serialize(io, opr.mem)
Serialization.deserialize(io::IO, ::Type{SymGlaOprVac}) = SymGlaOprVac(deserialize(io, GlaVacOprMem))
Serialization.serialize(io::IO, opr::MulRegGlaOprVac) = serialize(io, opr.oprMat)
# The blocks go through the generic serializer, which rebuilds their FFTW plans
# on load, see vacuum/glaVacOprMem.jl
Serialization.deserialize(io::IO, ::Type{MulRegGlaOprVac}) = MulRegGlaOprVac(deserialize(io))
function Serialization.serialize(io::IO, opr::GlaSndOprVac)
    serialize(io, opr.opr)
    serialize(io, opr.trgRat)
    serialize(io, opr.srcRat)
    serialize(io, opr.wgt)
end
function Serialization.deserialize(io::IO, ::Type{GlaSndOprVac})
    innOpr = deserialize(io, GlaOprVac)
    trgRat = deserialize(io)
    srcRat = deserialize(io)
    wgt = deserialize(io)
    return GlaSndOprVac(innOpr, trgRat, srcRat, wgt)
end
function Serialization.serialize(io::IO, opr::GlaCmpOprVac)
    serialize(io, opr.trgCvl)
    serialize(io, opr.srcCvl)
    # The blocks go through the generic serializer, as for MulRegGlaOprVac above
    serialize(io, opr.blkMat)
end
function Serialization.deserialize(io::IO, ::Type{GlaCmpOprVac})
    trgCvl = deserialize(io)
    srcCvl = deserialize(io)
    blkMat = deserialize(io)
    return GlaCmpOprVac(trgCvl, srcCvl, blkMat)
end
#= The written blocks already carry the Fourier coefficients of the part, so the
raw constructor is the one to read them back with. =#
Serialization.serialize(io::IO, opr::AsyGlaCmpOprVac) = serialize(io, opr.opr)
Serialization.deserialize(io::IO, ::Type{AsyGlaCmpOprVac}) =
    AsyGlaCmpOprVac(deserialize(io, GlaCmpOprVac), Val(:raw))
Serialization.serialize(io::IO, opr::SymGlaCmpOprVac) = serialize(io, opr.opr)
Serialization.deserialize(io::IO, ::Type{SymGlaCmpOprVac}) =
    SymGlaCmpOprVac(deserialize(io, GlaCmpOprVac), Val(:raw))
function Serialization.serialize(io::IO, opr::InvSctOpr)
    #= The vacuum operator is written by whichever method its runtime type
    selects, so its type leads the payload and picks the reader. =#
    serialize(io, typeof(opr.oprVac))
    serialize(io, opr.oprVac)
    sus = opr.sus
    if sus isa CuArray
        sus = Array(sus) # Convert to CPU array for serialization
    end
    serialize(io, sus)
end
function Serialization.deserialize(io::IO, ::Type{InvSctOpr})
    vacTyp = deserialize(io)
    oprVac = deserialize(io, vacTyp)
    sus = deserialize(io)
    return InvSctOpr(oprVac, sus)
end
function Serialization.serialize(io::IO, opr::SctOpr)
    serialize(io, opr.invSctOpr)
    serialize(io, opr.slv)
end
function Serialization.deserialize(io::IO, ::Type{SctOpr})
    invSctOpr = deserialize(io, InvSctOpr)
    slv = deserialize(io)
    return SctOpr(invSctOpr, slv)
end
Serialization.serialize(io::IO, opr::GlaOpr) = serialize(io, opr.sctOpr)
Serialization.deserialize(io::IO, ::Type{GlaOpr}) = GlaOpr(deserialize(io, SctOpr))

end # module
