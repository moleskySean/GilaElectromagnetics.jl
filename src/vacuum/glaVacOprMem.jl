using Base.Threads
using AbstractFFTs
using FFTW
using CUDA
using LinearAlgebra
using Serialization
using ..GilaVolumes

"""
    GlaVacOprMem

Memory structure for the vacuum Green function operator. This structure holds all the memory needed for computing the vacuum Green function operator. The structure is designed to minimize memory allocation during computation. The Fourier transform plans are used to efficiently compute the Green function integrals. The phase information is used to handle the splitting of Fourier transforms.

# Fields
- `cmpInf::GlaKerOpt`: Computation information, settings and kernel options, see `GlaKerOpt`
- `trgVol::GlaVol`: Target volume of Green function
- `srcVol::GlaVol`: Source volume of Green function
- `mixInf::GlaExtInf`: Information for matching source and target grids, see `GlaExtInf`
- `dimInf::NTuple{3,Integer}`: Dimension information for Green function volumes, host side
- `egoFur::AbstractVector{<:AbstractArray{ComplexF64}}`: Unique Fourier transform data for circulant Green function
- `fftPlnFwd::AbstractVector{<:AbstractFFTs.Plan}`: Forward Fourier transform plans
- `fftPlnRev::AbstractVector{<:AbstractFFTs.Plan}`: Reverse Fourier transform plans
- `adjFftPlnFwd::AbstractVector{<:AbstractFFTs.Plan}`: Forward Fourier transform plans for adjoint
- `adjFftPlnRev::AbstractVector{<:AbstractFFTs.Plan}`: Reverse Fourier transform plans for adjoint
- `phzInf::AbstractVector{<:AbstractArray{ComplexF64}}`: Phase vector for splitting Fourier transforms

# Notes
- This structure holds all the memory needed for computing the vacuum Green function operator
- The structure is designed to minimize memory allocation during computation
- The Fourier transform plans are used to efficiently compute the Green function integrals
- The phase information is used to handle the splitting of Fourier transforms
"""
mutable struct GlaVacOprMem
    cmpInf::GlaKerOpt
    trgVol::GlaVol
    srcVol::GlaVol
    mixInf::GlaExtInf
    dimInf::NTuple{3,Integer} 
    egoFur::AbstractVector{<:AbstractArray{ComplexF64}}
    fftPlnFwd::AbstractVector{<:AbstractFFTs.Plan}
    fftPlnRev::AbstractVector{<:AbstractFFTs.Plan}
    adjFftPlnFwd::AbstractVector{<:AbstractFFTs.Plan}
    adjFftPlnRev::AbstractVector{<:AbstractFFTs.Plan}
    phzInf::AbstractVector{<:AbstractArray{ComplexF64}}
end
#=
If intConTest.jl was failed the default intOrd used in the simplified constructor
may not be sufficient to insure that all integral values are properly converged.
It may be prudent to create the associated GlaVacOprMem with higher order. 
=#

"""
    GlaVacOprMem(cmpInf::GlaKerOpt, egoFur::AbstractVector{<:AbstractArray{ComplexF64}}, trgVol::GlaVol, srcVol::GlaVol=trgVol)

Prepare memory for Green function operator. When called with a single GlaVol, 
or identical source and target volumes, yields the self construction. 

# Arguments
- `cmpInf::GlaKerOpt`: Computation information, settings and kernel options, see `GlaKerOpt`.
- `egoFur::AbstractVector{<:AbstractArray{ComplexF64}}`: Unique Fourier transform data for Green function.
- `trgVol::GlaVol`: Target volume or self volume definition.
- `srcVol::Union{Nothing,GlaVol}=nothing`: Source volume for external construction. Nothing will generate the self construction.

# Returns
- `GlaVacOprMem`: The memory structure for the Green function operator.
"""
function GlaVacOprMem(cmpInf::GlaKerOpt, egoFur::AbstractVector{<:AbstractArray{ComplexF64}}, trgVol::GlaVol, srcVol::GlaVol=trgVol)
    mixInf = genEveExtInf(trgVol, srcVol)
    # branching depth of multiplication
    lvl = 3
    # number of multiplication branches     
    eoDim = 2^lvl
    # verify that egoFur contains only numeric values
    for eoItr ∈ eachindex(1:eoDim)
        if !all(isfinite, egoFur[eoItr])
            throw(ArgumentError("Fourier information contains non-numeric values."))
        end
    end
    return glaOprPrp(egoFur, trgVol, srcVol, mixInf, cmpInf)
end

include("glaVacOprMemGen.jl") # For genEgoCrc!

"""
    GlaVacOprMem(cmpInf::GlaKerOpt, trgVol::GlaVol, srcVol::GlaVol=trgVol)

Prepare memory for Green function operator. Automatically computes the Fourier transform data.

# Arguments
- `cmpInf::GlaKerOpt`: Computation information, settings and kernel options, see `GlaKerOpt`.
- `trgVol::GlaVol`: Target volume or self volume definition.
- `srcVol::Union{Nothing,GlaVol}=nothing`: Source volume for external construction. Nothing will generate the self construction.

# Returns
- `GlaVacOprMem`: The memory structure for the Green function operator.
"""
function GlaVacOprMem(cmpInf::GlaKerOpt, trgVol::GlaVol, srcVol::GlaVol=trgVol)
    mixInf = genEveExtInf(trgVol, srcVol)

    # total cells in circulant
    totCelCrc = mixInf.trgCel .+ mixInf.srcCel
    # total number of target and source partitions
    totParTrg = prod(mixInf.trgDiv)
    totParSrc = prod(mixInf.srcDiv)

    # memory for circulant green function vector. The integral kernels want
    # the 3×3 tensor block contiguous per cell, so the fill keeps this layout
    egoCrc = Array{ComplexF64}(undef, 3, 3, totCelCrc..., totParSrc, totParTrg)
    genEgoCrc!(egoCrc, trgVol, srcVol, mixInf, cmpInf)
    # verify that egoCrc contains numeric values
    if !all(isfinite, egoCrc)
        throw(ArgumentError("Computed circulant contains non-numeric values."))
    end
    # gather the six unique tensor components (real space symmetry under
    # transposition)---entries are xx, yy, zz, xy, xz, yz---into a cells-first
    # array so that a single batched plan transforms every component and
    # partition pair
    egoCrcCmp = Array{ComplexF64}(undef, totCelCrc..., 6, totParSrc, totParTrg)
    gthEgoCmp!(egoCrcCmp, egoCrc)
    # allow the 9-component fill array to be reclaimed before the transform
    egoCrc = nothing
    # number of unique elements in each cartesian index for a branch
    truInf = Array{Int}(undef, 3)
    for dirItr ∈ eachindex(1:3)
        # row and column entries are symmetric or anti-symmetric
        if mixInf.trgCel[dirItr] == mixInf.srcCel[dirItr] && all(mixInf.srcDiv .== 1) && all(mixInf.trgDiv .== 1) && trgVol.org[dirItr] == srcVol.org[dirItr]
            # store only necessary information
            truInf[dirItr] = max(Integer(ceil(mixInf.trgCel[dirItr] / 2)) + iseven(mixInf.trgCel[dirItr]), 2)
            continue
        end
        # genVolEve enforces that number of cells is even
        truInf[dirItr] = totCelCrc[dirItr] ÷ 2
    end
    # Fourier transform of the circulant and even/odd branch extraction on the
    # backend selected by cmpInf
    egoFur = genEgoFur(egoCrcCmp, truInf, cmpInf)
    return GlaVacOprMem(cmpInf, egoFur, trgVol, srcVol)
end

# (row, col) position in the 3×3 tensor block of each of the six unique
# components, in the xx, yy, zz, xy, xz, yz storage order of egoFur
const egoCmpPos = ((1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3))

#=
Gather the six unique tensor components of the circulant from the
(3, 3, cells...) fill layout into the cells-first component layout
(cells..., 6, partition pairs) used by the Fourier stage.
=#
function gthEgoCmp!(egoCrcCmp::AbstractArray{ComplexF64,6}, egoCrc::AbstractArray{ComplexF64,7})
    gthItr = CartesianIndices((size(egoCrc, 5), 6, size(egoCrc, 6), size(egoCrc, 7)))
    @threads for gthInd ∈ gthItr
        zItr, cmpItr, srcItr, trgItr = Tuple(gthInd)
        rowItr, colItr = egoCmpPos[cmpItr]
        @views egoCrcCmp[:, :, zItr, cmpItr, srcItr, trgItr] .= egoCrc[rowItr, colItr, :, :, zItr, srcItr, trgItr]
    end
    return nothing
end

#=
Extract one even/odd Fourier branch: copy the strided branch corner of the
transformed circulant into the branch array. Views plus broadcast express the
copy once for both Array (CPU) and CuArray (GPU) backends. The trailing
(component / partition) dimensions of the two arrays must agree. When a
dimension of the circulant holds a single unique coefficient the length-1 view
broadcasts up to the minimum branch extent of 2.
=#
function extEgoBrn!(egoFurBrn::AbstractArray{ComplexF64}, egoFurPrp::AbstractArray{ComplexF64}, eoItr::Integer)
    totCelCrc = size(egoFurPrp)[1:3]
    truInf = size(egoFurBrn)[1:3]
    # even/odd offsets of the three axes: eoItr bit 2 → x, bit 1 → y, bit 0 → z
    # first division is along smallest stride -> largest binary division
    eoOff = (mod(div(eoItr, 4), 2), mod(div(eoItr, 2), 2), mod(eoItr, 2))
    brnRng = ntuple(dirItr -> (1 + eoOff[dirItr]):2:(totCelCrc[dirItr] - 1 + eoOff[dirItr]), 3)
    # keep only the unique leading corner (self operators store roughly half)
    truRng = ntuple(dirItr -> length(brnRng[dirItr]) > truInf[dirItr] ?
        brnRng[dirItr][1:truInf[dirItr]] : brnRng[dirItr], 3)
    trlCln = ntuple(_ -> Colon(), ndims(egoFurPrp) - 3)
    egoFurBrn .= @view egoFurPrp[truRng..., trlCln...]
    return nothing
end

#=
Fourier stage of operator creation: transform the gathered six-component
circulant over its three cell dimensions with one batched in-place plan (all
6 × totParSrc × totParTrg blocks in a single plan execution) and extract the
eight even/odd branches. Dispatch on cmpInf selects the backend.
=#
function genEgoFur(egoCrcCmp::Array{ComplexF64,6}, truInf::AbstractVector{<:Integer}, cmpInf::CPUKerOpt)
    ddDim, totParSrc, totParTrg = size(egoCrcCmp)[4:6]
    # thread the creation-time FFT, restoring the global FFTW state afterwards
    fftwThr = FFTW.get_num_threads()
    FFTW.set_num_threads(Threads.nthreads())
    try
        plan_fft!(egoCrcCmp, 1:3) * egoCrcCmp
    finally
        FFTW.set_num_threads(fftwThr)
    end
    # verify integrity of Fourier transform data
    if !all(isfinite, egoCrcCmp)
        throw(ArgumentError("Fourier transform of circulant contains non-numeric values."))
    end
    # final Fourier coefficients for a given branch
    egoFur = Array{Array{ComplexF64}}(undef, 8)
    # only one eighth of the green function is unique; the eight branch
    # extractions are independent
    @threads for eoItr ∈ 0:7
        egoFurBrn = Array{ComplexF64}(undef, truInf..., ddDim, totParSrc, totParTrg)
        extEgoBrn!(egoFurBrn, egoCrcCmp, eoItr)
        egoFur[eoItr + 1] = egoFurBrn
    end
    return egoFur
end

#=
GPU Fourier stage: upload the gathered circulant once and run the transform
(CUFFT) and branch extraction on device. The branches already end up as
CuArrays, so this removes the former big host → device copy of the Fourier
coefficients rather than adding transfers. When the full circulant does not
fit on device next to the branches (large multi-partition external operators),
fall back to batching per partition pair with a single reused plan.
=#
function genEgoFur(egoCrcCmp::Array{ComplexF64,6}, truInf::AbstractVector{<:Integer}, cmpInf::GPUKerOpt)
    ddDim, totParSrc, totParTrg = size(egoCrcCmp)[4:6]
    egoFur = Array{CuArray{ComplexF64}}(undef, 8)
    for eoItr ∈ 0:7
        egoFur[eoItr + 1] = CuArray{ComplexF64}(undef, truInf..., ddDim, totParSrc, totParTrg)
    end
    # leave headroom: the transform itself needs device workspace
    if sizeof(egoCrcCmp) <= 3 * (CUDA.available_memory() ÷ 4)
        egoFurPrp = CuArray(egoCrcCmp)
        plan_fft!(egoFurPrp, 1:3) * egoFurPrp
        # verify integrity of Fourier transform data
        if !all(isfinite, egoFurPrp)
            throw(ArgumentError("Fourier transform of circulant contains non-numeric values."))
        end
        for eoItr ∈ 0:7
            extEgoBrn!(egoFur[eoItr + 1], egoFurPrp, eoItr)
        end
        CUDA.unsafe_free!(egoFurPrp)
    else
        # one slab and one plan reused for every partition pair
        slbDev = CuArray{ComplexF64}(undef, size(egoCrcCmp)[1:4])
        slbPln = plan_fft!(slbDev, 1:3)
        slbLen = length(slbDev)
        egoCrcVec = vec(egoCrcCmp)
        for parItr ∈ 0:(totParSrc * totParTrg - 1)
            srcItr = mod(parItr, totParSrc) + 1
            trgItr = div(parItr, totParSrc) + 1
            copyto!(slbDev, 1, egoCrcVec, 1 + parItr * slbLen, slbLen)
            slbPln * slbDev
            # verify integrity of Fourier transform data
            if !all(isfinite, slbDev)
                throw(ArgumentError("Fourier transform of circulant contains non-numeric values."))
            end
            for eoItr ∈ 0:7
                extEgoBrn!(view(egoFur[eoItr + 1], :, :, :, :, srcItr, trgItr), slbDev, eoItr)
            end
        end
        CUDA.unsafe_free!(slbDev)
    end
    return egoFur
end

# Create Fourier transform plans
function fftPlnGen(fwdSze::NTuple, revSze::NTuple, dir::Int, ::CPUKerOpt)
    # Fourier transform planning area
    fftWrkFwd = Array{ComplexF64}(undef, fwdSze...)
    fftWrkRev = Array{ComplexF64}(undef, revSze...)
    # create Fourier transform plans
    fftPlnFwd = plan_fft!(fftWrkFwd, [dir]; flags=FFTW.MEASURE)
    fftPlnRev = plan_ifft!(fftWrkRev, [dir]; flags=FFTW.MEASURE)
    adjFftPlnFwd = plan_fft!(fftWrkRev, [dir]; flags=FFTW.MEASURE)
    adjFftPlnRev = plan_ifft!(fftWrkFwd, [dir]; flags=FFTW.MEASURE)
    return fftPlnFwd, fftPlnRev, adjFftPlnFwd, adjFftPlnRev
end

function fftPlnGen(fwdSze::NTuple, revSze::NTuple, dir::Int, ::GPUKerOpt)
    # Fourier transform planning area
    fftWrkFwdDev = CuArray{ComplexF64}(undef, fwdSze...)
    fftWrkRevDev = CuArray{ComplexF64}(undef, revSze...)
    # create Fourier transform plans
    fftPlnFwdDev = plan_fft!(fftWrkFwdDev, [dir])
    fftPlnRevDev = plan_ifft!(fftWrkRevDev, [dir])
    adjFftPlnFwdDev = plan_fft!(fftWrkRevDev, [dir])
    adjFftPlnRevDev = plan_ifft!(fftWrkFwdDev, [dir])
    return fftPlnFwdDev, fftPlnRevDev, adjFftPlnFwdDev, adjFftPlnRevDev
end

# Memory preparation sub-protocol
function glaOprPrp(egoFur::AbstractVector{<:AbstractArray{ComplexF64}}, trgVol::GlaVol, srcVol::GlaVol, mixInf::GlaExtInf, cmpInf::GlaKerOpt)
    # number of embedding levels---dimensionality of ambient space
    lvls = 3
    # operator dimensions---unique vector information does not typically 
    # match operator size for distinct source and target volumes
    # sum of source and target volumes being divisible by 2 is guaranteed by 
    # genVolEve in GlaVacOprMem
    brnSze = div.(mixInf.trgCel .+ mixInf.srcCel, 2)
    phzInf = Array{arrTyp(cmpInf)}(undef, lvls)
    # Fourier transform plans
    fftPlnFwd = Array{AbstractFFTs.Plan}(undef, lvls)
    fftPlnRev = Array{AbstractFFTs.Plan}(undef, lvls)
    adjFftPlnFwd = Array{AbstractFFTs.Plan}(undef, lvls)
    adjFftPlnRev = Array{AbstractFFTs.Plan}(undef, lvls)

    # initialize Fourier transform plans
    for dir ∈ eachindex(1:lvls)
        # size of vector changes throughout application for external Green 
        vecSzeFwd = ntuple(x -> x <= dir ? brnSze[x] : mixInf.srcCel[x], 3)
        vecSzeRev = ntuple(x -> x > dir ? mixInf.trgCel[x] : brnSze[x], 3)
        fwdSze = (vecSzeFwd..., lvls, prod(mixInf.srcDiv))
        revSze = (vecSzeRev..., lvls, prod(mixInf.trgDiv))
        fftPlnFwd[dir], fftPlnRev[dir], adjFftPlnFwd[dir], adjFftPlnRev[dir] = fftPlnGen(fwdSze, revSze, dir, cmpInf)
    end
    # phase transformations (internal for block Toeplitz transformations)
    for itr ∈ eachindex(1:lvls)
        # allows calculation odd coefficient numbers
        phzInfHst = ComplexF64.([cispi(-k / brnSze[itr]) for k ∈ 0:(brnSze[itr] - 1)])
        phzInf[itr] = similar(first(egoFur), brnSze[itr])
        copyto!(phzInf[itr], phzInfHst)
    end
    return GlaVacOprMem(cmpInf, trgVol, srcVol, mixInf, brnSze, egoFur, fftPlnFwd, fftPlnRev, adjFftPlnFwd, adjFftPlnRev, phzInf)
end

isadjoint(vacOprMem::GlaVacOprMem) = adjMod(vacOprMem.cmpInf)

# deepcopy must regenerate FFTW plans rather than copying the raw C pointers.
# Two Julia plan objects that share the same C pointer both register a finalizer
# calling fftw_destroy_plan; when either is GC'd the other becomes a dangling
# pointer, causing a segfault on the next plan execution.
#
# FFTW plans are always created for the non-adjoint (original) volume orientation.
# If the mem is currently in adjoint state the volumes are swapped back before
# plan creation, then the adjoint state is re-applied to the new mem.
function Base.deepcopy_internal(mem::GlaVacOprMem, stackdict::IdDict)
    haskey(stackdict, mem) && return stackdict[mem]::GlaVacOprMem
    egoFur = deepcopy(mem.egoFur) # data arrays are safe to deepcopy
    cmpInf = deepcopy(mem.cmpInf) # mutable struct
    if isadjoint(mem)
        # Volumes are currently swapped relative to plan creation order.
        # Swap back so that GlaVacOprMem creates plans for the original orientation.
        cmpInf.adjMod = false
        new_mem = GlaVacOprMem(cmpInf, egoFur, mem.srcVol, mem.trgVol)
        # Re-apply the adjoint state (swap volumes, update mixInf, set adjMod).
        new_mem.trgVol, new_mem.srcVol = new_mem.srcVol, new_mem.trgVol
        new_mem.mixInf = GlaExtInf(new_mem.trgVol, new_mem.srcVol)
        new_mem.cmpInf.adjMod = true
    else
        new_mem = GlaVacOprMem(cmpInf, egoFur, mem.trgVol, mem.srcVol)
    end
    stackdict[mem] = new_mem
    return new_mem
end

function useCpu!(mem::GlaVacOprMem)
    if mem.cmpInf isa CPUKerOpt
        return
    end

    # Convert to CPU
    mem.egoFur = collect(map(Array, mem.egoFur))
    mem.cmpInf = useCpu(mem.cmpInf)
    fwdPln = Array{AbstractFFTs.Plan}(undef, 3)
    revPln = Array{AbstractFFTs.Plan}(undef, 3)
    ajdFwdPln = Array{AbstractFFTs.Plan}(undef, 3)
    ajdRevPln = Array{AbstractFFTs.Plan}(undef, 3)
    for dir in 1:3
        pln = fftPlnGen(size(mem.fftPlnFwd[dir]), size(mem.fftPlnRev[dir]), dir, CPUKerOpt())
        fwdPln[dir] = pln[1]
        revPln[dir] = pln[2]
        ajdFwdPln[dir] = pln[3]
        ajdRevPln[dir] = pln[4]
    end
    mem.fftPlnFwd = fwdPln
    mem.fftPlnRev = revPln
    mem.adjFftPlnFwd = ajdFwdPln
    mem.adjFftPlnRev = ajdRevPln
    mem.phzInf = collect(map(Array, mem.phzInf))

    return mem
end

function useGpu!(mem::GlaVacOprMem)
    if mem.cmpInf isa GPUKerOpt
        return
    end

    # Convert to GPU
    mem.egoFur = collect(map(CuArray, mem.egoFur))
    mem.cmpInf = useGpu(mem.cmpInf)
    fwdPln = Array{AbstractFFTs.Plan}(undef, 3)
    revPln = Array{AbstractFFTs.Plan}(undef, 3)
    ajdFwdPln = Array{AbstractFFTs.Plan}(undef, 3)
    ajdRevPln = Array{AbstractFFTs.Plan}(undef, 3)
    for dir in 1:3
        pln = fftPlnGen(size(mem.fftPlnFwd[dir]), size(mem.fftPlnRev[dir]), dir, GPUKerOpt())
        fwdPln[dir] = pln[1]
        revPln[dir] = pln[2]
        ajdFwdPln[dir] = pln[3]
        ajdRevPln[dir] = pln[4]
    end
    mem.fftPlnFwd = fwdPln
    mem.fftPlnRev = revPln
    mem.adjFftPlnFwd = ajdFwdPln
    mem.adjFftPlnRev = ajdRevPln
    mem.phzInf = collect(map(CuArray, mem.phzInf))

    return mem
end

# As with deepcopy above, serialization must regenerate the FFTW plans rather
# than write out the raw C pointers, which are only meaningful inside the
# process that created them. Only the Fourier data, volumes, and kernel options
# are written, and glaOprPrp rebuilds the plans on load. These methods cover
# mems reached through the generic serializer (struct fields, array elements);
# the io::IO methods below are the top level format used for preload files.
function Serialization.serialize(s::AbstractSerializer, mem::GlaVacOprMem)
    Serialization.serialize_type(s, GlaVacOprMem)
    serialize(s, mem.cmpInf isa GPUKerOpt ? collect(map(Array, mem.egoFur)) : mem.egoFur)
    cmpInf = useCpu(mem.cmpInf)
    serialize(s, cmpInf.frqPhz)
    serialize(s, cmpInf.intOrd)
    serialize(s, cmpInf.adjMod)
    serialize(s, mem.trgVol)
    serialize(s, mem.srcVol)
    serialize(s, mem.mixInf)
end

function Serialization.deserialize(s::AbstractSerializer, ::Type{GlaVacOprMem})
    egoFur = deserialize(s)
    frqPhz = deserialize(s)
    intOrd = deserialize(s)
    adjMod = deserialize(s)
    trgVol = deserialize(s)
    srcVol = deserialize(s)
    mixInf = deserialize(s)
    return glaOprPrp(egoFur, trgVol, srcVol, mixInf, CPUKerOpt(frqPhz, intOrd, adjMod, CPU()))
end

function Serialization.serialize(io::IO, mem::GlaVacOprMem)
    wasGpu = false
    if mem.cmpInf isa GPUKerOpt
        wasGpu = true
        useCpu!(mem) # Convert to CPU for serialization
    end
    serialize(io, mem.egoFur)
    serialize(io, mem.cmpInf)
    serialize(io, mem.trgVol)
    serialize(io, mem.srcVol)
    serialize(io, mem.mixInf)

    if wasGpu
        useGpu!(mem) # Convert back to GPU after serialization
    end
end

function Serialization.deserialize(io::IO, ::Type{GlaVacOprMem})
    egoFur = deserialize(io)
    cmpInf = deserialize(io, CPUKerOpt)
    trgVol = deserialize(io)
    srcVol = deserialize(io)
    mixInf = deserialize(io)
     
    # Reconstruct the full operator
    return glaOprPrp(egoFur, trgVol, srcVol, mixInf, cmpInf)
end
