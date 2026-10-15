using Base.Threads
using AbstractFFTs
using FFTW
using Adapt
using KernelAbstractions
import VkFFT
using LinearAlgebra
using Serialization
using ..GilaVolumes
using ..GilaVolumes: _ovrLap, _lwrEdg

"""
    GlaVacOprMem{T<:AbstractFloat}

Everything a vacuum operator needs to act: the options `cmpInf`, the two volumes,
the grid matching `mixInf`, the Fourier coefficients `egoFur` of the circulant
kernel (generated at `genPrc(cmpInf)`, rounded once to `Complex{T}`), the FFT
plans of the operator and of its adjoint, which `adjoint!` swaps, and the phases `phzInf` of the even/odd
branch split at each level.
"""
mutable struct GlaVacOprMem{T<:AbstractFloat}
    cmpInf::GlaKerOpt{T}
    trgVol::GlaVol
    srcVol::GlaVol
    mixInf::GlaExtInf
    brnSze::NTuple{3,Int}
    egoFur::AbstractVector{<:AbstractArray{Complex{T}}}
    fftPlnFwd::AbstractVector{<:AbstractFFTs.Plan}
    fftPlnRev::AbstractVector{<:AbstractFFTs.Plan}
    adjFftPlnFwd::AbstractVector{<:AbstractFFTs.Plan}
    adjFftPlnRev::AbstractVector{<:AbstractFFTs.Plan}
    phzInf::AbstractVector{<:AbstractArray{Complex{T}}}
end
"""
    GlaVacOprMem(cmpInf::GlaKerOpt{T}, egoFur::AbstractVector{<:AbstractArray{Complex{T}}}, trgVol::GlaVol, srcVol::GlaVol=trgVol)

Operator memory around precomputed Fourier coefficients `egoFur`, which must be
finite. One volume gives the self construction.
"""
function GlaVacOprMem(cmpInf::GlaKerOpt{T}, egoFur::AbstractVector{<:AbstractArray{Complex{T}}}, trgVol::GlaVol, srcVol::GlaVol=trgVol) where T<:AbstractFloat
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
    GlaVacOprMem(cmpInf::GlaKerOpt{T}, trgVol::GlaVol, srcVol::GlaVol=trgVol; shpCch, prxWrn, hrmPrt, blk)

Generate the Fourier coefficients and the operator memory. One volume gives the
self construction.

# Keywords
- `shpCch::Bool=false`: cache the far-field geometry table of each cell shape in a
  scratch space; a table is 16-19 MB and takes 12-16 s to build against 0.6-0.7 s to read
- `prxWrn::Bool=true`: warn when separated volumes are closer than a third of a wavelength
- `hrmPrt=nothing`: keep only the entry by entry real (`:sym`) or imaginary (`:asy`)
  part of the kernel, as `SymGlaOprVac` and `AsyGlaOprVac` do
- `blk=nothing`: the Bloch block `(H, A)` blcCmp gave a pair of composite regions
"""
function GlaVacOprMem(cmpInf::GlaKerOpt{T}, trgVol::GlaVol, srcVol::GlaVol=trgVol; shpCch::Bool = false, prxWrn::Bool = true, hrmPrt::Union{Nothing,Symbol} = nothing, blk = nothing) where T<:AbstractFloat
    hrmPrt ∈ (nothing, :sym, :asy) || throw(ArgumentError("hrmPrt is :sym, :asy or nothing, got $(repr(hrmPrt))."))
    prxWrn && prxChk(frqPhz(cmpInf), trgVol, srcVol)
    mixInf = genEveExtInf(trgVol, srcVol)
    # a Bloch kernel is not complex symmetric: genEgoSlf! takes its parts H and A from the fill
    cmpInf.blc === nothing || (cmpInf.blc = merge(cmpInf.blc, (; hrmPrt)))

    # total cells in circulant
    totCelCrc = mixInf.trgCel .+ mixInf.srcCel
    # total number of target and source partitions
    totParTrg = prod(mixInf.trgDiv)
    totParSrc = prod(mixInf.srcDiv)

    # memory for circulant green function vector. The integral kernels want
    # the 3×3 tensor block contiguous per cell, so the fill keeps this layout
    genCpx = Complex{genPrc(cmpInf)}
    egoCrc = Array{genCpx}(undef, 3, 3, totCelCrc..., totParSrc, totParTrg)
    genEgoCrc!(egoCrc, trgVol, srcVol, mixInf, cmpInf; shpCch, blk)
    # verify that egoCrc contains numeric values
    if !all(isfinite, egoCrc)
        throw(ArgumentError("Computed circulant contains non-numeric values."))
    end
    # gather the six unique tensor components (real space symmetry under
    # transposition)---entries are xx, yy, zz, xy, xz, yz---into a cells-first
    # array so that a single batched plan transforms every component and
    # partition pair
    egoCrcCmp = Array{genCpx}(undef, totCelCrc..., 6, totParSrc, totParTrg)
    gthEgoCmp!(egoCrcCmp, egoCrc)
    #= The Hermitian and anti-Hermitian parts are the entry by entry real and
    imaginary parts of the kernel, taken before the transform rather than after:
    imag(fft(g)) leaves ‖imag g‖ behind a subtraction of two terms the size of
    ‖real g‖, which costs eps * ‖real g‖ / ‖imag g‖ and grows as (k cell)^-3. =#
    if cmpInf.blc === nothing
        hrmPrt === :sym && (egoCrcCmp .= complex.(real.(egoCrcCmp)))
        hrmPrt === :asy && (egoCrcCmp .= complex.(imag.(egoCrcCmp)))
    end
    # allow the 9-component fill array to be reclaimed before the transform
    egoCrc = nothing
    # number of unique elements in each cartesian index for a branch
    truInf = Array{Int}(undef, 3)
    for dirItr ∈ eachindex(1:3)
        # row and column entries are symmetric or anti-symmetric, except along a periodic axis
        if mixInf.trgCel[dirItr] == mixInf.srcCel[dirItr] && all(mixInf.srcDiv .== 1) && all(mixInf.trgDiv .== 1) && trgVol.org[dirItr] == srcVol.org[dirItr] &&
            (cmpInf.blc === nothing || dirItr ∉ cmpInf.blc.ax)
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
function gthEgoCmp!(egoCrcCmp::AbstractArray{<:Complex,6}, egoCrc::AbstractArray{<:Complex,7})
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
transformed circulant into the branch array. The trailing (component /
partition) dimensions of the two arrays must agree. When a
dimension of the circulant holds a single unique coefficient the length-1 view
broadcasts up to the minimum branch extent of 2.
=#
function extEgoBrn!(egoFurBrn::AbstractArray{<:Complex}, egoFurPrp::AbstractArray{<:Complex}, eoItr::Integer)
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
eight even/odd branches. Dispatch on cmpInf selects the backend. The transform
runs at generation precision (eltype of egoCrcCmp) while the branch arrays are
Complex{T}, so the extraction copy is the single point of narrowing.
=#
function genEgoFur(egoCrcCmp::Array{<:Complex,6}, truInf::AbstractVector{<:Integer}, cmpInf::GlaKerOpt{T, CPU}) where T<:AbstractFloat
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
    egoFur = Array{Array{Complex{T}}}(undef, 8)
    # only one eighth of the green function is unique; the eight branch
    # extractions are independent
    @threads for eoItr ∈ 0:7
        egoFurBrn = Array{Complex{T}}(undef, truInf..., ddDim, totParSrc, totParTrg)
        extEgoBrn!(egoFurBrn, egoCrcCmp, eoItr)
        egoFur[eoItr + 1] = egoFurBrn
    end
    return egoFur
end

#=
GPU Fourier stage: upload the gathered circulant once, then run the transform
(VkFFT) and branch extraction on device, so the branches already end up on
device. When the full circulant does not fit on device next to the branches
(large multi-partition external operators), fall back to batching per
partition pair with a single reused plan. VkFFT transforms Float64 at most, so
a backend without Float64, or a Double64 circulant, transforms on the CPU and
receives the narrowed branches.
=#
function genEgoFur(egoCrcCmp::Array{<:Complex,6}, truInf::AbstractVector{<:Integer}, cmpInf::GlaKerOpt{T, <:GPU}) where T<:AbstractFloat
    dev = cmpInf.dev
    supports_float64(dev) && eltype(egoCrcCmp) === ComplexF64 || return map(brn -> adapt(dev, brn), genEgoFur(egoCrcCmp, truInf, GlaKerOpt{T}(cmpInf, CPU())))
    ddDim, totParSrc, totParTrg = size(egoCrcCmp)[4:6]
    egoFur = map(_ -> allocate(dev, Complex{T}, truInf..., ddDim, totParSrc, totParTrg), 1:8)
    # leave headroom: the transform itself needs device workspace
    if sizeof(egoCrcCmp) <= 3 * (devMem(dev) ÷ 4)
        egoFurPrp = adapt(dev, egoCrcCmp)
        VkFFT.plan_fft!(egoFurPrp, 1:3) * egoFurPrp
        # verify integrity of Fourier transform data
        if !all(isfinite, egoFurPrp)
            throw(ArgumentError("Fourier transform of circulant contains non-numeric values."))
        end
        for eoItr ∈ 0:7
            extEgoBrn!(egoFur[eoItr + 1], egoFurPrp, eoItr)
        end
        memCln!(egoFurPrp)
    else
        # one slab and one plan reused for every partition pair
        slbDev = allocate(dev, eltype(egoCrcCmp), size(egoCrcCmp)[1:4])
        slbPln = VkFFT.plan_fft!(slbDev, 1:3)
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
        memCln!(slbDev)
    end
    return egoFur
end

# Create Fourier transform plans
function fftPlnGen(fwdSze::NTuple, revSze::NTuple, dir::Int, ::GlaKerOpt{T, CPU}) where T<:AbstractFloat
    # Fourier transform planning area
    fftWrkFwd = Array{Complex{T}}(undef, fwdSze...)
    fftWrkRev = Array{Complex{T}}(undef, revSze...)
    # create Fourier transform plans
    fftPlnFwd = plan_fft!(fftWrkFwd, [dir]; flags=FFTW.MEASURE)
    fftPlnRev = plan_ifft!(fftWrkRev, [dir]; flags=FFTW.MEASURE)
    adjFftPlnFwd = plan_fft!(fftWrkRev, [dir]; flags=FFTW.MEASURE)
    adjFftPlnRev = plan_ifft!(fftWrkFwd, [dir]; flags=FFTW.MEASURE)
    return fftPlnFwd, fftPlnRev, adjFftPlnFwd, adjFftPlnRev
end

function fftPlnGen(fwdSze::NTuple, revSze::NTuple, dir::Int, cmpInf::GlaKerOpt{T, <:GPU}) where T<:AbstractFloat
    # Fourier transform planning area
    fftWrkFwdDev = allocate(cmpInf.dev, Complex{T}, fwdSze)
    fftWrkRevDev = allocate(cmpInf.dev, Complex{T}, revSze)
    # create Fourier transform plans
    fftPlnFwdDev = VkFFT.plan_fft!(fftWrkFwdDev, dir)
    fftPlnRevDev = VkFFT.plan_ifft!(fftWrkRevDev, dir)
    adjFftPlnFwdDev = VkFFT.plan_fft!(fftWrkRevDev, dir)
    adjFftPlnRevDev = VkFFT.plan_ifft!(fftWrkFwdDev, dir)
    return fftPlnFwdDev, fftPlnRevDev, adjFftPlnFwdDev, adjFftPlnRevDev
end

# Memory preparation sub-protocol
function glaOprPrp(egoFur::AbstractVector{<:AbstractArray{Complex{T}}}, trgVol::GlaVol, srcVol::GlaVol, mixInf::GlaExtInf, cmpInf::GlaKerOpt{T}) where T<:AbstractFloat
    # number of embedding levels---dimensionality of ambient space
    lvls = 3
    # operator dimensions---unique vector information does not typically 
    # match operator size for distinct source and target volumes
    # sum of source and target volumes being divisible by 2 is guaranteed by 
    # genVolEve in GlaVacOprMem
    brnSze = div.(mixInf.trgCel .+ mixInf.srcCel, 2)
    phzInf = Array{AbstractVector{Complex{T}}}(undef, lvls)
    # Fourier transform plans
    fftPlnFwd = Array{AbstractFFTs.Plan}(undef, lvls)
    fftPlnRev = Array{AbstractFFTs.Plan}(undef, lvls)
    adjFftPlnFwd = Array{AbstractFFTs.Plan}(undef, lvls)
    adjFftPlnRev = Array{AbstractFFTs.Plan}(undef, lvls)
    (; srcCel, trgCel, srcDiv, trgDiv) = mixInf
    # initialize Fourier transform plans
    for dir ∈ eachindex(1:lvls)
        # size of vector changes throughout application for external Green 
        vecSzeFwd = ntuple(x -> x <= dir ? brnSze[x] : srcCel[x], 3)
        vecSzeRev = ntuple(x -> x > dir ? trgCel[x] : brnSze[x], 3)
        fwdSze = (vecSzeFwd..., lvls, prod(srcDiv))
        revSze = (vecSzeRev..., lvls, prod(trgDiv))
        fftPlnFwd[dir], fftPlnRev[dir], adjFftPlnFwd[dir], adjFftPlnRev[dir] = fftPlnGen(fwdSze, revSze, dir, cmpInf)
    end
    # phase transformations (internal for block Toeplitz transformations)
    for itr ∈ eachindex(1:lvls)
        # allows calculation odd coefficient numbers
        phzInfHst = Complex{T}.([cispi(-k / brnSze[itr]) for k ∈ 0:(brnSze[itr] - 1)])
        phzInf[itr] = similar(first(egoFur), brnSze[itr])
        copyto!(phzInf[itr], phzInfHst)
    end
    return GlaVacOprMem{T}(cmpInf, trgVol, srcVol, mixInf, brnSze, egoFur, fftPlnFwd, fftPlnRev, adjFftPlnFwd, adjFftPlnRev, phzInf)
end

isadjoint(vacOprMem::GlaVacOprMem) = adjMod(vacOprMem.cmpInf)

"""
    setBlc!(mem::GlaVacOprMem, θ::Tuple, blk = nothing)

Refill the Fourier data of a Bloch operator in place at the Bloch phases `θ`, reusing its plans and its `BlcSet`, or the block `blk = (H, A)` blcCmp gave a composite pair.
"""
function setBlc!(mem::GlaVacOprMem, θ::Tuple, blk = nothing)
    opt = mem.cmpInf
    opt.blc === nothing && throw(ArgumentError("setBlc! needs an operator built with a Bloch lattice."))
    opt.blc = merge(opt.blc, (; θ))
    totCelCrc = mem.mixInf.trgCel .+ mem.mixInf.srcCel
    egoCrc = Array{Complex{genPrc(opt)}}(undef, 3, 3, totCelCrc..., 1, 1)
    genEgoCrc!(egoCrc, mem.trgVol, mem.srcVol, mem.mixInf, opt; blk)
    egoCrcCmp = Array{Complex{genPrc(opt)}}(undef, totCelCrc..., 6, 1, 1)
    gthEgoCmp!(egoCrcCmp, egoCrc)
    egoFur = genEgoFur(egoCrcCmp, collect(size(first(mem.egoFur))[1:3]), opt)
    foreach((dst, src) -> dst .= adjMod(opt) ? conj.(src) : src, mem.egoFur, egoFur)
    return mem
end

"""
    GlaVacOprMem{T}(mem::GlaVacOprMem)

Convert the storage precision of a vacuum Green function operator memory to `T`, rebuilding plans and phase data without regenerating the Fourier coefficients.
"""
GlaVacOprMem{T}(mem::GlaVacOprMem{T}) where T<:AbstractFloat = mem
function GlaVacOprMem{T}(mem::GlaVacOprMem) where T<:AbstractFloat
    egoFur = map(mem.egoFur) do fur
        furCnv = similar(fur, Complex{T})
        # broadcast, not copyto!, so that the eltype change also works on device
        furCnv .= fur
        furCnv
    end
    cmpInf = GlaKerOpt{T}(mem.cmpInf)
    return glaOprPrp(egoFur, mem.trgVol, mem.srcVol, mem.mixInf, cmpInf)
end

# deepcopy must regenerate FFTW plans rather than copying the raw C pointers.
# Two Julia plan objects that share the same C pointer both register a finalizer
# calling fftw_destroy_plan; when either is GC'd the other becomes a dangling
# pointer, causing a segfault on the next plan execution.
function Base.deepcopy_internal(mem::GlaVacOprMem, stackdict::IdDict)
    haskey(stackdict, mem) && return stackdict[mem]::GlaVacOprMem
    stackdict[mem] = glaOprPrp(deepcopy(mem.egoFur), mem.trgVol, mem.srcVol, deepcopy(mem.mixInf), deepcopy(mem.cmpInf))
    return stackdict[mem]
end

KernelAbstractions.get_backend(mem::GlaVacOprMem) = mem.cmpInf.dev

function Adapt.adapt_structure(to, mem::GlaVacOprMem)
    egoFur = map(fur -> adapt(to, fur), mem.egoFur)
    cmpInf = GlaKerOpt{real(eltype(first(egoFur)))}(mem.cmpInf, get_backend(first(egoFur)))
    return glaOprPrp(egoFur, mem.trgVol, mem.srcVol, mem.mixInf, cmpInf)
end

# As with deepcopy above, serialization must regenerate the FFTW plans rather
# than write out the raw C pointers, which are only meaningful inside the
# process that created them. Only the Fourier data, volumes, and kernel options
# (and the Bloch lattice, not its per-frequency BlcSet cache) are written, and glaOprPrp rebuilds the plans on load.
function Serialization.serialize(s::AbstractSerializer, mem::GlaVacOprMem)
    # serialize_type needs the concrete GlaVacOprMem{T}, not the UnionAll
    Serialization.serialize_type(s, typeof(mem))
    serialize(s, map(fur -> adapt(CPU(), fur), mem.egoFur))
    serialize(s, mem.cmpInf.frqPhz)
    serialize(s, mem.cmpInf.genPrc)
    serialize(s, mem.cmpInf.qssApx)
    serialize(s, mem.cmpInf.adjMod)
    serialize(s, mem.cmpInf.blc)
    serialize(s, mem.trgVol)
    serialize(s, mem.srcVol)
    serialize(s, mem.mixInf)
end

function Serialization.deserialize(s::AbstractSerializer, ::Type{<:GlaVacOprMem})
    egoFur = deserialize(s)
    frqPhz = deserialize(s)
    genPrc = deserialize(s)
    qssApx = deserialize(s)
    adjMod = deserialize(s)
    blc = deserialize(s)
    trgVol = deserialize(s)
    srcVol = deserialize(s)
    mixInf = deserialize(s)
    # storage precision is recovered from the written Fourier data
    prc = real(eltype(first(egoFur)))
    return glaOprPrp(egoFur, trgVol, srcVol, mixInf, GlaKerOpt{prc}(frqPhz, genPrc, qssApx, adjMod, CPU(), blc))
end
