"""
    GlaSndOprVac

One block of a composite operator, computed on the finer of the two meshes it
connects.

The block wraps a same-scale `GlaOprVac` between the target and source regions
represented at a common fine cell size. A coarse side carries an integer cell
ratio: on the target side the fine fields are summed over each `trgRat` block of
cells, and on the source side each coarse coefficient is repeated over its
`srcRat` block of fine cells at weight one, which is what a pulse basis function
of unit density looks like on the fine mesh. A ratio of `(1, 1, 1)` leaves that
side alone.

`wgt` holds every scalar of the block at once: the `1 / prod(trgRat)` that turns
the sum over the target block into a mean, and the `sqrt(ΔV_trg / ΔV_src)` of the
normalized basis. Written this way the block matrix is `wgt` times a product of
two zero-one matrices with the inner operator between them, so the adjoint is the
same shape with the two ratios exchanged and `wgt` conjugated.

# Fields
- `opr::GlaOprVac`: The same-scale operator between the two regions on the fine
  mesh
- `trgRat::NTuple{3,Int}`: Fine cells per target cell in each dimension
- `srcRat::NTuple{3,Int}`: Fine cells per source cell in each dimension
- `wgt::ComplexF64`: The scalar multiplying the block
"""
struct GlaSndOprVac <: AbstractGlaVacOpr
    opr::GlaOprVac
    trgRat::NTuple{3,Int}
    srcRat::NTuple{3,Int}
    wgt::ComplexF64
end

"""
    GlaCmpOprVac

The vacuum Green function operator between two composite volumes.

The operator is the block matrix of pairwise vacuum operators over the regions of
the two tilings. Pulse basis functions of disjoint regions do not overlap, so the
block matrix is the Galerkin discretization of the Green operator on the
non-uniform mesh, with no stitching at region boundaries.

Blocks work in the √ΔV normalized basis of `GlaFld`, so block `(i, j)` carries
a factor `sqrt(ΔV_i / ΔV_j)`. A region has one cell size, so that factor is one
scalar per block. In this basis the Euclidean inner product is the physical L²
pairing and the self operator is complex-symmetric.

# Fields
- `trgCvl::GlaCmpVol`: The composite volume the fields land on
- `srcCvl::GlaCmpVol`: The composite volume the currents live on
- `blkMat::Matrix{AbstractGlaOpr}`: The block matrix. Same-scale pairs and
  separated cross-scale pairs are a `GlaOprVac`, cross-scale pairs in contact are
  a `GlaSndOprVac`, and same-scale pairs whose kind of contact the external
  construction does not cover are a `GlaOprVac` on the union of the two regions
"""
struct GlaCmpOprVac <: AbstractGlaVacOpr
    trgCvl::GlaCmpVol
    srcCvl::GlaCmpVol
    blkMat::Matrix{AbstractGlaOpr}
end

const CompositeVacuumGreenOperator = GlaCmpOprVac

"""
    AsyGlaCmpOprVac(opr::GlaCmpOprVac)

The anti-Hermitian part `Asym(G₀) = (G₀ - G₀†) / 2i` of the vacuum Green function
operator over a composite volume, built from the self composite operator `opr`.

A self composite operator is complex-symmetric in the √ΔV basis, so its
anti-Hermitian part is its entry by entry imaginary part, which is again a block
matrix over the same tiling. Every block carries the Fourier coefficients of the
imaginary part of its own kernel, so a matvec costs one application of `G₀`
rather than the two of the difference. `opr` itself is left untouched.

# Fields
- `opr::GlaCmpOprVac`: The block matrix, every block replaced by its entry by
  entry imaginary part

# Throws
- `ArgumentError`: If the operator maps between two different tilings, or is in
  adjoint mode
"""
struct AsyGlaCmpOprVac <: AbstractGlaVacOpr
    opr::GlaCmpOprVac

    AsyGlaCmpOprVac(opr::GlaCmpOprVac) = new(_hrmOpr(opr, true))
    #= Blocks that already carry the transformed coefficients, as read back by
    the deserializer. Taking the part again would apply it twice. =#
    AsyGlaCmpOprVac(opr::GlaCmpOprVac, ::Val{:raw}) = new(opr)
end

"""
    SymGlaCmpOprVac(opr::GlaCmpOprVac)

The Hermitian part `Sym(G₀) = (G₀ + G₀†) / 2` of the vacuum Green function
operator over a composite volume, built the same way as `AsyGlaCmpOprVac` from
the entry by entry real part of the blocks.

# Fields
- `opr::GlaCmpOprVac`: The block matrix, every block replaced by its entry by
  entry real part

# Throws
- `ArgumentError`: If the operator maps between two different tilings, or is in
  adjoint mode
"""
struct SymGlaCmpOprVac <: AbstractGlaVacOpr
    opr::GlaCmpOprVac

    SymGlaCmpOprVac(opr::GlaCmpOprVac) = new(_hrmOpr(opr, false))
    # Raw path, see AsyGlaCmpOprVac
    SymGlaCmpOprVac(opr::GlaCmpOprVac, ::Val{:raw}) = new(opr)
end

const AsymCompositeVacuumGreenOperator = AsyGlaCmpOprVac
const SymCompositeVacuumGreenOperator = SymGlaCmpOprVac

# Start of each region block in the flat layout, plus the total length at the end
function _dofOff(cvol::GlaCmpVol)
    off = zeros(Int, nregions(cvol) + 1)
    for (idx, reg) in enumerate(regions(cvol))
        off[idx + 1] = off[idx] + 3 * prod(reg.cel)
    end
    return off
end

# Normalization of block (i, j) in the √ΔV basis
_nrmWgt(trgReg::GlaVol, srcReg::GlaVol) =
    sqrt(Float64(prod(trgReg.scl) // prod(srcReg.scl)))

#= Check for contact between two regions to then run the contact quadrature
subroutine (sandwich). =#
_cntChk(trgReg::GlaVol, srcReg::GlaVol) =
    all(max.(_lwrEdg(trgReg), _lwrEdg(srcReg)) .<= min.(_uprEdg(trgReg), _uprEdg(srcReg)))

#= The closest cross pair of two tilings, warned about once. Regions of one
tiling touch each other by construction, so only target against source pairs
are measured, and touching cross pairs are skipped rather than exempting the
whole check. =#
function prxChk(trgCvl::GlaCmpVol, srcCvl::GlaCmpVol)
    gapMin = nothing
    for trgReg in regions(trgCvl), srcReg in regions(srcCvl)
        gap = _prxGap(trgReg, srcReg)
        isnothing(gap) && continue
        (isnothing(gapMin) || gap[1] < gapMin[1]) && (gapMin = gap)
    end
    (isnothing(gapMin) || gapMin[1] >= prxCelMin) && return nothing

    celStr = (isone(denominator(gapMin[1])) ? string(numerator(gapMin[1])) : string(round(Float64(gapMin[1]); digits=2))) * (isone(gapMin[1]) ? " cell" : " cells")
    @warn "The target and source are separated by $celStr of the coarser $(gapMin[2]) interface scale. Operator entries this close are quadrature limited, and the relative error only reaches about 1e-8 at a gap of $prxCelMin cells. It is strongly recommended to refine the facing surfaces with `refine` to increase the accuracy of the operator."
    return nothing
end

# The cross-scale contact block, computed on the finer of the two meshes
function _sndBlk(trgReg::GlaVol, srcReg::GlaVol, useGpu::Bool)
    sclFin = min.(trgReg.scl, srcReg.scl)
    trgRat = ntuple(dir -> Int(trgReg.scl[dir] // sclFin[dir]), 3)
    srcRat = ntuple(dir -> Int(srcReg.scl[dir] // sclFin[dir]), 3)
    trgFin = GlaVol(Tuple(trgReg.cel .* trgRat), sclFin, trgReg.org)
    srcFin = GlaVol(Tuple(srcReg.cel .* srcRat), sclFin, srcReg.org)
    innOpr = GlaOprVac(trgFin, srcFin; useGpu=useGpu, prxWrn=false)
    return GlaSndOprVac(innOpr, trgRat, srcRat,
        _nrmWgt(trgReg, srcReg) / prod(trgRat))
end

function _cmpBlk(trgReg::GlaVol, srcReg::GlaVol, isSlf::Bool, slfCmp::Bool,
    trgIdx::Integer, srcIdx::Integer, useGpu::Bool)
    isSlf && return GlaOprVac(trgReg; useGpu=useGpu)
    if !slfCmp && _ovrLap(trgReg, srcReg)
        throw(ArgumentError("Target region $trgIdx spans $(_lwrEdg(trgReg)) to $(_uprEdg(trgReg)) and source region $srcIdx spans $(_lwrEdg(srcReg)) to $(_uprEdg(srcReg)), so the two overlap. A composite operator between two bodies needs the bodies to be disjoint."))
    end
    trgReg.scl != srcReg.scl && _cntChk(trgReg, srcReg) &&
        return _sndBlk(trgReg, srcReg, useGpu)
    opr = GlaOprVac(trgReg, srcReg; useGpu=useGpu, prxWrn=false)
    nrm = _nrmWgt(trgReg, srcReg)
    # A real scalar on the Fourier coefficients survives adjoint! untouched
    nrm != 1 && map!(fur -> nrm .* fur, opr.mem.egoFur)
    return opr
end

"""
    GlaCmpOprVac(trgCvl::GlaCmpVol, srcCvl::GlaCmpVol; useGpu::Bool=false, prxWrn::Bool=true)

Construct the vacuum Green function operator between two composite volumes.

Each pair of regions gets the block that its geometry allows. Two regions of the
same cell size use the ordinary external path, in contact or not, unless they
touch in a way the contact correction does not cover, in which case the block is
built on the union of the two. Two regions of different cell size use the
partitioned cross-scale path when they are far enough apart, and the fine mesh
construction of `GlaSndOprVac` when they are close enough that the cross-scale
contact quadrature would run. A pair of identical regions, which only happens on
the diagonal of a self operator, uses the self path.

Passing the same composite volume twice gives the self operator of one body.
Passing two different ones gives the operator between two bodies, which have to
be disjoint. A warning is emitted for two bodies with fewer than 6 cells of
separation.

# Arguments
- `trgCvl::GlaCmpVol`: The composite volume the fields land on
- `srcCvl::GlaCmpVol`: The composite volume the currents live on
- `useGpu::Bool=false`: Whether to build the blocks on the GPU
- `prxWrn::Bool=true`: Whether to warn about a separation under the accuracy bound

# Returns
- `GlaCmpOprVac`: The composite operator

# Throws
- `ArgumentError`: If a region of one volume overlaps a region of the other
"""
function GlaCmpOprVac(trgCvl::GlaCmpVol, srcCvl::GlaCmpVol; useGpu::Bool=false,
    prxWrn::Bool=true)
    slfCmp = trgCvl === srcCvl || trgCvl == srcCvl
    prxWrn && !slfCmp && prxChk(trgCvl, srcCvl)
    trgRegs, srcRegs = regions(trgCvl), regions(srcCvl)
    blkMat = Matrix{AbstractGlaOpr}(undef, length(trgRegs), length(srcRegs))
    for trgIdx in eachindex(trgRegs), srcIdx in eachindex(srcRegs)
        blkMat[trgIdx, srcIdx] = _cmpBlk(trgRegs[trgIdx], srcRegs[srcIdx],
            slfCmp && trgIdx == srcIdx, slfCmp, trgIdx, srcIdx, useGpu)
    end
    return GlaCmpOprVac(trgCvl, srcCvl, blkMat)
end

"""
    GlaCmpOprVac(cvol::GlaCmpVol; useGpu::Bool=false)

Construct the self vacuum Green function operator of a composite volume.

# Arguments
- `cvol::GlaCmpVol`: The composite volume
- `useGpu::Bool=false`: Whether to build the blocks on the GPU

# Returns
- `GlaCmpOprVac`: The composite operator
"""
GlaCmpOprVac(cvol::GlaCmpVol; useGpu::Bool=false) = GlaCmpOprVac(cvol, cvol; useGpu=useGpu)

"""
    GlaOprVac(cvol::GlaCmpVol; useGpu::Bool=false)

Construct the self vacuum Green function operator of a composite volume.

A composite volume needs a block matrix rather than one circulant, so the result
is a `GlaCmpOprVac` and not a `GlaOprVac`. Both are `AbstractGlaVacOpr`.

# Arguments
- `cvol::GlaCmpVol`: The composite volume
- `useGpu::Bool=false`: Whether to build the blocks on the GPU

# Returns
- `GlaCmpOprVac`: The composite operator
"""
GlaOprVac(cvol::GlaCmpVol; useGpu::Bool=false) = GlaCmpOprVac(cvol; useGpu=useGpu)

"""
    GlaOprVac(trgCvl::GlaCmpVol, srcCvl::GlaCmpVol; useGpu::Bool=false, prxWrn::Bool=true)

Construct the vacuum Green function operator between two composite volumes.

The result is a `GlaCmpOprVac`, for the reason given in the single volume method.

# Arguments
- `trgCvl::GlaCmpVol`: The composite volume the fields land on
- `srcCvl::GlaCmpVol`: The composite volume the currents live on
- `useGpu::Bool=false`: Whether to build the blocks on the GPU
- `prxWrn::Bool=true`: Whether to warn about a separation under the accuracy bound

# Returns
- `GlaCmpOprVac`: The composite operator
"""
GlaOprVac(trgCvl::GlaCmpVol, srcCvl::GlaCmpVol; useGpu::Bool=false,
    prxWrn::Bool=true) =
    GlaCmpOprVac(trgCvl, srcCvl; useGpu=useGpu, prxWrn=prxWrn)

#= Sign picked up by each stored tensor component when the real space kernel is
reflected in the directions flagged in dirRfl. Storage order is xx, yy, zz, xy,
xz, yz, and a reflection flips a component once for each of its two tensor
indices that lies along it. =#
_rflSgn(dirRfl::NTuple{3,Bool}, cmp::Integer) =
    prod(dir -> dirRfl[dir] && count(==(dir), egoCmpPos[cmp]) == 1 ? -1 : 1, 1:3)

#= Fourier coefficients of the entry by entry real or imaginary part of a block.
A block is the leading corner of a circulant, so taking the real or imaginary
part entry by entry is the circulant of the real or imaginary part of its kernel,
whose coefficients are (fur[m] ± conj(fur[-m])) / 2 (with an extra 1/i for the
imaginary part). Negating the circulant index reverses a direction held in full,
and is the reflection sign of the component in a direction stored only up to that
reflection. A branch is even or odd in each direction, which shifts the even
reversal by one, and never mixes with the other branches. =#
function _hrmFur!(mem::GlaVacOprMem, isAsy::Bool)
    brnSze = div.(mem.mixInf.trgCel .+ mem.mixInf.srcCel, 2)
    for bId in 0:7
        fur = mem.egoFur[bId + 1]
        dirRfl = ntuple(dir -> size(fur, dir) != brnSze[dir], 3)
        negFur = conj.(fur)
        for dir in 1:3
            dirRfl[dir] && continue
            negFur = reverse(negFur; dims=dir)
            isodd(bId >> (3 - dir)) && continue
            negFur = circshift(negFur,
                ntuple(idx -> idx == dir ? 1 : 0, ndims(negFur)))
        end
        for cmp in 1:6
            sgn = _rflSgn(dirRfl, cmp)
            sgn == 1 || (view(negFur, :, :, :, cmp, :, :) .*= sgn)
        end
        fur .= isAsy ? (fur .- negFur) ./ 2im : (fur .+ negFur) ./ 2
    end
    return mem
end

function _hrmBlk(opr::GlaOprVac, isAsy::Bool)
    mem = _hrmFur!(deepcopy(opr.mem), isAsy)
    return GlaOprVac(mem, opr.srcMsk, opr.trgMsk)
end

#= wgt is real and the aggregation and injection matrices hold zeros and ones, so
they all commute with taking the real or imaginary part of the block. =#
_hrmBlk(opr::GlaSndOprVac, isAsy::Bool) =
    GlaSndOprVac(_hrmBlk(opr.opr, isAsy), opr.trgRat, opr.srcRat, opr.wgt)

function _hrmOpr(opr::GlaCmpOprVac, isAsy::Bool)
    nam = isAsy ? "AsyGlaCmpOprVac" : "SymGlaCmpOprVac"
    if !isselfoperator(opr)
        throw(ArgumentError("$nam can only be constructed from a GlaCmpOprVac with identical target and source tilings, and this operator maps between two different ones."))
    end
    if isadjoint(opr)
        throw(ArgumentError("$nam can only be constructed from an operator that is not in adjoint mode. Call adjoint! on it to restore it first."))
    end
    blkMat = Matrix{AbstractGlaOpr}(undef, size(opr.blkMat))
    for idx in eachindex(opr.blkMat)
        blkMat[idx] = _hrmBlk(opr.blkMat[idx], isAsy)
    end
    return GlaCmpOprVac(opr.trgCvl, opr.srcCvl, blkMat)
end

"""
    AsyGlaCmpOprVac(cvol::GlaCmpVol; useGpu::Bool=false)

Construct the anti-Hermitian part of the self vacuum Green function operator of a
composite volume.

# Arguments
- `cvol::GlaCmpVol`: The composite volume
- `useGpu::Bool=false`: Whether to build the blocks on the GPU

# Returns
- `AsyGlaCmpOprVac`: The anti-Hermitian part of the composite operator
"""
AsyGlaCmpOprVac(cvol::GlaCmpVol; useGpu::Bool=false) =
    AsyGlaCmpOprVac(GlaCmpOprVac(cvol; useGpu=useGpu))

"""
    SymGlaCmpOprVac(cvol::GlaCmpVol; useGpu::Bool=false)

Construct the Hermitian part of the self vacuum Green function operator of a
composite volume.

# Arguments
- `cvol::GlaCmpVol`: The composite volume
- `useGpu::Bool=false`: Whether to build the blocks on the GPU

# Returns
- `SymGlaCmpOprVac`: The Hermitian part of the composite operator
"""
SymGlaCmpOprVac(cvol::GlaCmpVol; useGpu::Bool=false) =
    SymGlaCmpOprVac(GlaCmpOprVac(cvol; useGpu=useGpu))

"""
    asym(opr::GlaCmpOprVac)

Construct the anti-Hermitian part of a self composite vacuum Green function
operator.

# Arguments
- `opr::GlaCmpOprVac`: The composite operator, which has to be a self operator

# Returns
- `AsyGlaCmpOprVac`: The anti-Hermitian part of the composite operator
"""
asym(opr::GlaCmpOprVac) = AsyGlaCmpOprVac(opr)

"""
    sym(opr::GlaCmpOprVac)

Construct the Hermitian part of a self composite vacuum Green function operator.

# Arguments
- `opr::GlaCmpOprVac`: The composite operator, which has to be a self operator

# Returns
- `SymGlaCmpOprVac`: The Hermitian part of the composite operator
"""
sym(opr::GlaCmpOprVac) = SymGlaCmpOprVac(opr)

glaSze(opr::GlaSndOprVac) =
    ((glaSze(opr.opr, 1)[1:3] .÷ opr.trgRat..., 3),
     (glaSze(opr.opr, 2)[1:3] .÷ opr.srcRat..., 3))
glaSze(opr::GlaCmpOprVac) = glaSze.(opr.blkMat)
glaSze(opr::GlaCmpOprVac, dim::Int) = map(sze -> sze[dim], glaSze(opr))
glaSze(opr::Union{AsyGlaCmpOprVac, SymGlaCmpOprVac}) = glaSze(opr.opr)
glaSze(opr::Union{AsyGlaCmpOprVac, SymGlaCmpOprVac}, dim::Int) =
    glaSze(opr.opr, dim)

Base.size(opr::GlaCmpOprVac) =
    (sum(3 * prod(reg.cel) for reg in regions(opr.trgCvl)),
     sum(3 * prod(reg.cel) for reg in regions(opr.srcCvl)))
Base.size(opr::GlaCmpOprVac, dim::Int) = size(opr)[dim]
Base.size(opr::Union{AsyGlaCmpOprVac, SymGlaCmpOprVac}) = size(opr.opr)
Base.size(opr::Union{AsyGlaCmpOprVac, SymGlaCmpOprVac}, dim::Int) =
    size(opr.opr, dim)

function mulAct!(opr::GlaSndOprVac, act::AbstractVector{ComplexF64})
    finAct = act
    # Repeat every coarse source value over the fine cells it covers
    if !all(opr.srcRat .== 1)
        rat, cel = opr.srcRat, glaSze(opr, 2)[1:3]
        finInn = similar(act, rat[1], cel[1], rat[2], cel[2], rat[3], cel[3], 3)
        finInn .= reshape(act, 1, cel[1], 1, cel[2], 1, cel[3], 3)
        finAct = vec(finInn)
    end
    out = reshape(mulAct!(opr.opr, finAct), glaSze(opr.opr, 1))
    # Sum every block of fine cells into the coarse target cell holding it
    if !all(opr.trgRat .== 1)
        rat = opr.trgRat
        cel = size(out)[1:3] .÷ rat
        blk = reshape(out, rat[1], cel[1], rat[2], cel[2], rat[3], cel[3], 3)
        out = reshape(sum(blk; dims=(1, 3, 5)), cel..., 3)
    end
    return rmul!(vec(out), opr.wgt)
end

#= Block row sums over the flat layout. A region block of the buffer is already
the input a block operator wants, so a slice needs no permutation. =#
function mulAct!(opr::GlaCmpOprVac, act::AbstractVector{ComplexF64})
    trgOff, srcOff = _dofOff(opr.trgCvl), _dofOff(opr.srcCvl)
    if length(act) != srcOff[end]
        throw(ArgumentError("An input of length $(length(act)) does not fit the source volume of this operator, which has $(srcOff[end]) degrees of freedom."))
    end
    outDat = fill!(similar(act, trgOff[end]), zero(ComplexF64))
    for srcIdx in axes(opr.blkMat, 2), trgIdx in axes(opr.blkMat, 1)
        # A block eats its buffer, so every block gets its own copy of the slice
        innBlk = copy(view(act, (srcOff[srcIdx] + 1):srcOff[srcIdx + 1]))
        view(outDat, (trgOff[trgIdx] + 1):trgOff[trgIdx + 1]) .+=
            mulAct!(opr.blkMat[trgIdx, srcIdx], innBlk)
    end
    return outDat
end

mulAct!(opr::Union{AsyGlaCmpOprVac, SymGlaCmpOprVac},
    act::AbstractVector{ComplexF64}) = mulAct!(opr.opr, act)

"""
    *(opr::GlaCmpOprVac, fld::GlaFld)

Apply a composite operator to a composite field.

# Arguments
- `opr::GlaCmpOprVac`: The operator
- `fld::GlaFld`: The field, which must live on the source volume of `opr`

# Returns
- `GlaFld`: The result, on the target volume of `opr`

# Throws
- `ArgumentError`: If the field lives on a different tiling than the source
  volume of the operator
"""
function Base.:*(opr::GlaCmpOprVac, fld::GlaFld)
    if !(fld.cvol === opr.srcCvl || fld.cvol == opr.srcCvl)
        throw(ArgumentError("The field lives on a different composite volume than the source volume of the operator. An operator only applies to fields on the tiling it was built for."))
    end
    return GlaFld(opr * fld.dat, opr.trgCvl)
end

"""
    *(opr::AsyGlaCmpOprVac, fld::GlaFld)

Apply a Hermitian or anti-Hermitian part of a composite operator to a composite
field.

# Arguments
- `opr::AsyGlaCmpOprVac`: The operator, either part
- `fld::GlaFld`: The field, which must live on the volume of `opr`

# Returns
- `GlaFld`: The result, on the same tiling

# Throws
- `ArgumentError`: If the field lives on a different tiling than the operator
"""
Base.:*(opr::Union{AsyGlaCmpOprVac, SymGlaCmpOprVac}, fld::GlaFld) = opr.opr * fld

adjoint!(opr::GlaSndOprVac) =
    GlaSndOprVac(adjoint!(opr.opr), opr.srcRat, opr.trgRat, conj(opr.wgt))

function adjoint!(opr::GlaCmpOprVac)
    adjMat = Matrix{AbstractGlaOpr}(undef, reverse(size(opr.blkMat)))
    for trgIdx in axes(opr.blkMat, 1), srcIdx in axes(opr.blkMat, 2)
        adjMat[srcIdx, trgIdx] = adjoint!(opr.blkMat[trgIdx, srcIdx])
    end
    return GlaCmpOprVac(opr.srcCvl, opr.trgCvl, adjMat)
end
# Both parts are Hermitian (self-adjoint)
adjoint!(opr::Union{AsyGlaCmpOprVac, SymGlaCmpOprVac}) = opr

function useCpu!(opr::GlaSndOprVac)
    useCpu!(opr.opr)
    return opr
end

function useGpu!(opr::GlaSndOprVac)
    useGpu!(opr.opr)
    return opr
end

function useCpu!(opr::GlaCmpOprVac)
    useCpu!.(opr.blkMat)
    return opr
end

function useGpu!(opr::GlaCmpOprVac)
    useGpu!.(opr.blkMat)
    return opr
end

function useCpu!(opr::Union{AsyGlaCmpOprVac, SymGlaCmpOprVac})
    useCpu!(opr.opr)
    return opr
end

function useGpu!(opr::Union{AsyGlaCmpOprVac, SymGlaCmpOprVac})
    useGpu!(opr.opr)
    return opr
end

GilaVacuum.arrTyp(opr::GlaSndOprVac) = arrTyp(opr.opr)
GilaVacuum.arrTyp(opr::GlaCmpOprVac) = arrTyp(first(opr.blkMat))
GilaVacuum.arrTyp(opr::Union{AsyGlaCmpOprVac, SymGlaCmpOprVac}) = arrTyp(opr.opr)

isadjoint(opr::GlaSndOprVac) = isadjoint(opr.opr)
isadjoint(opr::GlaCmpOprVac) = all(isadjoint, opr.blkMat)
isadjoint(::Union{AsyGlaCmpOprVac, SymGlaCmpOprVac}) = false
isselfoperator(opr::GlaSndOprVac) = false
isselfoperator(opr::GlaCmpOprVac) =
    opr.trgCvl === opr.srcCvl || opr.trgCvl == opr.srcCvl
isselfoperator(::Union{AsyGlaCmpOprVac, SymGlaCmpOprVac}) = true
isexternaloperator(opr::GlaSndOprVac) = true
isexternaloperator(opr::GlaCmpOprVac) = !isselfoperator(opr)
isexternaloperator(::Union{AsyGlaCmpOprVac, SymGlaCmpOprVac}) = false
isgpu(opr::GlaSndOprVac) = isgpu(opr.opr)
isgpu(opr::GlaCmpOprVac) = all(isgpu, opr.blkMat)
isgpu(opr::Union{AsyGlaCmpOprVac, SymGlaCmpOprVac}) = isgpu(opr.opr)

_strKnd(opr::GlaSndOprVac) = "fine mesh G₀"
_strKnd(opr::GlaCmpOprVac) = "composite G₀"
_strKnd(opr::AsyGlaCmpOprVac) = "composite Asym(G₀)"
_strKnd(opr::SymGlaCmpOprVac) = "composite Sym(G₀)"

function Base.show(io::IO, opr::GlaSndOprVac)
    isadjoint(opr) && print(io, "Adjoint ")
    print(io, isgpu(opr) ? "GPU " : "CPU ")
    print(io, "fine mesh G₀ for $(eltype(opr)) (" *
        join(glaSze(opr, 2)[1:3], "×") * ") -> (" *
        join(glaSze(opr, 1)[1:3], "×") * ") volumes ")
    print(io, "on a (" * join(_srcVol(opr.opr).scl, "×") * ")λ³ mesh")
end
Base.show(io::IO, ::MIME"text/plain", opr::GlaSndOprVac) = show(io, opr)

#= The susceptibility of one cell, repeated over the three vector components of
that cell. A region block of the flat layout is the vec of a (cel..., 3) array,
so the three copies of a block sit one after the other. =#
function _expSus(cvol::GlaCmpVol, susCel::Vector{ComplexF64})
    susDof = Vector{ComplexF64}(undef, 3 * length(susCel))
    celOff, dofOff = 0, 0
    for reg in regions(cvol)
        celNum = prod(reg.cel)
        blk = view(susCel, (celOff + 1):(celOff + celNum))
        for dir in 1:3
            copyto!(view(susDof, (dofOff + (dir - 1) * celNum + 1):(dofOff + dir * celNum)), blk)
        end
        celOff += celNum
        dofOff += 3 * celNum
    end
    return susDof
end

# Per-region susceptibility tensors, checked against the cell counts of the tiling
function _tenSus(cvol::GlaCmpVol, sus::AbstractVector)
    regs = regions(cvol)
    if length(sus) != length(regs)
        throw(ArgumentError("Got $(length(sus)) susceptibility tensors for a composite volume of $(length(regs)) regions."))
    end
    susCel = Vector{ComplexF64}(undef, sum(prod(reg.cel) for reg in regs))
    celOff = 0
    for (idx, reg) in enumerate(regs)
        ten = sus[idx]
        if size(ten) != Tuple(reg.cel)
            throw(ArgumentError("The susceptibility tensor of region $idx has size $(size(ten)), but region $idx has $(join(reg.cel, "×")) cells."))
        end
        celNum = prod(reg.cel)
        copyto!(view(susCel, (celOff + 1):(celOff + celNum)), vec(Array{ComplexF64}(ten)))
        celOff += celNum
    end
    return susCel
end

# Any accepted susceptibility, read into the flat degree of freedom layout
function _cmpSus(cvol::GlaCmpVol, sus, useGpu::Bool=false)
    celNum = sum(prod(reg.cel) for reg in regions(cvol))
    susDof = if sus isa Number
        _expSus(cvol, fill(ComplexF64(sus), celNum))
    elseif sus isa AbstractVector{<:Number}
        length(sus) in (celNum, 3 * celNum) || throw(ArgumentError("A susceptibility vector of length $(length(sus)) fits neither the $(3 * celNum) degrees of freedom nor the $celNum cells of this composite volume."))
        length(sus) == 3 * celNum ? Vector{ComplexF64}(Array(sus)) :
            _expSus(cvol, Vector{ComplexF64}(Array(sus)))
    elseif sus isa AbstractArray{<:Number, 3}
        if nregions(cvol) != 1
            throw(ArgumentError("A single susceptibility tensor only fits a composite volume of one region, and this one has $(nregions(cvol)). Pass one tensor per region as a vector."))
        end
        _expSus(cvol, _tenSus(cvol, [sus]))
    elseif sus isa AbstractVector
        _expSus(cvol, _tenSus(cvol, sus))
    else
        # Anything else is a function of the cell center
        _expSus(cvol, [ComplexF64(sus(pos)) for (pos, _, _) in coordinates(cvol)])
    end
    return useGpu ? CuArray(susDof) : susDof
end

"""
    InvSctOpr(cvol::GlaCmpVol, sus; useGpu::Bool=false)

Construct the inverse scattering operator `(I - XG₀)` over a composite volume.

The vacuum operator is built for the tiling, and the susceptibility is stored in
the flat degree of freedom layout of `GlaFld`. Since the susceptibility and the
√ΔV normalization are both diagonal, they commute, and the operator in the
normalized basis is the same expression as on a uniform mesh.

The susceptibility is one scalar per cell, and it can be given as a number for a
uniform medium, a function of the cell center returning that scalar, a vector of
one 3-tensor per region, a vector of one value per cell in layout order, or a
vector already in the degree of freedom layout. Every other composite scattering
constructor takes the same forms.

# Arguments
- `cvol::GlaCmpVol`: The composite volume
- `sus`: The susceptibility, in any of the forms above
- `useGpu::Bool=false`: Whether to build the operator on the GPU

# Returns
- `InvSctOpr`: The inverse scattering operator

# Throws
- `ArgumentError`: If the shape of `sus` does not fit the tiling
"""
InvSctOpr(cvol::GlaCmpVol, sus; useGpu::Bool=false) =
    InvSctOpr(GlaCmpOprVac(cvol; useGpu=useGpu), sus)

"""
    SctOpr(cvol::GlaCmpVol, sus; useGpu::Bool=false, slv::GlaSlv=BiCGStabSolver())

Construct the scattering operator `(I - XG₀)⁻¹` over a composite volume.

# Arguments
- `cvol::GlaCmpVol`: The composite volume
- `sus`: The susceptibility, in any of the forms `InvSctOpr(::GlaCmpVol, sus)` takes
- `useGpu::Bool=false`: Whether to build the operator on the GPU
- `slv::GlaSlv=BiCGStabSolver()`: The solver used for the inverse

# Returns
- `SctOpr`: The scattering operator
"""
SctOpr(cvol::GlaCmpVol, sus; useGpu::Bool=false, slv::GlaSlv=BiCGStabSolver()) =
    SctOpr(InvSctOpr(cvol, sus; useGpu=useGpu), slv)

"""
    SctOpr(opr::GlaCmpOprVac, sus; slv::GlaSlv=BiCGStabSolver())

Construct the scattering operator from a composite vacuum operator.

# Arguments
- `opr::GlaCmpOprVac`: The composite vacuum operator, which has to be a self operator
- `sus`: The susceptibility, in any of the forms `InvSctOpr(::GlaCmpVol, sus)` takes
- `slv::GlaSlv=BiCGStabSolver()`: The solver used for the inverse

# Returns
- `SctOpr`: The scattering operator
"""
SctOpr(opr::GlaCmpOprVac, sus; slv::GlaSlv=BiCGStabSolver()) =
    SctOpr(InvSctOpr(opr, sus), slv)

"""
    GlaOpr(cvol::GlaCmpVol, sus; useGpu::Bool=false, slv::GlaSlv=BiCGStabSolver())

Construct the full Green function operator `G₀(I - XG₀)⁻¹` over a composite volume.

# Arguments
- `cvol::GlaCmpVol`: The composite volume
- `sus`: The susceptibility, in any of the forms `InvSctOpr(::GlaCmpVol, sus)` takes
- `useGpu::Bool=false`: Whether to build the operator on the GPU
- `slv::GlaSlv=BiCGStabSolver()`: The solver used for the inverse

# Returns
- `GlaOpr`: The full Green function operator
"""
GlaOpr(cvol::GlaCmpVol, sus; useGpu::Bool=false, slv::GlaSlv=BiCGStabSolver()) =
    GlaOpr(SctOpr(cvol, sus; useGpu=useGpu, slv=slv))

"""
    GlaOpr(opr::GlaCmpOprVac, sus; slv::GlaSlv=BiCGStabSolver())

Construct the full Green function operator from a composite vacuum operator.

# Arguments
- `opr::GlaCmpOprVac`: The composite vacuum operator, which has to be a self operator
- `sus`: The susceptibility, in any of the forms `InvSctOpr(::GlaCmpVol, sus)` takes
- `slv::GlaSlv=BiCGStabSolver()`: The solver used for the inverse

# Returns
- `GlaOpr`: The full Green function operator
"""
GlaOpr(opr::GlaCmpOprVac, sus; slv::GlaSlv=BiCGStabSolver()) =
    GlaOpr(SctOpr(opr, sus; slv=slv))

#= The tiling a scattering operator reads its input on, checked against the
field. A plain vacuum operator carries a single volume, which is a tiling of one
region. =#
function _chkSctFld(oprVac::AbstractGlaVacOpr, fld::GlaFld)
    if oprVac isa GlaCmpOprVac
        if !(fld.cvol === oprVac.srcCvl || fld.cvol == oprVac.srcCvl)
            throw(ArgumentError("The field lives on a different composite volume than the operator. An operator only applies to fields on the tiling it was built for."))
        end
        return oprVac.srcCvl
    end
    srcVol = oprVac.mem.srcVol
    if nregions(fld.cvol) != 1 || regions(fld.cvol)[1] != srcVol
        throw(ArgumentError("The field does not live on the source volume of the operator, which is a ($(join(srcVol.cel, "×"))) cell volume of ($(join(srcVol.scl, "×")))λ³ cells."))
    end
    return fld.cvol
end

"""
    *(opr::InvSctOpr, fld::GlaFld)

Apply an inverse scattering operator to a field.

# Arguments
- `opr::InvSctOpr`: The operator, which has to be built on a composite volume
- `fld::GlaFld`: The field, which has to live on the tiling of `opr`

# Returns
- `GlaFld`: The result, on the same tiling

# Throws
- `ArgumentError`: If the field lives on a different tiling
"""
function Base.:*(opr::InvSctOpr, fld::GlaFld)
    cvol = _chkSctFld(opr.oprVac, fld)
    return GlaFld(opr * fld.dat, cvol)
end

"""
    *(opr::SctOpr, fld::GlaFld)

Apply a scattering operator to a field, which runs the iterative solve.

# Arguments
- `opr::SctOpr`: The operator, which has to be built on a composite volume
- `fld::GlaFld`: The field, which has to live on the tiling of `opr`

# Returns
- `GlaFld`: The result, on the same tiling

# Throws
- `ArgumentError`: If the field lives on a different tiling
"""
function Base.:*(opr::SctOpr, fld::GlaFld)
    cvol = _chkSctFld(opr.invSctOpr.oprVac, fld)
    return GlaFld(solve(opr.invSctOpr, fld.dat, opr.slv), cvol)
end

#= The full operator on a field is the vacuum operator after the solve, the two
the other way around in adjoint mode. Both halves take a field, so the tiling
rides along and the checks happen inside them. =#
Base.:*(opr::GlaOpr, fld::GlaFld) = isadjoint(opr) ?
    opr.sctOpr * (opr.sctOpr.invSctOpr.oprVac * fld) :
    opr.sctOpr.invSctOpr.oprVac * (opr.sctOpr * fld)

function Base.show(io::IO, opr::GlaCmpOprVac)
    numTrg, numSrc = size(opr.blkMat)
    isadjoint(opr) && print(io, "Adjoint ")
    print(io, isgpu(opr) ? "GPU " : "CPU ")
    print(io, isselfoperator(opr) ? "self " : "external ")
    print(io, "composite G₀ ($numTrg target region", numTrg == 1 ? "" : "s",
        " × $numSrc source region", numSrc == 1 ? "" : "s", ")")
    numSnd = count(blk -> blk isa GlaSndOprVac, opr.blkMat)
    print(io, "\n  $(size(opr, 1)) × $(size(opr, 2)) degrees of freedom, ",
        "$numSnd fine mesh block", numSnd == 1 ? "" : "s")
    print(io, "\n  targets: ", opr.trgCvl)
    isselfoperator(opr) || print(io, "\n  sources: ", opr.srcCvl)
end
Base.show(io::IO, ::MIME"text/plain", opr::GlaCmpOprVac) = show(io, opr)

function Base.show(io::IO, opr::Union{AsyGlaCmpOprVac, SymGlaCmpOprVac})
    numReg = nregions(opr.opr.srcCvl)
    print(io, isgpu(opr) ? "GPU " : "CPU ")
    print(io, "self ", _strKnd(opr), " ($numReg region",
        numReg == 1 ? "" : "s", ")")
    numSnd = count(blk -> blk isa GlaSndOprVac, opr.opr.blkMat)
    print(io, "\n  $(size(opr, 1)) × $(size(opr, 2)) degrees of freedom, ",
        "$numSnd fine mesh block", numSnd == 1 ? "" : "s")
    print(io, "\n  ", opr.opr.srcCvl)
end
Base.show(io::IO, ::MIME"text/plain",
    opr::Union{AsyGlaCmpOprVac, SymGlaCmpOprVac}) = show(io, opr)
