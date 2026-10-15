
"""
    GlaVol

Basic spatial memory structure for a rectangular prism volume.

# Fields
- `cel::NTuple{3,Int}`: Number of cells in each dimension of the volume
- `scl::NTuple{3,Rational{Int}}`: Relative side length of a cuboid voxel (cell) compared to the wavelength
- `org::NTuple{3,Rational{Int}}`: Center position of the domain
- `grd::NTuple{3,StepRange{Rational{Int},Rational{Int}}}`: Spatial location of the center of each cell contained in the volume
"""
struct GlaVol
    cel::NTuple{3,Int}
    scl::NTuple{3,Rational{Int}}
    org::NTuple{3,Rational{Int}}
    grd::NTuple{3,StepRange{Rational{Int},Rational{Int}}}
end

"""
    GlaVol(cel::Array{<:Integer,1}, celScl::NTuple{3,Rational}, org::NTuple{3,Rational}, grdScl::NTuple{3,Rational}=celScl)

Constructor for Gila Volumes.

# Arguments
- `cel::Array{<:Integer,1}`: Array of the number of cells in each dimension of the volume
- `celScl::NTuple{3,Rational}`: The size of each cell in each dimensions of the volume (in units of wavelength)
- `org::NTuple{3,Rational}=(0//1, 0//1, 0//1)`: The origin of the volume in each dimension (in units of wavelength)
- `grdScl::NTuple{3,Rational}=celScl`: Spatial location of the center of each cell contained in the volume

# Returns
- `GlaVol`: A new volume with the specified parameters

# Throws
- `ErrorException`: If the cell scale is larger than the grid scale
"""
function GlaVol(cel::Union{Array{<:Integer,1},NTuple{3,Integer}}, 
    celScl::NTuple{3,Rational}, org::NTuple{3,Rational}=(0//1, 0//1, 0//1),
    grdScl::NTuple{3,Rational}=celScl)
    
    if !all(celScl .<= grdScl)
        error("The cell scale must be smaller than the grid scale to avoid partially overlapping basis elements.")
    end
    cel = Tuple(cel) # a Vector cel would broadcast the grid into a Vector
    brd = grdScl .* (Rational.(floor.(cel ./ 2)) .- (iseven.(cel) .// 2))
    grd = map(StepRange, org .- brd, grdScl, org .+ brd)
    return GlaVol(cel, celScl, org, grd)
end

# A cell scale, as one rational times λ when isotropic, else one per dimension
_sclStr(scl::NTuple{3,Rational}) = allequal(scl) ? "$(first(scl)) λ" : join(scl, "×") * " λ"

# A rational shown as a plain integer when it has no fractional part
_ratStr(r::Rational) = isinteger(r) ? string(numerator(r)) : string(r)

#= A volume's cell grid and physical extent, joined by sep. Shared with the
"volume" row of an operator's show. =#
_volDesc(vol::GlaVol, sep::AbstractString) = string(join(vol.cel, "×"), " cells of ",
    _sclStr(vol.scl), sep, join(vol.scl .* vol.cel, " × "), " λ³ at (",
    join(_ratStr.(vol.org), ", "), ")")

#= The grid step only differs from the cell scale for a lattice with gaps
between cells (grdScl > celScl in the constructor), which is otherwise
invisible — so it is only printed then. =#
function Base.show(io::IO, vol::GlaVol)
    print(io, "GlaVol: ", _volDesc(vol, " — "))
    grdScl = step.(vol.grd)
    grdScl == vol.scl || print(io, ", grid step ", _sclStr(grdScl))
end

# Create the union volume of two overlapping volumes
function uniVol(vol1::GlaVol, vol2::GlaVol)
    sclMin = min.(vol1.scl, vol2.scl)
    sclMax = max.(vol1.scl, vol2.scl)
    sclRat = sclMax .// sclMin
    all(isinteger, sclRat) || throw(ArgumentError("Volumes must share a common scale grid for overlap handling"))
    scl = sclMin # Pick the finer scale as the common scale

    minEdg = min.(_lwrEdg(vol1), _lwrEdg(vol2))
    maxEdg = max.(_uprEdg(vol1), _uprEdg(vol2))

    cel = (maxEdg .- minEdg) .// scl
    all(isinteger, cel) || throw(ArgumentError("Computed union volume cell counts must be integers"))
    cel = numerator.(cel)
    org = minEdg .+ (cel .* scl .// 2)
    return GlaVol(cel, scl, org)
end

Base.union(vols::GlaVol...) = reduce(uniVol, vols)

"""
    GlaExtInf 

Information for mapping between general source and target volumes.

# Fields
- `minScl::NTuple{3,Rational{Int}}`: Common minimum cell size (in units of wavelength)
- `trgDiv::NTuple{3,Int}`: Divisions in each cartesian index of target volume
- `srcDiv::NTuple{3,Int}`: Divisions in each cartesian index of source volume
- `trgCel::NTuple{3,Int}`: Cells in a target partition
- `srcCel::NTuple{3,Int}`: Cells in a source partition
"""
struct GlaExtInf
    minScl::NTuple{3,Rational{Int}}
    trgDiv::NTuple{3,Int}
    srcDiv::NTuple{3,Int}
    trgCel::NTuple{3,Int}
    srcCel::NTuple{3,Int}
end

# Grid offset of partition itr among the div partitions of a volume, zero based
_parOff(div::NTuple{3,Int}, itr::Integer) = Tuple(CartesianIndices(div)[itr]) .- 1

"""
    GlaExtInf(trgVol::GlaVol, srcVol::GlaVol)

Constructor for `GlaExtInf`.

Treats grid mismatch if the volumes do not share a common scale grid.

# Arguments
- `trgVol::GlaVol`: Target volume definition
- `srcVol::GlaVol`: Source volume definition

# Returns
- `GlaExtInf`: The transfer information

# Throws
- `ErrorException`: If the volumes do not share a common scale grid or if volume sizes are incompatible
"""
function GlaExtInf(trgVol::GlaVol, srcVol::GlaVol)::GlaExtInf
    # test that cell scales are compatible 
    if prod(isinteger.(srcVol.scl ./ trgVol.scl) .+ isinteger.(trgVol.scl ./ 
        srcVol.scl)) == 0
        error("Volume pair must share a common scale grid.")
    end
    # common scale 
    minScl = gcd.(srcVol.scl, trgVol.scl)
    # maximal scale
    maxScl = lcm.(srcVol.scl, trgVol.scl)
    # grid divisions for the source and target volumes
    trgDivGrd = ntuple(itr -> maxScl[itr] .÷ trgVol.scl[itr], 3)
    srcDivGrd = ntuple(itr -> maxScl[itr] .÷ srcVol.scl[itr], 3)
    # number of cells in each source (target) division 
    trgDivCel = Tuple(trgVol.cel .÷ trgDivGrd)
    srcDivCel = Tuple(srcVol.cel .÷ srcDivGrd)
    # confirm subdivision of source and target volumes
    if prod(trgDivCel) == 0 || prod(srcDivCel) == 0
        error("Volume sizes are incompatible---volume smaller than cell.")
    end
    # create transfer information
    return GlaExtInf(minScl, trgDivGrd, srcDivGrd, trgDivCel, srcDivCel)
end

#=
The branching Fourier algorithm needs an even sum of target and source cells in
every dimension. For a cross-scale pair the counts that matter are the ones per
partition, not the ones of the full volumes. Having an odd sum of cells would
normally give incorrect results without throw an error, so here we throw.
Returns mixInf when it passes.
=#
function chkParExtInf(mixInf::GlaExtInf)
    celSum = mixInf.trgCel .+ mixInf.srcCel
    if any(isodd.(celSum))
        badDir = findall(isodd.(celSum))
        throw(ArgumentError("Cells per partition sum to an odd number in dimension(s) $(badDir): target $(mixInf.trgCel) plus source $(mixInf.srcCel) gives $(celSum), with $(mixInf.trgDiv) target and $(mixInf.srcDiv) grid divisions. Adjust the cell counts of the volumes, or the refinement factor between them, so that every dimension sums to an even number."))
    end
    return mixInf
end

"""
    genCntVol(trgVol::GlaVol, srcVol::GlaVol)

Create contact volume for self Green function calculations.

# Arguments
- `trgVol::GlaVol`: Target volume
- `srcVol::GlaVol`: Source volume

# Returns
- `GlaVol`: A new volume representing the contact region

# Notes
- The contact volume is created by taking the minimum cell size between source and target
  and combining their divisions
"""
function genCntVol(trgVol::GlaVol, srcVol::GlaVol)
    # scale for greatest common division of source and target cells
    cntScl = min.(trgVol.scl, srcVol.scl)
    # divisions of target and source cells
    trgDiv = Integer.(trgVol.scl ./ cntScl)
    srcDiv = Integer.(srcVol.scl ./ cntScl)
    # create padded self interaction domain for simplification of code
    return GlaVol(trgDiv .+ srcDiv, cntScl, (0//1, 0//1, 0//1))
end

"""
    sepGrd(trgVol::GlaVol, srcVol::GlaVol, flip::Integer)

Create grid of spanning separations for a pair of volumes.

# Arguments
- `trgVol::GlaVol`: Target volume
- `srcVol::GlaVol`: Source volume
- `flip::Integer`: Direction of separation (1 for source grid, 2 for target grid)

# Returns
- `Array{<:StepRange,1}`: Array of step ranges representing the separation grid

# Notes
- The flipped separation grid is used when generating the circulant vector
- The step direction is automatically adjusted based on the separation orientation
"""
function sepGrd(trgVol::GlaVol, srcVol::GlaVol, 
    flip::Integer)
    # grid over the source volume
    if flip == 1
        sep = step.(srcVol.grd)
        str = first.(trgVol.grd) .- last.(srcVol.grd)
        stp = first.(trgVol.grd) .- first.(srcVol.grd)
    # grid over the target volume
    else
        sep = step.(trgVol.grd)
        str = first.(trgVol.grd) .- first.(srcVol.grd)
        stp = last.(trgVol.grd) .- first.(srcVol.grd)
    end 
    # match step to separation orientation
    sep = ntuple(dir -> stp[dir] < str[dir] ? -sep[dir] : sep[dir], 3)
    return collect(map(StepRange, str, sep, stp))
end

"""
    crcIndClc(cntVol::GlaVol, trgInd::CartesianIndex{3}, srcInd::CartesianIndex{3})

Calculate circulant index for self Green function vector.

# Arguments
- `cntVol::GlaVol`: Contact volume
- `trgInd::CartesianIndex{3}`: Target index
- `srcInd::CartesianIndex{3}`: Source index

# Returns
- `CartesianIndex{3}`: The circulant index

# Notes
- This function is used to compute the index in the circulant vector for self-interactions
- The index is adjusted to handle negative separations by wrapping around the volume
"""
@inline function crcIndClc(cntVol::GlaVol, trgInd::CartesianIndex{3}, srcInd::CartesianIndex{3})
    # separation in terms of cells
    cellInd = trgInd - srcInd 
    # cellInd[dir] is always added because the value is negative if true
    return cellInd + CartesianIndex(ntuple(itr -> (cellInd[itr] < 0) * 2 * 
        cntVol.cel[itr] + 1, 3))
end

"""
    GlaCmpVol

A set of disjoint `GlaVol` regions (potentially with different scales, and
potentially separated) made to "feel" like a single volume.

# Fields
- `regions::Vector{GlaVol}`: The regions of the tiling, in the order that fixes
  the flat degree of freedom layout

# Flat degree of freedom layout

Fields over a composite volume are stored in one flat vector. Region `r`
contributes a block of `3 * prod(r.cel)` entries and the blocks are concatenated
in `regions` order. Inside a block the ordering is `vec` of an array of size
`(r.cel..., 3)`, so the first Cartesian index runs fastest and the vector
component index runs slowest. This matches the reshape that `GlaOprVac`'s `*`
applies to a flat vector, which means a region block can be reshaped into the
4-tensor an operator expects without any permutation.

`coordinates` walks the cells in this order but yields one entry per cell rather
than per degree of freedom, so it is three times shorter than the flat vector.
`cellvolumes` repeats each cell volume across the three vector components, so it
lines up with the degrees of freedom entry by entry.

# Invariants

The regions must be nonempty, have even cell counts, have pairwise commensurate
scales, sit on a common grid, have disjoint interiors (face, edge, and corner
contact is fine and expected), and satisfy the per-partition parity condition of
`chkParExtInf` for every pair. Gaps between regions are allowed: a tiling may
hold several separated bodies.
"""
struct GlaCmpVol
    regions::Vector{GlaVol}

    function GlaCmpVol(regs::Vector{GlaVol})
        chkCmpVol(regs)
        return new(copy(regs))
    end
end

const CompositeVolume = GlaCmpVol

"""
    GlaCmpVol(vol::GlaVol)

Construct the one-region composite volume around a single `GlaVol`.

# Arguments
- `vol::GlaVol`: The volume to wrap

# Returns
- `GlaCmpVol`: A composite volume with `vol` as its only region
"""
GlaCmpVol(vol::GlaVol) = GlaCmpVol([vol])

# Lower and upper corner of the cuboid occupied by a volume
_lwrEdg(vol::GlaVol) = first.(vol.grd) .- (vol.scl .// 2)
_uprEdg(vol::GlaVol) = last.(vol.grd) .+ (vol.scl .// 2)

#= Interiors share volume. Contact along a face, an edge, or a corner returns
false, since the external construction has contact corrections for it. =#
function _ovrLap(vol1::GlaVol, vol2::GlaVol)
    lwr = max.(_lwrEdg(vol1), _lwrEdg(vol2))
    upr = min.(_uprEdg(vol1), _uprEdg(vol2))
    return all(lwr .< upr)
end

# Closer than a third of a wavelength a separated pair's discretisation error
# is several times its wide separation value.
const prxSepMin = 1 / 3

# Face to face distance between two volumes, or nothing when they meet
function prxGap(volA::GlaVol, volB::GlaVol)
    sep = max.(_lwrEdg(volB) .- _uprEdg(volA), _lwrEdg(volA) .- _uprEdg(volB))
    all(sep .<= 0) && return nothing
    return sqrt(sum(x -> Float64(max(x, 0//1))^2, sep))
end

function prxChk(frq::Number, trgVol::GlaVol, srcVol::GlaVol)
    gap = prxGap(trgVol, srcVol)
    isnothing(gap) && return nothing
    sep = gap * real(frq)
    sep >= prxSepMin && return nothing
    why = iszero(imag(frq)) ?
        "A separated pair needs at least a third of a wavelength for its discretisation error to settle." :
        "The background is lossy, which needs more than a third of a wavelength; how much more depends on the problem, so measure it against a refined mesh."
    @warn "The target and source are separated by $(round(sep; sigdigits = 3)) wavelengths. $why Refine the facing surfaces with `refine`, or separate the volumes further."
    return nothing
end

"""
    regions(cvol::GlaCmpVol)

The regions of a composite volume, in flat layout order.

# Arguments
- `cvol::GlaCmpVol`: The composite volume

# Returns
- `Vector{GlaVol}`: The regions
"""
regions(cvol::GlaCmpVol) = cvol.regions

"""
    nregions(cvol::GlaCmpVol)

The number of regions in a composite volume.

# Arguments
- `cvol::GlaCmpVol`: The composite volume

# Returns
- The region count
"""
nregions(cvol::GlaCmpVol) = length(cvol.regions)

# Start of each region block in the flat layout, plus the total length at the end
function _dofOff(cvol::GlaCmpVol)
    off = zeros(Int, nregions(cvol) + 1)
    for (idx, reg) in enumerate(regions(cvol))
        off[idx + 1] = off[idx] + 3 * prod(reg.cel)
    end
    return off
end

"""
    finest(cvol::GlaCmpVol)

The smallest cell scale present in a composite volume, taken elementwise.

# Arguments
- `cvol::GlaCmpVol`: The composite volume

# Returns
- `NTuple{3,Rational}`: The elementwise minimum of the region cell scales
"""
finest(cvol::GlaCmpVol) = reduce((sclA, sclB) -> min.(sclA, sclB), (reg.scl for reg in cvol.regions))

Base.:(==)(cvolA::GlaCmpVol, cvolB::GlaCmpVol) = cvolA.regions == cvolB.regions

function Base.show(io::IO, cvol::GlaCmpVol)
    numReg = nregions(cvol)
    print(io, "Composite volume ($numReg region", numReg == 1 ? "" : "s", ")")
    for (idx, reg) in enumerate(cvol.regions)
        print(io, "\n  [$idx] (" * join(reg.cel, "×") * ") cells, (" *
            join(reg.scl, "×") * ")λ³, center (" * join(reg.org, ", ") * ")λ")
    end
end
Base.show(io::IO, ::MIME"text/plain", cvol::GlaCmpVol) = show(io, cvol)

function chkCmpVol(regs::Vector{GlaVol})
    if isempty(regs)
        throw(ArgumentError("A composite volume needs at least one region."))
    end
    for (idx, reg) in enumerate(regs)
        if any(reg.cel .< 1)
            throw(ArgumentError("Region $idx is empty: cell counts $(reg.cel). Every region must have at least one cell in every dimension."))
        end
        if any(Rational.(step.(reg.grd)) .!= reg.scl)
            throw(ArgumentError("Region $idx has grid step $(Rational.(step.(reg.grd))) but cell scale $(reg.scl). A composite volume region must be a solid cuboid, so the grid pitch has to equal the cell size."))
        end
        if any(isodd.(reg.cel))
            badDir = findall(isodd.(reg.cel))
            throw(ArgumentError("Region $idx has an odd number of cells in dimension(s) $(badDir): $(reg.cel). Gila demands even cell counts in every region."))
        end
    end
    gcdScl = reduce((sclA, sclB) -> gcd.(sclA, sclB), (reg.scl for reg in regs))
    for idxA in 1:length(regs), idxB in (idxA + 1):length(regs)
        regA, regB = regs[idxA], regs[idxB]
        chkCmnScl(regA, regB, idxA, idxB)
        if _ovrLap(regA, regB)
            throw(ArgumentError("Regions $idxA and $idxB overlap: region $idxA spans $(_lwrEdg(regA)) to $(_uprEdg(regA)) and region $idxB spans $(_lwrEdg(regB)) to $(_uprEdg(regB)). Regions may touch, but their interiors must be disjoint."))
        end
        offset = (_lwrEdg(regA) .- _lwrEdg(regB)) .// gcdScl
        if any(.!isinteger.(offset))
            throw(ArgumentError("Regions $idxA and $idxB do not share a common grid: their lower corners differ by $(Tuple(_lwrEdg(regA) .- _lwrEdg(regB))), which is $(Tuple(offset)) common cells of size $(Tuple(gcdScl)). Region corners must be on the common grid."))
        end
        chkParCmpVol(regA, regB, idxA, idxB)
    end
    return nothing
end

#= Larger over smaller has to be a whole number in every dimension, otherwise
no common grid exists for the pair. =#
function chkCmnScl(regA::GlaVol, regB::GlaVol, idxA::Integer, idxB::Integer)
    sclRat = max.(regA.scl, regB.scl) .// min.(regA.scl, regB.scl)
    if any(.!isinteger.(sclRat))
        badDir = findall(.!isinteger.(sclRat))
        throw(ArgumentError("Regions $idxA and $idxB have incommensurate cell scales in dimension(s) $(badDir): $(regA.scl) and $(regB.scl) give the ratio(s) $(Tuple(sclRat)). One scale must be an integer multiple of the other in every dimension."))
    end
    return nothing
end

#= The per-pair version of chkParExtInf, in region indices. Divisibility comes
first: GlaExtInf throws a bare error for a region smaller than one partition. =#
function chkParCmpVol(regA::GlaVol, regB::GlaVol, idxA::Integer, idxB::Integer)
    maxScl = lcm.(regA.scl, regB.scl)
    for (idx, reg) in ((idxA, regA), (idxB, regB))
        divPar = maxScl .÷ reg.scl
        badDir = findall(!iszero, mod.(reg.cel, divPar))
        isempty(badDir) || throw(ArgumentError("Region $idx cannot be partitioned against region $(idx == idxA ? idxB : idxA) in dimension(s) $(badDir): $(reg.cel) cells do not divide into $(divPar) sub-lattices. Adjust the refinement box so that the region spans a whole number of coarse cells."))
    end
    mixInf = GlaExtInf(regA, regB)
    celSum = mixInf.trgCel .+ mixInf.srcCel
    badDir = findall(isodd, celSum)
    isempty(badDir) || throw(ArgumentError("Regions $idxA and $idxB violate the partition parity condition in dimension(s) $(badDir): $(mixInf.trgCel) cells per partition in region $idxA plus $(mixInf.srcCel) in region $idxB gives $(celSum). Move the refinement box or change the refinement factor so that every dimension sums to an even number."))
    return nothing
end

_facTup(factor::Integer) = ntuple(_ -> Int(factor), 3)
_facTup(factor::NTuple{3,Integer}) = Int.(factor)

# Corners of the region to refine
_boxBnd(box::GlaVol) = (_lwrEdg(box), _uprEdg(box))
_boxBnd(box::Tuple{NTuple{3,Rational},NTuple{3,Rational}}) =
    (box[1] .- (box[2] .// 2), box[1] .+ (box[2] .// 2))

#= Cell index bounds of a box inside a region, half open and zero based. The
bounds grow outward to cell boundaries and then to even cell indices, keeping 
the core and every complement slab at an even cell count. =#
function _snpBox(vol::GlaVol, boxLwr, boxUpr)
    volLwr = _lwrEdg(vol)
    rawLwr = ntuple(dir ->
        floor(Int, (boxLwr[dir] - volLwr[dir]) // vol.scl[dir]), 3)
    rawUpr = ntuple(dir ->
        ceil(Int, (boxUpr[dir] - volLwr[dir]) // vol.scl[dir]), 3)
    idxLwr = ntuple(dir -> 2 * fld(max(rawLwr[dir], 0), 2), 3)
    idxUpr = ntuple(dir -> 2 * cld(min(rawUpr[dir], Int(vol.cel[dir])), 2), 3)
    return idxLwr, idxUpr
end

#= The piece of vol spanning cells idxLwr+1 through idxUpr, at the parent cell
scale divided by fac =#
function _subVol(vol::GlaVol, idxLwr::NTuple{3,Int}, idxUpr::NTuple{3,Int}, fac::NTuple{3,Int}=(1, 1, 1))
    volLwr = _lwrEdg(vol)
    lwr = ntuple(dir -> volLwr[dir] + idxLwr[dir] * vol.scl[dir], 3)
    upr = ntuple(dir -> volLwr[dir] + idxUpr[dir] * vol.scl[dir], 3)
    cel = ntuple(dir -> (idxUpr[dir] - idxLwr[dir]) * fac[dir], 3)
    scl = ntuple(dir -> vol.scl[dir] // fac[dir], 3)
    org = ntuple(dir -> (lwr[dir] + upr[dir]) // 2, 3)
    return GlaVol(cel, scl, org)
end

#= Split a region into the refined core plus the complement slabs, in the fixed
order core, xlo, xhi, ylo, yhi, zlo, zhi. The x slabs span the full cross
section, the y slabs only the middle in x, and the z slabs only the middle in x
and y, so the pieces are disjoint and fill the region. Empty slabs are
dropped. =#
function _crvVol(vol::GlaVol, boxLwr, boxUpr, fac::NTuple{3,Int})
    idxLwr, idxUpr = _snpBox(vol, boxLwr, boxUpr)
    cel = ntuple(dir -> Int(vol.cel[dir]), 3)
    pcs = GlaVol[_subVol(vol, idxLwr, idxUpr, fac)]
    if idxLwr[1] > 0
        push!(pcs, _subVol(vol, (0, 0, 0), (idxLwr[1], cel[2], cel[3])))
    end
    if idxUpr[1] < cel[1]
        push!(pcs, _subVol(vol, (idxUpr[1], 0, 0), (cel[1], cel[2], cel[3])))
    end
    if idxLwr[2] > 0
        push!(pcs, _subVol(vol, (idxLwr[1], 0, 0),
            (idxUpr[1], idxLwr[2], cel[3])))
    end
    if idxUpr[2] < cel[2]
        push!(pcs, _subVol(vol, (idxLwr[1], idxUpr[2], 0),
            (idxUpr[1], cel[2], cel[3])))
    end
    if idxLwr[3] > 0
        push!(pcs, _subVol(vol, (idxLwr[1], idxLwr[2], 0),
            (idxUpr[1], idxUpr[2], idxLwr[3])))
    end
    if idxUpr[3] < cel[3]
        push!(pcs, _subVol(vol, (idxLwr[1], idxLwr[2], idxUpr[3]),
            (idxUpr[1], idxUpr[2], cel[3])))
    end
    return pcs
end

"""
    refine(cvol::GlaCmpVol, box; factor=2)

Refine a composite volume inside a box by carving.

Every region whose interior meets the box is replaced by a refined core plus the
complement slabs that fill the rest of the region. The core takes `factor` times
as many cells at `factor` times the resolution over the same physical extent.
Regions the box misses are left alone and keep their position in the region
list, so the pieces of region `i` appear where region `i` used to be, core
first, then the nonempty slabs in the order xlo, xhi, ylo, yhi, zlo, zhi.

Snapping happens per region. The box is first intersected with the region, then
grown outward to that region's cell boundaries, then grown outward again to even
cell indices so that the core and every slab keep an even cell count. A box that
crosses regions of different scale therefore snaps differently in each of them.
The core always covers at least the requested box.

You may refine a volume multiple times, even in nested a fashion.

# Arguments
- `cvol::GlaCmpVol`: The composite volume to refine
- `box`: The refinement box, either a `GlaVol` whose outer bounds are used, or a
  tuple `(org, dims)` of rational center and side lengths
- `factor=2`: The refinement factor, an `Integer` or an `NTuple{3,Integer}` with
  every entry at least 1. An entry of 1 leaves that direction at the parent
  resolution while still carving the region.

# Returns
- `GlaCmpVol`: The refined composite volume, validated by the constructor

# Throws
- `ArgumentError`: If any refinement factor is less than 1, or if the resulting
  tiling violates an invariant of `GlaCmpVol`
"""
function refine(cvol::GlaCmpVol, box::Union{GlaVol, Tuple{NTuple{3,Rational},NTuple{3,Rational}}};
    factor::Union{Integer, NTuple{3,Integer}}=2)
    fac = _facTup(factor)
    if any(fac .< 1)
        throw(ArgumentError("Refinement factors must be at least 1, got $(fac)."))
    end
    boxLwr, boxUpr = _boxBnd(box)
    newRegs = GlaVol[]
    for reg in cvol.regions
        lwr = max.(boxLwr, _lwrEdg(reg))
        upr = min.(boxUpr, _uprEdg(reg))
        if all(lwr .< upr)
            append!(newRegs, _crvVol(reg, lwr, upr, fac))
        else
            push!(newRegs, reg)
        end
    end
    return GlaCmpVol(newRegs)
end

"""
    refine(vol::GlaVol, box; factor=2)

Refine a single volume inside a box, returning a composite volume.

# Arguments
- `vol::GlaVol`: The volume to refine
- `box`: The refinement box, either a `GlaVol` or a tuple `(org, dims)`
- `factor=2`: The refinement factor, an `Integer` or an `NTuple{3,Integer}`

# Returns
- `GlaCmpVol`: The refined composite volume
"""
refine(vol::GlaVol, box; factor=2) = refine(GlaCmpVol(vol), box; factor)

struct GlaCmpCrd
    cvol::GlaCmpVol
end

"""
    coordinates(cvol::GlaCmpVol)

Iterate over the cells of a composite volume in flat layout order.

The iterator yields one `(pos, vol, idx)` triple per cell: `pos` is the cell
center in wavelengths, `vol` is the cell volume in cubic wavelengths, and `idx`
is the index of the region the cell belongs to. Cells come out region by region
and, inside a region, in column major order over the cell grid, which is the
same order the flat field layout uses. There is no repetition over the three
vector components, so the iterator is a third as long as a field vector.

# Arguments
- `cvol::GlaCmpVol`: The composite volume

# Returns
- An iterator of `Tuple{NTuple{3,Float64},Float64,Int}`
"""
coordinates(cvol::GlaCmpVol) = GlaCmpCrd(cvol)

Base.eltype(::Type{GlaCmpCrd}) = Tuple{NTuple{3,Float64},Float64,Int}
Base.length(itr::GlaCmpCrd) = sum(prod(reg.cel) for reg in itr.cvol.regions)
Base.IteratorSize(::Type{GlaCmpCrd}) = Base.HasLength()

function Base.iterate(itr::GlaCmpCrd, stt::Tuple{Int,Int}=(1, 1))
    regIdx, celIdx = stt
    regs = itr.cvol.regions
    regIdx > length(regs) && return nothing
    reg = regs[regIdx]
    if celIdx > prod(reg.cel)
        return iterate(itr, (regIdx + 1, 1))
    end
    crtInd = CartesianIndices(Tuple(reg.cel))[celIdx]
    pos = ntuple(dir -> Float64(reg.grd[dir][crtInd[dir]]), 3)
    return ((pos, Float64(prod(reg.scl)), regIdx), (regIdx, celIdx + 1))
end

"""
    cellvolumes(cvol::GlaCmpVol)

Cell volumes of a composite volume, one entry per degree of freedom.

The result has length `sum(3 * prod(r.cel) for r in regions(cvol))` and lines up
entry by entry with a flat field vector: each cell volume is repeated across the
three vector components of that cell. A region has a single cell size, so the
whole block of a region carries one value.

# Arguments
- `cvol::GlaCmpVol`: The composite volume

# Returns
- `Vector{Float64}`: The cell volumes in cubic wavelengths
"""
function cellvolumes(cvol::GlaCmpVol)
    celVol = Float64[]
    for reg in cvol.regions
        append!(celVol, fill(Float64(prod(reg.scl)), 3 * prod(reg.cel)))
    end
    return celVol
end
