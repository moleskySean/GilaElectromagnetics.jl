"""
    GlaCmpVol

A set of disjoint `GlaVol` regions (potentially with different scales) tiling
one rectangular domain but made to "feel" like single volume.

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
contact is fine and expected), exactly tile the bounding box of their union, and
satisfy the per-partition parity condition of `chkParExtInf` for every pair.
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

#= Lower and upper corner of the cuboid occupied by a volume. Written in terms
of the grid and the cell scale, the same way ovrChk measures edges. =#
_lwrEdg(vol::GlaVol) = Tuple(first.(vol.grd) .- (vol.scl .// 2))
_uprEdg(vol::GlaVol) = Tuple(last.(vol.grd) .+ (vol.scl .// 2))

#= Interiors share volume. Contact along a face, an edge, or a corner returns
false, since the external construction has contact corrections for it. =#
function _ovrLap(vol1::GlaVol, vol2::GlaVol)
    lwr = max.(_lwrEdg(vol1), _lwrEdg(vol2))
    upr = min.(_uprEdg(vol1), _uprEdg(vol2))
    return all(lwr .< upr)
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

"""
    finest(cvol::GlaCmpVol)

The smallest cell scale present in a composite volume, taken elementwise.

# Arguments
- `cvol::GlaCmpVol`: The composite volume

# Returns
- `NTuple{3,Rational}`: The elementwise minimum of the region cell scales
"""
function finest(cvol::GlaCmpVol)
    scl = reduce((sclA, sclB) -> min.(sclA, sclB),
        (reg.scl for reg in cvol.regions))
    return Tuple(scl)
end

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
            throw(ArgumentError("Region $idx has grid step $(Tuple(Rational.(step.(reg.grd)))) but cell scale $(reg.scl). A composite volume region must be a solid cuboid, so the grid pitch has to equal the cell size."))
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
    chkTilCmpVol(regs)
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

#= The per-pair version of chkParExtInf: count the cells in one partition the
way GlaExtInf does and require an even sum. =#
function chkParCmpVol(regA::GlaVol, regB::GlaVol, idxA::Integer, idxB::Integer)
    maxScl = lcm.(regA.scl, regB.scl)
    divA = ntuple(dir -> maxScl[dir] ÷ regA.scl[dir], 3)
    divB = ntuple(dir -> maxScl[dir] ÷ regB.scl[dir], 3)
    for (idx, reg, divPar) in ((idxA, regA, divA), (idxB, regB, divB))
        if any(mod.(reg.cel, divPar) .!= 0)
            badDir = findall(mod.(reg.cel, divPar) .!= 0)
            throw(ArgumentError("Region $idx cannot be partitioned against region $(idx == idxA ? idxB : idxA) in dimension(s) $(badDir): $(reg.cel) cells do not divide into $(divPar) sub-lattices. Adjust the refinement box so that the region spans a whole number of coarse cells."))
        end
    end
    celParA = regA.cel .÷ divA
    celParB = regB.cel .÷ divB
    celSum = celParA .+ celParB
    if any(isodd.(celSum))
        badDir = findall(isodd.(celSum))
        throw(ArgumentError("Regions $idxA and $idxB violate the partition parity condition in dimension(s) $(badDir): $(celParA) cells per partition in region $idxA plus $(celParB) in region $idxB gives $(celSum). Move the refinement box or change the refinement factor so that every dimension sums to an even number."))
    end
    return nothing
end

#= Disjoint regions whose volumes add up to the volume of their bounding box
tile that box exactly. =#
function chkTilCmpVol(regs::Vector{GlaVol})
    minEdg = reduce((edgA, edgB) -> min.(edgA, edgB), _lwrEdg.(regs))
    maxEdg = reduce((edgA, edgB) -> max.(edgA, edgB), _uprEdg.(regs))
    bndVol = prod(maxEdg .- minEdg)
    regVol = sum(prod(reg.cel .* reg.scl) for reg in regs)
    if regVol != bndVol
        throw(ArgumentError("The regions do not tile their bounding box: they fill $(regVol) λ³ of a box of $(bndVol) λ³ spanning $(Tuple(minEdg)) to $(Tuple(maxEdg)). A composite volume is one solid cuboid, so gaps are not allowed. Use one composite volume per body."))
    end
    return nothing
end

_facTup(factor::Integer) = ntuple(_ -> Int(factor), 3)
_facTup(factor::NTuple{3,Integer}) = Int.(factor)
_facTup(factor) = throw(ArgumentError("A refinement factor must be an Integer or an NTuple{3,Integer}, got $(typeof(factor))."))

# Corners of the region to refine
_boxBnd(box::GlaVol) = (_lwrEdg(box), _uprEdg(box))
_boxBnd(box::Tuple{NTuple{3,Rational},NTuple{3,Rational}}) =
    (box[1] .- (box[2] .// 2), box[1] .+ (box[2] .// 2))
_boxBnd(box) = throw(ArgumentError("A refinement box must be a GlaVol or a tuple (org, dims) of rational center and side lengths, got $(typeof(box))."))

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
    refine(cvol::GlaCmpVol, box; factor=2, snap::Symbol=:outward)

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
- `snap::Symbol=:outward`: The snapping mode. Only `:outward` exists.

# Returns
- `GlaCmpVol`: The refined composite volume, validated by the constructor

# Throws
- `ArgumentError`: If `snap` is not `:outward`, if any refinement factor is less
  than 1, if `box` is not a supported type, or if the resulting tiling violates
  an invariant of `GlaCmpVol`
"""
function refine(cvol::GlaCmpVol, box; factor=2, snap::Symbol=:outward)
    if snap !== :outward
        throw(ArgumentError("Unknown snapping mode :$(snap). Only :outward is implemented."))
    end
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
    refine(vol::GlaVol, box; factor=2, snap::Symbol=:outward)

Refine a single volume inside a box, returning a composite volume.

# Arguments
- `vol::GlaVol`: The volume to refine
- `box`: The refinement box, either a `GlaVol` or a tuple `(org, dims)`
- `factor=2`: The refinement factor, an `Integer` or an `NTuple{3,Integer}`
- `snap::Symbol=:outward`: The snapping mode

# Returns
- `GlaCmpVol`: The refined composite volume
"""
refine(vol::GlaVol, box; factor=2, snap::Symbol=:outward) =
    refine(GlaCmpVol(vol), box; factor=factor, snap=snap)

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
