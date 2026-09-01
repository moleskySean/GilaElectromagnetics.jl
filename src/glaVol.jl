module GilaVolumes
"""
    GilaVolumes

This module provides memory structures for representing and manipulating spatial volumes
in the Gila package. It includes functionality for creating and managing cuboid volumes,
handling grid layouts, and computing volume interactions.

# Types
- `GlaVol`: Basic spatial memory structure for a rectangular prism volume
- `GlaExtInf`: Information for mapping between general source and target volumes

# Functions
- `genVolEve`: Regenerate a volume with an even number of cells
- `genEveExtInf`: Generate mixed information for source and target volumes
- `genCntVol`: Create contact volume for self Green function calculations
- `sepGrd`: Create grid of spanning separations for a pair of volumes
- `crcIndClc`: Calculate circulant index for self Green function vector
"""

export GlaVol, GlaExtInf, genVolEve, genEveExtInf, genCntVol, sepGrd, crcIndClc, uniVol
export GlaCmpVol, CompositeVolume, refine, regions, nregions, coordinates, cellvolumes, finest

"""
    GlaVol

Basic spatial memory structure for a rectangular prism volume.

# Fields
- `cel::NTuple{3,Integer}`: Number of cells in each dimension of the volume
- `scl::NTuple{3,Rational}`: Relative side length of a cuboid voxel (cell) compared to the wavelength
- `org::NTuple{3,Rational}`: Center position of the domain
- `grd::Array{<:StepRange,1}`: Spatial location of the center of each cell contained in the volume
"""
struct GlaVol
    cel::NTuple{3,Integer}
    scl::NTuple{3,Rational}
    org::NTuple{3,Rational}
    grd::Array{<:StepRange,1}
    # boundary conditions here?
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
    
    if !prod(celScl .<= grdScl)
        error("The cell scale must be smaller than the grid scale to avoid 
        partially overlapping basis elements.")
    end 
    brd = grdScl .* (Rational.(floor.(cel ./ 2)) .- (iseven.(cel) .// 2))
    grd = map(StepRange, org .- brd, grdScl, org .+ brd)
    return GlaVol(Tuple(cel), celScl, org, [grd...])
end

function Base.:(==)(a::GlaVol, b::GlaVol)
    return a.cel == b.cel && a.scl == b.scl && a.org == b.org && a.grd == b.grd
end

# Create the union volume of two overlapping volumes
function uniVol(vol1::GlaVol, vol2::GlaVol)
    sclMin = min.(vol1.scl, vol2.scl)
    sclMax = max.(vol1.scl, vol2.scl)
    sclRat = sclMax .// sclMin
    @assert all(isinteger.(sclRat)) "Volumes must share a common scale grid for overlap handling"
    scl = sclMin # Pick the finer scale as the common scale

    # Upper/lower edges of the volumes
    lwrEdg1 = first.(vol1.grd) .- (vol1.scl .// 2)
    uprEdg1 = last.(vol1.grd) .+ (vol1.scl .//2)
    lwrEdg2 = first.(vol2.grd) .- (vol2.scl .// 2)
    uprEdg2 = last.(vol2.grd) .+ (vol2.scl .//2)
    minEdg = min.(lwrEdg1, lwrEdg2) # Lower edge of the union volume
    maxEdg = max.(uprEdg1, uprEdg2) # Upper edge of the union volume

    cel = (maxEdg .- minEdg) .// scl
    @assert all(isinteger.(cel)) "Computed union volume cell counts must be integers"
    cel = Tuple(numerator.(cel))
    org = Tuple(minEdg .+ (cel .* scl .// 2))
    return GlaVol(cel, scl, org)
end

Base.union(vol1::GlaVol, vol2::GlaVol) = uniVol(vol1, vol2)
Base.union(vols::GlaVol...) where {T<:GlaVol} = reduce(uniVol, vols)

"""
    genVolEve(glaVol::GlaVol)

Regenerate a volume with an even number of cells.

# Arguments
- `glaVol::GlaVol`: The volume to regenerate

# Returns
- `GlaVol`: The regenerated volume with an even number of cells

# Notes
- If the volume already has an even number of cells in all dimensions, it is returned unchanged
- If any dimension has an odd number of cells, the volume is regenerated with one additional cell
  in those dimensions and the cell sizes are adjusted accordingly
"""
function genVolEve(glaVol::GlaVol)
    # check that the number of cells in each Cartesian dimension is even
    celParVec = iseven.(glaVol.cel)
    # number of cells is odd in some direction
    if any(celParVec .== false)
        # warn user that the volume is being regenerated.
        @warn "A volume has been regenerated to have an even number of cells---the size of a cell has changed. The sum of the number of source and target cells must be even for the algorithm to function."
        # determine dimensions where cells will be scaled. 
        parVec = .!celParVec
        # adjust number of cells
        newCelNum = glaVol.cel .+ parVec
        # adjust size of cells
        newCelScl = map(//, numerator.(glaVol.scl) .* glaVol.cel, 
            denominator.(glaVol.scl) .* newCelNum)  
        # adjust grid scale
        oldGrdScl = Rational.(step.(glaVol.grd)) 
        newGrdScl = map(//, numerator.(oldGrdScl) .* glaVol.cel, 
            denominator.(oldGrdScl) .* newCelNum)
        # regenerate volume
        return GlaVol(Tuple(newCelNum), newCelScl, glaVol.org, Tuple(newGrdScl))
    # otherwise, everything is fine
    else
        return glaVol
    end
end

"""
    GlaExtInf 

Information for mapping between general source and target volumes.

# Fields
- `minScl::NTuple{3,Rational}`: Common minimum cell size (in units of wavelength)
- `trgDiv::NTuple{3,Integer}`: Divisions in each cartesian index of source volume
- `srcDiv::NTuple{3,Integer}`: Divisions in each cartesian index of target volume
- `trgCel::NTuple{3,Integer}`: Cells in a target partition
- `srcCel::NTuple{3,Integer}`: Cells in a source partition
- `trgPar::CartesianIndices`: Identification of volume partition with grid offsets in target volume
- `srcPar::CartesianIndices`: Identification of volume partition with grid offsets in source volume
"""
mutable struct GlaExtInf
    minScl::NTuple{3,Rational}
    trgDiv::NTuple{3,Integer}
    srcDiv::NTuple{3,Integer}
    trgCel::NTuple{3,Integer}
    srcCel::NTuple{3,Integer}
    trgPar::CartesianIndices
    srcPar::CartesianIndices
end

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
    # association of volume partition with grid offset
    trgPar = CartesianIndices(tuple(trgDivGrd...)) .- CartesianIndex(1, 1, 1)
    srcPar = CartesianIndices(tuple(srcDivGrd...)) .- CartesianIndex(1, 1, 1)
    # create transfer information
    return GlaExtInf(minScl, trgDivGrd, srcDivGrd, trgDivCel, srcDivCel, 
        trgPar, srcPar)
end

"""
    genEveExtInf(trgVol::GlaVol, srcVol::GlaVol)

Generate mixed information for source and target volumes, regenerating the volumes if necessary.

# Arguments
- `trgVol::GlaVol`: Target volume
- `srcVol::GlaVol`: Source volume

# Returns
- `GlaExtInf`: The transfer information

# Notes
- If the sum of cells in source and target volumes is odd, both volumes are regenerated
  to have an even number of cells

# Throws
- `ArgumentError`: If any dimension has an odd sum of target and source cells per partition
"""
function genEveExtInf(trgVol::GlaVol, srcVol::GlaVol)
    # ensure that sum number of cells is even, regenerate if not
    if sum(mod.(trgVol.cel .+ srcVol.cel, 2)) != 0
        # regenerate target volume so that number of cells is even
        trgVol = genVolEve(trgVol)
        # regenerate source volume so that number of cells is even
        srcVol = genVolEve(srcVol)
    end
    # useful information for aligning source and target volumes
    mixInf = GlaExtInf(trgVol, srcVol)
    chkParExtInf(mixInf) # Make sure we have an even sum of cells per partition
    return mixInf
end

#=
The branching Fourier algorithm needs an even sum of target and source cells in
every dimension. For a cross-scale pair the counts that matter are the ones per
partition, not the ones of the full volumes. Having an odd sum of cells would
normally give incorrect results without throw an error, so here we throw.
=#
function chkParExtInf(mixInf::GlaExtInf)
    celSum = mixInf.trgCel .+ mixInf.srcCel
    if any(isodd.(celSum))
        badDir = findall(isodd.(celSum))
        throw(ArgumentError("Cells per partition sum to an odd number in dimension(s) $(badDir): target $(mixInf.trgCel) plus source $(mixInf.srcCel) gives $(celSum), with $(mixInf.trgDiv) target and $(mixInf.srcDiv) grid divisions. Adjust the cell counts of the volumes, or the refinement factor between them, so that every dimension sums to an even number."))
    end
    return nothing
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
    cntCel = (trgDiv[1] + srcDiv[1], trgDiv[2] + srcDiv[2], 
        trgDiv[3] + srcDiv[3])
    return GlaVol(cntCel, cntScl, (0//1, 0//1, 0//1))
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
        sep = getproperty.(srcVol.grd, :step)
        str = getproperty.(trgVol.grd, :start) .- 
            getproperty.(srcVol.grd, :stop)
        stp = getproperty.(trgVol.grd, :start) .- 
            getproperty.(srcVol.grd, :start)
    # grid over the target volume
    else
        sep = getproperty.(trgVol.grd, :step)
        str = getproperty.(trgVol.grd, :start) .- 
                getproperty.(srcVol.grd, :start)
        stp = getproperty.(trgVol.grd, :stop) .- 
                getproperty.(srcVol.grd, :start)
    end 
    # match step to separation orientation
    for dir ∈ 1:3
        if stp[dir] < str[dir]
            sep[dir] *= -1
        end
    end
    return map(StepRange, str, sep, stp) 
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

include("glaCmpVol.jl")

end # module
