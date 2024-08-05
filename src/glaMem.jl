"""
  GlaVol

Basic spatial memory structure for a volume.

# Fields
- `cel::NTuple{3,Integer}`: Tuple of cells in rectangular prism.
- `scl::NTuple{3,Rational}`: Relative side length of a cuboid voxel (cell) compared to the wavelength.
- `org::NTuple{3,Rational}`: Center position of the domain.
- `grd::Array{<:StepRange,1}`: Spatial location of the center of each cell contained in the volume.
"""
struct GlaVol

	cel::NTuple{3,Integer}
	scl::NTuple{3,Rational}
	org::NTuple{3,Rational}
	grd::Array{<:StepRange,1}
	# boundary conditions here?
end
#=
To simply code operation Gila internally enforces an even number of cells 
during the memory preparation phase---see GlaOprMem in glaMemSup.jl.
=#

"""
    GlaExtInf 

Information for mapping between general source and target volumes.

# Fields
- `minScl::NTuple{3,Rational}`: Common minimum cell size (in units of wavelength).
- `trgDiv::NTuple{3,Integer}`: Divisions in each cartesian index of source volume.
- `srcDiv::NTuple{3,Integer}`: Divisions in each cartesian index of target volume.
- `trgCel::NTuple{3,Integer}`: Cells in a target partition.
- `srcCel::NTuple{3,Integer}`: Cells in a source partition.
- `trgPar::CartesianIndices`: Identification of volume partition with grid offsets in target volume.
- `srcPar::CartesianIndices`: Identification of volume partition with grid offsets in source volume.
"""
struct GlaExtInf

	minScl::NTuple{3,Rational}
	trgDiv::NTuple{3,Integer}
	srcDiv::NTuple{3,Integer}
	trgCel::NTuple{3,Integer}
	srcCel::NTuple{3,Integer}
	trgPar::CartesianIndices
	srcPar::CartesianIndices
end
"""
  GlaKerOpt

Green function operator assembly and kernel operation options.

# Fields
- `frqPhz::Number`: Multiplicative scaling factor allowing for complex frequencies.
- `intOrd::Integer`: Gauss-Legendre integration order for cells in contact.
- `adjMod::Bool`: Flip between operator and operator adjoint.
- `devMod::Union{Bool,Array{<:Bool,1}}`: Boolean vector representing activation of GPUs
- `numTrd::Union{Tuple{},NTuple{3,Integer}}`: Number of threads to use when running GPU kernels.
- `numBlk::Union{Tuple{},NTuple{3,Integer}}`: Number of threads to use when running GPU kernels.
"""
struct GlaKerOpt

	frqPhz::Number
	intOrd::Integer
	adjMod::Bool
	devMod::Union{Bool,Array{<:Bool,1}}
	numTrd::Union{Tuple{},NTuple{3,Integer}}
	numBlk::Union{Tuple{},NTuple{3,Integer}}
end
#=
If intConTest.jl was failed the default intOrd used in the simplified constructor
may not be sufficient to insure that all integral values are properly converged.
It may be prudent to create the associated GlaKerOpt with higher order. 
=#

"""
  GlaOprMem

Storage structure for a Green's function operator.

# Fields
- `cmpInf::GlaKerOpt`: Computation information, settings and kernel options, see `GlaKerOpt`.
- `trgVol::GlaVol`: Target volume of Green function.
- `srcVol::GlaVol`: Source volume of Green function.
- `mixInf::GlaExtInf`: Information for matching source and target grids, see `GlaExtInf`.
- `dimInf::NTuple{3,Integer}`: Dimension information for Green function volumes, host side.
- `egoFur::AbstractArray{<:AbstractArray{T},1}`: Unique Fourier transform data for circulant Green function.
- `fftPlnFwd::AbstractArray{<:AbstractFFTs.Plan,1}`: Forward Fourier transform plans.
- `fftPlnRev::AbstractArray{<:AbstractFFTs.ScaledPlan,1}`: Reverse Fourier transform plans.
- `phzInf::AbstractArray{<:AbstractArray{T},1}`: Phase vector for splitting Fourier transforms.
"""
struct GlaOprMem
	
	cmpInf::GlaKerOpt
	trgVol::GlaVol
	srcVol::GlaVol
	mixInf::GlaExtInf
	dimInf::NTuple{3,Integer} 
	egoFur::AbstractArray{<:AbstractArray{T},1} where 
	T<:Union{ComplexF64,ComplexF32}
	fftPlnFwd::AbstractArray{<:AbstractFFTs.Plan,1}
	fftPlnRev::AbstractArray{<:AbstractFFTs.ScaledPlan,1}
	phzInf::AbstractArray{<:AbstractArray{T},1} where 
	T<:Union{ComplexF64,ComplexF32}
end

"""
	GlaOpr

Abstraction wrapper for GlaOprMem. 

# Fields
- `mem::GlaOprMem`: Data to process the Green function, see `GlaOprMem`.
"""
struct GlaOpr

	mem::GlaOprMem
end

#=
Constructors
=#
"""
	  GlaVol(cel::Array{<:Integer,1}, celScl::NTuple{3,Rational}, 
	org::NTuple{3,Rational}, grdScl::NTuple{3,Rational}=celScl)::GlaVol

Constructor for Gila Volumes.

# Arguments
- `cel::Array{<:Integer,1}`: Array of the number of cells in each dimension of the volume.
- `celScl::NTuple{3,Rational}`: The size of each cell in each dimensions of the volume (in units of wavelength).
- `org::NTuple{3,Rational}`: The origin of the volume in each dimension (in units of wavelength).
- `grdScl::NTuple{3,Rational}=celScl`: Spatial location of the center of each cell contained in the volume.
"""
function GlaVol(cel::Union{Array{<:Integer,1},NTuple{3,Integer}}, 
	celScl::NTuple{3,Rational}, org::NTuple{3,Rational}, 
	grdScl::NTuple{3,Rational}=celScl)::GlaVol
	
	if !prod(celScl .<= grdScl)
		error("The cell scale must be smaller than the grid scale to avoid 
		partially overlapping basis elements.")
	end	
	brd = grdScl .* (Rational.(floor.(cel ./ 2)) .- (iseven.(cel) .// 2))
	grd = map(StepRange, org .- brd, grdScl, org .+ brd)
	return GlaVol(Tuple(cel), celScl, org, [grd...])
end
#=
Regenerate a GlaVol enforcing that the number of cells is even. Called in 
GlaOprMem constructor.
=#
function glaVolEveGen(glaVol::GlaVol)::GlaVol
	# check that the number of cells in each Cartesian dimension is even
	celParVec = iseven.(glaVol.cel)
	# number of cells is odd in some direction
	if prod(celParVec) != 1
		# warn user that the volume is being regenerated.
		println("Warning! A volume has been regenerated to have an even number of cells---the size of a cell has changed. The sum of the number of source and target cells must be even for the algorithm to function.")
		# determine dimensions where cells will be scaled. 
		parVec = map(!, celParVec)
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

#=
Internal constructor for external pair information, treating grid mismatch. 
=#
"""
    GlaExtInf(trgVol::GlaVol, srcVol::GlaVol)::GlaExtInf

Constructor for `GlaExtInf, a memory structure for translating between distinct grid layouts. Treats grid mismatch.

# Arguments
- `trgVol::GlaVol`: Target volume definition, see `GlaVol`.
- `srcVol::GlaVol`: Source volume definition, see `GlaVol`.
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
    GlaKerOpt(devStt::Bool)

Simplified GlaKerOpt constructor where default parameters are given depending on the GPU activation.

# Arguments
- `devStt::Bool`: Whether to activate the GPU (true) or CPU (false).
"""
function GlaKerOpt(devStt::Bool)

	if devStt == true
		return GlaKerOpt(1.0 + 0.0im, 32, false, true, (128, 2, 1), 
			(1, 128, 256))
	else
		return GlaKerOpt(1.0 + 0.0im, 32, false, false, (), ())
	end
end
