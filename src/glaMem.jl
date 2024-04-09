"""
GlaVol

Basic spatial memory structure.
.cel---tuple of cells in rectangular prism 
.scl--relative side length of a cuboid voxel compared to the wavelength 
.org---center position of the domain 
.grd---spatial location of the center of each cell contained in the volume 
"""
#=
To simply code operation Gila internally enforces an even number of cells 
during the memory preparation phase---see GlaOprMem in glaMemSup.jl.
=#
struct GlaVol

	cel::NTuple{3,<:Integer}
	scl::NTuple{3,<:Rational}
	org::NTuple{3,<:Rational}
	grd::Array{<:StepRange,1}
	# boundary conditions here?
end
#=
GlaExtInf 

Information for mapping between general source and target volumes.
.minScl---common minimum cell size
.srcDiv---divisions in each Cartesian index of source volume 
.trgDiv---divisions in each Cartesian index of target volume
.trgCel---cells in a target partition  
.srcCel---cells in a source partition
.trgPar---identification of volume partition with grid offsets in target volume
.srcPar---identification of volume partition with grid offsets in source volume
=#
struct GlaExtInf

	minScl::NTuple{3,<:Rational}
	trgDiv::NTuple{3,<:Integer}
	srcDiv::NTuple{3,<:Integer}
	trgCel::NTuple{3,<:Integer}
	srcCel::NTuple{3,<:Integer}
	trgPar::CartesianIndices
	srcPar::CartesianIndices
end
"""
GlaKerOpt

Green function operator assembly and kernel operation options.
.frqPhz---multiplicative scaling factor allowing for complex frequencies
.intOrd---Gauss-Legendre integration order for cells in contact
.adjMod---flip between operator and operator adjoint
.devMod---boolean vector representing activation of GPUs
.numTrd---number of threads to use when running GPU kernels
.numBlk---number of threads to use when running GPU kernels 
"""
#=
If inConTest.jl was failed the default intOrd used in the simplified constructor
may not be sufficient to insure that all integral values are properly converged.
It may be prudent to create the associated GlaKerOpt with higher order. 
=#
struct GlaKerOpt

	frqPhz::Number
	intOrd::Integer
	adjMod::Bool
	devMod::Union{Bool,Array{<:Bool,1}}
	numTrd::Union{Tuple{},NTuple{3,<:Integer}}
	numBlk::Union{Tuple{},NTuple{3,<:Integer}}
end
"""
GlaOprMem

Storage structure for a Green function operator.
.cmpInf---computation information see GlaKerOpt
.trgVol---target volume of Green function
.srcVol---source volume of Green function
.mixInf---information for matching source and target grids, see GlaExtInf
.dimInfC---dimension information for Green function volumes, host side
.dimInfD---dimension information for Green function volumes, device side
.egoFur---unique Fourier transform data for circulant Green function
.fftPlnFwd---forward Fourier transform plans
.fftPlnRev---reverse Fourier transform plans
.phzInf---phase vector for splitting Fourier transforms
"""
struct GlaOprMem
	
	cmpInf::GlaKerOpt
	trgVol::GlaVol
	srcVol::GlaVol
	mixInf::GlaExtInf
	dimInf::NTuple{3,<:Integer} 
	egoFur::AbstractArray{<:AbstractArray{T},1} where 
	T<:Union{ComplexF64,ComplexF32}
	fftPlnFwd::AbstractArray{<:AbstractFFTs.Plan,1}
	fftPlnRev::AbstractArray{<:AbstractFFTs.ScaledPlan,1}
	phzInf::AbstractArray{<:AbstractArray{T},1} where 
	T<:Union{ComplexF64,ComplexF32}
end
#=
Constructors
=#
"""
	GlaVol(cel::Array{<:Integer,1}, celScl::NTuple{3,<:Rational}, 
	org::NTuple{3,<:Rational}, grdScl::NTuple{3,<:Rational}=celScl)::GlaVol

Constructor for Gila Volumes.
"""
function GlaVol(cel::Union{Array{<:Integer,1},NTuple{3,<:Integer}}, 
	celScl::NTuple{3,<:Rational}, org::NTuple{3,<:Rational}, 
	grdScl::NTuple{3,<:Rational}=celScl)::GlaVol
	
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

Simplified GlaKerOpt constructor.
"""
function GlaKerOpt(devStt::Bool)

	if devStt == true
		return GlaKerOpt(1.0 + 0.0im, 32, false, true, (128, 2, 1), 
			(1, 128, 256))
	else
		return GlaKerOpt(1.0 + 0.0im, 32, false, false, (), ())
	end
end