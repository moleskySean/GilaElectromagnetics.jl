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

	cel::Array{<:Integer,1}
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
.trgPar---identification of volume partition with grid position in target volume
.srcPar---identification of volume partition with grid position in source volume
=#
struct GlaExtInf

	minScl::NTuple{3,<:Rational}
	trgDiv::NTuple{3,<:Integer}
	srcDiv::NTuple{3,<:Integer}
	trgCel::Array{<:Integer,1}
	srcCel::Array{<:Integer,1}
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
struct GlaKerOpt

	frqPhz::Number
	intOrd::Integer
	adjMod::Bool
	devMod::Union{Bool, Array{<:Bool,1}}
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
	dimInfC::AbstractArray{<:Integer,1} 
	dimInfD::AbstractArray{<:Integer,1} 
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
function GlaVol(cel::Array{<:Integer,1}, celScl::NTuple{3,<:Rational}, 
	org::NTuple{3,<:Rational}, grdScl::NTuple{3,<:Rational}=celScl)::GlaVol
	
	if !prod(celScl .<= grdScl)
		error("The cell scale must be smaller than the grid scale to avoid 
		partially overlapping basis elements.")
	end	
	brd = @. grdScl * (cel - 1) // 2
	grd = map(StepRange, org .- brd, grdScl, org .+ brd)
	return GlaVol(cel, celScl, org, grd)
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
		map(!, celParVec)
		# adjust number of cells
		newCelNum = glaVol.cel .+ celParVec
		# adjust size of cells
		newCelScl = map(//, numerator.(glaVol.scl), denominator.(glaVol.scl) 
			.+ celParVec)	
		# adjust grid scale
		oldGrdScl = Rational.(step.(glaVol.grd)) 
		newGrdScl = map(//, numerator.(oldGrdScl), denominator.(oldGrdScl) 
			.+ celParVec)
		# regenerate volume
		return GlaVol(newCelnum, newCelScl, glaVol.org, newGrdScl)
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
		error("Volume pair must share a common scale grid in order to construct 
		Green function.")
	end
	# common scale 
	minScl = gcd.(srcVol.scl, trgVol.scl)
	# maximal scale
	maxScl = lcm.(srcVol.scl, trgVol.scl)
	# grid divisions for the source and target volumes
	trgDivGrd = Int.(ntuple(itr -> maxScl[itr] ./ trgVol.scl[itr], 3))
	srcDivGrd = Int.(ntuple(itr -> maxScl[itr] ./ srcVol.scl[itr], 3))
	# number of cells in each source (target) division 
	trgDivCel = Int.(trgVol.cel ./ trgDivGrd)
	srcDivCel = Int.(srcVol.cel ./ srcDivGrd)
	# association of volume partition with grid position
	trgPar = CartesianIndices(tuple(trgDivGrd...))
	srcPar = CartesianIndices(tuple(srcDivGrd...))
	# create transfer information
	return GlaExtInf(minScl, trgDivGrd, srcDivGrd, trgDivCel, srcDivCel, 
		trgPar,srcPar)
end
"""
GlaKerOpt(devStt::Bool)

Simplified GlaKerOpt constructor.
"""
function GlaKerOpt(devStt::Bool)

	if devStt == true
		return GlaKerOpt(1.0 + 0.0im, 48, false, true, (128, 2, 1), 
			(1, 128, 256))
	else
		return GlaKerOpt(1.0 + 0.0im, 48, false, false, (), ())
	end
end