"""
The GilaStruc module defines the layout of the computational structures used 
in Gila. The code distributed under GNU LGPL.

Author: Sean Molesky 

Reference: Sean Molesky, Gila documentation section ...
"""
module GilaMem
using AbstractFFTs
export GlaVol, GlaKerOpt, GlaOprMem, GlaTrfInf
"""
Basic Gila spatial memory structure.
.cel---tuple of cells for defining a rectangular prism 
.scl--relative side length of a cuboid voxel compared to the wavelength 
.org---center position of the domain 
.grd---spatial location of the center of each cell contained in the volume 
"""
struct GlaVol

	cel::Array{<:Integer,1}
	scl::NTuple{3,<:Rational}
	org::NTuple{3,<:Rational}
	grd::Array{<:StepRangeLen,1}
	# boundary conditions here?
end
"""
Simplified GlaVol constructor.
"""
function GlaVol(cel::Array{<:Integer,1}, celScl::NTuple{3,<:Rational}, 
	grdScl::NTuple{3,<:Rational}=celScl, org::NTuple{3,<:Rational})
	
	if !prod(celScl .< grdScl)
		error("The cell scale must be smaller than the grid scale to avoid 
		partially overlapping basis elements.")
	end	
	brd = @. grdScl * (cel - 1) // 2
	grd = [(org[1] - brd[1]):grdScl[1]:(org[1] + brd[1]),
		(org[2] - brd[2]):grdScl[2]:(org[2] + brd[2]),
		(org[3] - brd[3]):grdScl[3]:(org[3] + brd[3])]
	return GlaVol(cel, scale, org, grid)
end	
"""
Descriptors for mapping between general source and target volumes.
.minScl---common minimum cell size
.srcDiv---division in each Cartesian index of source body 
.trgDiv---division in each Cartesian index of target body
.srcCel---cells in a source division
.trgCel---cells in a target division  
"""
struct GlaExtInf

	minScl::NTuple{3,<:Rational}
	trgDiv::NTuple{3,<:Integer}
	srcDiv::NTuple{3,<:Integer}
	trgCel::Array{<:Integer,1}
	scrCel::Array{<:Integer,1}
end
"""
Information for Green function assembly and kernel operation.  
"""
struct GlaKerOpt
	# multiplicative scaling factor allowing for complex frequencies
	frqPhz::Number
	# Gauss-Legendre integration order for cells in contact
	glOrd::Integer
	# flip between operator and operator adjoint
	adjMod::Bool
	# boolean vector representing activation of GPUs
	dev::Array{<:Bool,1} 
	# number of threads to use when running GPU kernels
	numTrd::Union{Tuple{},NTuple{3,<:Integer}}
	# number of threads to use when running GPU kernels 
	numBlk::Union{Tuple{},NTuple{3,<:Integer}}
end
"""
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
"""
Memory pinned to a Green function operator.
"""
struct GlaOprMem
	# computation information
	cmpInf::GlaKerOpt
	# descriptions for source and target volumes
	egoVol::Union{GlaVol,Tuple{GlaVol,GlaVol}}
	# information for mapping from source to target
	extInf::GlaExtInf
	# dimension information for Green function volumes
	dimInfC::AbstractArray{<:Integer,2} 
	dimInfD::AbstractArray{<:Integer,2} 
	# Fourier transform of circulant Green function
	egoFur::AbstractArray{<:AbstractArray{T},3} where 
	T<:Union{ComplexF64,ComplexF32}
	# Fourier transform plans
	fftPlnFwd::AbstractArray{<:AbstractFFTs.Plan,2}
	fftPlnRev::AbstractArray{<:AbstractFFTs.ScaledPlan,2}
	# internal phase information for implementation of operator action
	phzInf::AbstractArray{<:AbstractArray{T},2} where 
	T<:Union{ComplexF64,ComplexF32}
end
end