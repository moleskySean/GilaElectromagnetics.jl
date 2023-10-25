"""
The GilaStruc module defines the layout of the computational structures used 
in Gila. The code distributed under GNU LGPL.

Author: Sean Molesky 

Reference: Sean Molesky, Gila documentation section ...
"""
module GilaStruc
using AbstractFFTs
export GlaVol, GlaKerOpt, GlaOprMem
"""
Basic Gila spatial memory structure.
.cells---tuple of cells for defining a rectangular prism 
.totalCells---total number of cells contained in the volume
.scl--relative side length of a cuboid voxel compared to the wavelength 
.coord---center position of the domain 
.grd---spatial location of the center of each cell contained in the volume 
"""
struct GlaVol

	cells::Array{<:Integer,1}
	totalCells::Integer
	scl::NTuple{3,<:Real}
	coord::NTuple{3,<:Real}
	grd::Array{<:StepRangeLen,1}
	# boundary conditions here?
end
"""
Simplified GlaVol constructor.
"""
function GlaVol(cells::Array{<:Integer,1}, scale::NTuple{3,<:Real}, 
	coord::NTuple{3,<:Real})

	bounds = @. scale * (cells - 1) / 2.0 
	grid = [(round(-bounds[1] + coord[1], digits = 12):scale[1]: 
	round(bounds[1] + coord[1], digits = 12)), 
	(round(-bounds[2] + coord[2], digits = 12):scale[2]: 
	round(bounds[2] + coord[2], digits = 12)), 
	(round(-bounds[3] + coord[3], digits = 12):scale[3]: 
	round(bounds[3] + coord[3], digits = 12))]
	return GlaVol(cells, prod(cells), scale, coord, grid)
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
	# boolean representing activation of GPU
	dev::Bool 
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
	# interacting volumes 
	trgVol::GlaVol
	srcVol::GlaVol
	# dimension information for Green function volumes
	dimInfC::AbstractVector{<:Integer} 
	dimInfD::AbstractVector{<:Integer} 
	# vector that will be acted on by the Green function
	actVec::AbstractArray{T,4} where (T <: Union{ComplexF64,ComplexF32})
	# Fourier transform of circulant Green function
	egoFur::AbstractArray{<:AbstractArray{T}} where 
	T<:Union{ComplexF64,ComplexF32}
	# Fourier transform plans
	fftPlnFwd::AbstractArray{<:AbstractFFTs.Plan}
	fftPlnRev::AbstractArray{<:AbstractFFTs.ScaledPlan}
	# internal phase informations for implementation of operator action
	phzInf::AbstractArray{<:AbstractArray{T}} where 
	T<:Union{ComplexF64,ComplexF32}
end
end