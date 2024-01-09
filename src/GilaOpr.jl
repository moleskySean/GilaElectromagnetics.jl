"""
The GilaOpr module provides support functions for implementing Green function 
matrix vector product. Documentation for the module is currently being worked
on by Alexandre Siron. 

The code is distributed under GNU LGPL.

Author: Sean Molesky 
"""
module GilaOpr
using CUDA, Base.Threads, LinearAlgebra, AbstractFFTs, FFTW, GilaMem, GilaCrc
export egoOpr!, GlaOprMemGenSlf
###MEMORY 
#=
egoMemPrp.jl included two basic functions for creating GlaOprMem structures:

GlaOprMemGenExt(trgVol::GlaVol, srcVol::GlaVol)::GlaOprMem

GlaOprMemGenSlf(cmpInf::GlaKerOpt, slfVol::GlaVol)::GlaOprMem

GlaOprMemGenSlf(cmpInf::GlaKerOpt, egoFur::AbstractArray{<:AbstractArray{T}},
	slfVol::GlaVol)::GlaOprMem where (T <: Union{ComplexF64,ComplexF32})
=#
include("egoMemPrp.jl")
###PROCEDURE
"""
	
	egoOpr!(egoMem::GlaOprMem)::Nothing

Act with the electric Green function on the memory location linked to egoMem. 
"""
#=
egoOpr! has no internal check for NaN entries
=#
function egoOpr!(egoMem::GlaOprMem)::Nothing
	
	egoBrn!(egoMem, 0, 0, egoMem.actVec)
	return nothing
end
include("egoAct.jl")
end