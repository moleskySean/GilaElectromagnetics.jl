###MEMORY 
#=

	function GlaOprMem(cmpInf::GlaKerOpt, trgVol::GlaVol,
	srcVol::Union{GlaVol,Nothing}=nothing, 
	egoFur::Union{AbstractArray{<:AbstractArray{T}},
	Nothing}=nothing)::GlaOprMem where T<:Union{ComplexF64,ComplexF32}

Prepare memory for green function operator---when called with a single GlaVol, 
or identical source and target volumes, the function yields the self green 
function construction. 
=#
include("glaMemSup.jl")
###PROCEDURE
"""
	
	egoOpr!(egoMem::GlaOprMem)::Nothing

Act with the electric Green function on the memory location linked to egoMem. 
"""
#=
egoOpr! has no internal check for NaN entries---checks are done by GlaOprMem. 
=#
function egoOpr!(egoMem::GlaOprMem, 
	actVec::AbstractArray{T,4})::AbstractArray{T,4} where 
	T<:Union{ComplexF64,ComplexF32}
	
	egoBrn!(egoMem, 0, 0, actVec)
	return nothing
end
include("glaActSup.jl")