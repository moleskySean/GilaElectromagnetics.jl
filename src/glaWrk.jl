#=
Restructure and divide actVec match provided partition information. 
=#
function genPrt(egoMem::GlaOprMem, parVecEve::AbstractArray{T}, 
	parVecOdd::AbstractArray{T}, actVec::AbstractArray{T},)::Nothing where 
	T<:Union{ComplexF64,ComplexF32}
	# number of elements in a branch
	brnSze = div.(egoMem.mixInf.trgCel .+ egoMem.mixInf.srcCel, 2)
	# number of source partitions
	parNum = prod(egoMem.mixInf.srcDiv)
	rngStd = CartesianIndices(Tuple(brnSze))
	# rearrange source vector into appropriate partitions; perform branch split
	for parItr ∈ 1:parNum
		# bottom range is always smaller than top range
		rngTop = CartesianIndices(Tuple(map(StepRange, 
			Tuple(egoMem.mixInf.srcPar[parItr]), egoMem.mixInf.srcDiv, 
			egoMem.mixInf.srcDiv .* brnSze)))
		# overlap range
		rngBot = CartesianIndices(Tuple(map(StepRange, egoMem.mixInf.srcDiv .* 
			brnSze .+ Tuple(egoMem.mixInf.srcPar[parItr]), 
			egoMem.mixInf.srcDiv, egoMem.mixInf.srcDiv .* 
			egoMem.mixInf.srcCel)))
		# copy range where source vector spills over brnSze
		@inbounds @threads for botItr ∈ 1:length(rngBot)
			# even branch generation
			parVecEve[rngStd[botItr],parItr,:] .= actVec[rngTop[botItr],:] .+ 
				actVec[rngBot[botItr],:]
			# odd branch generation
			parVecOdd[rngStd[botItr],parItr,:] .= 
				egoMem.phzInf[1][rngStd[botItr][1]] *
				(actVec[rngTop[botItr],:] .- actVec[rngBot[botItr],:])
		end
		# spill over of source vector has been exhausted
		@inbounds @threads for topItr ∈ (length(rngBot) + 1):length(rngTop)
			# even branch generation
			parVecEve[rngStd[topItr],parItr,:] .= actVec[rngTop[topItr],:]
			# odd branch generation
			parVecOdd[rngStd[topItr],parItr,:] .= 
				egoMem.phzInf[1][rngStd[topItr][1]] * actVec[rngTop[topItr],:]
		end
	end
	return nothing
end

function 