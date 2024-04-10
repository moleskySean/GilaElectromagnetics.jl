###UTILITY LOADING
# include("./test/preamble.jl")
#=
Basic code to examine the (dense) matrix of a self Green function. For a 
standard desktop computer, memory issues will likely begin around a thousand 
cells---16^3 is 4096.
=#
"""

	function genEgoMat(celScl::{3,<:Rational}, 
	celNum::Ntuple{3,<:Integer})::Array{ComplexF64,2}

Generate dense matrix of a Green function. 
"""
function genEgoMat(celScl::NTuple{3,<:Rational}, 
	celNum::NTuple{3,<:Integer})::Array{ComplexF64,2}
	# copy tolerance---do not copy values if below threshold
	cpyTol = 1.0e-15
	# boiler plate compute information, alter to tighten tolerances 
	cmpInf = GlaKerOpt(false)
	# create self volume at the origin
	volOrg = (0//1, 0//1, 0//1)
	slfVol = GlaVol(celNum, celScl, volOrg)
	# interaction information
	mixInf = GlaExtInf(slfVol, slfVol)
	# generate circulant green function---GilaCrc module
	egoCrc = Array{ComplexF64}(undef, 3, 3, (mixInf.trgCel .+ mixInf.srcCel)...)
	genEgoSlf!(egoCrc, slfVol, cmpInf)
	# verify that egoCrc contains numeric values
	if maximum(isnan.(egoCrc)) == 1 || maximum(isinf.(egoCrc)) == 1
		error("Computed circulant contains non-numeric values.")
	end
	# allocate matrix memory---mixInf.srcCel == mixInf.trgCel
	egoMat = zeros(ComplexF64, 3 .* prod(mixInf.srcCel), 3 .* prod(mixInf.srcCel))
	# linear indices, offset by 1, for accessing matrix
	linInd = LinearIndices(mixInf.srcCel) .- 1
	# copy seed Toeplitz information into matrix
	@threads for crtInd ∈ CartesianIndices(Tuple(mixInf.srcCel))
		# linear offset
		off	= 3 * linInd[crtInd]
		#xx
		if abs(egoCrc[1,1,crtInd]) > cpyTol
			egoMat[1 + off, 1] = egoCrc[1,1,crtInd]
		end
		#xy
		if abs(egoCrc[1,2,crtInd]) > cpyTol
			egoMat[1 + off, 2] = egoCrc[1,2,crtInd]
		end
		#xz
		if abs(egoCrc[1,3,crtInd]) > cpyTol
			egoMat[1 + off, 3] = egoCrc[1,3,crtInd]
		end
		#yx
		egoMat[2 + off, 1] = egoMat[1 + off, 2]
		#yy
		if abs(egoCrc[2,2,crtInd]) > cpyTol
			egoMat[2 + off, 2] = egoCrc[2,2,crtInd]
		end
		#yz
		if abs(egoCrc[2,3,crtInd]) > cpyTol
			egoMat[2 + off, 3] = egoCrc[2,3,crtInd]
		end
		#zx
		egoMat[3 + off, 1] = egoMat[1 + off, 3]
		#zy
		egoMat[3 + off, 2] = egoMat[2 + off, 3]
		if abs(egoCrc[3,3,crtInd]) > cpyTol
			egoMat[3 + off, 3] = egoCrc[3,3,crtInd]
		end
	end
	# expand block Toeplitz structure
	for dimItr ∈ eachindex(1:2)
		wrtSymToeBlc!(egoMat, (3 * prod(celNum[1:(dimItr - 1)]), 
			3 * prod(celNum[1:(dimItr - 1)])), celNum[dimItr])
	end
	return egoMat			
end
#=
Utility function adding a level of (symmetric) block Toeplitz structure. blcSze 
is the number of low-level data elements contained in a entry of the current 
structure. blcDim is the the dimension of the current structure. 
=#
function wrtSymToeBlc!(toeMat::Array{<:Number,2}, blcSze::NTuple{2,<:Integer}, 
	blcDim::Integer)::Nothing
	# size of a copy unit 
	untSze = blcSze[1] * blcDim
	# number of block partitions contained in matrix
	numPrt = size(toeMat)[1] / untSze
	# check that result is sensible
	if !isinteger(numPrt)
		error("Proposed dimensions are not logically consistent.")
	end
	numPrt = Integer(numPrt)
	# unit copy loop
	for supItr ∈ range(0, (numPrt - 1))
		untOff = untSze * supItr
		# dimension copy loop
		for dimItr ∈ eachindex(1:(blcDim - 1))
			rowOff = blcSze[1] * dimItr
			colOff = blcSze[2] * dimItr
			# offset source range
			srcCrtRngStd = CartesianIndices((
				(1 + untOff):(untOff + untSze - rowOff), 
				blcSze[2]))
			# modulo source range
			srcCrtRngMod = CartesianIndices((
				(1 + untOff + untSze - rowOff):(untOff + untSze), 
				blcSze[2]))
			# offset copy range
			cpyCrtRngStd = CartesianIndices((
				(1 + rowOff + untOff):(untOff + untSze), 
				(1 + colOff):(colOff + blcSze[2])))
			# modulo copy range
			cpyCrtRngMod = CartesianIndices((
				(1 + untOff):(rowOff + untOff), 
				(1 + colOff):(colOff + blcSze[2])))
			# standard element copy loop
			for (cpyItr, srcItr) ∈ zip(cpyCrtRngStd, srcCrtRngStd)
				toeMat[cpyItr] = toeMat[srcItr]
			end
			# shifted element copy loop
			for (cpyItr, srcItr) ∈ zip(cpyCrtRngMod, srcCrtRngMod)
				toeMat[cpyItr] = toeMat[srcItr]
			end
		end
	end
	return nothing
end