#=
The GilaCrc module calculates the unique elements of the electromagnetic 
Green functions, embedded in circulant form. The code is distributed under 
GNU LGPL.

Author: Sean Molesky 

Reference: Sean Molesky, MaxG documentation sections II and IV.
=#
module GilaCrc
using Base.Threads, Cubature, FastGaussQuadrature, LinearAlgebra, 
GilaStruc, GilaWInt, BenchmarkTools
export genEgoExt!, genEgoSlf!
# settings for cubature integral evaluation: relative tolerance and absolute 
# tolerance
const cubRelTol = 1e-6;
# const cubRelTol = 1e-3;
const cubAbsTol = 1e-9;
# const cubAbsTol = 1e-3;
# tolerance for cell contact
const cntTol = 1e-8;
"""

	genEgoExt!(egoCrc::AbstractArray{T}, trgVol::GlaVol, 
	srcVol::GlaVol, cmpInf::GlaKerOpt)::Nothing where 
	T<:Union{ComplexF64,ComplexF32}

Calculate circulant vector for the Green function between a target volume, 
trgVol, and source volume, srcVol.
"""
function genEgoExt!(egoCrc::AbstractArray{T,5}, trgVol::GlaVol, 
	srcVol::GlaVol, cmpInf::GlaKerOpt)::Nothing where 
	T<:Union{ComplexF64,ComplexF32}

	facPar = GilaWInt.facPar()
	trgFac = GilaWInt.cubFac(trgVol.scl)
	srcFac = GilaWInt.cubFac(srcVol.scl)
	sepGrdTrg = sepGrd(trgVol, srcVol, 0)
	sepGrdSrc = sepGrd(trgVol, srcVol, 1)
	# calculate Green function values
	asbEgoCrcExt!(egoCrc, trgVol, srcVol, sepGrdTrg, sepGrdSrc, trgFac, 
		srcFac, facPar, cmpInf)
	return nothing
end
"""
	
	genEgoSlf!(egoCrc::Array{ComplexF64}, slfVol::GlaVol, 
	cmpInf::GlaKerOpt)::Nothing

Calculate circulant vector of the Green function on a single domain.
"""
function genEgoSlf!(egoCrc::AbstractArray{T,5}, slfVol::GlaVol, 
	cmpInf::GlaKerOpt)::Nothing where T<:Union{ComplexF64,ComplexF32}

	facPar = GilaWInt.facPar()
	trgFac = GilaWInt.cubFac(slfVol.scl)
	srcFac = GilaWInt.cubFac(slfVol.scl)
	srcGrd = sepGrd(slfVol, slfVol, 0)
	# calculate Green function values
	asbEgoCrcSlf!(slfVol, egoCrc, srcGrd, trgFac, srcFac, facPar, cmpInf)
	return nothing
end
#=
Generate the circulant vector of the Green function between a pair of distinct 
domains, checking for possible domain contact. For the procedure to function 
correctly, whenever the domains are in contact, the ratio of scales between the 
source and target volumes must all be integers, and cell corners must match up.  
=#
function asbEgoCrcExt!(egoCrcExt::AbstractArray{T,5}, 
	trgVol::GlaVol, srcVol::GlaVol, sepGrdTrg::Array{<:StepRangeLen,1}, 
	sepGrdSrc::Array{<:StepRangeLen,1}, trgFac::Array{<:Number,3}, 
	srcFac::Array{<:Number,3}, facPar::Array{<:Integer,2}, 
	cmpInf::GlaKerOpt)::Nothing where T<:Union{ComplexF64,ComplexF32}
	## calculate domain separation
	# upper and lower edges of the source volume
	srcEdgs = (srcVol.coord .+ (srcVol.scl .* srcVol.cells ./ 2), 
		srcVol.coord .- (srcVol.scl .* srcVol.cells ./ 2))
	# upper and lower edges of the target volume
	trgEdgs = (trgVol.coord .+ (trgVol.scl .* trgVol.cells ./ 2), 
		trgVol.coord .- (trgVol.scl .* trgVol.cells ./ 2))
	# preparatory separation vector
	srcTrgPS = (abs.(srcEdgs[1] .- trgEdgs[2]), abs.(srcEdgs[2] .- trgEdgs[1])) 
	# calculate separation vector
	srcTrgSep = min.(srcTrgPS[1], srcTrgPS[2])
	# check for volume overlap
	if ~prod((ones(3) .* cntTol) .< srcTrgSep)
		if srcTrgSep[1] < -cntTol || srcTrgSep[2] < -cntTol || 
			srcTrgSep[3] < -cntTol
			error("Source and target volumes are overlapping.")
		end
		# number of cells needed
		stRatio = Int.(ceil.(trgVol.scl ./ srcVol.scl))
		tsRatio = Int.(ceil.(srcVol.scl ./ trgVol.scl))
		# generate self Green function for contact cells
		cntVol = genCntVol(trgVol, srcVol) 
		egoCrcCnt = Array{eltype(egoCrcExt)}(undef, 3, 3, 2 * cntVol.cells[1], 
			2 * cntVol.cells[2], 2 * cntVol.cells[3])
		genEgoSlf!(egoCrcCnt, cntVol, cmpInf)
		# pull values for cells in contact 
		@threads for posItr ∈ CartesianIndices(egoCrcExt[1,1,:,:,:])
			egoFunExtCnt!(cntVol, view(egoCrcExt, :, :, posItr), egoCrcCnt, 
				posItr, trgVol.cells, sepGrdTrg, sepGrdSrc, trgVol.scl, 
				srcVol.scl, trgFac, srcFac, facPar, cmpInf)
		end
	else
		@threads for posItr ∈ CartesianIndices(egoCrcExt[1,1,:,:,:])
			egoFunExt!(view(egoCrcExt, :, :, posItr), posItr, trgVol.cells, 
				sepGrdTrg, sepGrdSrc, trgVol.scl, srcVol.scl, trgFac, srcFac, 
				facPar, cmpInf)
		end
	end
	return nothing
end
#=
Generate circulant vector for self Green function.
=#
function asbEgoCrcSlf!(slfVol::GlaVol, egoCrc::AbstractArray{T,5},  
	srcGrd::Array{<:StepRangeLen,1}, trgFac::Array{<:AbstractFloat,3}, 
	srcFac::Array{<:AbstractFloat,3}, facPar::Array{<:Integer,2}, 
	cmpInf::GlaKerOpt)::Nothing where T<:Union{ComplexF64,ComplexF32}
	# allocate intermediate storage for Toeplitz interaction vector
	egoToe = Array{eltype(egoCrc)}(undef, 3, 3, slfVol.cells[1], 
		slfVol.cells[2], slfVol.cells[3])
	# write Green function, ignoring weakly singular integrals
	@threads for outItr ∈ CartesianIndices(egoToe[1,1,1,:,:])
		@inbounds for innItr ∈ CartesianIndices(egoToe[1,1,:,1,1])
		 egoFunInn!(egoToe, CartesianIndex(innItr, outItr), srcGrd, slfVol.scl, 
		 	trgFac, srcFac, facPar, cmpInf)
		end
	end
	# Gauss-Legendre quadrature
	glQud = gaussQuad1(cmpInf.glOrd) 
	# correction values for singular integrals
	# return order of normal faces is xx yy zz
	wS = (^(prod(slfVol.scl), -1) .* 
		GilaWInt.wekS(slfVol.scl, glQud, cmpInf))
	# return order of normal faces is xxY xxZ yyX yyZ zzX zzY xy xz yz
	wE = (^(prod(slfVol.scl), -1) .* 
		GilaWInt.wekE(slfVol.scl, glQud, cmpInf))
	# return order of normal faces is xx yy zz xy xz yz
	wV = (^(prod(slfVol.scl), -1) .* 
		GilaWInt.wekV(slfVol.scl, glQud, cmpInf))
	# correct singular integrals for coincident and adjacent cells
	for posItr ∈ CartesianIndices((min(slfVol.cells[1], 2), 
		min(slfVol.cells[2], 2), min(slfVol.cells[3], 2)))
		egoFunSng!(view(egoToe, :, :, posItr), posItr, wS, wE, wV, 
			srcGrd, slfVol.scl, trgFac, srcFac, facPar, cmpInf)
	end
	# include identity component
	egoToe[1,1,1,1,1] -= 1 / (cmpInf.frqPhz^2)
	egoToe[2,2,1,1,1] -= 1 / (cmpInf.frqPhz^2)
	egoToe[3,3,1,1,1] -= 1 / (cmpInf.frqPhz^2)
	# embed self Toeplitz vector into a circulant vector
	@threads for outItr ∈ CartesianIndices(egoCrc[1,1,1,:,:])
		@inbounds for innItr ∈ CartesianIndices(egoCrc[1,1,:,1,1])
			egoToeCrc!(view(egoCrc, :, :, CartesianIndex(innItr, outItr)), 
				egoToe, CartesianIndex(innItr, outItr), 
				div.(size(egoCrc)[3:5], 2))
		end
	end
	return nothing
end
#=
Generate circulant self Green function from Toeplitz self Green function. The 
implemented mask takes into account the relative flip in the assumed dipole 
direction under a coordinate reflection. 
=#
function egoToeCrc!(egoCrc::AbstractArray{T,2}, egoToe::AbstractArray{T,5}, 
	posInd::CartesianIndex{3}, indSpt::Tuple{Vararg{<:Integer}})::Nothing where 
	T<:Union{ComplexF64,ComplexF32}
	
	if posInd[1] == (indSpt[1] + 1) || posInd[2] == (indSpt[2] + 1) || 
		posInd[3] == (indSpt[3] + 1)
		egoCrc[:,:] .= zeros(eltype(egoCrc), 3, 3)
	else		
		# flip field under coordinate reflection
		fi = indFlp(posInd[1], indSpt[1])
		fj = indFlp(posInd[2], indSpt[2])
		fk = indFlp(posInd[3], indSpt[3])
		# embedding
		egoCrc[:,:] .= view(egoToe, :, :, 
			indSel(posInd[1], indSpt[1]), indSel(posInd[2], indSpt[2]), 
			indSel(posInd[3], indSpt[3])) .* 
			[1.0 (fi * fj)  (fi * fk); (fj * fi) 1.0 (fj * fk); 
			(fk * fi) (fk * fj) 1.0]
	end
	return nothing
end
#=
Write Green function element for a pair of cubes in distinct domains. Recall 
that grids span the separations between a pair of volumes. No field flips are 
required when calculating an external Green function. 
=#
@inline function egoFunExt!(egoCrc::AbstractArray{T,2}, 
	posInd::CartesianIndex{3}, indSpt::Array{<:Integer,1}, 
	sepGrdTrg::Array{<:StepRangeLen,1}, sepGrdSrc::Array{<:StepRangeLen,1}, 
	sclTrg::NTuple{3,<:AbstractFloat}, sclSrc::NTuple{3,<:AbstractFloat}, 
	trgFac::Array{<:AbstractFloat,3}, srcFac::Array{<:AbstractFloat,3}, 
	facPar::Array{<:Integer,2}, cmpInf::GlaKerOpt)::Nothing where 
	T<:Union{ComplexF64,ComplexF32}
	
	if posInd[1] == (indSpt[1] + 1) || posInd[2] == (indSpt[2] + 1) || 
			posInd[3] == (indSpt[3] + 1)
		egoCrc[:,:,posInd] .= zeros(eltype(egoCrc), 3, 3)
	else		
		egoFunOut!(egoCrc, [grdSel(posInd[1], indSpt[1], 1, 
			sepGrdTrg, sepGrdSrc), grdSel(posInd[2], indSpt[2], 2, 
			sepGrdTrg, sepGrdSrc), grdSel(posInd[3], indSpt[3], 3, 
			sepGrdTrg, sepGrdSrc)], sclTrg, sclSrc, trgFac, srcFac, facPar, 
			cmpInf)
	end
	return nothing
end
#=
Write Green element for a pair of cubes in distinct domains, checking for the 
possibility of contact. Separation grids span the separations between the pair 
of volumes. No field flip is required for external Green function. 
=#
function egoFunExtCnt!(cntVol::GlaVol, egoCrc::AbstractArray{T,2},
	egoCrcCnt::AbstractArray{T,5}, posInd::CartesianIndex{3}, 
	indSpt::Array{<:Integer,1}, sepGrdTrg::Array{<:StepRangeLen,1}, 
	sepGrdSrc::Array{<:StepRangeLen,1}, sclTrg::NTuple{3,<:AbstractFloat}, 
	sclSrc::NTuple{3,<:AbstractFloat}, trgFac::Array{<:AbstractFloat,3}, 
	srcFac::Array{<:AbstractFloat,3}, facPar::Array{<:Integer,2}, 
	cmpInf::GlaKerOpt)::Nothing where T<:Union{ComplexF64,ComplexF32}
	# separation vector
	sepVec = [grdSel(posInd[1], indSpt[1], 1, sepGrdTrg, sepGrdSrc), 
		grdSel(posInd[2], indSpt[2], 2, sepGrdTrg, sepGrdSrc), 
		grdSel(posInd[3], indSpt[3], 3, sepGrdTrg, sepGrdSrc)]
	# absolute separation
	absSep = abs.(sepVec)
	# branch between contact and non-contact cases
	# contact case
	if prod(absSep .< (ones(3) .* cntTol .+ ((sclTrg .+ sclSrc) ./ 2.0)))
		if posInd[1] == (indSpt[1] + 1) || posInd[2] == (indSpt[2] + 1) || 
			posInd[3] == (indSpt[3] + 1) 
			egoCrc[:,:] .= zeros(eltype(egoCrc), 3, 3)
		else
			egoCntCmp!(cntVol, egoCrcCnt, egoCrc, posInd, sclTrg, sclSrc, 
				sepVec)
		end
	# non-contact case
	else
		if posInd[1] == (indSpt[1] + 1) || posInd[2] == (indSpt[2] + 1) || 
			posInd[3] == (indSpt[3] + 1)
			egoCrc[:,:] .= zeros(eltype(egoCrc), 3, 3)
		else
			egoFunOut!(egoCrc, sepVec, sclTrg, sclSrc, trgFac, 
				srcFac, facPar, cmpInf)		
		end
	end
	return nothing
end
#=
General external Green function interaction element. 
=#
function egoFunOut!(egoCrc::AbstractArray{T,5}, grd::Array{<:AbstractFloat,1}, 
	sclTrg::NTuple{3,<:AbstractFloat}, sclSrc::NTuple{3,<:AbstractFloat}, 
	trgFac::Array{<:AbstractFloat,3}, srcFac::Array{<:AbstractFloat,3}, 
	facPar::Array{<:Integer,2}, cmpInf::GlaKerOpt)::Nothing where 
	T<:Union{ComplexF64,ComplexF32}

	srfMat = zeros(eltype{egoCrc}, 36)
	# calculate interaction contributions between all cube faces
	egoSrfAdp!(grd[1], grd[2], grd[3], srfMat, trgFac, srcFac, 1:36, 
		facPar, GilaWInt.srfScl(sclTrg, sclSrc), cmpInf)
	# sum contributions depending on source and target current orientation 
	srfSum!(egoCrc, srfMat)
	return nothing
end
#=
Compute Green function element for cells in contact. 
=#
function egoCntCmp!(cntVol::GlaVol, egoCrcCnt::Array{T,5}, 
	egoCrc::AbstractArray{T,2}, posInd::CartesianIndex{3}, 
	sclTrg::NTuple{3,<:AbstractFloat}, sclSrc::NTuple{3,<:AbstractFloat}, 
	sepVec::Array{<:AbstractFloat,1})::Nothing where
	T<:Union{ComplexF64,ComplexF32}
	# safety zero local section of the Green function
	egoCrc[:,:] .= zeros(ComplexF64, 3, 3)
	# contact cell locations
	srcCellLocs = [[1,1,1], 
	[Int(sclSrc[dir] / cntVol.scl[dir]) for dir ∈ 1:3]]
	trgCellSpan = [Int(sclTrg[dir] / cntVol.scl[dir]) for dir ∈ 1:3]
	trgCellLocs = [[1,1,1], [1,1,1]]
	sepCellSpan = [sepVec[dir] / cntVol.scl[dir] for dir ∈ 1:3]
	# positions of target cells
	# lower cell boundaries
	trgCellLocs[1] = Int.(round.(sepCellSpan + (srcCellLocs[2] ./ 2.0) - 
		(trgCellSpan ./ 2.0))) + [1,1,1]
	# upper cell boundaries
	trgCellLocs[2] = Int.(round.(sepCellSpan + (srcCellLocs[2] ./ 2.0) + 
		(trgCellSpan ./ 2.0)))
	# shift cell locations into contact volume
	for dir ∈ 1:3
		if trgCellLocs[1][dir] < 1
			# must update target cell lower bound last
			for ind ∈ 1:2
				srcCellLocs[ind][dir] = srcCellLocs[ind][dir] + 
					abs(trgCellLocs[1][dir]) + 1
			end
			# target cells
			for ind ∈ 2:-1:1
				trgCellLocs[ind][dir]  = trgCellLocs[ind][dir] + 
					abs(trgCellLocs[1][dir]) + 1
			end
		end
	end
	# loop over contact cells
	for indSrc ∈ CartesianIndices((srcCellLocs[1][1]:srcCellLocs[2][1], 
			srcCellLocs[1][2]:srcCellLocs[2][2], 
			srcCellLocs[1][3]:srcCellLocs[2][3]))

		for indTrg ∈ CartesianIndices((trgCellLocs[1][1]:trgCellLocs[2][1], 
				trgCellLocs[1][2]:trgCellLocs[2][2],
				trgCellLocs[1][3]:trgCellLocs[2][3]))

			# add result to element calculation
			egoCrc[:,:] .+= egoCrcCnt[:,:,
				crcIndClc(cntVol, indTrg, indSrc)]
		end
	end
	totTrgCells = (trgCellLocs[2][3] - trgCellLocs[1][3] + 1) * 
		(trgCellLocs[2][2] - trgCellLocs[1][2] + 1) * 
		(trgCellLocs[2][1] - trgCellLocs[1][1] + 1)
	egoCrc[:,:] .= egoCrc[:,:] ./ totTrgCells
	return nothing
end
#=
Create contact volume for self Green function calculations. 
=#
function genCntVol(trgVol::GlaVol, srcVol::GlaVol)::GlaVol
	# scale for greatest common division of source and target cells
	cntScl = min.(trgVol.scl, srcVol.scl)
	# divisions of target and source cells
	trgDiv = Int.(trgVol.scl ./ cntScl)
	srcDiv = Int.(srcVol.scl ./ cntScl)
	# create padded self interaction domain for simplification of code
	cntCells = [trgDiv[1] + srcDiv[1],trgDiv[2] + srcDiv[2],
		trgDiv[3] + srcDiv[3]]
	return GilaStruc.GlaVol(cntCells, cntScl, (0.0, 0.0, 0.0))
end
#=
General self Green function interaction element. 
=#
function egoFunInn!(egoToe::AbstractArray{T,5}, posInd::CartesianIndex{3},
	srcGrd::Array{<:StepRangeLen,1}, scl::NTuple{3,<:AbstractFloat}, 
	trgFac::Array{<:AbstractFloat,3}, srcFac::Array{<:AbstractFloat,3}, 
	facPar::Array{<:Integer,2}, cmpInf::GlaKerOpt)::Nothing where 
	T<:Union{ComplexF64,ComplexF32}

	srfMat = zeros(eltype(egoToe), 36)
	# calculate interaction contributions between all cube faces
	if (posInd[1] > 2) || (posInd[2] > 2) || (posInd[3] > 2)
		egoSrfAdp!(srcGrd[1][posInd[1]], srcGrd[2][posInd[2]], 
			srcGrd[3][posInd[3]], srfMat, trgFac, srcFac, 1:36, facPar, 
			GilaWInt.srfScl(scl, scl), cmpInf)
		# add contributions based on source and target current orientation 
		srfSum!(view(egoToe, :, :, posInd), srfMat)
	end
	return nothing
end
#=
Calculate Green function elements for adjacent cubes, assumed to be in the same 
domain. wS, wE, and wV refer to self-intersecting, edge intersecting, and 
vertex intersecting cube face integrals respectively. In the wE and wV cases, 
the first value returned is for in-plane faces, and the second value is for 
``cornered'' faces. 

The convention by which facePairs are generated begins by looping over the 
source faces. Because of this choice, the transpose of the mask  follows the 
standard source to target matrix convention used elsewhere. 
=#
function egoFunSng!(egoCrc::AbstractArray{T,2}, posInd::CartesianIndex{3}, 
	wS::Vector{R}, wE::Vector{R}, wV::Vector{R}, 
	srcGrd::Array{<:StepRangeLen,1}, scl::NTuple{3,<:AbstractFloat}, 
	trgFac::Array{<:AbstractFloat,3}, srcFac::Array{<:AbstractFloat,3}, 
	facPar::Array{<:Integer,2}, cmpInf::GlaKerOpt)::Nothing where 
	{T<:Union{ComplexF64,ComplexF32}, R<:Union{ComplexF64,ComplexF32}}

	srfMat = zeros(eltype(egoCrc), 36)
	# linear index conversion
	linCon = LinearIndices((1:6, 1:6))
	# face convention yzL yzU (x-faces) xzL xzU (y-faces) xyL xyU (z-faces)
	# index based corrections.
	if posInd == CartesianIndex(1, 1, 1)
		corVal = [
		wS[1]  0.0    wE[7]  wE[7]  wE[8]  wE[8]
		0.0    wS[1]  wE[7]  wE[7]  wE[8]  wE[8]
		wE[7]  wE[7]  wS[2]   0.0   wE[9]  wE[9]
		wE[7]  wE[7]  0.0    wS[2]  wE[9]  wE[9]
		wE[8]  wE[8]  wE[9]  wE[9]  wS[3]  0.0
		wE[8]  wE[8]  wE[9]  wE[9]  0.0    wS[3]]
		mask =[
		1 0 1 1 1 1
		0 1 1 1 1 1
		1 1 1 0 1 1
		1 1 0 1 1 1
		1 1 1 1 1 0
		1 1 1 1 0 1]
		pairListUn = linCon[findall(iszero, transpose(mask))]
		# uncorrected surface integrals
		egoSrfAdp!(srcGrd[1][posInd[1]], srcGrd[2][posInd[2]], 
			srcGrd[3][posInd[3]], srfMat, trgFac, srcFac, pairListUn, facPar, 
			GilaWInt.srfScl(scl, scl), cmpInf)
	elseif posInd == CartesianIndex(2, 1, 1) 
		corVal = [
		0.0  wS[1]  wE[7]  wE[7]  wE[8]  wE[8]
		0.0  0.0    0.0    0.0    0.0    0.0
		0.0  wE[7]  wE[3]  0.0    wV[6]  wV[6]
		0.0  wE[7]  0.0    wE[3]  wV[6]  wV[6]
		0.0  wE[8]  wV[6]  wV[6]  wE[5]  0.0
		0.0  wE[8]  wV[6]  wV[6]  0.0    wE[5]]
		mask = [
		0 1 1 1 1 1
		0 0 0 0 0 0
		0 1 1 0 1 1
		0 1 0 1 1 1
		0 1 1 1 1 0
		0 1 1 1 0 1]
		pairListUn = linCon[findall(iszero, transpose(mask))]
		# uncorrected surface integrals
		egoSrfAdp!(srcGrd[1][posInd[1]], srcGrd[2][posInd[2]], 
			srcGrd[3][posInd[3]], srfMat, trgFac, srcFac, pairListUn, facPar, 
			GilaWInt.srfScl(scl, scl), cmpInf)
	elseif (posInd[1], posInd[2], posInd[3]) == (2, 1, 2)
		corVal = [
		0.0  wE[2]  wV[4]  wV[4]  0.0  wE[8]
		0.0  0.0    0.0    0.0    0.0  0.0
		0.0  wV[4]  wV[2]  0.0    0.0  wV[6]
		0.0  wV[4]  0.0    wV[2]  0.0  wV[6]
		0.0  wE[8]  wV[6]  wV[6]  0.0  wE[5]
		0.0  0.0    0.0    0.0    0.0  0.0]
		mask = [
		0 1 1 1 0 1
		0 0 0 0 0 0
		0 1 1 0 0 1
		0 1 0 1 0 1
		0 1 1 1 0 1
		0 0 0 0 0 0] 
		pairListUn = linCon[findall(iszero, transpose(mask))]
		# uncorrected surface integrals
		egoSrfAdp!(srcGrd[1][posInd[1]], srcGrd[2][posInd[2]], 
			srcGrd[3][posInd[3]], srfMat, trgFac, srcFac, pairListUn, facPar, 
			GilaWInt.srfScl(scl, scl), cmpInf)
	elseif posInd == CartesianIndex(1, 1, 2) 
		corVal = [
		wE[2]  0.0    wV[4]  wV[4]  0.0  wE[8]
		0.0    wE[2]  wV[4]  wV[4]  0.0  wE[8]
		wV[4]  wV[4]  wE[4]  0.0    0.0  wE[9]
		wV[4]  wV[4]  0.0    wE[4]  0.0  wE[9]
		wE[8]  wE[8]  wE[9]  wE[9]  0.0  wS[3]
		0.0    0.0    0.0    0.0    0.0  0.0]
		mask = [
		1 0 1 1 0 1
		0 1 1 1 0 1
		1 1 1 0 0 1
		1 1 0 1 0 1
		1 1 1 1 0 1
		0 0 0 0 0 0]
		pairListUn = linCon[findall(iszero, transpose(mask))]
		# uncorrected surface integrals
		egoSrfAdp!(srcGrd[1][posInd[1]], srcGrd[2][posInd[2]], 
			srcGrd[3][posInd[3]], srfMat, trgFac, srcFac, pairListUn, facPar, 
			GilaWInt.srfScl(scl, scl), cmpInf)
	elseif posInd == CartesianIndex(1, 2, 1) 
		corVal = [
		wE[1]  0.0    0.0  wE[7]  wV[5]  wV[5]
		0.0    wE[1]  0.0  wE[7]  wV[5]  wV[5]
		wE[7]  wE[7]  0.0  wS[2]  wE[9]  wE[9]
		0.0    0.0    0.0  0.0    0.0    0.0
		wV[5]  wV[5]  0.0  wE[9]  wE[6]  0.0
		wV[5]  wV[5]  0.0  wE[9]  0.0    wE[6]]
		mask = [
		1 0 0 1 1 1
		0 1 0 1 1 1
		1 1 0 1 1 1
		0 0 0 0 0 0
		1 1 0 1 1 0
		1 1 0 1 0 1]
		pairListUn = linCon[findall(iszero, transpose(mask))]
		# uncorrected surface integrals
		egoSrfAdp!(srcGrd[1][posInd[1]], srcGrd[2][posInd[2]], 
			srcGrd[3][posInd[3]], srfMat, trgFac, srcFac, pairListUn, facPar, 
			GilaWInt.srfScl(scl, scl), cmpInf)
	elseif posInd == CartesianIndex(2, 2, 1) 
		corVal = [
		0.0  wE[1]  0.0  wE[7]  wV[5]  wV[5]
		0.0  0.0    0.0  0.0    0.0    0.0
		0.0  wE[7]  0.0  wE[3]  wV[6]  wV[6]
		0.0  0.0    0.0  0.0    0.0    0.0
		0.0  wV[5]  0.0  wV[6]  wV[3]  0.0
		0.0  wV[5]  0.0  wV[6]  0.0    wV[3]]
		mask = [
		0 1 0 1 1 1
		0 0 0 0 0 0
		0 1 0 1 1 1
		0 0 0 0 0 0
		0 1 0 1 1 0
		0 1 0 1 0 1]  
		pairListUn = linCon[findall(iszero, transpose(mask))]
		# uncorrected surface integrals
		egoSrfAdp!(srcGrd[1][posInd[1]], srcGrd[2][posInd[2]], 
			srcGrd[3][posInd[3]], srfMat, trgFac, srcFac, pairListUn, facPar, 
			GilaWInt.srfScl(scl, scl), cmpInf)
	elseif posInd == CartesianIndex(1, 2, 2) 
		corVal = [
		wV[1]  0.0    0.0  wV[4]  0.0  wV[5]
		0.0    wV[1]  0.0  wV[4]  0.0  wV[5]
		wV[4]  wV[4]  0.0  wE[4]  0.0  wE[9]
		0.0    0.0    0.0  0.0    0.0  0.0
		wV[5]  wV[5]  0.0  wE[9]  0.0  wE[6]
		0.0    0.0    0.0  0.0    0.0  0.0]
		mask = [
		1 0 0 1 0 1
		0 1 0 1 0 1
		1 1 0 1 0 1
		0 0 0 0 0 0
		1 1 0 1 0 1
		0 0 0 0 0 0]  
		pairListUn = linCon[findall(iszero, transpose(mask))]
		# uncorrected surface integrals
		egoSrfAdp!(srcGrd[1][posInd[1]], srcGrd[2][posInd[2]], 
			srcGrd[3][posInd[3]], srfMat, trgFac, srcFac, pairListUn, facPar, 
			GilaWInt.srfScl(scl, scl), cmpInf)
	elseif posInd == CartesianIndex(2, 2, 2) 
		corVal = [
		0.0  wV[1]  0.0  wV[4]  0.0  wV[5]
		0.0  0.0    0.0  0.0    0.0  0.0
		0.0  wV[4]  0.0  wV[2]  0.0  wV[6]
		0.0  0.0    0.0  0.0    0.0  0.0
		0.0  wV[5]  0.0  wV[6]  0.0  wV[3]
		0.0  0.0    0.0  0.0    0.0  0.0]
		mask = [
		0 1 0 1 0 1
		0 0 0 0 0 0
		0 1 0 1 0 1
		0 0 0 0 0 0
		0 1 0 1 0 1
		0 0 0 0 0 0]
		pairListUn = linCon[findall(iszero, transpose(mask))]
		# uncorrected surface integrals
		egoSrfAdp!(srcGrd[1][posInd[1]], srcGrd[2][posInd[2]], 
			srcGrd[3][posInd[3]], srfMat, trgFac, srcFac, pairListUn, facPar, 
			GilaWInt.srfScl(scl, scl), cmpInf)
	else
		println(posInd[1], posInd[2], posInd[3])
		error("Attempted to access improper case.")
	end
	# correct values of srfMat where needed
	for fp ∈ 1:36
		if mask[facPar[fp,1], facPar[fp,2]] == 1
			srfMat[fp] = corVal[facPar[fp,1], facPar[fp,2]]
		end
	end
	# overwrite problematic elements of Green function matrix
	srfSum!(egoCrc, srfMat)
	return nothing
end
#=
Update egoCrc to hold Green function interactions. The storage format of egoCrc 
is [[ii, ji, ki]^{T}; [ij, jj, kj]^{T}; [ik, jk, kk]^{T}]. 
See documentation for explanation.
=#
function srfSum!(egoCrc::AbstractArray{T,2},srfMat::Array{T,1})::Nothing where 
	T<:Union{ComplexF64,ComplexF32}
	# ii
	egoCrc[1,1] = srfMat[15] - srfMat[16] - srfMat[21] + 
	srfMat[22] + srfMat[29] - srfMat[30] - srfMat[35] + srfMat[36]
	# ji
	egoCrc[2,1] = - srfMat[13] + srfMat[14] + srfMat[19] - srfMat[20] 
	# ki
	egoCrc[3,1] = - srfMat[25] + srfMat[26] + srfMat[31] - srfMat[32] 
	# ij
	egoCrc[1,2] = - srfMat[3] + srfMat[4] + srfMat[9] - srfMat[10]
	# jj
	egoCrc[2,2] = srfMat[1] - srfMat[2] - srfMat[7] + srfMat[8] + 
	srfMat[29] - srfMat[30] - srfMat[35] + srfMat[36]
	# kj
	egoCrc[3,2] = - srfMat[27] + srfMat[28] + srfMat[33] - srfMat[34]
	# ik
	egoCrc[1,3] = - srfMat[5] + srfMat[6] + srfMat[11] - srfMat[12]
	# jk
	egoCrc[2,3] = - srfMat[17] + srfMat[18] + srfMat[23] - srfMat[24]
	# kk
	egoCrc[3,3] = srfMat[1] - srfMat[2] - srfMat[7] + srfMat[8] + 
	srfMat[15] - srfMat[16] - srfMat[21] + srfMat[22]
	return nothing
end
#=
Adaptive integration of the Green function over face pairs. 
=#
function egoSrfAdp!(grdX::AbstractFloat, grdY::AbstractFloat, 
	grdZ::AbstractFloat, srfMat::Array{T,1}, 
	trgFac::Array{<:AbstractFloat,3}, srcFac::Array{<:AbstractFloat,3}, 
	pairList::Union{UnitRange{<:Integer},Array{<:Integer,1}}, 
	facPar::Array{<:Integer,2}, srfScales::Array{<:AbstractFloat,1}, 
	cmpInf::GlaKerOpt)::Nothing where T<:Union{ComplexF64,ComplexF32}
	# container for intermediate integral evaluation
	intVal = [0.0,0.0]
	@inbounds for fp ∈ pairList
		srfMat[fp] = 0.0 + 0.0im
		# define integration kernel  
		intKer = (ordVec::Array{<:AbstractFloat,1}, 
			vals::Array{<:AbstractFloat,1}) -> srfKer(ordVec, vals, grdX, grdY, 
			grdZ, fp, trgFac, srcFac, facPar, cmpInf)
		# surface integration
		intVal[:] = hcubature(2, intKer, [0.0,0.0,0.0,0.0], 
			[1.0,1.0,1.0,1.0], reltol = cubRelTol, abstol = cubAbsTol, 
			maxevals = 0, error_norm = Cubature.INDIVIDUAL)[1];
		srfMat[fp] = intVal[1] + im * intVal[2]
		# scaling correction
		srfMat[fp] *= srfScales[fp]
	end
	return nothing
end
#=
Integration kernel for Green function surface integrals.
=#
function srfKer(ordVec::Array{<:AbstractFloat,1}, vals::Array{<:AbstractFloat,1}, 
	grdX::AbstractFloat, grdY::AbstractFloat, grdZ::AbstractFloat, fp::Integer, 
	trgFac::Array{<:AbstractFloat,3}, srcFac::Array{<:AbstractFloat,3}, 
	facPar::Array{<:Integer,2}, cmpInf::GlaKerOpt)::Nothing
	# value of scalar Green function
	z = GilaWInt.sclEgo(GilaWInt.dstMag(
		GilaWInt.cubVecAltAdp(1, ordVec, fp, trgFac, srcFac, facPar) + grdX, 
		GilaWInt.cubVecAltAdp(2, ordVec, fp, trgFac, srcFac, facPar) + grdY, 
		GilaWInt.cubVecAltAdp(3, ordVec, fp, trgFac, srcFac, facPar) + grdZ), 
		cmpInf.frqPhz)
	vals[:] = [real(z),imag(z)]
	return nothing
end
#=
Create grid of spanning separations for a pair of volumes. The flipped 
separation grid is used when generating the circulant vector.  
=#
function sepGrd(trgVol::GlaVol, srcVol::GlaVol, 
	flip::Integer)::Array{<:StepRangeLen,1}
	start = zeros(3)
	stop = zeros(3)
	sep = zeros(3)
	gridS = srcVol.grd
	gridT = trgVol.grd
	if flip == 1
		for dir ∈ 1:3
			sep[dir] = round(srcVol.scl[dir], digits = 8)		
			start[dir] = round(gridT[dir][1] - gridS[dir][end], digits = 8)
			stop[dir] = round(gridT[dir][1] - gridS[dir][1], digits = 8)
			if stop[dir] < start[dir]
				sep[dir] *= -1.0
			end
		end
		return [start[1]:sep[1]:stop[1],start[2]:sep[2]:stop[2], 
		start[3]:sep[3]:stop[3]]
	else
		sep = round.(trgVol.scl, digits = 8)
		for dir ∈ 1:3
			start[dir] = round(gridT[dir][1] - gridS[dir][1], digits = 8)
			stop[dir] =  round(gridT[dir][end] - gridS[dir][1], digits = 8)
			if stop[dir] < start[dir]
				sep[dir] *= -1.0; 
			end
		end
		return [start[1]:sep[1]:stop[1], start[2]:sep[2]:stop[2], 
		start[3]:sep[3]:stop[3]]
	end 
end
#=
Return the separation between two elements from circulant embedding indices and 
domain grids. 
=#
@inline function grdSel(ind::Integer, indSpt::Integer, dir::Integer, 
	gridT::Array{<:StepRangeLen,1}, 
	gridS::Array{<:StepRangeLen,1})::Float64
	
	if ind <= indSpt
		return gridT[dir][ind]
	else	
		if ind > (1 + indSpt)
			ind -= 1
		end		
		return gridS[dir][ind - indSpt]
	end
end
#=
Return a reference index relative to the embedding index of the Green function. 
=#
@inline function indSel(ind::Integer, indSpt::Integer)::Int
		
	if ind <= indSpt
		return ind
	else
		if ind == (1 + indSpt)
			ind -= 1
		else
			ind -= 2
		end
		return 2 * indSpt - ind
	end
end
#=
Flip dipole direction based on index values. 
=#
@inline function indFlp(ind::Integer, indSpt::Integer)::Float64

	if ind <= indSpt
		return 1.0
	else
		return -1.0
	end
end
#=
Calculate circulant index for self Green function vector 
=#
@inline function crcIndClc(cntVol::GlaVol, trgInd::CartesianIndex{3}, 
	srcInd::CartesianIndex{3})::CartesianIndex{3}
	# separation in terms of cells
	cellInd = trgInd - srcInd 
	# cellInd[dir] is always added because the value is negative if true
	return cellInd + CartesianIndex(ntuple(itr -> (cellInd[itr] < 0) * 2 * 
		cntVol.cells[itr] + 1, 3))
end
#=
Returns locations and weights for 1D Gauss-Legendre quadrature. Order must be  
an integer between 1 and 64. The first column of the returned array is 
positions, on the interval [-1,1], the second column contains the associated 
weights.

Options:
gausschebyshev(), gausslegendre(), gaussjacobi(), gaussradau(), gausslobatto(), 
gausslaguerre(), gausshermite()
=#
function gaussQuad1(ord::Int64)::Array{Float64,2}

	pos, val = gausslegendre(ord)
	return [pos ;; val]
end
end