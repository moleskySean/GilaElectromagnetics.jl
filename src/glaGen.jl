# settings for cubature integral evaluation: relative tolerance and absolute 
# tolerance
const cubRelTol = 1e-6;
# const cubRelTol = 1e-3;
const cubAbsTol = 1e-9;
# const cubAbsTol = 1e-3;
# tolerance for cell contact
const cntTol = 1e-8;
#=
    genEgoExt!(egoCrcExt::AbstractArray{T,5}, trgVol::GlaVol, 
	srcVol::GlaVol, cmpInf::GlaKerOpt)::Nothing where 
	T<:Union{ComplexF64,ComplexF32}
	
Calculate circulant vector for the Green function between a target volume, 
trgVol, and source volume, srcVol.
=#
function genEgoExt!(egoCrcExt::AbstractArray{T,5}, trgVol::GlaVol, 
	srcVol::GlaVol, cmpInf::GlaKerOpt)::Nothing where 
	T<:Union{ComplexF64,ComplexF32}

	facLst = facPar()
	trgFac = Float64.(cubFac(trgVol.scl))
	srcFac = Float64.(cubFac(srcVol.scl))
	sepGrdTrg = sepGrd(trgVol, srcVol, 0)
	sepGrdSrc = sepGrd(trgVol, srcVol, 1)
	# calculate Green function values
	genEgoCrcExt!(egoCrcExt, trgVol, srcVol, sepGrdTrg, sepGrdSrc, trgFac, 
		srcFac, facLst, cmpInf)
	return nothing
end
#=
	
	genEgoSlf!(egoCrc::Array{ComplexF64}, slfVol::GlaVol, 
	cmpInf::GlaKerOpt)::Nothing

Calculate circulant vector of the Green function on a single domain.
=#
function genEgoSlf!(egoCrcSlf::AbstractArray{T,5}, slfVol::GlaVol, 
	cmpInf::GlaKerOpt)::Nothing where T<:Union{ComplexF64,ComplexF32}

	facLst = facPar()
	trgFac = Float64.(cubFac(slfVol.scl))
	srcFac = Float64.(cubFac(slfVol.scl))
	srcGrd = sepGrd(slfVol, slfVol, 0)
	# calculate Green function values
	genEgoCrcSlf!(slfVol, egoCrcSlf, srcGrd, trgFac, srcFac, facLst, cmpInf)
	return nothing
end
#=
Generate the circulant vector of the Green function between a pair of distinct 
domains, checking for possible domain contact. For the procedure to function 
correctly, whenever the domains are in contact, the ratio of scales between the 
source and target volumes must all be integers, and cell corners must match up.  
=#
function genEgoCrcExt!(egoCrcExt::AbstractArray{T,5}, 
	trgVol::GlaVol, srcVol::GlaVol, sepGrdTrg::Array{<:StepRange,1}, 
	sepGrdSrc::Array{<:StepRange,1}, trgFac::Array{<:Number,3}, 
	srcFac::Array{<:Number,3}, facPar::Array{<:Integer,2}, 
	cmpInf::GlaKerOpt)::Nothing where T<:Union{ComplexF64,ComplexF32}
	## calculate domain separation
	# grid scales of source and target volumes
	srcGrdScl = getproperty.(srcVol.grd, :step)
	trgGrdScl = getproperty.(trgVol.grd, :step)
	# upper and lower edges of the source volume
	srcEdg = (getproperty.(srcVol.grd, :start) .- (srcVol.scl .//2), 
			getproperty.(srcVol.grd, :stop) .+ (srcVol.scl .//2))
	# upper and lower edges of the target volume
	trgEdg = (getproperty.(trgVol.grd, :start) .- (trgVol.scl .//2), 
			getproperty.(trgVol.grd, :stop) .+ (trgVol.scl .//2))
	# check for volume coincidence
	cinChk = prod((srcEdg[1] .<= trgEdg[1]) .* (trgEdg[1] .<= srcEdg[2])) || 
		prod((srcEdg[1] .<= trgEdg[2]) .* (trgEdg[2] .<= srcEdg[2]))
	# check for volume overlap
	ovrChk = sum((srcEdg[1] .< trgEdg[1]) .* (trgEdg[1] .< srcEdg[2])) +
		sum((srcEdg[1] .< trgEdg[2]) .* (trgEdg[2] .< srcEdg[2]))
	if cinChk 
		if ovrChk > 0
			error("Source and target volumes are overlapping.")
		end
		# generate self Green function for contact cells
		cntVol = genCntVol(trgVol, srcVol) 
		egoCrcCnt = Array{eltype(egoCrcExt)}(undef, 3, 3, (2 .* cntVol.cel)...)
		genEgoSlf!(egoCrcCnt, cntVol, cmpInf)
		# pull values for cells in contact 
		@threads for posItr ∈ CartesianIndices(egoCrcExt[1,1,:,:,:])
			egoFunExtCnt!(cntVol, view(egoCrcExt, :, :, posItr), egoCrcCnt, 
				posItr, trgVol.cel, sepGrdTrg, sepGrdSrc, trgVol.scl, 
				srcVol.scl, trgFac, srcFac, facPar, cmpInf)
		end
	else
		@threads for posItr ∈ CartesianIndices(egoCrcExt[1,1,:,:,:])
			egoFunExt!(view(egoCrcExt, :, :, posItr), posItr, trgVol.cel, 
				sepGrdTrg, sepGrdSrc, trgVol.scl, srcVol.scl, trgFac, srcFac, 
				facPar, cmpInf)
		end
	end
	return nothing
end
#=
Generate circulant vector for self Green function.
=#
function genEgoCrcSlf!(slfVol::GlaVol, egoCrc::AbstractArray{T,5},  
	srcGrd::Array{<:StepRange,1}, trgFac::Array{<:AbstractFloat,3}, 
	srcFac::Array{<:AbstractFloat,3}, facPar::Array{<:Integer,2}, 
	cmpInf::GlaKerOpt)::Nothing where T<:Union{ComplexF64,ComplexF32}
	# allocate intermediate storage for Toeplitz interaction vector
	egoToe = Array{eltype(egoCrc)}(undef, 3, 3, slfVol.cel...)
	# write Green function, ignoring weakly singular integrals
	@threads for crtItr ∈ CartesianIndices(egoToe[1,1,:,:,:])
		@inbounds egoFunInn!(egoToe, crtItr, srcGrd, slfVol.scl, trgFac, 
			srcFac, facPar, cmpInf)
	end
	# Gauss-Legendre quadrature
	glQud = gauQud(cmpInf.intOrd) 
	# correction values for singular integrals
	# return order of normal faces is xx yy zz
	wS = (^(prod(slfVol.scl), -1) .* wekS(slfVol.scl, glQud, cmpInf))
	# return order of normal faces is xxY xxZ yyX yyZ zzX zzY xy xz yz
	wE = (^(prod(slfVol.scl), -1) .* wekE(slfVol.scl, glQud, cmpInf))
	# return order of normal faces is xx yy zz xy xz yz
	wV = (^(prod(slfVol.scl), -1) .* wekV(slfVol.scl, glQud, cmpInf))
	# correct singular integrals for coincident and adjacent cells
	for posItr ∈ CartesianIndices(ntuple(itr -> min(slfVol.cel[itr], 2), 3))
		egoFunSng!(view(egoToe, :, :, posItr), posItr, wS, wE, wV, 
			srcGrd, slfVol, trgFac, srcFac, facPar, cmpInf)
	end
	# include identity component
	egoToe[1,1,1,1,1] -= 1 / (cmpInf.frqPhz^2)
	egoToe[2,2,1,1,1] -= 1 / (cmpInf.frqPhz^2)
	egoToe[3,3,1,1,1] -= 1 / (cmpInf.frqPhz^2)
	# embed self Toeplitz vector into a circulant vector
	@threads for crtItr ∈ CartesianIndices(egoCrc[1,1,:,:,:])
		@inbounds egoToeCrc!(view(egoCrc, :, :, crtItr), egoToe, crtItr, 
			div.(size(egoCrc)[3:5], 2))
	end
	return nothing
end
#=
Generate circulant self Green function from Toeplitz self Green function. The 
implemented mask takes into account the relative flip in the assumed dipole 
direction under a coordinate reflection. 
=#
function egoToeCrc!(egoCrc::AbstractArray{T,2}, egoToe::AbstractArray{T,5}, 
	posInd::CartesianIndex{3}, indSpt::Tuple{Vararg{Integer}})::Nothing where 
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
@inline function egoFunExt!(egoCrcExt::AbstractArray{T,2}, 
	posInd::CartesianIndex{3}, 
	indSpt::Union{Array{<:Integer,1},NTuple{3,Integer}}, 
	sepGrdTrg::Array{<:StepRange,1}, sepGrdSrc::Array{<:StepRange,1}, 
	sclTrg::NTuple{3,Number}, sclSrc::NTuple{3,Number}, 
	trgFac::Array{<:Number,3}, srcFac::Array{<:AbstractFloat,3}, 
	facPar::Array{<:Integer,2}, cmpInf::GlaKerOpt)::Nothing where 
	T<:Union{ComplexF64,ComplexF32}
	
	if posInd[1] == (indSpt[1] + 1) || posInd[2] == (indSpt[2] + 1) || 
			posInd[3] == (indSpt[3] + 1)
		egoCrcExt[:,:] .= zeros(eltype(egoCrcExt), 3, 3)
	else		
		egoFunOut!(egoCrcExt, [grdSel(posInd[1], indSpt[1], 1, 
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
	indSpt::NTuple{3,Integer}, sepGrdTrg::Array{<:StepRange,1}, 
	sepGrdSrc::Array{<:StepRange,1}, sclTrg::NTuple{3,Number}, 
	sclSrc::NTuple{3,Number}, trgFac::Array{<:AbstractFloat,3}, 
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
			egoCntOut!(cntVol, egoCrcCnt, egoCrc, posInd, sclTrg, sclSrc, 
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
function egoFunOut!(egoCrc::AbstractArray{T,2}, grd::Array{<:AbstractFloat,1}, 
	sclTrg::NTuple{3,Number}, sclSrc::NTuple{3,Number}, 
	trgFac::Array{<:AbstractFloat,3}, srcFac::Array{<:AbstractFloat,3}, 
	facPar::Array{<:Integer,2}, cmpInf::GlaKerOpt)::Nothing where 
	T<:Union{ComplexF64,ComplexF32}

	srfMat = zeros(eltype(egoCrc), 36)
	# calculate interaction contributions between all cube faces
	egoSrfAdp!(grd[1], grd[2], grd[3], srfMat, trgFac, srcFac, 1:36, 
		facPar, srfScl(sclTrg, sclSrc), cmpInf)
	# sum contributions depending on source and target current orientation 
	srfSum!(egoCrc, srfMat)
	return nothing
end
#=
Compute Green function element for cells in contact. 
=#
function egoCntOut!(cntVol::GlaVol, egoCrcCnt::Array{T,5}, 
	egoCrc::AbstractArray{T,2}, posInd::CartesianIndex{3}, 
	sclTrg::NTuple{3,Number}, sclSrc::NTuple{3,Number}, 
	sepVec::Array{<:AbstractFloat,1})::Nothing where
	T<:Union{ComplexF64,ComplexF32}
	# safety zero local section of the Green function
	egoCrc[:,:] .= zeros(ComplexF64, 3, 3)
	# contact cell locations
	srcCellLocs = [[1,1,1], 
	[Int(sclSrc[dir] / cntVol.scl[dir]) for dir ∈ 1:3]]
	trgCellSpan = [Int(sclTrg[dir] / cntVol.scl[dir]) for dir ∈ 1:3]
	trgCellLocs = [[1,1,1], [1,1,1]]
	sepCellSpan = [sepVec[dir] / Float64(cntVol.scl[dir]) for dir ∈ 1:3]
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
	trgDiv = Integer.(trgVol.scl ./ cntScl)
	srcDiv = Integer.(srcVol.scl ./ cntScl)
	# create padded self interaction domain for simplification of code
	cntCel = (trgDiv[1] + srcDiv[1], trgDiv[2] + srcDiv[2], 
		trgDiv[3] + srcDiv[3])
	return GlaVol(cntCel, cntScl, (0//1, 0//1, 0//1))
end
#=
General self Green function interaction element. 
=#
function egoFunInn!(egoToe::AbstractArray{T,5}, posInd::CartesianIndex{3},
	srcGrd::Array{<:StepRange,1}, scl::NTuple{3,Number}, 
	trgFac::Array{<:AbstractFloat,3}, srcFac::Array{<:AbstractFloat,3}, 
	facPar::Array{<:Integer,2}, cmpInf::GlaKerOpt)::Nothing where 
	T<:Union{ComplexF64,ComplexF32}

	srfMat = zeros(eltype(egoToe), 36)
	# calculate interaction contributions between all cube faces
	if (posInd[1] > 2) || (posInd[2] > 2) || (posInd[3] > 2)
		egoSrfAdp!(Float64(srcGrd[1][posInd[1]]), Float64(srcGrd[2][posInd[2]]), 
			Float64(srcGrd[3][posInd[3]]), srfMat, trgFac, srcFac, 1:36, facPar, 
			Float64.(srfScl(scl, scl)), cmpInf)
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
	srcGrd::Array{<:StepRange,1}, slfVol::GlaVol, 
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
	elseif posInd == CartesianIndex(2, 1, 1) && slfVol.scl[1] == 
		getproperty(slfVol.grd[1], :step)
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
	elseif posInd == CartesianIndex(2, 1, 2) && slfVol.scl[1] == 
		getproperty(slfVol.grd[1], :step) && slfVol.scl[3] == 
		getproperty(slfVol.grd[3], :step) 
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
	elseif posInd == CartesianIndex(1, 1, 2) && slfVol.scl[3] == 
		getproperty(slfVol.grd[3], :step)
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
	elseif posInd == CartesianIndex(1, 2, 1) && slfVol.scl[2] == 
		getproperty(slfVol.grd[2], :step)
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
	elseif posInd == CartesianIndex(2, 2, 1) && slfVol.scl[1] == 
		getproperty(slfVol.grd[1], :step) && slfVol.scl[2] == 
		getproperty(slfVol.grd[2], :step)
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
	elseif posInd == CartesianIndex(1, 2, 2) && slfVol.scl[2] == 
		getproperty(slfVol.grd[2], :step) && slfVol.scl[3] == 
		getproperty(slfVol.grd[3], :step) 
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
	elseif posInd == CartesianIndex(2, 2, 2) && slfVol.scl[1] == 
		getproperty(slfVol.grd[1], :step) && slfVol.scl[2] == 
		getproperty(slfVol.grd[2], :step) && slfVol.scl[3] == 
		getproperty(slfVol.grd[3], :step)
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
	else
		println("empty case selected.")
		corVal = [
		0.0  0.0  0.0  0.0  0.0  0.0
		0.0  0.0  0.0  0.0  0.0  0.0
		0.0  0.0  0.0  0.0  0.0  0.0
		0.0  0.0  0.0  0.0  0.0  0.0
		0.0  0.0  0.0  0.0  0.0  0.0
		0.0  0.0  0.0  0.0  0.0  0.0]
		mask = [
		0 0 0 0 0 0
		0 0 0 0 0 0
		0 0 0 0 0 0
		0 0 0 0 0 0
		0 0 0 0 0 0
		0 0 0 0 0 0]
	end
	# include uncorrected surface integrals
	pairListUn = linCon[findall(iszero, transpose(mask))]
	egoSrfAdp!(Float64(srcGrd[1][posInd[1]]), Float64(srcGrd[2][posInd[2]]), 
		Float64(srcGrd[3][posInd[3]]), srfMat, trgFac, srcFac, pairListUn, 
		facPar, Float64.(srfScl(slfVol.scl, slfVol.scl)), cmpInf)
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
function srfSum!(egoCrc::AbstractArray{T,2}, srfMat::Array{T,1})::Nothing where 
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
	z = sclEgo(dstMag(
		cubVecAltAdp(1, ordVec, fp, trgFac, srcFac, facPar) + grdX, 
		cubVecAltAdp(2, ordVec, fp, trgFac, srcFac, facPar) + grdY, 
		cubVecAltAdp(3, ordVec, fp, trgFac, srcFac, facPar) + grdZ), 
		cmpInf.frqPhz)
	vals[:] = [real(z),imag(z)]
	return nothing
end
#=
Create grid of spanning separations for a pair of volumes. The flipped 
separation grid is used when generating the circulant vector.  
=#
function sepGrd(trgVol::GlaVol, srcVol::GlaVol, 
	flip::Integer)::Array{<:StepRange,1}
	# grid over the source volume
	if flip == 1
		sep = getproperty.(srcVol.grd, :step)
		str = getproperty.(trgVol.grd, :start) .- 
			getproperty.(srcVol.grd, :stop)
		stp = getproperty.(trgVol.grd, :start) .- 
			getproperty.(srcVol.grd, :start)
	# grid over the target volume
	else
		sep = getproperty.(trgVol.grd, :step)
		str = getproperty.(trgVol.grd, :start) .- 
				getproperty.(srcVol.grd, :start)
		stp = getproperty.(trgVol.grd, :stop) .- 
				getproperty.(srcVol.grd, :start)
	end 
	# match step to separation orientation
	for dir ∈ 1:3
		if stp[dir] < str[dir]
			sep[dir] *= -1
		end
	end
	return map(StepRange, str, sep, stp) 
end
#=
Return the separation between two elements from circulant embedding indices and 
domain grids. 
=#
@inline function grdSel(ind::Integer, indSpt::Integer, dir::Integer, 
	trgGrd::Array{<:StepRange,1}, srcGrd::Array{<:StepRange,1})::Float64
	
	if ind <= indSpt
		return Float64(trgGrd[dir][ind])
	else	
		if ind > (1 + indSpt)
			ind -= 1
		end		
		return Float64(srcGrd[dir][ind - indSpt])
	end
end
#=
Return a reference index relative to the embedding index of the Green function. 
=#
@inline function indSel(posInd::T, indSpt::R)::CartesianIndex where 
	{T<:Union{CartesianIndex, Tuple{Vararg{Integer}}, Integer}, 
	R<:Union{CartesianIndex, Tuple{Vararg{Integer}}, Integer}}
		
	return CartesianIndex(map((x, y) -> x <= y ? x : 
		(x == y + 1 ? 2 * y - x + 1 : 2 * y - x + 2), 
		Tuple(posInd), Tuple(indSpt)))
end
#=
Flip dipole direction based on index values. 
=#
@inline function indFlp(posInd::Integer, indSpt::Integer)::Float64

	return posInd <= indSpt ? 1.0 : -1.0
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
		cntVol.cel[itr] + 1, 3))
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
function gauQud(ord::Int64)::Array{Float64,2}

	pos, val = gausslegendre(ord)
	return [pos ;; val]
end
