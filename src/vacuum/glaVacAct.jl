using KernelAbstractions

"""
    egoOpr!(egoMem::GlaVacOprMem, actVec::AbstractArray{<:Complex})::AbstractArray{<:Complex}

Apply the Green operator to a `(srcCel..., 3)` current density of eltype
`Complex{T}`, returning the `(trgCel..., 3)` field. Overwrites `actVec`.
"""
function egoOpr!(egoMem::GlaVacOprMem, actVec::AbstractArray{<:Complex})
    # srcCel counts the cells of one source partition, so the full input size is
    # the per-partition count times the number of partitions in each direction
    srcCelTot = egoMem.mixInf.srcCel .* egoMem.mixInf.srcDiv
    if (srcCelTot..., 3) != size(actVec)
        throw(ArgumentError("Size of input vector does not match size of source volume. Required data layout is (srcCelX, srcCelY, srcCelZ, 3)."))
    end
    return egoBrn!(egoMem, 0, 0, actVec, egoMem.cmpInf)
end

# Support functions for operator action
include("glaVacActSup.jl")

function egoBrn!(egoMem::GlaVacOprMem, lvl::Integer, bId::Integer, actVec::AbstractArray{<:Complex}, cmpInf::GlaKerOpt)
    # generate branch pair to initiate operator
    # size of circulant vector
    brnSze = div.(egoMem.mixInf.trgCel .+ egoMem.mixInf.srcCel, 2)
    # number of source partitions
    parNumSrc = prod(egoMem.mixInf.srcDiv)
    # number of target partitions
    parNumTrg = prod(egoMem.mixInf.trgDiv)
    # if necessary, reshape source vector into partitions
    if lvl == 0 && sum(egoMem.mixInf.srcCel .!= size(actVec)[1:3]) > 0 
        # reshape current vector 
        # orgVec and actVec share the same underlying data
        orgVec = genPrt!(actVec, cmpInf, egoMem.mixInf, parNumSrc)
    elseif lvl == 0
        orgVec = reshape(actVec, egoMem.mixInf.srcCel..., 3, parNumSrc)
    else
        # partitions have been created, simply rename memory
        orgVec = actVec
    end
    # perform forward Fourier transform
    if lvl > 0
        # forward FFT
        # possibility of changing vector size accounted for in FFT plans
        fwdFFT!(egoMem, lvl, orgVec, cmpInf)
    end 
    # split branch
    if lvl < length(egoMem.dimInf)  
        # check if size of current active vector dimension is compatible 
        if size(orgVec)[lvl + 1] == brnSze[lvl + 1]
            prgVec = similar(orgVec)
            # split branch, includes phase operation and stream sync
            sptBrn!(prgVec, lvl + 1, egoMem.phzInf[lvl+1], parNumSrc, orgVec, egoMem.cmpInf)
            # rename vectors for consistent convention with general branching
            prgVecEve = orgVec 
            prgVecOdd = prgVec
        # size of progeny vectors must be changed
        else
            # current and progeny vector sizes
            curSze = size(orgVec)
            prgSze = ntuple(x -> x == (lvl + 1) ? brnSze[x] : curSze[x], 3)
            # allocate progeny vectors
            prgVecEve = similar(orgVec, prgSze..., 3, parNumSrc)
            prgVecOdd = similar(orgVec, prgSze..., 3, parNumSrc)
            synchronize(cmpInf.dev)
            # memory previously associated with orgVec is freed internally
            sptBrn!(prgVecEve, prgVecOdd, lvl + 1, egoMem.phzInf[lvl+1],
                egoMem.mixInf, parNumSrc, orgVec, egoMem.cmpInf)
        end
    end
    # branch until depth of block structure
    if lvl < length(egoMem.dimInf)
        # execute split branches
        eveVec, oddVec = egoBrnExe!(egoMem, prgVecEve, prgVecOdd, lvl, bId, cmpInf)
        # check if size of vector must change on merge
        if size(eveVec)[lvl + 1] == egoMem.mixInf.trgCel[lvl + 1]
            # merge branches, includes phase operation and stream sync
			mrgBrn!(eveVec, lvl + 1, parNumTrg, egoMem.phzInf[lvl + 1], oddVec, egoMem.cmpInf)
            # rename eveVec to match return name used elsewhere
            retVec = eveVec
        # merge and resize
        else
            # size of current vector
            curSze = size(eveVec)
            # size of cartesian indices of merged vector
            mrgSze = ntuple(x -> x == (lvl + 1) ? egoMem.mixInf.trgCel[x] : 
                curSze[x], 3)
            # allocate memory for merge
            retVec = similar(eveVec, mrgSze..., 3, parNumTrg)
            # merge branches, even and odd vectors are freed, stream synced 
			mrgBrn!(retVec, lvl + 1, parNumTrg, 
				egoMem.phzInf[lvl + 1], eveVec, oddVec, egoMem.cmpInf)
        end
    else
        retVec = similar(orgVec, brnSze..., 3, parNumTrg)
        fill!(retVec, zero(eltype(retVec)))
        synchronize(cmpInf.dev)
        # multiply by Toeplitz vector, sequentially: every source partition adds into the same target slice
        for trgItr in eachindex(1:parNumTrg) 
            for srcItr in eachindex(1:parNumSrc) 
                mulBrn!(egoMem.mixInf, bId, selectdim(retVec, 5, trgItr),
                view(egoMem.egoFur[bId + 1], :, :, :, :, srcItr, trgItr), 
                selectdim(orgVec, 5, srcItr), egoMem.cmpInf)
            end
        end
        # free orgVec
        memCln!(orgVec) # FIXME: Do we really want to do this, or can we leave it up to the GC
    end
    # perform reverse Fourier transform
    if lvl > 0
        # inverse FFT
        # possibility of changing vector size accounted for in FFT plans
        invFFT!(egoMem, lvl, retVec, cmpInf)
    end
    # terminate task and return control to previous level 
    synchronize(cmpInf.dev)
    # zero level return
    if lvl == 0
        # check if there are partitions that need to be merged
        if parNumTrg == 1
            return reshape(retVec, egoMem.mixInf.trgCel..., 3)
        else
            return mrgPrt!(egoMem.mixInf, egoMem.cmpInf, parNumTrg, retVec)
        end
    end
    return retVec
end

function fwdFFT!(egoMem::GlaVacOprMem, lvl::Int, actVec::AbstractArray{<:Complex}, cmpInf::GlaKerOpt)
    # forward FFT
    if isadjoint(egoMem)
        egoMem.adjFftPlnFwd[lvl] * actVec
    else
        egoMem.fftPlnFwd[lvl] * actVec
    end
    # wait for completion of FFT
    synchronize(cmpInf.dev)
end

function invFFT!(egoMem::GlaVacOprMem, lvl::Int, actVec::AbstractArray{<:Complex}, cmpInf::GlaKerOpt)
    # inverse FFT
    if isadjoint(egoMem)
        egoMem.adjFftPlnRev[lvl] * actVec
    else
        egoMem.fftPlnRev[lvl] * actVec
    end
    # wait for completion of FFT
    synchronize(cmpInf.dev)
end

function egoBrnExe!(egoMem::GlaVacOprMem, prgVecEve::AbstractArray{<:Complex}, prgVecOdd::AbstractArray{<:Complex}, lvl::Int, bId::Int, cmpInf::GlaKerOpt)
    # Sequential: tasks sharing an FFTW plan segfault on its scratch space
    eveVec = egoBrn!(egoMem, lvl + 1, bId, prgVecEve, cmpInf)
    if lvl < length(egoMem.dimInf) - 1
        synchronize(cmpInf.dev)
    end
    oddVec = egoBrn!(egoMem, lvl + 1, nxtBrnId(length(egoMem.dimInf), lvl, bId), prgVecOdd, cmpInf)
    return eveVec, oddVec
end

function mulBrn!(mixInf::GlaExtInf, bId::Integer, prdVec::AbstractArray{<:Complex, 4}, vecMod::AbstractArray{<:Complex, 4}, orgVec::AbstractArray{<:Complex, 4}, cmpInf::GlaKerOpt)
	# size of branch
	brnSze = (mixInf.trgCel .+ mixInf.srcCel) .÷ 2
	# unique information of Green function
	egoSze = size(vecMod)[1:3]
	# zero if branch is even in that direction, one if branch is odd
    brnSym = isodd.(bId .>> (2:-1:0))
	# declare memory 
    dimInf = Array{Int32}(undef, 3, 8)
    dirSym = Array{Float32}(undef, 3, 8)
	# selection ranges for Green function and vector
	modRng = Array{StepRange}(undef, 3, 8)
	vecRng = Array{StepRange}(undef, 3, 8)
	# sign symmetry for Green function multiplication
	symSgn = Array{Float32}(undef, 3, 8)
	# perform Hadamard---forward and reverse operation in each direction
	for divItr ∈ 0:7
		# one for reverse direction, zero for forward
        revSwt = isodd.(divItr .>> (0:2))
		# set copy ranges
		for dirItr ∈ 1:3
			# switch between full and partial stored information cases
			if egoSze[dirItr] == brnSze[dirItr]
				if revSwt[dirItr] == 0
					# Green function range
					modRng[dirItr, divItr + 1] = 1:1:brnSze[dirItr]
					vecRng[dirItr, divItr + 1] = 1:1:brnSze[dirItr]
                    symSgn[dirItr, divItr + 1] = one(eltype(symSgn))
				else
					modRng[dirItr, divItr + 1] = 1:1:0
					vecRng[dirItr, divItr + 1] = 1:1:0
                    symSgn[dirItr, divItr + 1] = zero(eltype(symSgn))
				end
			# partial information case
			else
				# switch between even and odd branch cases
				if brnSym[dirItr] == 0 
					# switch between source and target portions 
					if revSwt[dirItr] == 0
						begInd = 1
						endInd = Integer.(ceil.(mixInf.trgCel[dirItr] / 
							2)) + iseven(mixInf.trgCel[dirItr])
					# source coefficients
					else
						begInd = 2
						endInd = Integer.(floor.(mixInf.srcCel[dirItr] / 
							2)) + 1 - iseven(mixInf.trgCel[dirItr])
					end
				# odd coefficient branch
				else
					begInd = 1 
					# target coefficients
					if revSwt[dirItr] == 0
						endInd = Integer.(ceil.(mixInf.trgCel[dirItr] / 2)) 
					# source coefficients
					else
						endInd = Integer.(floor.(mixInf.srcCel[dirItr] / 2)) 
					end
				end
				# Green function range
				modRng[dirItr,divItr + 1] = begInd:1:endInd
				# vector range
				if revSwt[dirItr] == 0
					vecRng[dirItr,divItr + 1] = 1:1:(1 + endInd - begInd)
				else
					vecRng[dirItr,divItr + 1] = 
						brnSze[dirItr]:-1:(brnSze[dirItr] - endInd + begInd)
				end
				# sign symmetry for Green function multiplication
				symSgn[dirItr, divItr + 1] = 1.0 - 2.0 * revSwt[dirItr]
			end
		end
		# check that range is not empty
		if !isempty(CartesianIndices(tuple(modRng[:,divItr + 1]...)))
            # number of elements
            copyto!(view(dimInf, :, divItr + 1), 
                getproperty.(modRng[:,divItr + 1], :stop) .-
                getproperty.(modRng[:,divItr + 1], :start) .+ 1)
            # symmetry information
			copyto!(view(dirSym, :, divItr + 1), 
				eltype(dirSym).(
                [symSgn[1,divItr + 1] * symSgn[2,divItr + 1],
                 symSgn[1,divItr + 1] * symSgn[3,divItr + 1],
                 symSgn[2,divItr + 1] * symSgn[3,divItr + 1]]))
            # perform Hadamard product
            krn = mulKer!(cmpInf.dev)
            krn(view(prdVec, vecRng[1, divItr + 1], vecRng[2, divItr + 1], vecRng[3, divItr + 1], :),
                view(vecMod, modRng[1, divItr + 1], modRng[2, divItr + 1], modRng[3, divItr + 1], :),
                view(orgVec, vecRng[1, divItr + 1], vecRng[2, divItr + 1], vecRng[3, divItr + 1], :),
                Tuple(dirSym[:, divItr + 1]);
                ndrange=tuple(dimInf[:, divItr + 1]...))
		end
	end
	return nothing
end
@kernel function mulKer!(prdVec::AbstractArray{<:Complex, 4}, vecMod::AbstractArray{<:Complex, 4}, orgVec::AbstractArray{<:Complex, 4}, dirSym::NTuple{3, Float32})
    # Linear index
    itr = @index(Global)
    
    # Convert linear index to 3D coordinates
    sze = size(prdVec)
    itrZ, rem = divrem(itr-1, sze[1]*sze[2])
    itrY, itrX = divrem(rem, sze[1])

    # Convert to 1-based indexing
    itrX += 1
    itrY += 1 
    itrZ += 1

    # vecMod follows ii, jj, kk, ij, ik, jk storage format
    # i vector direction
    @inbounds prdVec[itrX, itrY, itrZ, 1] += 
                    vecMod[itrX, itrY, itrZ, 1] * orgVec[itrX, itrY, itrZ, 1] +
        dirSym[1] * vecMod[itrX, itrY, itrZ, 4] * orgVec[itrX, itrY, itrZ, 2] +
        dirSym[2] * vecMod[itrX, itrY, itrZ, 5] * orgVec[itrX, itrY, itrZ, 3]
    # j vector direction
    @inbounds prdVec[itrX, itrY, itrZ, 2] += 
        dirSym[1] * vecMod[itrX, itrY, itrZ, 4] * orgVec[itrX, itrY, itrZ, 1] +
                    vecMod[itrX, itrY, itrZ, 2] * orgVec[itrX, itrY, itrZ, 2] +
        dirSym[3] * vecMod[itrX, itrY, itrZ, 6] * orgVec[itrX, itrY, itrZ, 3]
    # k vector direction
    @inbounds prdVec[itrX, itrY, itrZ, 3] += 
        dirSym[2] * vecMod[itrX, itrY, itrZ, 5] * orgVec[itrX, itrY, itrZ, 1] +
        dirSym[3] * vecMod[itrX, itrY, itrZ, 6] * orgVec[itrX, itrY, itrZ, 2] +
                    vecMod[itrX, itrY, itrZ, 3] * orgVec[itrX, itrY, itrZ, 3]
end
