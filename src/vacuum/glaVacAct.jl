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
    return egoBrn!(egoMem, 0, 0, actVec)
end

# Support functions for operator action
include("glaVacActSup.jl")

function egoBrn!(egoMem::GlaVacOprMem, lvl::Integer, bId::Integer, actVec::AbstractArray{<:Complex})
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
        orgVec = genPrt!(actVec, egoMem.mixInf, parNumSrc)
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
        fwdFFT!(egoMem, lvl, orgVec)
    end 
    # split branch
    if lvl < length(egoMem.dimInf)  
        # check if size of current active vector dimension is compatible 
        if size(orgVec)[lvl + 1] == brnSze[lvl + 1]
            prgVec = similar(orgVec)
            # split branch, includes phase operation and stream sync
            sptBrn!(prgVec, lvl + 1, egoMem.phzInf[lvl+1], parNumSrc, orgVec)
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
            synchronize(egoMem.cmpInf.dev)
            # memory previously associated with orgVec is freed internally
            sptBrn!(prgVecEve, prgVecOdd, lvl + 1, egoMem.phzInf[lvl+1],
                egoMem.mixInf, parNumSrc, orgVec)
        end
    end
    # branch until depth of block structure
    if lvl < length(egoMem.dimInf)
        # execute split branches
        eveVec, oddVec = egoBrnExe!(egoMem, prgVecEve, prgVecOdd, lvl, bId)
        # check if size of vector must change on merge
        if size(eveVec)[lvl + 1] == egoMem.mixInf.trgCel[lvl + 1]
            # merge branches, includes phase operation and stream sync
			mrgBrn!(eveVec, lvl + 1, parNumTrg, egoMem.phzInf[lvl + 1], oddVec)
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
				egoMem.phzInf[lvl + 1], eveVec, oddVec)
        end
    else
        retVec = similar(orgVec, brnSze..., 3, parNumTrg)
        fill!(retVec, zero(eltype(retVec)))
        synchronize(egoMem.cmpInf.dev)
        # multiply by Toeplitz vector, sequentially: every source partition adds into the same target slice
        for trgItr in eachindex(1:parNumTrg) 
            for srcItr in eachindex(1:parNumSrc) 
                mulBrn!(egoMem.mixInf, bId, selectdim(retVec, 5, trgItr),
                view(egoMem.egoFur[bId + 1], :, :, :, :, srcItr, trgItr), 
                selectdim(orgVec, 5, srcItr))
            end
        end
        # free orgVec
        memCln!(orgVec) # FIXME: Do we really want to do this, or can we leave it up to the GC
    end
    # perform reverse Fourier transform
    if lvl > 0
        # inverse FFT
        # possibility of changing vector size accounted for in FFT plans
        invFFT!(egoMem, lvl, retVec)
    end
    # terminate task and return control to previous level 
    synchronize(egoMem.cmpInf.dev)
    # zero level return
    if lvl == 0
        # check if there are partitions that need to be merged
        if parNumTrg == 1
            return reshape(retVec, egoMem.mixInf.trgCel..., 3)
        else
            return mrgPrt!(egoMem.mixInf, parNumTrg, retVec)
        end
    end
    return retVec
end

function fwdFFT!(egoMem::GlaVacOprMem, lvl::Int, actVec::AbstractArray{<:Complex})
    # forward FFT
    egoMem.fftPlnFwd[lvl] * actVec
    # wait for completion of FFT
    synchronize(egoMem.cmpInf.dev)
end

function invFFT!(egoMem::GlaVacOprMem, lvl::Int, actVec::AbstractArray{<:Complex})
    # inverse FFT
    egoMem.fftPlnRev[lvl] * actVec
    # wait for completion of FFT
    synchronize(egoMem.cmpInf.dev)
end

function egoBrnExe!(egoMem::GlaVacOprMem, prgVecEve::AbstractArray{<:Complex}, prgVecOdd::AbstractArray{<:Complex}, lvl::Int, bId::Int)
    # Sequential: tasks sharing an FFTW plan segfault on its scratch space
    eveVec = egoBrn!(egoMem, lvl + 1, bId, prgVecEve)
    if lvl < length(egoMem.dimInf) - 1
        synchronize(egoMem.cmpInf.dev)
    end
    oddVec = egoBrn!(egoMem, lvl + 1, nxtBrnId(length(egoMem.dimInf), lvl, bId), prgVecOdd)
    return eveVec, oddVec
end

#= The stored Green function range, the vector range and the reflection sign of
one direction of one octant. Only half the coefficients are stored in a direction
where the branch is shorter than the source and target together. =#
function _octRng(mixInf::GlaExtInf, brnSze, egoSze, dir::Int, brnOdd::Bool, rev::Bool)
    if egoSze[dir] == brnSze[dir]
        rng = rev ? (1:0) : (1:brnSze[dir])
        return rng, rng, 1f0
    end
    trgCel, srcCel = mixInf.trgCel[dir], mixInf.srcCel[dir]
    if brnOdd
        begInd, endInd = 1, rev ? fld(srcCel, 2) : cld(trgCel, 2)
    else
        begInd, endInd = rev ? (2, fld(srcCel, 2) + 1 - iseven(trgCel)) : (1, cld(trgCel, 2) + iseven(trgCel))
    end
    vecRng = rev ? (brnSze[dir]:-1:(brnSze[dir] - endInd + begInd)) : (1:(1 + endInd - begInd))
    return begInd:endInd, vecRng, rev ? -1f0 : 1f0
end

# The Hadamard product of a branch with its Green function, forward and reverse in each direction
function mulBrn!(mixInf::GlaExtInf, bId::Integer, prdVec::AbstractArray{<:Complex, 4}, vecMod::AbstractArray{<:Complex, 4}, orgVec::AbstractArray{<:Complex, 4})
    brnSze = (mixInf.trgCel .+ mixInf.srcCel) .÷ 2
    egoSze = size(vecMod)[1:3]
    krn = mulKer!(get_backend(prdVec))
    for oct in 0:7
        rngs = ntuple(dir -> _octRng(mixInf, brnSze, egoSze, dir, isodd(bId >> (3 - dir)), isodd(oct >> (dir - 1))), 3)
        modRng, vecRng, sgn = getindex.(rngs, 1), getindex.(rngs, 2), getindex.(rngs, 3)
        any(isempty, modRng) && continue
        krn(view(prdVec, vecRng..., :), view(vecMod, modRng..., :), view(orgVec, vecRng..., :),
            (sgn[1] * sgn[2], sgn[1] * sgn[3], sgn[2] * sgn[3]); ndrange = length.(modRng))
    end
    return nothing
end
@kernel function mulKer!(prdVec::AbstractArray{<:Complex, 4}, vecMod::AbstractArray{<:Complex, 4}, orgVec::AbstractArray{<:Complex, 4}, dirSym::NTuple{3, Float32})
    itrX, itrY, itrZ = @index(Global, NTuple)

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
