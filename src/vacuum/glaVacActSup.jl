#=
restructure input vector to match partitioned Green function data
=#
function genPrt!(actVec::AbstractArray{<:Complex, 4}, mixInf::GlaExtInf, parNum::Integer)
    maxItr = mixInf.srcCel
    orgVec = similar(actVec, maxItr..., 3, parNum)
    ker = genPrtKer!(get_backend(actVec))
    for parItr in 1:parNum, dirItr in 1:3
        stp = Tuple(mixInf.srcDiv)
        off = _parOff(mixInf.srcDiv, parItr)
        ker(stp, off, dirItr, parItr, actVec, orgVec; ndrange=maxItr)
    end
    memCln!(actVec)
    return orgVec
end
@kernel function genPrtKer!(stp::NTuple{3,Integer}, off::NTuple{3,Integer}, dirItr::Integer, parItr::Integer, actVec::AbstractArray{<:Complex,4}, orgVec::AbstractArray{<:Complex,5})
    itr = @index(Global, NTuple)
    srcItr = (itr .- 1) .* stp .+ off .+ 1
    @inbounds orgVec[itr..., dirItr, parItr] = actVec[srcItr..., dirItr]
end

#=
split a branch so that even and odd Fourier coefficients are independent
=#
function sptBrn!(prgVec::AbstractArray{<:Complex, 5}, sptDir::Integer, phzVec::AbstractVector{<:Complex}, parNum::Integer, orgVec::AbstractArray{<:Complex, 5})
    vecSze = size(orgVec)[1:3]
    ker = sptKer!(get_backend(orgVec))
    for parItr in 1:parNum
        ker(sptDir, phzVec, parItr, orgVec, prgVec; ndrange=vecSze)
    end
    return nothing
end
@kernel function sptKer!(sptDir::Integer, phzVec::AbstractVector{<:Complex}, parItr::Integer, orgVec::AbstractArray{<:Complex, 5}, prgVec::AbstractArray{<:Complex, 5})
    itr = @index(Global, NTuple)
    @inbounds begin
        phz = phzVec[itr[sptDir]]
        # apply it to all three Fourier‐branch slots
        prgVec[itr..., 1, parItr] = phz * orgVec[itr..., 1, parItr]
        prgVec[itr..., 2, parItr] = phz * orgVec[itr..., 2, parItr]
        prgVec[itr..., 3, parItr] = phz * orgVec[itr..., 3, parItr]
    end
end
#=
split branches for external Green function
=#
function sptBrn!(prgVecEve::AbstractArray{<:Complex,5}, prgVecOdd::AbstractArray{<:Complex,5}, dirSpt::Integer, phzVec::AbstractVector{<:Complex}, mixInf::GlaExtInf, parNum::Integer, orgVec::AbstractArray{<:Complex,5})
    brnSze = div.(mixInf.trgCel .+ mixInf.srcCel, 2)
    curSze = size(orgVec)[1:3]
    prgSze = ntuple(x -> x==dirSpt ? brnSze[x] : curSze[x], 3)

    ovrRng  = ntuple(i-> i==dirSpt ?
                     (1 : mixInf.srcCel[i]-brnSze[i]) :
                     (1 : prgSze[i]), 3)
    stdRng  = ntuple(i-> i==dirSpt ?
                     (last(ovrRng[i])+1 : min(mixInf.srcCel[i],brnSze[i])) :
                     (1 : prgSze[i]), 3)
    ovrOff  = ntuple(i-> i==dirSpt ? brnSze[dirSpt] : 0, 3)

    # Zero the progeny vectors
    fill!(prgVecEve, zero(eltype(prgVecEve)))
    fill!(prgVecOdd, zero(eltype(prgVecOdd)))

    # Launch one kernel per (par,dir) over the entire prgSze
    ker = sptBrnExt!(get_backend(orgVec))
    for parItr in 1:parNum, dirItr in 1:3
      ker(ovrRng, stdRng, ovrOff, dirSpt, dirItr, parItr,
          phzVec, orgVec, prgVecEve, prgVecOdd;
          ndrange=prgSze)
    end
    return nothing
end
@kernel function sptBrnExt!(ovrRng::NTuple{3,UnitRange{Int}}, stdRng::NTuple{3,UnitRange{Int}}, ovrOff::NTuple{3,Int}, dirSpt::Int, dirItr::Int, parItr::Int, phzVec::AbstractVector{<:Complex}, orgVec::AbstractArray{<:Complex,5}, eveVec::AbstractArray{<:Complex,5}, oddVec::AbstractArray{<:Complex,5})
    itr = @index(Global, NTuple)
    @inbounds begin
        phz = phzVec[itr[dirSpt]]
        # is it in the spill‑over region?
        if all(map(in, itr, ovrRng))
            # the other index
            sprItr = itr .+ ovrOff
            eveVec[itr..., dirItr, parItr] = orgVec[itr..., dirItr, parItr] + orgVec[sprItr..., dirItr, parItr]
            oddVec[itr..., dirItr, parItr] = phz * (orgVec[itr..., dirItr, parItr] - orgVec[sprItr..., dirItr, parItr])
        # else if it's in the standard region
        elseif all(map(in, itr, stdRng))
            eveVec[itr..., dirItr, parItr] = orgVec[itr..., dirItr, parItr]
            oddVec[itr..., dirItr, parItr] = phz * orgVec[itr..., dirItr, parItr]
        # else: skip (they're already zero)
        end
    end
end
#=
merge two branches, eliminating unused coefficients
=#
function mrgBrn!(orgVec::AbstractArray{<:Complex,5}, mrgDir::Integer, parNumTrg::Integer, phzVec::AbstractVector{<:Complex}, prgVec::AbstractArray{<:Complex,5})
    # launch one fused kernel per (partition, direction)
    vecSze = size(orgVec)[1:3]
    ker = mrgKer!(get_backend(orgVec))

    for prtItr in 1:parNumTrg, dirItr in 1:3
        ker(mrgDir, phzVec, dirItr, prtItr, orgVec, prgVec;
            ndrange = vecSze)
    end

    # free the progeny vector on GPU
    memCln!(prgVec)
    return nothing
end
@kernel function mrgKer!(mrgDir::Integer, phzVec::AbstractVector{<:Complex}, dirItr::Integer, prtItr::Integer, orgVec::AbstractArray{<:Complex,5}, prgVec::AbstractArray{<:Complex,5})
    itr = @index(Global, NTuple)
    @inbounds begin
        ph = phzVec[itr[mrgDir]]
        # merge branches in place
        orgVec[itr..., dirItr, prtItr] = (orgVec[itr..., dirItr, prtItr] + conj(ph) * prgVec[itr..., dirItr, prtItr]) / 2
    end
end
#=
generalized host merge allowing for different output size
=#
function mrgBrn!(mrgVec::AbstractArray{<:Complex,5}, mrgDir::Integer, parNumTrg::Integer, phzVec::AbstractVector{<:Complex}, eveVec::AbstractArray{<:Complex,5}, oddVec::AbstractArray{<:Complex,5})
    # 3D sizes
    mrgSze3 = size(mrgVec)[1:3]
    curSze3 = size(eveVec)[1:3]

    # how much of the top comes from eve/odd
    topStop = min(mrgSze3[mrgDir], curSze3[mrgDir])
    offSet  = topStop

    # kernel launcher
    ker = mrgKerGnl!(get_backend(mrgVec))

    # launch one kernel per (partition, direction)
    for prtItr in 1:parNumTrg, dirItr in 1:3
        ker(topStop, offSet,
            mrgDir, dirItr, prtItr,
            phzVec, eveVec, oddVec, mrgVec;
            ndrange = mrgSze3)
    end

    # free intermediate buffers
    memCln!(eveVec)
    memCln!(oddVec)
    return nothing
end
@kernel function mrgKerGnl!(topStop::Integer, offSet::Integer, mrgDir::Integer, dirItr::Integer, prtItr::Integer, phzVec::AbstractVector{<:Complex}, eveVec::AbstractArray{<:Complex,5}, oddVec::AbstractArray{<:Complex,5}, mrgVec::AbstractArray{<:Complex,5})
    itr = @index(Global, NTuple)
    top = itr[mrgDir] <= topStop
    # map global→input coords (for accessing eve/odd)
    inp = top ? itr : ntuple(dir -> dir == mrgDir ? itr[dir] - offSet : itr[dir], 3)
    ph = phzVec[inp[mrgDir]]
    @inbounds begin
        if top
            mrgVec[itr..., dirItr, prtItr] = (eveVec[inp..., dirItr, prtItr] + conj(ph) * oddVec[inp..., dirItr, prtItr]) / 2
        else
            mrgVec[itr..., dirItr, prtItr] = (eveVec[inp..., dirItr, prtItr] - conj(ph) * oddVec[inp..., dirItr, prtItr]) / 2
        end
    end
end
#=
merge partitions and return output vector 
=#
function mrgPrt!(mixInf::GlaExtInf, parNum::Integer, prtVec::AbstractArray{<:Complex, 5})
    maxItr = mixInf.trgCel
    mrgVec = similar(prtVec, (mixInf.trgDiv .* mixInf.trgCel)..., 3)
    synchronize(get_backend(prtVec))
    ker = mrgPrtKer!(get_backend(prtVec))
    for parItr in 1:parNum, dirItr in 1:3
        stp = Tuple(mixInf.trgDiv)
        off = _parOff(mixInf.trgDiv, parItr)
        ker(stp, off, dirItr, parItr, mrgVec, prtVec; ndrange=maxItr)
    end
    synchronize(get_backend(prtVec))
    memCln!(prtVec)
    return mrgVec
end
@kernel function mrgPrtKer!(stp::NTuple{3,Integer}, off::NTuple{3,Integer}, dirItr::Integer, parItr::Integer, mrgVec::AbstractArray{<:Complex,4}, prtVec::AbstractArray{<:Complex,5})
    itr = @index(Global, NTuple)
    trgItr = (itr .- 1) .* stp .+ off .+ 1
    @inbounds mrgVec[trgItr..., dirItr] = prtVec[itr..., dirItr, parItr]
end

# Release a device buffer ahead of the GC, a no-op on the CPU
function memCln!(vec::AbstractArray)
    dev = get_backend(vec)
    synchronize(dev)
    finalize(vec)
    synchronize(dev)
    return nothing
end
