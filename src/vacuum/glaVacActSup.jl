# The phases of axis dir, shaped to broadcast along it
_phzDir(phzVec::AbstractVector{<:Complex}, dir::Integer) = reshape(phzVec, ntuple(d -> d == dir ? length(phzVec) : 1, 3))

#=
restructure input vector to match partitioned Green function data: cell i of
partition o sits at (i - 1) * div + o + 1, partitions in column-major order
=#
function genPrt!(actVec::AbstractArray{<:Complex, 4}, mixInf::GlaExtInf)
    div, cel = mixInf.srcDiv, mixInf.srcCel
    orgVec = similar(actVec, cel..., 3, prod(div))
    permutedims!(reshape(orgVec, cel..., 3, div...),
        reshape(actVec, div[1], cel[1], div[2], cel[2], div[3], cel[3], 3), (2, 4, 6, 7, 1, 3, 5))
    memCln!(actVec)
    return orgVec
end

#=
split a branch so that even and odd Fourier coefficients are independent: the
even branch stays in orgVec, the odd one is its twiddled copy
=#
function sptBrn!(prgVec::AbstractArray{<:Complex, 5}, dir::Integer, phzVec::AbstractVector{<:Complex}, orgVec::AbstractArray{<:Complex, 5})
    prgVec .= _phzDir(phzVec, dir) .* orgVec
    return nothing
end
#=
split branches for external Green function: source cells past the branch length
fold back onto its start, and the progeny past the source are zero
=#
function sptBrn!(eveVec::AbstractArray{<:Complex,5}, oddVec::AbstractArray{<:Complex,5}, dir::Integer, phzVec::AbstractVector{<:Complex}, mixInf::GlaExtInf, orgVec::AbstractArray{<:Complex,5})
    brn, src = div(mixInf.trgCel[dir] + mixInf.srcCel[dir], 2), mixInf.srcCel[dir]
    phz = _phzDir(phzVec, dir)
    sel(arr, rng) = selectdim(arr, dir, rng)
    ovr = 1:(src - brn)
    std = (length(ovr) + 1):min(src, brn)
    fill!(eveVec, zero(eltype(eveVec)))
    fill!(oddVec, zero(eltype(oddVec)))
    sel(eveVec, ovr) .= sel(orgVec, ovr) .+ sel(orgVec, ovr .+ brn)
    sel(oddVec, ovr) .= sel(phz, ovr) .* (sel(orgVec, ovr) .- sel(orgVec, ovr .+ brn))
    sel(eveVec, std) .= sel(orgVec, std)
    sel(oddVec, std) .= sel(phz, std) .* sel(orgVec, std)
    return nothing
end
#=
merge two branches, eliminating unused coefficients
=#
function mrgBrn!(orgVec::AbstractArray{<:Complex,5}, dir::Integer, phzVec::AbstractVector{<:Complex}, prgVec::AbstractArray{<:Complex,5})
    orgVec .= (orgVec .+ conj.(_phzDir(phzVec, dir)) .* prgVec) ./ 2
    # free the progeny vector on GPU
    memCln!(prgVec)
    return nothing
end
#=
generalized merge allowing for different output size: the output past the
branch length comes from the start of the branches, with the opposite sign
=#
function mrgBrn!(mrgVec::AbstractArray{<:Complex,5}, dir::Integer, phzVec::AbstractVector{<:Complex}, eveVec::AbstractArray{<:Complex,5}, oddVec::AbstractArray{<:Complex,5})
    phz = _phzDir(phzVec, dir)
    sel(arr, rng) = selectdim(arr, dir, rng)
    top = 1:min(size(mrgVec, dir), size(eveVec, dir))
    bot = 1:(size(mrgVec, dir) - length(top))
    sel(mrgVec, top) .= (sel(eveVec, top) .+ conj.(sel(phz, top)) .* sel(oddVec, top)) ./ 2
    sel(mrgVec, bot .+ length(top)) .= (sel(eveVec, bot) .- conj.(sel(phz, bot)) .* sel(oddVec, bot)) ./ 2
    # free intermediate buffers
    memCln!(eveVec)
    memCln!(oddVec)
    return nothing
end
#=
merge partitions and return output vector, the inverse of genPrt!
=#
function mrgPrt!(mixInf::GlaExtInf, prtVec::AbstractArray{<:Complex, 5})
    div, cel = mixInf.trgDiv, mixInf.trgCel
    mrgVec = similar(prtVec, (div .* cel)..., 3)
    synchronize(get_backend(prtVec))
    permutedims!(reshape(mrgVec, div[1], cel[1], div[2], cel[2], div[3], cel[3], 3),
        reshape(prtVec, cel..., 3, div...), (5, 1, 6, 2, 7, 3, 4))
    synchronize(get_backend(prtVec))
    memCln!(prtVec)
    return mrgVec
end

# Release a device buffer ahead of the GC, a no-op on the CPU
function memCln!(vec::AbstractArray)
    dev = get_backend(vec)
    synchronize(dev)
    finalize(vec)
    synchronize(dev)
    return nothing
end
