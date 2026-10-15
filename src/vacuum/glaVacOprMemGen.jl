include("integrals/glaVacIntMom.jl") # near-field moments and contact block
include("integrals/glaVacIntFar.jl") # far-field expansion and block fills

function genEgoCrc!(egoCrc::AbstractArray{<:Complex}, trgVol::GlaVol, srcVol::GlaVol, mixInf::GlaExtInf, cmpInf::GlaKerOpt; shpCch::Bool = false, blk = nothing)
    # self green function case
    if srcVol == trgVol
        return genEgoSlf!(selectdim(selectdim(egoCrc, 7, 1), 6, 1), trgVol, cmpInf; shpCch, blk)
    end
    # a Bloch pair of a composite: the block blcCmp gave, in this layout
    cmpInf.blc === nothing || return view(egoCrc, :, :, :, :, :, 1, 1) .= blcKer(cmpInf.blc, blk...)
    # external green function case
    # total number of target and source partitions
    totParTrg = prod(mixInf.trgDiv)
    totParSrc = prod(mixInf.srcDiv)
    # partition source and target for consistent distance offsets
    for trgItr ∈ 1:totParTrg
        # target grid offset
        trgGrdOff = Tuple(mixInf.trgPar[trgItr]) 
        # offset of center of partition from center of volume
        trgOrgOff = Rational.((trgGrdOff .- (mixInf.trgDiv .- 1) .// 2) .* trgVol.scl .+ trgVol.org)
        # grid scale of partition
        trgGrdScl = mixInf.trgDiv .* trgVol.scl
        # create target volume partition
        trgVolPar = GlaVol(mixInf.trgCel, trgVol.scl, trgOrgOff, trgGrdScl)
        for srcItr ∈ 1:totParSrc 
            # relative grid position
            srcGrdOff = Tuple(mixInf.srcPar[srcItr]) 
            # offset of center of partition from center of volume
            srcOrgOff = Rational.((srcGrdOff .- (mixInf.srcDiv .- 1) .// 2) .* srcVol.scl .+ srcVol.org)
            # grid scale of partition
            srcGrdScl = mixInf.srcDiv .* srcVol.scl
            # create target volume partition
            srcVolPar = GlaVol(mixInf.srcCel, srcVol.scl, srcOrgOff, srcGrdScl)
            # generate green function information for partition pair
            genEgoExt!(selectdim(selectdim(egoCrc, 7, trgItr), 6, srcItr), trgVolPar, srcVolPar, cmpInf; shpCch = shpCch)
        end
    end
    return egoCrc
end

#=
    genEgoSlf!(egoCrc::AbstractArray{<:Complex,5}, slfVol::GlaVol,
    cmpInf::GlaKerOpt)::Nothing

Calculate circulant vector of the Green function on a single domain: the contact block from the
closed-form moments, every pair two or more cells apart from the far-field expansion, then the
identity term and the circulant embedding. Under qssApx the same two blocks are taken at zero
frequency, giving S = f^2 G, and the single 1/f^2 at the end keeps that identity exact. With a
Bloch lattice one blcFil call, or the block blk = (H, A) blcCmp gave a composite region, is the
whole block, identity term included.
=#
# The Bloch kernel hrmPrt asks for: H, A or G = H + iA
blcKer(blc, H, A) = blc.hrmPrt === :sym ? H : blc.hrmPrt === :asy ? A : H .+ im .* A

function genEgoSlf!(egoCrc::AbstractArray{<:Complex,5}, slfVol::GlaVol, cmpInf::GlaKerOpt;
    shpCch::Bool = false, blk = nothing)

    prc = real(eltype(egoCrc))
    blc = cmpInf.blc
    stp = ntuple(dir -> Int(step(slfVol.grd[dir]) // slfVol.scl[dir]), 3)
    if blk !== nothing
        egoToe = blcKer(blc, blk...)
    elseif blc !== nothing
        frq = Complex{prc}(frqPhz(cmpInf))
        # one BlcSet per frequency and unit cell, kept for setBlc!
        st = cmpInf.blcSet
        st !== nothing && (st.f, st.N) == (frq, slfVol.cel) ||
            (st = cmpInf.blcSet = blcSet(slfVol.scl, frq, blc.ax, slfVol.cel, blc.P))
        H, A, _ = blcFil(st, blc.θ)
        egoToe = blcKer(blc, H, A)
    elseif qssApx(cmpInf)
        egoToe = Array{eltype(egoCrc)}(undef, 3, 3, slfVol.cel...)
        momBlkQss!(egoToe, slfVol, [(Tuple(itr) .- 1) .* stp # the contact offsets
            for itr ∈ vec(CartesianIndices(min.(slfVol.cel, 2)))])
        ner = farBlkQss!(egoToe, slfVol.scl; tol = farTol(prc), disk = shpCch, stp = stp)
        momBlkQss!(egoToe, slfVol, ner) # the offsets the multipole refuses, left NaN
        all(isfinite, egoToe) || error("genEgoSlf!: quasistatic near band left unfilled.")
        for dir ∈ 1:3
            egoToe[dir,dir,1,1,1] -= 1
        end
        egoToe ./= frqPhz(cmpInf)^2
    else
        egoToe = Array{eltype(egoCrc)}(undef, 3, 3, slfVol.cel...)
        cntBlk!(egoToe, slfVol, cmpInf)
        farBlk!(egoToe, slfVol.scl, Complex{prc}(frqPhz(cmpInf)); tol = farTol(prc),
            disk = shpCch, stp = stp)
        for dir ∈ 1:3
            egoToe[dir,dir,1,1,1] -= 1 / (frqPhz(cmpInf)^2)
        end
    end
    prd = ntuple(dir -> blc !== nothing && dir in blc.ax, 3)
    @threads for crtItr ∈ CartesianIndices(axes(egoCrc)[3:5])
        @inbounds egoToeCrc!(view(egoCrc, :, :, crtItr), egoToe, crtItr,
            div.(size(egoCrc)[3:5], 2), prd)
    end
    return nothing
end
#=
Integer cell offset of every circulant index of an equal-cell pair, per dimension, with the
sub-cell shift a pair off the cell lattice carries. Both separation grids step by a whole number
of cells, so one shift covers the block; nothing when they do not.
=#
function extOff(nCrc::NTuple{3,Integer}, cel::NTuple{3,Integer}, scl::NTuple{3,Rational},
    sepGrdTrg::AbstractVector{<:StepRange}, sepGrdSrc::AbstractVector{<:StepRange})

    sep(dir, ind) = Rational(grdSep(ind, cel[dir], dir, sepGrdTrg, sepGrdSrc))
    # half a cell rounds up, not to even, so a half-cell shift stays one shift
    celOff(dir, ind) = (n = floor(Int, sep(dir, ind) / scl[dir] + 1//2);
        (n, sep(dir, ind) - n * scl[dir]))
    lst = ntuple(dir -> [celOff(dir, ind) for ind ∈ 1:nCrc[dir]], 3)
    off = ntuple(dir -> lst[dir][1][2], 3)
    for dir ∈ 1:3, ind ∈ 1:nCrc[dir]
        ind != cel[dir] + 1 && lst[dir][ind][2] != off[dir] && return nothing
    end
    return (ntuple(dir -> first.(lst[dir]), 3), off)
end
#=
    genEgoExt!(egoCrcExt::AbstractArray{<:Complex,5}, trgVol::GlaVol,
    srcVol::GlaVol, cmpInf::GlaKerOpt)::Nothing

Calculate circulant vector for the Green function between a target volume, trgVol, and source
volume, srcVol. Volumes that share interior are rejected. Cells in contact average the gcd self
block; every separated pair takes the far-field expansion, on the cell lattice for equal cells and
on the exact rational offsets of the pair's own trapezoid otherwise.
=#
function genEgoExt!(egoCrcExt::AbstractArray{<:Complex,5}, trgVol::GlaVol, srcVol::GlaVol,
    cmpInf::GlaKerOpt; shpCch::Bool = false)

    # the far blocks below stay Helmholtz, so a quasistatic contact block would silently mix
    qssApx(cmpInf) && throw(ArgumentError("The quasistatic Green function is self volume only: " *
        "$(trgVol.cel) cells at $(trgVol.org) and $(srcVol.cel) cells at $(srcVol.org) are an external pair."))
    sepGrdTrg = sepGrd(trgVol, srcVol, 0)
    sepGrdSrc = sepGrd(trgVol, srcVol, 1)
    _ovrLap(trgVol, srcVol) && throw(ArgumentError("Source and target volumes are overlapping."))
    itrSpc = CartesianIndices(axes(egoCrcExt)[3:5])
    lat = trgVol.scl == srcVol.scl ? extOff(size(egoCrcExt)[3:5], trgVol.cel, trgVol.scl,
        sepGrdTrg, sepGrdSrc) : nothing
    prc = real(eltype(egoCrcExt))
    cntPos = CartesianIndex{3}[]
    farPos = CartesianIndex{3}[]
    farOff = NTuple{3,Int}[]
    farSep = NTuple{3,QI}[]
    # the closed cells meet in every direction: contact, and everything else is separated
    cntLim = (trgVol.scl .+ srcVol.scl) .// 2
    fill!(egoCrcExt, zero(eltype(egoCrcExt)))
    for posInd ∈ itrSpc
        any(dir -> posInd[dir] == trgVol.cel[dir] + 1, 1:3) && continue
        sep = ntuple(dir -> QI(grdSep(posInd[dir], trgVol.cel[dir], dir, sepGrdTrg,
            sepGrdSrc)), 3)
        if all(dir -> abs(sep[dir]) <= cntLim[dir], 1:3)
            push!(cntPos, posInd)
        else
            push!(farPos, posInd)
            lat === nothing ? push!(farSep, sep) :
                push!(farOff, ntuple(dir -> lat[1][dir][posInd[dir]], 3))
        end
    end
    if lat !== nothing
        farFil!(farOff, ntuple(dir -> QI(trgVol.scl[dir]), 3),
            Complex{prc}(frqPhz(cmpInf)), ntuple(dir -> prc(lat[2][dir]), 3);
            tol = farTol(prc), disk = shpCch) do q
            view(egoCrcExt, :, :, farPos[q])
        end
    elseif !isempty(farPos)
        # the pair's own trapezoid expansion on the exact rational offsets of a cross-scale pair
        egoFar = Array{eltype(egoCrcExt)}(undef, 3, 3, length(farPos))
        farBlkX!(egoFar, farSep, trgVol.scl, srcVol.scl, Complex{prc}(frqPhz(cmpInf));
            tol = farTol(prc), disk = shpCch)
        for posItr ∈ eachindex(farPos)
            @inbounds copyto!(view(egoCrcExt, :, :, farPos[posItr]),
                view(egoFar, :, :, posItr))
        end
    end
    if !isempty(cntPos)
        # the contact average is over cells of the gcd, which both grids must sit on
        all(isinteger, max.(trgVol.scl, srcVol.scl) .// min.(trgVol.scl, srcVol.scl)) &&
            all(isinteger, (_lwrEdg(trgVol) .- _lwrEdg(srcVol)) .// gcd.(trgVol.scl, srcVol.scl)) ||
            throw(ArgumentError("Cells in contact must sit on a common lattice."))
        cntVol = genCntVol(trgVol, srcVol)
        egoCrcCnt = Array{eltype(egoCrcExt)}(undef, 3, 3, (2 .* cntVol.cel)...)
        genEgoSlf!(egoCrcCnt, cntVol, cmpInf; shpCch = shpCch)
        @threads for posInd ∈ cntPos
            cntSep = ntuple(dir -> grdSep(posInd[dir], trgVol.cel[dir], dir, sepGrdTrg, sepGrdSrc), 3)
            egoCntOut!(cntVol, egoCrcCnt, view(egoCrcExt, :, :, posInd), trgVol.scl, srcVol.scl, cntSep)
        end
    end
    return nothing
end
#=
Generate circulant self Green function from Toeplitz self Green function. The 
implemented mask takes into account the relative flip in the assumed dipole 
direction under a coordinate reflection. A periodic axis (prd) is no reflection
symmetry: its Toeplitz block holds the signed offsets m = -(N-1):(N-1) at m + N.
=#
function egoToeCrc!(egoCrc::AbstractArray{<:Complex,2}, egoToe::AbstractArray{<:Complex,5},
    posInd::CartesianIndex{3}, indSpt::Tuple{Vararg{Integer}}, prd::NTuple{3,Bool} = (false, false, false))
    
    if posInd[1] == (indSpt[1] + 1) || posInd[2] == (indSpt[2] + 1) ||
        posInd[3] == (indSpt[3] + 1)
        fill!(egoCrc, zero(eltype(egoCrc)))
    else
        # flip field under coordinate reflection
        fi, fj, fk = ntuple(dir -> prd[dir] || posInd[dir] <= indSpt[dir] ? 1.0 : -1.0, 3)
        # embedding
        egoCrc .= view(egoToe, :, :, ntuple(dir -> prd[dir] ?
            mod(posInd[dir] - 1 + indSpt[dir], 2 * indSpt[dir]) : indSel(posInd[dir], indSpt[dir]), 3)...) .*
            SMatrix{3,3,Float64}(1.0, (fj * fi), (fk * fi),
            (fi * fj), 1.0, (fk * fj),
            (fi * fk), (fj * fk), 1.0)
    end
    return nothing
end
#= The Green function element between two cells in contact: the sum over the
source cell of the contact cells tiling it, averaged over the target cell. sep is
the exact separation of the two cell centers. =#
function egoCntOut!(cntVol::GlaVol, egoCrcCnt::AbstractArray{<:Complex,5}, egoCrc::AbstractMatrix{<:Complex},
    sclTrg::NTuple{3,Rational}, sclSrc::NTuple{3,Rational}, sep::NTuple{3,Rational})
    fill!(egoCrc, zero(eltype(egoCrc)))
    srcSpn = Int.(sclSrc .// cntVol.scl)
    trgSpn = Int.(sclTrg .// cntVol.scl)
    # first contact cell of the target, counted from the first of the source
    trgLwr = Int.(sep .// cntVol.scl .+ (srcSpn .- trgSpn) .// 2) .+ 1
    # shift both into the contact volume when the target starts below the source
    off = max.(0, 1 .- trgLwr)
    srcRng = map((spn, o) -> (1:spn) .+ o, srcSpn, off)
    trgRng = map((lwr, spn, o) -> (lwr:(lwr + spn - 1)) .+ o, trgLwr, trgSpn, off)
    for indSrc ∈ CartesianIndices(srcRng), indTrg ∈ CartesianIndices(trgRng)
        egoCrc .+= view(egoCrcCnt, :, :, crcIndClc(cntVol, indTrg, indSrc))
    end
    egoCrc ./= length(CartesianIndices(trgRng))
    return nothing
end

#=
Return the separation between two elements from circulant embedding indices and 
domain grids. 
=#
@inline function grdSep(ind::Integer, indSpt::Integer, dir::Integer,
    trgGrd::AbstractVector{<:StepRange}, srcGrd::AbstractVector{<:StepRange})

    if ind <= indSpt
        return trgGrd[dir][ind]
    end
    if ind > (1 + indSpt)
        ind -= 1
    end
    return srcGrd[dir][ind - indSpt]
end
# Reference index relative to the embedding index of the Green function. This is the index (in the Toeplitz world) the circulant embedding reflects to.
indSel(posInd::Integer, indSpt::Integer) = posInd <= indSpt ? posInd : posInd == indSpt + 1 ? indSpt : 2 * indSpt - posInd + 2
