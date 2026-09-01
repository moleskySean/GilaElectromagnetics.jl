using FastGaussQuadrature
using HCubature

# settings for cubature integral evaluation: relative tolerance and absolute 
# tolerance
const cubRelTol = 1e-6;
# const cubRelTol = 1e-3;
const cubAbsTol = 1e-9;
# const cubAbsTol = 1e-3;
# tolerance for cell contact
const cntTol = 1e-8;

include("glaVacOprMemInt.jl") # For integrals

#=
Threaded loop over cell indices for the adaptive-cubature fill. Per-cell cost
is wildly non-uniform (near cells subdivide heavily, far cells exit at the
absolute tolerance), so on Julia ≥ 1.11 greedy scheduling is used to keep the
near-corner cells from serializing on a single thread.
=#
function thrCubFil!(celFun::F, itrSpc::CartesianIndices{3}) where {F<:Function}
    # TODO: Eventually we should work towards fixed quadrature orders for big
    # separation cells (maybe 4+?) to even out the threading load. For this
    # though, we will need to do some testing to see what order is sufficient
    # for the tolerances we are interested in (the DIRECTFN paper has some
    # starting points for us here)
    @static if VERSION >= v"1.11"
        @threads :greedy for posItr ∈ itrSpc
            celFun(posItr)
        end
    else
        @threads for posItr ∈ itrSpc
            celFun(posItr)
        end
    end
    return nothing
end

function genEgoCrc!(egoCrc::AbstractArray{ComplexF64}, trgVol::GlaVol, srcVol::GlaVol, mixInf::GlaExtInf, cmpInf::GlaKerOpt)
    # self green function case
    if srcVol == trgVol
        return genEgoSlf!(selectdim(selectdim(egoCrc, 7, 1), 6, 1), trgVol, cmpInf)
    end
    # external green function case
    # total number of target and source partitions
    totParTrg = prod(mixInf.trgDiv)
    totParSrc = prod(mixInf.srcDiv)
    # partition source and target for consistent distance offsets
    for trgItr ∈ eachindex(1:totParTrg)
        # target grid offset
        trgGrdOff = Tuple(mixInf.trgPar[trgItr]) 
        # offset of center of partition from center of volume
        trgOrgOff = Rational.((trgGrdOff .- (mixInf.trgDiv .- 1) .// 2) .* trgVol.scl .+ trgVol.org)
        # grid scale of partition
        trgGrdScl = mixInf.trgDiv .* trgVol.scl
        # create target volume partition
        trgVolPar = GlaVol(mixInf.trgCel, trgVol.scl, trgOrgOff, trgGrdScl)
        for srcItr ∈ eachindex(1:totParSrc) 
            # relative grid position
            srcGrdOff = Tuple(mixInf.srcPar[srcItr]) 
            # offset of center of partition from center of volume
            srcOrgOff = Rational.((srcGrdOff .- (mixInf.srcDiv .- 1) .// 2) .* srcVol.scl .+ srcVol.org)
            # grid scale of partition
            srcGrdScl = mixInf.srcDiv .* srcVol.scl
            # create target volume partition
            srcVolPar = GlaVol(mixInf.srcCel, srcVol.scl, srcOrgOff, srcGrdScl)
            # generate green function information for partition pair
            genEgoExt!(selectdim(selectdim(egoCrc, 7, trgItr), 6, srcItr), trgVolPar, srcVolPar, cmpInf)
        end
    end
    return egoCrc
end

#=
    genEgoExt!(egoCrcExt::AbstractArray{T,5}, trgVol::GlaVol, 
    srcVol::GlaVol, cmpInf::GlaKerOpt)::Nothing where 
    T<:Union{ComplexF64,ComplexF32}
    
Calculate circulant vector for the Green function between a target volume, 
trgVol, and source volume, srcVol.
=#
function genEgoExt!(egoCrcExt::AbstractArray{ComplexF64,5}, trgVol::GlaVol, srcVol::GlaVol, cmpInf::GlaKerOpt)
    facLst = facPar()
    trgFac = Float64.(cubFac(trgVol.scl))
    srcFac = Float64.(cubFac(srcVol.scl))
    sepGrdTrg = sepGrd(trgVol, srcVol, 0)
    sepGrdSrc = sepGrd(trgVol, srcVol, 1)
    # calculate Green function values
    return genEgoCrcExt!(egoCrcExt, trgVol, srcVol, sepGrdTrg, sepGrdSrc, trgFac, srcFac, facLst, cmpInf)
end
#=
    
    genEgoSlf!(egoCrc::Array{ComplexF64}, slfVol::GlaVol, 
    cmpInf::GlaKerOpt)::Nothing

Calculate circulant vector of the Green function on a single domain.
=#
function genEgoSlf!(egoCrcSlf::AbstractArray{ComplexF64,5}, slfVol::GlaVol, cmpInf::GlaKerOpt)
    facLst = facPar()
    trgFac = Float64.(cubFac(slfVol.scl))
    srcFac = Float64.(cubFac(slfVol.scl))
    srcGrd = sepGrd(slfVol, slfVol, 0)
    # calculate Green function values
    return genEgoCrcSlf!(slfVol, egoCrcSlf, srcGrd, trgFac, srcFac, facLst, cmpInf)
end
#=
Generate the circulant vector of the Green function between a pair of distinct
domains. Volumes that share interior are rejected. The contact branch runs when
the two bounding boxes come within cntTol plus half the sum of the cell scales
in every dimension, and the two grids sit on a common lattice: integer scale
ratios per dimension, and lower corners a whole number of gcd cells apart. Near
but unaligned pairs take the regular quadrature. Which cell pairs actually get
the contact treatment is decided inside egoFunExtCnt!.
=#
function genEgoCrcExt!(egoCrcExt::AbstractArray{ComplexF64,5}, 
    trgVol::GlaVol, srcVol::GlaVol, sepGrdTrg::AbstractVector{<:StepRange}, 
    sepGrdSrc::AbstractVector{<:StepRange}, trgFac::AbstractArray{<:Number,3}, 
    srcFac::AbstractArray{<:Number,3}, facPar::AbstractMatrix{<:Integer}, 
    cmpInf::GlaKerOpt)
    ## calculate domain separation
    # upper and lower edges of the source volume
    srcEdg = (getproperty.(srcVol.grd, :start) .- (srcVol.scl .//2), 
            getproperty.(srcVol.grd, :stop) .+ (srcVol.scl .//2))
    # upper and lower edges of the target volume
    trgEdg = (getproperty.(trgVol.grd, :start) .- (trgVol.scl .//2), 
            getproperty.(trgVol.grd, :stop) .+ (trgVol.scl .//2))
    # shared interior in every dimension is overlap
    if all(max.(srcEdg[1], trgEdg[1]) .< min.(srcEdg[2], trgEdg[2]))
        throw(ArgumentError("Source and target volumes are overlapping."))
    end
    # separation of the closed boxes, negative where the projections meet
    sepBox = max.(srcEdg[1] .- trgEdg[2], trgEdg[1] .- srcEdg[2])
    cntChk = all(sepBox .< (cntTol .+ ((trgVol.scl .+ srcVol.scl) ./ 2)))
    # contact quadrature needs both grids on a common lattice
    fitChk = all(isinteger, max.(trgVol.scl, srcVol.scl) .//
            min.(trgVol.scl, srcVol.scl)) &&
        all(isinteger, (trgEdg[1] .- srcEdg[1]) .// gcd.(trgVol.scl, srcVol.scl))
    if cntChk && fitChk
        # generate self Green function for contact cells
        cntVol = genCntVol(trgVol, srcVol) 
        egoCrcCnt = Array{eltype(egoCrcExt)}(undef, 3, 3, (2 .* cntVol.cel)...)
        genEgoSlf!(egoCrcCnt, cntVol, cmpInf)
        # pull values for cells in contact
        thrCubFil!(CartesianIndices(axes(egoCrcExt)[3:5])) do posItr
            egoFunExtCnt!(cntVol, view(egoCrcExt, :, :, posItr), egoCrcCnt,
                posItr, trgVol.cel, sepGrdTrg, sepGrdSrc, trgVol.scl,
                srcVol.scl, trgFac, srcFac, facPar, cmpInf)
        end
    else
        thrCubFil!(CartesianIndices(axes(egoCrcExt)[3:5])) do posItr
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
function genEgoCrcSlf!(slfVol::GlaVol, egoCrc::AbstractArray{ComplexF64,5},  
    srcGrd::AbstractVector{<:StepRange}, trgFac::AbstractArray{<:AbstractFloat,3}, 
    srcFac::AbstractArray{<:AbstractFloat,3}, facPar::AbstractMatrix{<:Integer}, 
    cmpInf::GlaKerOpt)
    # allocate intermediate storage for Toeplitz interaction vector
    egoToe = Array{eltype(egoCrc)}(undef, 3, 3, slfVol.cel...)
    # write Green function, ignoring weakly singular integrals
    thrCubFil!(CartesianIndices(axes(egoToe)[3:5])) do crtItr
        @inbounds egoFunInn!(egoToe, crtItr, srcGrd, slfVol.scl, trgFac,
            srcFac, facPar, cmpInf)
    end
    # Gauss-Legendre quadrature
    glQud = gauQud(intOrd(cmpInf))
    # correction values for singular integrals
    # return order of normal faces is xx yy zz
    wS = (^(prod(Float64.(slfVol.scl)), -1) .* wekS(slfVol.scl, glQud, cmpInf))
    # return order of normal faces is xxY xxZ yyX yyZ zzX zzY xy xz yz
    wE = (^(prod(Float64.(slfVol.scl)), -1) .* wekE(slfVol.scl, glQud, cmpInf))
    # return order of normal faces is xx yy zz xy xz yz
    wV = (^(prod(Float64.(slfVol.scl)), -1) .* wekV(slfVol.scl, glQud, cmpInf))
    # correct singular integrals for coincident and adjacent cells
    for posItr ∈ CartesianIndices(ntuple(itr -> min(slfVol.cel[itr], 2), 3))
        egoFunSng!(view(egoToe, :, :, posItr), posItr, wS, wE, wV, 
            srcGrd, slfVol, trgFac, srcFac, facPar, cmpInf)
    end
    # include identity component
    egoToe[1,1,1,1,1] -= 1 / (frqPhz(cmpInf)^2)
    egoToe[2,2,1,1,1] -= 1 / (frqPhz(cmpInf)^2)
    egoToe[3,3,1,1,1] -= 1 / (frqPhz(cmpInf)^2)
    # embed self Toeplitz vector into a circulant vector
    @threads for crtItr ∈ CartesianIndices(axes(egoCrc)[3:5])
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
function egoToeCrc!(egoCrc::AbstractArray{ComplexF64,2}, egoToe::AbstractArray{ComplexF64,5},
    posInd::CartesianIndex{3}, indSpt::Tuple{Vararg{Integer}})
    
    if posInd[1] == (indSpt[1] + 1) || posInd[2] == (indSpt[2] + 1) ||
        posInd[3] == (indSpt[3] + 1)
        fill!(egoCrc, zero(eltype(egoCrc)))
    else
        # flip field under coordinate reflection
        fi = indFlp(posInd[1], indSpt[1])
        fj = indFlp(posInd[2], indSpt[2])
        fk = indFlp(posInd[3], indSpt[3])
        # embedding
        egoCrc .= view(egoToe, :, :,
            indSel(posInd[1], indSpt[1]), indSel(posInd[2], indSpt[2]),
            indSel(posInd[3], indSpt[3])) .*
            SMatrix{3,3,Float64}(1.0, (fj * fi), (fk * fi),
            (fi * fj), 1.0, (fk * fj),
            (fi * fk), (fj * fk), 1.0)
    end
    return nothing
end
#=
Write Green function element for a pair of cubes in distinct domains. Recall 
that grids span the separations between a pair of volumes. No field flips are 
required when calculating an external Green function. 
=#
@inline function egoFunExt!(egoCrcExt::AbstractMatrix{ComplexF64}, 
    posInd::CartesianIndex{3}, 
    indSpt::Union{AbstractVector{<:Integer},NTuple{3,Integer}}, 
    sepGrdTrg::AbstractVector{<:StepRange}, sepGrdSrc::AbstractVector{<:StepRange}, 
    sclTrg::NTuple{3,Number}, sclSrc::NTuple{3,Number}, 
    trgFac::AbstractArray{<:Number,3}, srcFac::AbstractArray{<:AbstractFloat,3}, 
    facPar::AbstractMatrix{<:Integer}, cmpInf::GlaKerOpt)
    
    if posInd[1] == (indSpt[1] + 1) || posInd[2] == (indSpt[2] + 1) ||
            posInd[3] == (indSpt[3] + 1)
        fill!(egoCrcExt, zero(eltype(egoCrcExt)))
    else
        egoFunOut!(egoCrcExt, SVector(grdSel(posInd[1], indSpt[1], 1,
            sepGrdTrg, sepGrdSrc), grdSel(posInd[2], indSpt[2], 2,
            sepGrdTrg, sepGrdSrc), grdSel(posInd[3], indSpt[3], 3,
            sepGrdTrg, sepGrdSrc)), sclTrg, sclSrc, trgFac, srcFac, facPar,
            cmpInf)
    end
    return nothing
end
#=
Write Green element for a pair of cubes in distinct domains, checking for the 
possibility of contact. Separation grids span the separations between the pair 
of volumes. No field flip is required for external Green function. 
=#
function egoFunExtCnt!(cntVol::GlaVol, egoCrc::AbstractMatrix{ComplexF64},
    egoCrcCnt::AbstractArray{ComplexF64,5}, posInd::CartesianIndex{3}, 
    indSpt::NTuple{3,Integer}, sepGrdTrg::AbstractVector{<:StepRange}, 
    sepGrdSrc::AbstractVector{<:StepRange}, sclTrg::NTuple{3,Number}, 
    sclSrc::NTuple{3,Number}, trgFac::AbstractArray{<:AbstractFloat,3}, 
    srcFac::AbstractArray{<:AbstractFloat,3}, facPar::AbstractMatrix{<:Integer}, 
    cmpInf::GlaKerOpt)
    # separation vector
    sepVec = SVector(grdSel(posInd[1], indSpt[1], 1, sepGrdTrg, sepGrdSrc),
        grdSel(posInd[2], indSpt[2], 2, sepGrdTrg, sepGrdSrc),
        grdSel(posInd[3], indSpt[3], 3, sepGrdTrg, sepGrdSrc))
    # absolute separation
    absSep = abs.(sepVec)
    # branch between contact and non-contact cases
    # contact case
    if all(absSep .< (cntTol .+ ((sclTrg .+ sclSrc) ./ 2.0)))
        if posInd[1] == (indSpt[1] + 1) || posInd[2] == (indSpt[2] + 1) ||
            posInd[3] == (indSpt[3] + 1)
            fill!(egoCrc, zero(eltype(egoCrc)))
        else
            egoCntOut!(cntVol, egoCrcCnt, egoCrc, posInd, sclTrg, sclSrc,
                sepVec)
        end
    # non-contact case
    else
        if posInd[1] == (indSpt[1] + 1) || posInd[2] == (indSpt[2] + 1) ||
            posInd[3] == (indSpt[3] + 1)
            fill!(egoCrc, zero(eltype(egoCrc)))
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
function egoFunOut!(egoCrc::AbstractMatrix{ComplexF64}, grd::AbstractVector{<:AbstractFloat}, 
    sclTrg::NTuple{3,Number}, sclSrc::NTuple{3,Number}, 
    trgFac::AbstractArray{<:AbstractFloat,3}, srcFac::AbstractArray{<:AbstractFloat,3}, 
    facPar::AbstractMatrix{<:Integer}, cmpInf::GlaKerOpt)

    srfMat = zeros(MVector{36,ComplexF64})
    # calculate interaction contributions between all cube faces
    egoSrfAdp!(grd[1], grd[2], grd[3], srfMat, trgFac, srcFac, 1:36,
               facPar, srfScl(Float64.(sclTrg), Float64.(sclSrc)), cmpInf)
    # sum contributions depending on source and target current orientation 
    return srfSum!(egoCrc, srfMat)
end
#=
Compute Green function element for cells in contact. 
=#
function egoCntOut!(cntVol::GlaVol, egoCrcCnt::AbstractArray{ComplexF64,5}, 
    egoCrc::AbstractMatrix{ComplexF64}, posInd::CartesianIndex{3}, 
    sclTrg::NTuple{3,Number}, sclSrc::NTuple{3,Number}, 
    sepVec::AbstractVector{<:AbstractFloat})
    # safety zero local section of the Green function
    fill!(egoCrc, zero(eltype(egoCrc)))
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
            egoCrc .+= view(egoCrcCnt, :, :,
                crcIndClc(cntVol, indTrg, indSrc))
        end
    end
    totTrgCells = (trgCellLocs[2][3] - trgCellLocs[1][3] + 1) * 
        (trgCellLocs[2][2] - trgCellLocs[1][2] + 1) * 
        (trgCellLocs[2][1] - trgCellLocs[1][1] + 1)
    egoCrc ./= totTrgCells
    return nothing
end

#=
General self Green function interaction element. 
=#
function egoFunInn!(egoToe::AbstractArray{ComplexF64,5}, posInd::CartesianIndex{3},
    srcGrd::AbstractVector{<:StepRange}, scl::NTuple{3,Number}, 
    trgFac::AbstractArray{<:AbstractFloat,3}, srcFac::AbstractArray{<:AbstractFloat,3}, 
    facPar::AbstractMatrix{<:Integer}, cmpInf::GlaKerOpt)

    # calculate interaction contributions between all cube faces
    if (posInd[1] > 2) || (posInd[2] > 2) || (posInd[3] > 2)
        srfMat = zeros(MVector{36,ComplexF64})
        egoSrfAdp!(Float64(srcGrd[1][posInd[1]]), Float64(srcGrd[2][posInd[2]]),
            Float64(srcGrd[3][posInd[3]]), srfMat, trgFac, srcFac, 1:36, facPar, 
            Float64.(srfScl(Float64.(scl), Float64.(scl))), cmpInf)
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
function egoFunSng!(egoCrc::AbstractMatrix{ComplexF64}, posInd::CartesianIndex{3}, 
    wS::AbstractVector{ComplexF64}, wE::AbstractVector{ComplexF64}, wV::AbstractVector{ComplexF64}, 
    srcGrd::AbstractVector{<:StepRange}, slfVol::GlaVol, 
    trgFac::AbstractArray{<:AbstractFloat,3}, srcFac::AbstractArray{<:AbstractFloat,3}, 
    facPar::AbstractMatrix{<:Integer}, cmpInf::GlaKerOpt)

    srfMat = zeros(MVector{36,ComplexF64})
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
        @warn "empty case selected"
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
        facPar, Float64.(srfScl(Float64.(slfVol.scl), Float64.(slfVol.scl))), cmpInf)
    # correct values of srfMat where needed
    for fp ∈ 1:36
        if mask[facPar[1, fp], facPar[2, fp]] == 1
            srfMat[fp] = corVal[facPar[1, fp], facPar[2, fp]]
        end
    end
    # overwrite problematic elements of Green function matrix
    return srfSum!(egoCrc, srfMat)
end
#=
Update egoCrc to hold Green function interactions. The storage format of egoCrc 
is [[ii, ji, ki]^{T}; [ij, jj, kj]^{T}; [ik, jk, kk]^{T}]. 
See documentation for explanation.
=#
function srfSum!(egoCrc::AbstractMatrix{ComplexF64}, srfMat::AbstractVector{ComplexF64})
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
    grdZ::AbstractFloat, srfMat::AbstractVector{ComplexF64}, 
    trgFac::AbstractArray{<:AbstractFloat,3}, srcFac::AbstractArray{<:AbstractFloat,3}, 
    pairList::Union{UnitRange{<:Integer},AbstractVector{<:Integer}}, 
    facPar::AbstractMatrix{<:Integer}, srfScales::AbstractVector{<:AbstractFloat}, 
    cmpInf::GlaKerOpt)
    @inbounds for fp ∈ pairList
        srfMat[fp] = zero(eltype(srfMat))
        # define integration kernel  
        # intKer = (ordVec::AbstractVector{Float64}, 
        #     vals::AbstractVector{Float64}) -> srfKer(ordVec, vals, grdX, grdY, 
        #     grdZ, fp, trgFac, srcFac, facPar, cmpInf)
        intKer = (ordVec::AbstractVector{Float64}) -> srfKer(ordVec, grdX, grdY, 
            grdZ, fp, trgFac, srcFac, facPar, cmpInf)
        # surface integration
        srfMat[fp] = hcubature(intKer, SVector{4}(0.0, 0.0, 0.0, 0.0), 
            SVector{4}(1.0, 1.0, 1.0, 1.0), rtol=cubRelTol, atol=cubAbsTol)[1];
        # scaling correction
        srfMat[fp] *= srfScales[fp]
    end
    return nothing
end
#=
Integration kernel for Green function surface integrals.
=#
@inline function srfKer(ordVec::AbstractVector{<:AbstractFloat}, grdX::AbstractFloat,
        grdY::AbstractFloat, grdZ::AbstractFloat, fp::Integer,
        trgFac::AbstractArray{<:AbstractFloat,3},
        srcFac::AbstractArray{<:AbstractFloat,3},
        facPar::AbstractMatrix{<:Integer}, cmpInf::GlaKerOpt)
    # value of scalar Green function
    return sclEgo(dstMag(
                      cubVecAltAdp(1, ordVec, fp, trgFac, srcFac, facPar) + grdX, 
                      cubVecAltAdp(2, ordVec, fp, trgFac, srcFac, facPar) + grdY, 
                      cubVecAltAdp(3, ordVec, fp, trgFac, srcFac, facPar) + grdZ), 
               frqPhz(cmpInf))
end
#=
Return the separation between two elements from circulant embedding indices and 
domain grids. 
=#
@inline function grdSel(ind::Integer, indSpt::Integer, dir::Integer, 
    trgGrd::AbstractVector{<:StepRange}, srcGrd::AbstractVector{<:StepRange})
    
    if ind <= indSpt
        return Float64(trgGrd[dir][ind])
    end
    if ind > (1 + indSpt)
        ind -= 1
    end     
    return Float64(srcGrd[dir][ind - indSpt])
end
#=
Return a reference index relative to the embedding index of the Green function. 
=#
@inline function indSel(posInd::T, indSpt::R) where 
    {T<:Union{CartesianIndex, Tuple{Vararg{Integer}}, Integer}, 
    R<:Union{CartesianIndex, Tuple{Vararg{Integer}}, Integer}}
        
    return CartesianIndex(map((x, y) -> x <= y ? x : 
        (x == y + 1 ? 2 * y - x + 1 : 2 * y - x + 2), 
        Tuple(posInd), Tuple(indSpt)))
end
#=
Flip dipole direction based on index values. 
=#
@inline function indFlp(posInd::Integer, indSpt::Integer)
    return posInd <= indSpt ? 1.0 : -1.0
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
function gauQud(ord::Int64)
    pos, val = gausslegendre(ord)
    return [pos ;; val]
end
