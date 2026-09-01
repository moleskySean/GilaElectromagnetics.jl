# GilaVolumes tests
import GilaElectromagnetics.GilaVolumes:
    GlaExtInf, genVolEve, genEveExtInf, genCntVol, sepGrd, crcIndClc, uniVol

@testset "GlaVol construction" begin
    for dim in volSizes
        volObj = mkVol(dim)
        @test volObj.cel == Tuple(dim)
        @test volObj.scl == stdScl
        @test volObj.org == stdOrg
        @test length(volObj.grd) == 3
        @test all(length.(volObj.grd) .== dim)
        @test all(Rational.(step.(volObj.grd)) .== stdScl)
        # Accepts NTuple as well as Array cel arg
        volArr = GlaVol(collect(dim), stdScl, stdOrg)
        @test volArr.cel == Tuple(dim)
    end
end

@testset "Even/odd grid centers" begin
    # Even: centers are half-cell from edges, symmetric about org
    volEve = GlaVol((4, 4, 4), stdScl, stdOrg)
    for d in 1:3
        grd = volEve.grd[d]
        @test first(grd) == stdOrg[d] - (2 - 1//2) * stdScl[d]
        @test last(grd)  == stdOrg[d] + (2 - 1//2) * stdScl[d]
    end
    # Odd: center cell sits exactly at org
    volOdd = GlaVol((3, 3, 3), stdScl, stdOrg)
    for d in 1:3
        grd = volOdd.grd[d]
        @test stdOrg[d] ∈ grd
        @test first(grd) == stdOrg[d] - 1 * stdScl[d]
        @test last(grd)  == stdOrg[d] + 1 * stdScl[d]
    end
end

@testset "GlaVol equality" begin
    v1 = mkVol((4,4,4))
    v2 = mkVol((4,4,4))
    v3 = mkVol((4,4,4); org=extOrg)
    v4 = mkVol((4,4,4); scl=(1//16, 1//16, 1//16))
    v5 = mkVol((6,4,4))
    @test v1 == v2
    @test v1 != v3
    @test v1 != v4
    @test v1 != v5
end

@testset "GlaVol constructor error" begin
    # celScl > grdScl should throw
    @test_throws ErrorException GlaVol((4,4,4), (1//16, 1//16, 1//16), stdOrg, (1//32, 1//32, 1//32))
    # celScl == grdScl is fine
    @test_nowarn GlaVol((4,4,4), stdScl, stdOrg, stdScl)
end

@testset "uniVol / union" begin
    v1 = mkVol((4,4,4))
    # v2 shifted by 4 cells (touching but not overlapping)
    v2 = GlaVol((4,4,4), stdScl, (4//32, 0//1, 0//1))
    uni = uniVol(v1, v2)
    # Physical extent: from edge of v1 to edge of v2
    lwrEdg1 = first.(v1.grd) .- (v1.scl .// 2)
    uprEdg2 = last.(v2.grd) .+ (v2.scl .// 2)
    lwrEdgU = first.(uni.grd) .- (uni.scl .// 2)
    uprEdgU = last.(uni.grd) .+ (uni.scl .// 2)
    @test Tuple(lwrEdgU) == Tuple(min.(lwrEdg1, first.(v2.grd) .- v2.scl .// 2))
    @test Tuple(uprEdgU) == Tuple(uprEdg2)
    @test uni.scl == min.(v1.scl, v2.scl)
    @test all(isinteger.(uni.cel))
    # Base.union agrees
    @test union(v1, v2) == uniVol(v1, v2)
    # Three-way union
    v3 = GlaVol((4,4,4), stdScl, (8//32, 0//1, 0//1))
    @test union(v1, v2, v3) == reduce(uniVol, [v1, v2, v3])
    # Incommensurate scales throw
    vBad = GlaVol((4,4,4), (1//48, 1//48, 1//48), extOrg)
    @test_throws AssertionError uniVol(v1, vBad)
end

@testset "GlaExtInf equal-scale" begin
    v1 = mkVol((4,4,4))
    v2 = mkVol((4,4,4); org=extOrg)
    inf = GlaExtInf(v1, v2)
    @test inf.minScl == stdScl
    @test inf.trgDiv == (1,1,1)
    @test inf.srcDiv == (1,1,1)
    @test inf.trgCel == (4,4,4)
    @test inf.srcCel == (4,4,4)
    @test length(inf.trgPar) == 1
    @test length(inf.srcPar) == 1
end

@testset "GlaExtInf mixed-scale (ratio 2)" begin
    sclCoarse = (2//32, 2//32, 2//32)
    vCoarse = GlaVol((4,4,4), sclCoarse, stdOrg)
    vFine   = GlaVol((8,8,8), stdScl, extOrg)
    inf = GlaExtInf(vCoarse, vFine)
    # minScl = (1/32), maxScl = (2/32)
    # trgDiv = maxScl/coarseScl = 1; srcDiv = maxScl/fineScl = 2
    @test inf.minScl == stdScl
    @test inf.trgDiv == (1,1,1)
    @test inf.srcDiv == (2,2,2)
    @test inf.trgCel == (4,4,4)
    @test inf.srcCel == (4,4,4)
    @test length(inf.trgPar) == 1
    @test length(inf.srcPar) == 8
end

@testset "GlaExtInf error cases" begin
    # Incommensurate scales
    vA = mkVol((4,4,4))
    vB = GlaVol((4,4,4), (1//48, 1//48, 1//48), extOrg)
    @test_throws ErrorException GlaExtInf(vA, vB)
    # Volume smaller than cell (srcVol too small relative to scale ratio)
    vCoarse = GlaVol((1,1,1), (2//32, 2//32, 2//32), stdOrg)
    vFine   = GlaVol((1,1,1), stdScl, extOrg)
    @test_throws ErrorException GlaExtInf(vCoarse, vFine)
end

@testset "genEveExtInf" begin
    # Even+even → no regen, no warn
    v1 = mkVol((4,4,4))
    v2 = mkVol((4,4,4); org=extOrg)
    @test_nowarn genEveExtInf(v1, v2)
    inf = genEveExtInf(v1, v2)
    @test isa(inf, GlaExtInf)

    # Odd+odd where dim-wise sum is all-even → no regen, no warn
    vOdd  = mkVol((3,3,3))
    vOdd2 = mkVol((3,3,3); org=extOrg)
    @test_nowarn genEveExtInf(vOdd, vOdd2)

    # Source bug: when regen IS triggered (sum has odd elements), genVolEve changes cell scales
    # non-uniformly, making the resulting scales incommensurate between the two volumes.
    # GlaExtInf then throws "Volume pair must share a common scale grid."
    # Document with @test_throws: the regen path is broken.
    vOddS  = mkVol((3,3,3))
    vEvenS = mkVol((4,4,4); org=extOrg)
    @test_throws ErrorException genEveExtInf(vOddS, vEvenS)
end

@testset "genVolEve" begin
    # Even → returned unchanged
    vEve = mkVol((4,4,4))
    @test genVolEve(vEve) == vEve
    # Odd → warns, physical extent preserved
    vOdd = mkVol((3,3,3))
    eveVol = @test_logs (:warn,) genVolEve(vOdd)
    @test all(iseven.(eveVol.cel))
    @test all(Rational.(eveVol.cel) .* eveVol.scl .== Rational.(vOdd.cel) .* vOdd.scl)
    @test eveVol.org == vOdd.org
end

@testset "genCntVol" begin
    v1 = mkVol((4,4,4))
    v2 = mkVol((4,4,4); org=extOrg)
    cnt = genCntVol(v1, v2)
    @test cnt.scl == min.(v1.scl, v2.scl)
    @test cnt.org == (0//1, 0//1, 0//1)
    # cel = trgDiv + srcDiv for each dim; with equal scales, div=(1,1,1)
    @test cnt.cel == (2, 2, 2)
end

@testset "sepGrd" begin
    vSrc = mkVol((4,4,4))
    vTrg = mkVol((4,4,4); org=extOrg)
    # flip==1: grid over source
    sep1 = sepGrd(vTrg, vSrc, 1)
    @test length(sep1) == 3
    @test Tuple(length.(sep1)) == vSrc.cel
    # flip==2: grid over target
    sep2 = sepGrd(vTrg, vSrc, 2)
    @test length(sep2) == 3
    @test Tuple(length.(sep2)) == vTrg.cel
end

@testset "crcIndClc" begin
    cntVol = GlaVol((4,4,4), stdScl, stdOrg)
    # Positive separation: index is trgInd - srcInd + CartesianIndex(1,1,1)
    trgInd = CartesianIndex(3, 3, 3)
    srcInd = CartesianIndex(1, 1, 1)
    idx = crcIndClc(cntVol, trgInd, srcInd)
    @test idx == CartesianIndex(3, 3, 3)
    # Negative separation: wraps with +2*cel+1 per dim
    trgInd2 = CartesianIndex(1, 1, 1)
    srcInd2 = CartesianIndex(3, 3, 3)
    idx2 = crcIndClc(cntVol, trgInd2, srcInd2)
    sep = trgInd2 - srcInd2  # (-2,-2,-2)
    expected = CartesianIndex(ntuple(i -> -2 + 2*4 + 1, 3))  # = (7,7,7)
    @test idx2 == expected
end
