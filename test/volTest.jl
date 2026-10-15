# GilaVolumes tests
import GilaElectromagnetics.GilaVolumes: GlaExtInf, uniVol

@testset "GlaVol construction" begin
    for dim in [(2,2,2), (4,4,4), (6,6,6), (8,8,8), (6,4,8), (8,2,10)]
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
    v4 = mkVol((4,4,4); scl=scl16)
    v5 = mkVol((6,4,4))
    @test v1 == v2
    @test v1 != v3
    @test v1 != v4
    @test v1 != v5
end

@testset "GlaVol constructor error" begin
    # celScl > grdScl should throw
    @test_throws ErrorException GlaVol((4,4,4), scl16, stdOrg, stdScl)
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
    @test lwrEdgU == min.(lwrEdg1, first.(v2.grd) .- v2.scl .// 2)
    @test uprEdgU == uprEdg2
    @test uni.scl == min.(v1.scl, v2.scl)
    @test all(isinteger.(uni.cel))
    # Base.union agrees
    @test union(v1, v2) == uniVol(v1, v2)
    # Three-way union
    v3 = GlaVol((4,4,4), stdScl, (8//32, 0//1, 0//1))
    @test union(v1, v2, v3) == reduce(uniVol, [v1, v2, v3])
    # Incommensurate scales throw
    vBad = GlaVol((4,4,4), (1//48, 1//48, 1//48), extOrg)
    @test_throws ArgumentError uniVol(v1, vBad)
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

#= Known bug: an odd-cell volume against an even one is regenerated with a cell
size that no longer shares a grid with the other. Flip to a passing build when
the regeneration is fixed. =#
@testset "Odd and even volume pair" begin
    vOdd, vEve = mkVol((3,3,3)), mkVol((4,4,4); org=extOrg)
    @test_logs (:warn, r"regenerated") (@test_throws ErrorException GlaOprVac{Float64}(vOdd, vEve))
end

@testset "GlaVol show" begin
    @test shwChk(mkVol((4,4,4)))
end
