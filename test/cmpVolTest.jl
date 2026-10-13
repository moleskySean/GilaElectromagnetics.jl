# GlaCmpVol (composite volume) geometry tests
import GilaElectromagnetics.GilaVolumes: _lwrEdg, _uprEdg, _ovrLap

# Base region for the carving tests: 8 cells of 1/16 λ per side, centered on the
# origin, so it spans -1/4 to 1/4 in every dimension.
mkCmpVol() = GlaVol((8, 8, 8), scl16, stdOrg)

# Exact rational volume filled by the regions of a composite volume
cmpTilVol(cvol) = sum(prod(reg.cel .* reg.scl) for reg in regions(cvol))

# The regions of cvol tile vol exactly, and no two of them share interior
function cmpTilChk(cvol, vol)
    @test cmpTilVol(cvol) == prod(vol.cel .* vol.scl)
    regs = regions(cvol)
    @test !any(_ovrLap(regs[i], regs[j])
        for i in eachindex(regs) for j in (i + 1):length(regs))
end

@testset "GlaCmpVol trivial composite" begin
    vol = mkCmpVol()
    cvol = GlaCmpVol(vol)
    @test nregions(cvol) == 1
    @test regions(cvol) == [vol]
    @test cvol == CompositeVolume(vol)
    @test finest(cvol) == scl16
    # One entry per cell, three per degree of freedom
    crd = collect(coordinates(cvol))
    @test length(crd) == prod(vol.cel)
    @test length(coordinates(cvol)) == prod(vol.cel)
    @test length(cellvolumes(cvol)) == 3 * prod(vol.cel)
    @test all(trp -> trp[3] == 1, crd)
    domVol = Float64(prod(vol.cel .* vol.scl))
    @test sum(trp -> trp[2], crd) ≈ domVol
    @test sum(cellvolumes(cvol)) ≈ 3 * domVol
    # Every cell center of the region shows up exactly once
    @test Set(first.(crd)) ==
        Set(ntuple(dir -> Float64(vol.grd[dir][ind[dir]]), 3)
            for ind in CartesianIndices(vol.cel))
end

@testset "GlaCmpVol single refine" begin
    vol = mkCmpVol()
    cvol = refine(GlaCmpVol(vol), (stdOrg, (1//8, 1//8, 1//8)))
    # Core plus six slabs
    @test nregions(cvol) == 7
    cmpTilChk(cvol, vol)
    core = regions(cvol)[1]
    @test core.scl == stdScl
    @test core.cel == (8, 8, 8)
    @test all(mod.(core.cel, 4) .== 0)
    @test core.org == stdOrg
    @test finest(cvol) == stdScl
    # The slabs stay at the parent scale and keep even cell counts
    for slb in regions(cvol)[2:end]
        @test slb.scl == scl16
        @test all(iseven.(slb.cel))
        @test all(slb.cel .> 0)
    end
    # Fixed slab order: xlo, xhi, ylo, yhi, zlo, zhi
    @test [reg.cel for reg in regions(cvol)[2:end]] ==
        [(2, 8, 8), (2, 8, 8), (4, 2, 8), (4, 2, 8), (4, 4, 2), (4, 4, 2)]
    # The bounding box of the tiling is the original volume
    minEdg = reduce((a, b) -> min.(a, b), _lwrEdg.(regions(cvol)))
    maxEdg = reduce((a, b) -> max.(a, b), _uprEdg.(regions(cvol)))
    @test minEdg == _lwrEdg(vol)
    @test maxEdg == _uprEdg(vol)
    # Degree of freedom count is conserved by the refinement bookkeeping
    @test length(cellvolumes(cvol)) ==
        sum(3 * prod(reg.cel) for reg in regions(cvol))
end

@testset "GlaCmpVol box flush against a face" begin
    vol = mkCmpVol()
    # Box hugs the low x face of the domain, so the x-low slab is empty
    cvol = refine(GlaCmpVol(vol), ((-3//16, 0//1, 0//1), (1//8, 1//8, 1//8)))
    @test nregions(cvol) == 6
    cmpTilChk(cvol, vol)
    core = regions(cvol)[1]
    @test core.scl == stdScl
    @test _lwrEdg(core)[1] == _lwrEdg(vol)[1]
    @test all(reg -> all(reg.cel .> 0), regions(cvol))
end

@testset "GlaCmpVol box covering the domain" begin
    vol = mkCmpVol()
    cvol = refine(GlaCmpVol(vol), (stdOrg, (1//1, 1//1, 1//1)))
    @test nregions(cvol) == 1
    core = regions(cvol)[1]
    @test core.cel == (16, 16, 16)
    @test core.scl == stdScl
    @test core.org == stdOrg
    cmpTilChk(cvol, vol)
    # A box that misses everything leaves the tiling alone
    away = refine(GlaCmpVol(vol), ((10//1, 0//1, 0//1), (1//8, 1//8, 1//8)))
    @test away == GlaCmpVol(vol)
end

@testset "GlaCmpVol outward snapping" begin
    vol = mkCmpVol()
    # Box spans 0 to 1/32 per dimension: off the cell grid and an odd cell count
    boxOrg = (1//64, 1//64, 1//64)
    boxDim = (1//32, 1//32, 1//32)
    cvol = refine(GlaCmpVol(vol), (boxOrg, boxDim))
    core = regions(cvol)[1]
    boxLwr = boxOrg .- (boxDim .// 2)
    boxUpr = boxOrg .+ (boxDim .// 2)
    # The core covers at least the requested box
    @test all(_lwrEdg(core) .<= boxLwr)
    @test all(_uprEdg(core) .>= boxUpr)
    # Two parent cells per dimension, which is even, so every slab is even too
    @test all(core.cel .÷ 2 .== 2)
    @test all(iseven.(core.cel .÷ 2))
    @test all(reg -> all(iseven.(reg.cel)), regions(cvol))
    cmpTilChk(cvol, vol)
    # Snapping is per region, so the core sits on the parent grid
    @test all(isinteger.((_lwrEdg(core) .- _lwrEdg(vol)) .// vol.scl))
end

@testset "GlaCmpVol chained refine" begin
    vol = mkCmpVol()
    cvol = refine(GlaCmpVol(vol), (stdOrg, (1//8, 1//8, 1//8)))
    # The core spans -1/8 to 1/8; refine its high octant again
    cvol2 = refine(cvol, ((1//16, 1//16, 1//16), (1//8, 1//8, 1//8)))
    @test nregions(cvol2) == 10
    cmpTilChk(cvol2, vol)
    @test finest(cvol2) == (1//64, 1//64, 1//64)
    inner = regions(cvol2)[1]
    @test inner.scl == (1//64, 1//64, 1//64)
    @test inner.cel == (8, 8, 8)
    # Three levels of resolution coexist, and the untouched slabs keep their
    # position at the end of the list
    @test Set(reg.scl for reg in regions(cvol2)) ==
        Set([(1//64, 1//64, 1//64), stdScl, scl16])
    @test [reg.cel for reg in regions(cvol2)[5:end]] ==
        [reg.cel for reg in regions(cvol)[2:end]]

    # A sub-box centered on the core leaves 1/32 slabs two cells wide, which
    # gives an odd cell count per partition against the 1/16 slabs
    @test_throws ArgumentError refine(cvol,
        (stdOrg, (1//16, 1//16, 1//16)))
end

@testset "GlaCmpVol anisotropic factor" begin
    vol = mkCmpVol()
    box = (stdOrg, (1//8, 1//8, 1//8))
    cvolZ = refine(GlaCmpVol(vol), box; factor=(1, 1, 2))
    coreZ = regions(cvolZ)[1]
    @test coreZ.scl == (1//16, 1//16, 1//32)
    @test coreZ.cel == (4, 4, 8)
    @test nregions(cvolZ) == 7
    cmpTilChk(cvolZ, vol)
    @test finest(cvolZ) == (1//16, 1//16, 1//32)

    cvolA = refine(GlaCmpVol(vol), box; factor=(2, 2, 4))
    coreA = regions(cvolA)[1]
    @test coreA.scl == (1//32, 1//32, 1//64)
    @test coreA.cel == (8, 8, 16)
    cmpTilChk(cvolA, vol)

    # A factor of 1 everywhere still carves the region
    cvolOne = refine(GlaCmpVol(vol), box; factor=1)
    @test nregions(cvolOne) == 7
    @test all(reg -> reg.scl == scl16, regions(cvolOne))
    cmpTilChk(cvolOne, vol)
end

@testset "GlaCmpVol refine argument errors" begin
    vol = mkCmpVol()
    box = (stdOrg, (1//8, 1//8, 1//8))
    @test_throws ArgumentError refine(GlaCmpVol(vol), box; snap=:inward)
    @test_throws ArgumentError refine(GlaCmpVol(vol), box; factor=0)
    @test_throws ArgumentError refine(GlaCmpVol(vol), box; factor=(2, 0, 2))
    @test_throws ArgumentError refine(GlaCmpVol(vol), box; factor=2.0)
    @test_throws ArgumentError refine(GlaCmpVol(vol), (0.0, 1.0))
end

@testset "GlaCmpVol constructor errors" begin
    # No regions at all
    @test_throws ArgumentError GlaCmpVol(GlaVol[])
    # Odd cell count
    @test_throws ArgumentError GlaCmpVol(GlaVol((3, 4, 4), stdScl, stdOrg))
    # Overlapping interiors
    volA = GlaVol((4, 4, 4), stdScl, stdOrg)
    @test_throws ArgumentError GlaCmpVol([volA, volA])
    # Incommensurate scales: 1/16 against 1/24
    volBad = GlaVol((4, 4, 4), (1//24, 1//24, 1//24), (1//1, 0//1, 0//1))
    @test_throws ArgumentError GlaCmpVol([GlaVol((4, 4, 4), scl16, stdOrg),
        volBad])
    # Misaligned grids: neighbour shifted by half a cell
    volOff = GlaVol((4, 4, 4), stdScl, (9//64, 0//1, 0//1))
    @test_throws ArgumentError GlaCmpVol([volA, volOff])
    # Region whose grid pitch is coarser than its cells
    volPit = GlaVol((4, 4, 4), stdScl, stdOrg, scl16)
    @test_throws ArgumentError GlaCmpVol(volPit)
    # Partition parity violation: a (2,2,2) coarse region against a (2,2,2)
    # region at half the scale gives one cell per partition on both sides
    volCrs = GlaVol((2, 2, 2), scl16, stdOrg)
    volFin = GlaVol((2, 2, 2), stdScl, (1//16 + 1//32, 0//1, 0//1))
    @test_throws ArgumentError GlaCmpVol([volCrs, volFin])
end

@testset "GlaCmpVol gapped tiling" begin
    # Aligned and disjoint with four cells of gap in x between them
    volA = GlaVol((4, 4, 4), stdScl, stdOrg)
    volB = GlaVol((4, 4, 4), stdScl, (1//4, 0//1, 0//1))
    cvol = GlaCmpVol([volA, volB])
    @test nregions(cvol) == 2
    @test finest(cvol) == stdScl
    @test cmpTilVol(cvol) == 2 * prod(volA.cel .* volA.scl)
    # regrid spans the bounding box and leaves the gap at zero
    fld = zerofield(Float64, cvol)
    fld.dat .= one(ComplexF64)
    rsm = regrid(fld)
    @test size(rsm) == (12, 4, 4, 3)
    @test all(iszero, rsm[5:8, :, :, :])
    @test all(!iszero, rsm[1:4, :, :, :])
    @test all(!iszero, rsm[9:12, :, :, :])
end

@testset "GlaCmpVol flat layout" begin
    # Two touching regions of different scale, both 1/8 λ per side
    volCrs = GlaVol((4, 4, 4), stdScl, stdOrg)
    volFin = GlaVol((8, 8, 8), (1//64, 1//64, 1//64), (1//8, 0//1, 0//1))
    cvol = GlaCmpVol([volCrs, volFin])
    @test nregions(cvol) == 2
    @test cmpTilVol(cvol) == 2 * prod(volCrs.cel .* volCrs.scl)
    @test finest(cvol) == (1//64, 1//64, 1//64)

    crd = collect(coordinates(cvol))
    @test length(crd) == prod(volCrs.cel) + prod(volFin.cel)
    # The first block is region 1, in column major order over its own grid
    fstBlk = crd[1:prod(volCrs.cel)]
    @test all(trp -> trp[3] == 1, fstBlk)
    @test all(trp -> trp[3] == 2, crd[(prod(volCrs.cel) + 1):end])
    expPos = [ntuple(dir -> Float64(volCrs.grd[dir][ind[dir]]), 3)
        for ind in CartesianIndices(volCrs.cel)]
    @test first.(fstBlk) == vec(expPos)
    @test all(trp -> trp[2] ≈ Float64(prod(volCrs.scl)), fstBlk)
    # The second block reads off the fine region grid the same way
    sndBlk = crd[(prod(volCrs.cel) + 1):end]
    expPos2 = [ntuple(dir -> Float64(volFin.grd[dir][ind[dir]]), 3)
        for ind in CartesianIndices(volFin.cel)]
    @test first.(sndBlk) == vec(expPos2)

    # cellvolumes matches the degrees of freedom block by block
    celVol = cellvolumes(cvol)
    nCrs = 3 * prod(volCrs.cel)
    nFin = 3 * prod(volFin.cel)
    @test length(celVol) == nCrs + nFin
    @test all(celVol[1:nCrs] .≈ Float64(prod(volCrs.scl)))
    @test all(celVol[(nCrs + 1):end] .≈ Float64(prod(volFin.scl)))
    # The weighted sum over the degrees of freedom is three times the domain
    @test sum(celVol) ≈ 3 * Float64(cmpTilVol(cvol))
end

@testset "GlaCmpVol refine convenience form" begin
    vol = mkCmpVol()
    box = (stdOrg, (1//8, 1//8, 1//8))
    @test refine(vol, box; factor=(1, 1, 2)) ==
        refine(GlaCmpVol(vol), box; factor=(1, 1, 2))
    # A GlaVol works as a box: only its outer bounds matter
    boxVol = GlaVol((2, 2, 2), scl16, stdOrg)
    @test refine(vol, boxVol) == refine(vol, box)
end
