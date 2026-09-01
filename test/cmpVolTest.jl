# GlaCmpVol (composite volume) geometry tests
import GilaElectromagnetics.GilaVolumes: _lwrEdg, _uprEdg, _ovrLap

const cmpScl16 = (1//16, 1//16, 1//16)
const cmpScl32 = (1//32, 1//32, 1//32)
const cmpOrg0 = (0//1, 0//1, 0//1)

# Base region for the carving tests: 8 cells of 1/16 λ per side, centered on the
# origin, so it spans -1/4 to 1/4 in every dimension.
mkCmpVol() = GlaVol((8, 8, 8), cmpScl16, cmpOrg0)

# Exact rational volume filled by the regions of a composite volume
cmpTilVol(cvol) = sum(prod(reg.cel .* reg.scl) for reg in regions(cvol))

# True if no two regions of a composite volume share interior
function cmpDsjChk(cvol)
    regs = regions(cvol)
    for i in 1:length(regs), j in (i + 1):length(regs)
        _ovrLap(regs[i], regs[j]) && return false
    end
    return true
end

@testset "GlaCmpVol trivial composite" begin
    vol = mkCmpVol()
    cvol = GlaCmpVol(vol)
    @test nregions(cvol) == 1
    @test regions(cvol) == [vol]
    @test cvol == CompositeVolume(vol)
    @test CompositeVolume === GlaCmpVol
    @test finest(cvol) == cmpScl16
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
    cvol = refine(GlaCmpVol(vol), (cmpOrg0, (1//8, 1//8, 1//8)))
    # Core plus six slabs
    @test nregions(cvol) == 7
    @test cmpDsjChk(cvol)
    @test cmpTilVol(cvol) == prod(vol.cel .* vol.scl)
    core = regions(cvol)[1]
    @test core.scl == cmpScl32
    @test core.cel == (8, 8, 8)
    @test all(mod.(core.cel, 4) .== 0)
    @test core.org == cmpOrg0
    @test finest(cvol) == cmpScl32
    # The slabs stay at the parent scale and keep even cell counts
    for slb in regions(cvol)[2:end]
        @test slb.scl == cmpScl16
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
    @test cmpDsjChk(cvol)
    @test cmpTilVol(cvol) == prod(vol.cel .* vol.scl)
    core = regions(cvol)[1]
    @test core.scl == cmpScl32
    @test _lwrEdg(core)[1] == _lwrEdg(vol)[1]
    @test all(reg -> all(reg.cel .> 0), regions(cvol))
end

@testset "GlaCmpVol box covering the domain" begin
    vol = mkCmpVol()
    cvol = refine(GlaCmpVol(vol), (cmpOrg0, (1//1, 1//1, 1//1)))
    @test nregions(cvol) == 1
    core = regions(cvol)[1]
    @test core.cel == (16, 16, 16)
    @test core.scl == cmpScl32
    @test core.org == cmpOrg0
    @test cmpTilVol(cvol) == prod(vol.cel .* vol.scl)
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
    @test cmpTilVol(cvol) == prod(vol.cel .* vol.scl)
    @test cmpDsjChk(cvol)
    # Snapping is per region, so the core sits on the parent grid
    @test all(isinteger.((_lwrEdg(core) .- _lwrEdg(vol)) .// vol.scl))
end

@testset "GlaCmpVol chained refine" begin
    vol = mkCmpVol()
    cvol = refine(GlaCmpVol(vol), (cmpOrg0, (1//8, 1//8, 1//8)))
    # The core spans -1/8 to 1/8; refine its high octant again
    cvol2 = refine(cvol, ((1//16, 1//16, 1//16), (1//8, 1//8, 1//8)))
    @test nregions(cvol2) == 10
    @test cmpDsjChk(cvol2)
    @test cmpTilVol(cvol2) == prod(vol.cel .* vol.scl)
    @test finest(cvol2) == (1//64, 1//64, 1//64)
    inner = regions(cvol2)[1]
    @test inner.scl == (1//64, 1//64, 1//64)
    @test inner.cel == (8, 8, 8)
    # Three levels of resolution coexist, and the untouched slabs keep their
    # position at the end of the list
    @test Set(reg.scl for reg in regions(cvol2)) ==
        Set([(1//64, 1//64, 1//64), cmpScl32, cmpScl16])
    @test [reg.cel for reg in regions(cvol2)[5:end]] ==
        [reg.cel for reg in regions(cvol)[2:end]]

    # A sub-box centered on the core leaves 1/32 slabs two cells wide, which
    # gives an odd cell count per partition against the 1/16 slabs
    @test_throws ArgumentError refine(cvol,
        (cmpOrg0, (1//16, 1//16, 1//16)))
end

@testset "GlaCmpVol anisotropic factor" begin
    vol = mkCmpVol()
    box = (cmpOrg0, (1//8, 1//8, 1//8))
    cvolZ = refine(GlaCmpVol(vol), box; factor=(1, 1, 2))
    coreZ = regions(cvolZ)[1]
    @test coreZ.scl == (1//16, 1//16, 1//32)
    @test coreZ.cel == (4, 4, 8)
    @test nregions(cvolZ) == 7
    @test cmpTilVol(cvolZ) == prod(vol.cel .* vol.scl)
    @test cmpDsjChk(cvolZ)
    @test finest(cvolZ) == (1//16, 1//16, 1//32)

    cvolA = refine(GlaCmpVol(vol), box; factor=(2, 2, 4))
    coreA = regions(cvolA)[1]
    @test coreA.scl == (1//32, 1//32, 1//64)
    @test coreA.cel == (8, 8, 16)
    @test cmpTilVol(cvolA) == prod(vol.cel .* vol.scl)
    @test cmpDsjChk(cvolA)

    # A factor of 1 everywhere still carves the region
    cvolOne = refine(GlaCmpVol(vol), box; factor=1)
    @test nregions(cvolOne) == 7
    @test all(reg -> reg.scl == cmpScl16, regions(cvolOne))
    @test cmpTilVol(cvolOne) == prod(vol.cel .* vol.scl)
end

@testset "GlaCmpVol refine argument errors" begin
    vol = mkCmpVol()
    box = (cmpOrg0, (1//8, 1//8, 1//8))
    @test_throws ArgumentError refine(GlaCmpVol(vol), box; snap=:inward)
    @test_throws ArgumentError refine(GlaCmpVol(vol), box; factor=0)
    @test_throws ArgumentError refine(GlaCmpVol(vol), box; factor=(2, 0, 2))
    @test_throws ArgumentError refine(GlaCmpVol(vol), box; factor=2.0)
    @test_throws ArgumentError refine(GlaCmpVol(vol), (0.0, 1.0))
end

@testset "GlaCmpVol constructor rejections" begin
    # No regions at all
    @test_throws ArgumentError GlaCmpVol(GlaVol[])
    # Odd cell count
    @test_throws ArgumentError GlaCmpVol(GlaVol((3, 4, 4), cmpScl32, cmpOrg0))
    # Overlapping interiors
    volA = GlaVol((4, 4, 4), cmpScl32, cmpOrg0)
    @test_throws ArgumentError GlaCmpVol([volA, volA])
    # Incommensurate scales: 1/16 against 1/24
    volBad = GlaVol((4, 4, 4), (1//24, 1//24, 1//24), (1//1, 0//1, 0//1))
    @test_throws ArgumentError GlaCmpVol([GlaVol((4, 4, 4), cmpScl16, cmpOrg0),
        volBad])
    # Misaligned grids: neighbour shifted by half a cell
    volOff = GlaVol((4, 4, 4), cmpScl32, (9//64, 0//1, 0//1))
    @test_throws ArgumentError GlaCmpVol([volA, volOff])
    # Aligned and disjoint, but with a gap, so the regions do not tile
    volGap = GlaVol((4, 4, 4), cmpScl32, (1//4, 0//1, 0//1))
    @test_throws ArgumentError GlaCmpVol([volA, volGap])
    # Region whose grid pitch is coarser than its cells
    volPit = GlaVol((4, 4, 4), cmpScl32, cmpOrg0, cmpScl16)
    @test_throws ArgumentError GlaCmpVol(volPit)
    # Partition parity violation: a (2,2,2) coarse region against a (2,2,2)
    # region at half the scale gives one cell per partition on both sides
    volCrs = GlaVol((2, 2, 2), cmpScl16, cmpOrg0)
    volFin = GlaVol((2, 2, 2), cmpScl32, (1//16 + 1//32, 0//1, 0//1))
    @test_throws ArgumentError GlaCmpVol([volCrs, volFin])
end

@testset "GlaCmpVol flat layout" begin
    # Two touching regions of different scale, both 1/8 λ per side
    volCrs = GlaVol((4, 4, 4), cmpScl32, cmpOrg0)
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
    box = (cmpOrg0, (1//8, 1//8, 1//8))
    @test refine(vol, box) == refine(GlaCmpVol(vol), box)
    @test refine(vol, box; factor=(1, 1, 2)) ==
        refine(GlaCmpVol(vol), box; factor=(1, 1, 2))
    # A GlaVol works as a box: only its outer bounds matter
    boxVol = GlaVol((2, 2, 2), cmpScl16, cmpOrg0)
    @test refine(vol, boxVol) == refine(vol, (cmpOrg0, (1//8, 1//8, 1//8)))
    # A coarser box volume with the same bounds gives the same tiling
    boxCrs = GlaVol((2, 2, 2), (1//16, 1//16, 1//16), cmpOrg0)
    @test refine(vol, boxCrs) == refine(vol, boxVol)
end
