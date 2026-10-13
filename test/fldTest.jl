# GlaFld (field over a plain or composite volume) tests

# 8 cells of 1/16 λ per side, spanning -1/4 to 1/4, and the same domain refined
# in its middle octant
mkFldVol() = GlaVol((8, 8, 8), scl16, stdOrg)
mkFldUni() = GlaCmpVol(mkFldVol())
mkFldRef() = refine(mkFldUni(), (stdOrg, (1//8, 1//8, 1//8)))

fldLen(cvol) = sum(3 * prod(reg.cel) for reg in regions(cvol))

# Smooth density, so two meshes of the same domain only agree to quadrature error
fldSmt(pos) = (exp(2im * pi * pos[1]), cos(2 * pi * pos[2]) + 0im, 0.0im)

#= Density that is constant on every 1/16 λ cell, so the midpoint sums of two
meshes that both resolve 1/16 λ agree exactly. =#
function fldStp(pos)
    val = (isodd(floor(Int, pos[1] * 8)) ? 2.0 : 1.0) +
        (isodd(floor(Int, pos[2] * 8)) ? 0.5im : 0.0im)
    return (val, 2 * val, -val)
end

# Cell offset of the block of region idx in the flat layout
fldCelOff(cvol, idx) = sum(prod(regions(cvol)[r].cel) for r in 1:(idx - 1); init=0)

@testset "GlaFld zerofield" begin
    cvol = mkFldRef()
    fld = zerofield(Float64, cvol)
    @test fld isa GlaFld
    @test length(fld) == fldLen(cvol)
    @test all(iszero, fld)
    @test iszero(norm(fld))
    @test fld.cvol == cvol
    # A trivial composite has one block of the whole volume
    @test length(zerofield(Float64, mkFldUni())) == 3 * prod(mkFldVol().cel)
    # The buffer has to match the tiling
    @test_throws ArgumentError GlaFld(zeros(ComplexF64, 7), cvol)
end

@testset "GlaFld over a plain volume" begin
    vol = mkFldVol()
    fld = zerofield(Float64, vol)
    @test fld isa GlaFld
    @test nregions(fld.cvol) == 1
    @test regions(fld.cvol)[1] == vol
    @test length(fld) == 3 * prod(vol.cel)
    @test GlaFld(vol) == fld
    @test fld == zerofield(Float64, GlaCmpVol(vol))
    @test discretize!(zerofield(Float64, vol), fldStp) == discretize!(zerofield(Float64, GlaCmpVol(vol)), fldStp)
    @test shwChk(fld)
end

@testset "GlaFld zero and constructor" begin
    cvol = mkFldRef()
    fld = GlaFld(cvol)
    @test fld isa GlaFld
    @test all(iszero, fld)
    @test fld == zerofield(Float64, cvol)
    fld = discretize!(zerofield(Float64, cvol), pos -> (1.0 + 0im, 0, 0))
    zro = zero(fld)
    @test zro isa GlaFld
    @test zro.cvol == cvol
    @test iszero(norm(zro))
    @test !iszero(norm(fld))
end

@testset "GlaFld eachregion" begin
    cvol = mkFldRef()
    fld = discretize!(zerofield(Float64, cvol), pos -> (pos[1] + 0im, pos[2], pos[3]))
    vws = collect(eachregion(fld))
    @test length(vws) == nregions(cvol)
    for (idx, vw) in enumerate(vws)
        @test vw == regionview(fld, idx)
    end
    # The views write through to the field
    first(eachregion(fld))[1, 1, 1, 1] = 7
    @test fld[1] == 7
end

@testset "GlaFld mesh invariance" begin
    cvolUni = mkFldUni()
    cvolRef = mkFldRef()
    @test nregions(cvolRef) == 7
    @test finest(cvolRef) == stdScl
    # A smooth density only agrees to quadrature error
    smtUni = discretize!(zerofield(Float64, cvolUni), fldSmt)
    smtRef = discretize!(zerofield(Float64, cvolRef), fldSmt)
    @test abs(norm(smtRef) - norm(smtUni)) / norm(smtUni) < 1e-3
    # A density constant on every coarse cell agrees exactly
    stpUni = discretize!(zerofield(Float64, cvolUni), fldStp)
    stpRef = discretize!(zerofield(Float64, cvolRef), fldStp)
    @test norm(stpRef) ≈ norm(stpUni) rtol=1e-12
    # Both norms are the analytic L2 norm of the step density
    stpNrm = sqrt(6 * Float64(prod(mkFldVol().cel .* mkFldVol().scl)) *
        (abs2(1.0) + abs2(1.0 + 0.5im) + abs2(2.0) + abs2(2.0 + 0.5im)) / 4)
    @test norm(stpUni) ≈ stpNrm rtol=1e-12
end

@testset "GlaFld regionview" begin
    cvol = mkFldRef()
    fld = zerofield(Float64, cvol)
    for (idx, reg) in enumerate(regions(cvol))
        @test size(regionview(fld, idx)) == (Int.(reg.cel)..., 3)
    end
    # Writing through a view mutates the field
    idx = 3
    reg = regions(cvol)[idx]
    regVew = regionview(fld, idx)
    celInd = (1, 2, 3)
    dir = 2
    regVew[celInd..., dir] = 7.0 + 0im
    @test count(!iszero, fld) == 1
    # The flat index predicted by the layout contract
    lin = LinearIndices(Tuple(reg.cel))[celInd...]
    flt = 3 * fldCelOff(cvol, idx) + (dir - 1) * prod(reg.cel) + lin
    @test fld[flt] == 7.0 + 0im
    # Setting the flat entry is the same as setting the view entry
    fld[flt] = 9.0 + 0im
    @test regionview(fld, idx)[celInd..., dir] == 9.0 + 0im
    # The cell that lit up sits where coordinates says it does
    crd = collect(coordinates(cvol))
    pos, celVol, crdIdx = crd[fldCelOff(cvol, idx) + lin]
    @test crdIdx == idx
    @test pos == ntuple(dr -> Float64(reg.grd[dr][celInd[dr]]), 3)
    @test celVol ≈ Float64(prod(reg.scl))
end

@testset "GlaFld physical convention" begin
    cvol = mkFldRef()
    fld = discretize!(zerofield(Float64, cvol), pos -> (1, 0, 0))
    # Dividing a region block by the square root of its cell volume gives the
    # density back, whatever the scale of the region
    for (idx, reg) in enumerate(regions(cvol))
        dns = regionview(fld, idx) ./ sqrt(Float64(prod(reg.scl)))
        @test all(dns[:, :, :, 1] .≈ 1)
        @test all(iszero, dns[:, :, :, 2])
        @test all(iszero, dns[:, :, :, 3])
    end
    # Two regions of different scale carry different stored values
    sclSet = Set(regions(cvol)[idx].scl for idx in 1:nregions(cvol))
    @test length(sclSet) == 2
    @test regionview(fld, 1)[1, 1, 1, 1] != regionview(fld, 2)[1, 1, 1, 1]
    # The norm of a unit density is the square root of the domain volume
    @test norm(fld)^2 ≈ Float64(prod(mkFldVol().cel .* mkFldVol().scl)) rtol=1e-12
end

@testset "GlaFld regrid" begin
    cvolRef = mkFldRef()
    volFin = GlaVol((16, 16, 16), stdScl, stdOrg)
    stpRef = discretize!(zerofield(Float64, cvolRef), fldStp)
    rsm = regrid(stpRef)
    @test rsm isa Array{ComplexF64,4}
    @test size(rsm) == (16, 16, 16, 3)
    # Sampling on the uniform fine mesh directly, read back as a density
    stpFin = discretize!(zerofield(Float64, GlaCmpVol(volFin)), fldStp)
    dnsFin = regionview(stpFin, 1) ./ sqrt(Float64(prod(volFin.scl)))
    @test maximum(abs, rsm .- dnsFin) < 1e-12
    # The explicit scale is the default
    @test regrid(stpRef, stdScl) == rsm
    # A trivial composite regrids to itself
    stpUni = discretize!(zerofield(Float64, mkFldUni()), fldStp)
    dnsUni = regionview(stpUni, 1) ./ sqrt(Float64(prod(scl16)))
    @test maximum(abs, regrid(stpUni) .- dnsUni) < 1e-12
    # Finer than every region is allowed and just repeats values
    rsmFin = regrid(stpRef, (1//64, 1//64, 1//64))
    @test size(rsmFin) == (32, 32, 32, 3)
    @test rsmFin[1, 1, 1, 1] == rsm[1, 1, 1, 1]
    # Coarser than a region, and incommensurate with one
    @test_throws ArgumentError regrid(stpRef, scl16)
    @test_throws ArgumentError regrid(stpRef, (1//48, 1//48, 1//48))
    @test_throws ArgumentError regrid(stpRef, (1//32, 1//32, 1//24))
end

@testset "GlaFld linear algebra" begin
    cvol = mkFldRef()
    fldX = discretize!(zerofield(Float64, cvol), fldSmt)
    fldY = discretize!(zerofield(Float64, cvol), fldStp)
    @test dot(fldX, fldX) ≈ norm(fldX)^2
    @test dot(fldX, fldY) ≈ dot(fldX.dat, fldY.dat)
    @test norm(fldX, 1) ≈ norm(fldX.dat, 1)
    @test norm(fldX, Inf) ≈ norm(fldX.dat, Inf)
    # axpy! and axpby! match their broadcast equivalents
    axp = axpy!(2.0 + 1im, fldX, copy(fldY))
    @test axp isa GlaFld
    @test norm(axp .- ((2.0 + 1im) .* fldX .+ fldY)) < 1e-12 * norm(axp)
    axb = axpby!(2.0, fldX, 3.0, copy(fldY))
    @test norm(axb .- (2 .* fldX .+ 3 .* fldY)) < 1e-12 * norm(axb)
end

@testset "GlaFld broadcasting" begin
    cvol = mkFldRef()
    fldX = discretize!(zerofield(Float64, cvol), fldSmt)
    fldY = 2 .* fldX
    @test fldY isa GlaFld
    @test fldY.cvol == cvol
    @test norm(fldY) ≈ 2 * norm(fldX)
    @test norm(fldX .+ fldX .- fldY) < 1e-12 * norm(fldY)
    # In place forms
    fldZ = zerofield(Float64, cvol)
    fldZ .= 2 .* fldX
    @test fldZ isa GlaFld
    @test norm(fldZ .- fldY) < 1e-12 * norm(fldY)
    fldZ .= 0
    @test iszero(norm(fldZ))
    fldZ .+= fldX
    @test norm(fldZ) ≈ norm(fldX)
    @test norm(abs.(fldX)) ≈ norm(fldX)
    # A comparison drops the wrapper, since the buffer cannot hold Bools
    @test all(fldX .== fldX)
    @test (fldX .== fldX) isa AbstractVector{Bool}
end

@testset "GlaFld mismatched volumes" begin
    cvolA = GlaCmpVol(GlaVol((8, 8, 8), scl16, stdOrg))
    # Same number of degrees of freedom, different geometry
    cvolB = GlaCmpVol(GlaVol((8, 8, 8), stdScl, stdOrg))
    fldA = discretize!(zerofield(Float64, cvolA), fldSmt)
    fldB = discretize!(zerofield(Float64, cvolB), fldSmt)
    @test length(fldA) == length(fldB)
    @test_throws ArgumentError dot(fldA, fldB)
    @test_throws ArgumentError axpy!(1.0, fldA, fldB)
    @test_throws ArgumentError axpby!(1.0, fldA, 1.0, fldB)
    @test_throws ArgumentError copyto!(fldA, fldB)
    @test_throws ArgumentError fldA .+ fldB
    @test_throws ArgumentError fldA .= fldB
    # The refined tiling of the same domain is a different volume again
    fldRef = zerofield(Float64, mkFldRef())
    @test_throws ArgumentError dot(fldA, fldRef)
end
