# Composite operator tests
# The reference for a composite operator is the same operator on a uniform mesh
# at the finest cell size of the tiling. A composite target row is the mean of
# the fine rows it covers, a composite source column injects unit density into
# the fine cells it covers, and the √ΔV basis puts sqrt(ΔVᵢ/ΔVⱼ) on block (i, j).
import GilaElectromagnetics: _lwrEdg
import GilaElectromagnetics: _nrmWgt

# The smallest tiling this file uses: one region of 2×2×2 cells of 1/16 λ
const smlVol = GlaVol((2, 2, 2), scl16, stdOrg)
const smlCvl = GlaCmpVol(smlVol)

# Geometries closer than λ/3 warn once on construction; pin it instead of printing
prxBld(bld) = @test_logs (:warn, r"third of a wavelength") bld()

# Linear indices in refVol of the cells under each cell of cvol, in layout order
function cmpChdIdx(cvol::GlaCmpVol, refVol::GlaVol)
    lin = LinearIndices(Tuple(refVol.cel))
    refLwr = _lwrEdg(refVol)
    chd = Vector{Vector{Int}}()
    for reg in regions(cvol)
        rat = ntuple(dir -> Int(reg.scl[dir] // refVol.scl[dir]), 3)
        bas = ntuple(dir ->
            Int((_lwrEdg(reg)[dir] - refLwr[dir]) // refVol.scl[dir]), 3)
        for celInd in CartesianIndices(Tuple(reg.cel))
            push!(chd, vec([lin[ntuple(dir ->
                bas[dir] + (celInd[dir] - 1) * rat[dir] + off[dir], 3)...]
                for off in CartesianIndices(rat)]))
        end
    end
    return chd
end

# Cell index, vector component, and cell volume of every degree of freedom
function cmpDofInf(cvol::GlaCmpVol)
    celIdx, dirIdx, celVol = Int[], Int[], Float64[]
    celOff = 0
    for reg in regions(cvol)
        celNum = prod(reg.cel)
        for dir in 1:3, cel in 1:celNum
            push!(celIdx, celOff + cel)
            push!(dirIdx, dir)
            push!(celVol, Float64(prod(reg.scl)))
        end
        celOff += celNum
    end
    return celIdx, dirIdx, celVol
end

# Dense map onto the fine dofs of refVol: each composite dof hits the fine cells under its cell with val(cell volume, fine cell count)
function cmpMap(cvol::GlaCmpVol, refVol::GlaVol, val)
    chd = cmpChdIdx(cvol, refVol)
    cel, dir, vol = cmpDofInf(cvol)
    num = prod(refVol.cel)
    mat = zeros(3 * num, length(cel))
    for dof in eachindex(cel), fin in chd[cel[dof]]
        mat[(dir[dof] - 1) * num + fin, dof] = val(vol[dof], length(chd[cel[dof]]))
    end
    return mat
end

#= The composite operator predicted by the uniform reference refOpr: a source
dof injects unit density, a target dof takes the mean, and the √ΔV basis
divides by the source and multiplies by the target cell volume. =#
function cmpRef(trgCvl::GlaCmpVol, trgRef::GlaVol, srcCvl::GlaCmpVol, srcRef::GlaVol, refOpr)
    inj = cmpMap(srcCvl, srcRef, (vol, _) -> 1 / sqrt(vol))
    agr = transpose(cmpMap(trgCvl, trgRef, (vol, num) -> sqrt(vol) / num))
    return LinearMap(agr) * LinearMap(refOpr) * LinearMap(inj)
end

cmpRef(cvol::GlaCmpVol, refVol::GlaVol, refOpr) = cmpRef(cvol, refVol, cvol, refVol, refOpr)

# Self, external, masked union, and fine mesh blocks of a composite operator
cmpBlkCnt(opr) = (count(blk -> blk isa GlaOprVac && isselfoperator(blk), opr.blkMat),
    count(blk -> blk isa GlaOprVac && isexternaloperator(blk), opr.blkMat),
    count(blk -> blk isa GlaOprVac && isoverlappingoperator(blk), opr.blkMat),
    count(blk -> blk isa GlaSnd, opr.blkMat))

#= The money geometry: a (4,4,4) volume of 1/16 λ cells with its low x half
refined, so the tiling is one fine region face to face with one coarse one. =#
const mnyVol = GlaVol((4, 4, 4), scl16, stdOrg)
const mnyCvl = refine(GlaCmpVol(mnyVol), ((-1//16, 0//1, 0//1), (1//8, 1//4, 1//4)))
const mnyRef = GlaVol((8, 8, 8), stdScl, stdOrg)
const mnyOpr = GlaCmpOprVac{Float64}(mnyCvl)
const mnyAgr = cmpRef(mnyCvl, mnyRef, GlaOprVac{Float64}(mnyRef))

@testset "Composite operator routes" begin
    @test nregions(mnyCvl) == 2
    @test regions(mnyCvl)[1].cel == (4, 8, 8)
    @test regions(mnyCvl)[1].scl == stdScl
    @test regions(mnyCvl)[2].cel == (2, 4, 4)
    @test regions(mnyCvl)[2].scl == scl16
    @test isselfoperator(mnyOpr)
    @test !isexternaloperator(mnyOpr)
    # The two cross-scale blocks touch, so they take the fine mesh route
    @test cmpBlkCnt(mnyOpr) == (2, 0, 0, 2)
end

@testset "Composite operator against a uniform reference" begin
    @test prbErr(mnyOpr, mnyAgr) < 1e-12
    # A composite target row is a mean, not a sum, so the wrong convention shows
    @test prbErr(mnyOpr, 8 * mnyAgr) > 1e-3
    # A self operator is complex symmetric
    @test symErr(mnyOpr) < 1e-13
end

@testset "Composite operator on a field" begin
    fld = discretize!(zerofield(Float64, mnyCvl), tstDns)
    out = mnyOpr * fld
    @test out isa GlaFld
    @test out.cvol === mnyCvl
    @test length(out) == 864
    @test all(isfinite, out.dat)
    # The flat path and the field path are the same computation
    @test mnyOpr * collect(fld.dat) == out.dat
    # A field on another tiling does not fit
    @test_throws ArgumentError mnyOpr * zerofield(Float64, GlaCmpVol(mnyRef))
    @test_throws ArgumentError mnyOpr * zeros(ComplexF64, 863)
    # Five argument mul! comes from the generic fallback
    outMul = zerofield(Float64, mnyCvl)
    mul!(outMul, mnyOpr, fld, 2.0, 0.0)
    @test norm(outMul.dat - 2 .* out.dat) < 1e-12 * norm(out.dat)
    # Densification through getindex
    @test mnyOpr[1:4, 1:4] == (mnyOpr * Matrix{ComplexF64}(I, 864, 4))[1:4, :]
end

#= The two Hermitian parts are read entry by entry off the composite matrix,
which the complex symmetry of the self operator makes exact. Both are built from
the same block matrix, so a matvec costs one application of G₀. =#
@testset "Composite operator Hermitian parts" begin
    asyOpr, symOpr = asym(mnyOpr), glaSym(mnyOpr)
    @test asyOpr isa AsyGlaCmpOprVac
    @test symOpr isa SymGlaCmpOprVac
    @test isselfoperator(asyOpr)
    @test !isexternaloperator(asyOpr)
    @test !isoverlappingoperator(asyOpr)
    @test prbErr(adapt(CPU(), asyOpr), asyOpr) < 1e-12
    @test shwChk(asyOpr) && shwChk(symOpr)

    inp = randn(ComplexF64, 864, 3)
    out = mnyOpr * inp
    asyOut, symOut = asyOpr * inp, symOpr * inp
    @test frbErr(asyOut, asyPrb(mnyOpr, inp)) < 1e-12
    @test frbErr(symOut, symPrb(mnyOpr, inp)) < 1e-12
    # Both parts are Hermitian, and together they rebuild the operator
    @test adjErr(asyOpr, asyOpr) < 1e-12
    @test adjErr(symOpr, symOpr) < 1e-12
    @test frbErr(symOut + im .* asyOut, out) < 1e-12
    # The operator the part was taken from is untouched
    @test mnyOpr * inp == out

    # Hermitian, so adjoint! hands the operator back
    @test GilaElectromagnetics.adjoint!(asyOpr) === asyOpr

    fld = discretize!(zerofield(Float64, mnyCvl), tstDns)
    out = asyOpr * fld
    @test out isa GlaFld
    @test out.cvol === mnyCvl
    @test asyOpr * collect(fld.dat) == out.dat
    @test_throws ArgumentError asyOpr * zerofield(Float64, GlaCmpVol(mnyRef))
    @test_throws ArgumentError asyOpr * zeros(ComplexF64, 863)
    @test asyOpr[1:4, 1:4] == (asyOpr * Matrix{ComplexF64}(I, 864, 4))[1:4, :]

    # A part only makes sense for a self operator, and not in adjoint mode
    extOpr = GlaCmpOprVac{Float64}(smlCvl,
        GlaCmpVol(GlaVol((2, 2, 2), scl16, (1//1, 0//1, 0//1))))
    @test_throws ArgumentError asym(extOpr)
    @test_throws ArgumentError glaSym(extOpr)
    @test_throws ArgumentError asym(adjoint(mnyOpr))

    # The volume constructor builds the operator it needs
    @test AsyGlaCmpOprVac{Float64}(smlCvl) isa AsyGlaCmpOprVac
    @test SymGlaCmpOprVac{Float64}(smlCvl) isa SymGlaCmpOprVac
end

#= A coarse region on each side of a fine one, so a sandwich block appears in
both orientations and the two coarse regions see each other across a gap. =#
const triCvl = refine(GlaCmpVol(GlaVol((6, 4, 4), scl16, stdOrg)),
    (stdOrg, (1//8, 1//4, 1//4)))
const triRef = GlaVol((12, 8, 8), stdScl, stdOrg)

@testset "Composite operator three regions" begin
    @test nregions(triCvl) == 3
    @test regions(triCvl)[1].cel == (4, 8, 8)
    @test all(reg -> reg.cel == (2, 4, 4), regions(triCvl)[2:3])
    triOpr = GlaCmpOprVac{Float64}(triCvl)
    @test size(triOpr) == (960, 960)
    # Three self blocks, the two coarse regions apart in x, four sandwiches
    @test cmpBlkCnt(triOpr) == (3, 2, 0, 4)
    @test prbErr(triOpr, cmpRef(triCvl, triRef, GlaOprVac{Float64}(triRef))) < 1e-13
    @test symErr(triOpr) < 1e-13
    # Sandwich blocks in both orientations, plus a same-scale external pair
    inp = randn(ComplexF64, 960, 3)
    @test frbErr(asym(triOpr) * inp, asyPrb(triOpr, inp)) < 1e-12
    @test frbErr(glaSym(triOpr) * inp, symPrb(triOpr, inp)) < 1e-12
end

@testset "Composite operator between two bodies" begin
    srcCvl = GlaCmpVol(GlaVol((2, 2, 2), scl16, (1//2, 0//1, 0//1)))
    opr = prxBld(() -> GlaCmpOprVac{Float64}(mnyCvl, srcCvl))
    @test isexternaloperator(opr)
    @test !isselfoperator(opr)
    @test size(opr) == (864, 24)
    # Apart and at the same scale, so both blocks take the ordinary external route
    @test cmpBlkCnt(opr) == (0, 2, 0, 0)
    srcRef = GlaVol((4, 4, 4), stdScl, (1//2, 0//1, 0//1))
    refOpr = prxBld(() -> GlaOprVac{Float64}(mnyRef, srcRef))
    @test prbErr(opr, cmpRef(mnyCvl, mnyRef, srcCvl, srcRef, refOpr)) < 1e-13
    # The two argument constructor spelling
    @test prxBld(() -> GlaOprVac{Float64}(mnyCvl, srcCvl)) isa GlaCmpOprVac
end

@testset "Composite operator gapped tiling" begin
    # One tiling holding two separated regions of different cell size
    gapCvl = GlaCmpVol([smlVol, GlaVol((4, 4, 4), stdScl, (1//2, 0//1, 0//1))])
    opr = GlaCmpOprVac{Float64}(gapCvl)
    @test isselfoperator(opr)
    @test isselfoperator(opr.blkMat[1, 1])
    @test isselfoperator(opr.blkMat[2, 2])
    @test isexternaloperator(opr.blkMat[1, 2])
    @test isexternaloperator(opr.blkMat[2, 1])
    # Apart, so every block is a plain operator rather than a fine mesh one
    @test cmpBlkCnt(opr) == (2, 2, 0, 0)
    #= The dense form assembles from the blocks each region pair would build on
    its own, scaled by the √ΔV weight of the pair. =#
    regs = regions(gapCvl)
    off = cumsum([0; [3 * prod(reg.cel) for reg in regs]])
    ref = zeros(ComplexF64, off[end], off[end])
    for trgIdx in 1:2, srcIdx in 1:2
        blk = trgIdx == srcIdx ? GlaOprVac{Float64}(regs[trgIdx]) :
            GlaOprVac{Float64}(regs[trgIdx], regs[srcIdx]; prxWrn=false)
        ref[(off[trgIdx] + 1):off[trgIdx + 1], (off[srcIdx] + 1):off[srcIdx + 1]] .=
            _nrmWgt(regs[trgIdx], regs[srcIdx]) .* dnsMat(blk)
    end
    @test frbErr(dnsMat(opr), ref) < 1e-14
end

@testset "Composite operator cross-scale near pair" begin
    # A coarse volume and a fine one, one coarse cell apart in x
    trgCvl = GlaCmpVol(GlaVol((4, 4, 4), stdScl, (3//16, 0//1, 0//1)))
    opr = prxBld(() -> GlaCmpOprVac{Float64}(trgCvl, smlCvl))
    # A gap of one coarse cell keeps the pair off the contact path
    @test opr.blkMat[1, 1] isa GlaOprVac
    srcRef = GlaVol((4, 4, 4), stdScl, stdOrg)
    refOpr = prxBld(() -> GlaOprVac{Float64}(regions(trgCvl)[1], srcRef))
    #= The cross-scale expansion is exact in the pulse basis, so the composite
    operator is the aggregated fine reference to the Float64 floor at any gap. =#
    @test prbErr(opr, cmpRef(trgCvl, regions(trgCvl)[1], smlCvl, srcRef, refOpr)) < 1e-12
    farCvl = GlaCmpVol(GlaVol((4, 4, 4), stdScl, (3//8, 0//1, 0//1)))
    farOpr = prxBld(() -> GlaCmpOprVac{Float64}(farCvl, smlCvl))
    farRef = prxBld(() -> GlaOprVac{Float64}(regions(farCvl)[1], srcRef))
    @test prbErr(farOpr, cmpRef(farCvl, regions(farCvl)[1], smlCvl, srcRef, farRef)) < 1e-12
    # Moving the fine volume onto the coarse one switches the block over
    tchCvl = GlaCmpVol(GlaVol((4, 4, 4), stdScl, (1//8, 0//1, 0//1)))
    tchOpr = GlaCmpOprVac{Float64}(tchCvl, smlCvl)
    @test tchOpr.blkMat[1, 1] isa GlaSnd
    @test all(isfinite, tchOpr * randn(ComplexF64, size(tchOpr, 2)))
end

@testset "Composite operator contact block" begin
    #= A small volume face to face with a taller one. The external construction
    corrects for the cells in contact, and the answer is the self operator of the
    union of the two, masked down. =#
    srcCvl = GlaCmpVol(GlaVol((2, 4, 4), scl16, (1//8, 0//1, 0//1)))
    opr = GlaCmpOprVac{Float64}(smlCvl, srcCvl)
    @test cmpBlkCnt(opr) == (0, 1, 0, 0)
    mat = dnsMat(opr)
    @test size(mat) == (24, 96)
    @test all(isfinite, mat)
    # The same entries read off the densified self operator of the union
    uniMat = dnsMat(GlaOprVac{Float64}(GlaVol((4, 4, 4), scl16, (1//16, 0//1, 0//1))))
    lin = LinearIndices((4, 4, 4))
    rowCel = vec([lin[xItr, yItr, zItr] for xItr in 1:2, yItr in 2:3, zItr in 2:3])
    colCel = vec([lin[xItr, yItr, zItr] for xItr in 3:4, yItr in 1:4, zItr in 1:4])
    rowDof = vcat([(dir - 1) * 64 .+ rowCel for dir in 1:3]...)
    colDof = vcat([(dir - 1) * 64 .+ colCel for dir in 1:3]...)
    @test frbErr(mat, uniMat[rowDof, colDof]) < 1e-12
end

@testset "Composite operator overlapping bodies" begin
    cvolB = GlaCmpVol(GlaVol((2, 2, 2), scl16, (1//16, 0//1, 0//1)))
    @test_throws ArgumentError GlaCmpOprVac{Float64}(smlCvl, cvolB)
    # Two equal tilings are the same body, so the self operator is meant
    @test isselfoperator(GlaCmpOprVac{Float64}(smlCvl, GlaCmpVol(smlVol)))
end

@testset "Composite operator one region" begin
    opr = VacuumGreenOperator{Float64}(smlCvl)
    @test opr isa GlaCmpOprVac
    @test isselfoperator(opr)
    @test cmpBlkCnt(opr) == (1, 0, 0, 0)
    # One region means one cell volume, so the normalization is the identity
    @test prbErr(opr, GlaOprVac{Float64}(smlVol)) == 0
    # One region reproduces the plain anti-Hermitian operator
    @test prbErr(asym(opr), AsyGlaOprVac{Float64}(smlVol)) < 1e-12
    @test slv(opr) isa GlaSlv
    @test prbErr(adapt(CPU(), opr), opr) < 1e-12
    @test shwChk(opr)
end

@testset "Plain against composite on a field" begin
    #= A plain operator over one region is the one region composite, so the two
    have to agree on a field, including the sqrt(ΔV_trg / ΔV_src) of the basis. =#
    trgVol = GlaVol((4, 4, 4), stdScl, (3//8, 0//1, 0//1))
    fld = discretize!(zerofield(Float64, smlVol), tstDns)
    cmpOut = prxBld(() -> GlaCmpOprVac{Float64}(GlaCmpVol(trgVol), smlCvl)) * fld
    plnOut = prxBld(() -> GlaOprVac{Float64}(trgVol, smlVol)) * fld
    @test plnOut isa GlaFld
    @test regions(plnOut.cvol)[1] == trgVol
    @test norm(plnOut.dat - cmpOut.dat) < 1e-13 * norm(cmpOut.dat)
end
