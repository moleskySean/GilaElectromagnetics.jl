# Composite operator tests
# The reference for a composite operator is the same operator on a uniform mesh
# at the finest cell size of the tiling. A composite target row is the mean of
# the fine rows it covers, a composite source column injects unit density into
# the fine cells it covers, and the √ΔV basis puts sqrt(ΔVᵢ/ΔVⱼ) on block (i, j).
import GilaElectromagnetics.GilaVolumes: _lwrEdg
import GilaElectromagnetics.GilaOperators: _nrmWgt

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

# The composite matrix predicted by a uniform reference matrix
function cmpAgrRef(trgCvl::GlaCmpVol, trgRef::GlaVol, srcCvl::GlaCmpVol,
    srcRef::GlaVol, refMat::AbstractMatrix{ComplexF64})
    trgChd, srcChd = cmpChdIdx(trgCvl, trgRef), cmpChdIdx(srcCvl, srcRef)
    trgCel, trgDir, trgVol = cmpDofInf(trgCvl)
    srcCel, srcDir, srcVol = cmpDofInf(srcCvl)
    trgNum, srcNum = prod(trgRef.cel), prod(srcRef.cel)
    injMat = zeros(ComplexF64, 3 * trgNum, length(srcCel))
    for dofItr in eachindex(srcCel)
        off = (srcDir[dofItr] - 1) * srcNum
        for celItr in srcChd[srcCel[dofItr]]
            injMat[:, dofItr] .+= view(refMat, :, off + celItr)
        end
    end
    agrMat = zeros(ComplexF64, length(trgCel), length(srcCel))
    for dofItr in eachindex(trgCel)
        off = (trgDir[dofItr] - 1) * trgNum
        chd = trgChd[trgCel[dofItr]]
        for celItr in chd
            agrMat[dofItr, :] .+= view(injMat, off + celItr, :)
        end
        agrMat[dofItr, :] ./= length(chd)
    end
    for dofTrg in eachindex(trgCel), dofSrc in eachindex(srcCel)
        agrMat[dofTrg, dofSrc] *= sqrt(trgVol[dofTrg] / srcVol[dofSrc])
    end
    return agrMat
end

cmpAgrRef(cvol::GlaCmpVol, refVol::GlaVol, refMat::AbstractMatrix{ComplexF64}) =
    cmpAgrRef(cvol, refVol, cvol, refVol, refMat)

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
const mnyMat = dnsMat(mnyOpr)
const mnyAgr = cmpAgrRef(mnyCvl, mnyRef, dnsMat(GlaOprVac{Float64}(mnyRef)))

@testset "Composite operator traits" begin
    @test nregions(mnyCvl) == 2
    @test regions(mnyCvl)[1].cel == (4, 8, 8)
    @test regions(mnyCvl)[1].scl == stdScl
    @test regions(mnyCvl)[2].cel == (2, 4, 4)
    @test regions(mnyCvl)[2].scl == scl16
    @test size(mnyOpr) == (864, 864)
    @test size(mnyOpr, 1) == 864
    @test eltype(mnyOpr) == ComplexF64
    @test isselfoperator(mnyOpr)
    @test !isexternaloperator(mnyOpr)
    @test !isadjoint(mnyOpr)
    @test get_backend(mnyOpr) isa CPU
    @test glaSze(mnyOpr) == glaSze.(mnyOpr.blkMat)
    @test glaSze(mnyOpr, 2)[1, 1] == (4, 8, 8, 3)
    # The two cross-scale blocks touch, so they take the fine mesh route
    @test cmpBlkCnt(mnyOpr) == (2, 0, 0, 2)
    @test CompositeVacuumGreenOperator === GlaCmpOprVac
end

@testset "Composite operator against a uniform reference" begin
    @test all(isfinite, mnyMat)
    @test frbErr(mnyMat, mnyAgr) < 1e-12
    # A composite target row is a mean, not a sum, so the wrong convention shows
    @test frbErr(mnyMat, 8 .* mnyAgr) > 1e-3
    # A self operator is complex symmetric
    @test frbErr(mnyMat, transpose(mnyMat)) < 1e-13
end

@testset "Composite operator adjoint" begin
    adjOpr = adjoint(mnyOpr)
    @test adjOpr isa GlaCmpOprVac
    @test isadjoint(adjOpr)
    @test !isadjoint(mnyOpr)
    @test size(adjOpr) == reverse(size(mnyOpr))
    @test frbErr(dnsMat(adjOpr), mnyMat') < 1e-13
    # The original operator is untouched
    @test dnsMat(mnyOpr) == mnyMat
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
    @test norm(out.dat - mnyMat * fld.dat) < 1e-12 * norm(out.dat)
    # A field on another tiling does not fit
    @test_throws ArgumentError mnyOpr * zerofield(Float64, GlaCmpVol(mnyRef))
    @test_throws ArgumentError mnyOpr * zeros(ComplexF64, 863)
    # Five argument mul! comes from the generic fallback
    outMul = zerofield(Float64, mnyCvl)
    mul!(outMul, mnyOpr, fld, 2.0, 0.0)
    @test norm(outMul.dat - 2 .* out.dat) < 1e-12 * norm(out.dat)
    # Densification through getindex
    @test frbErr(mnyOpr[1:4, 1:4], mnyMat[1:4, 1:4]) < 1e-12
end

#= The two Hermitian parts are read entry by entry off the composite matrix,
which the complex symmetry of the self operator makes exact. Both are built from
the same block matrix, so a matvec costs one application of G₀. =#
@testset "Composite operator Hermitian parts" begin
    asyOpr, symOpr = asym(mnyOpr), glaSym(mnyOpr)
    @test asyOpr isa AsyGlaCmpOprVac
    @test symOpr isa SymGlaCmpOprVac
    @test AsymCompositeVacuumGreenOperator === AsyGlaCmpOprVac
    @test SymCompositeVacuumGreenOperator === SymGlaCmpOprVac
    @test size(asyOpr) == (864, 864)
    @test size(asyOpr, 2) == 864
    @test eltype(asyOpr) == ComplexF64
    @test glaSze(asyOpr) == glaSze(mnyOpr)
    @test glaSze(asyOpr, 2)[1, 1] == (4, 8, 8, 3)
    @test isselfoperator(asyOpr)
    @test !isexternaloperator(asyOpr)
    @test !isadjoint(asyOpr)
    @test !isoverlappingoperator(asyOpr)
    @test get_backend(asyOpr) isa CPU
    @test frbErr(dnsMat(adapt(CPU(), asyOpr)), dnsMat(asyOpr)) < 1e-12
    @test occursin("composite Asym(G₀)", sprint(show, asyOpr))
    @test occursin("composite Sym(G₀)", sprint(show, symOpr))
    @test !occursin("\n", sprint(show, asyOpr))
    @test sprint(show, MIME"text/plain"(), asyOpr) != sprint(show, asyOpr)

    asyDns, symDns = dnsMat(asyOpr), dnsMat(symOpr)
    @test all(isfinite, asyDns)
    @test frbErr(asyDns, asymMat(mnyMat)) < 1e-12
    @test frbErr(symDns, symMat(mnyMat)) < 1e-12
    # Both parts are Hermitian, and together they rebuild the operator
    @test frbErr(asyDns, asyDns') < 1e-12
    @test frbErr(symDns, symDns') < 1e-12
    @test frbErr(symDns + im .* asyDns, mnyMat) < 1e-12
    # The operator the part was taken from is untouched
    @test dnsMat(mnyOpr) == mnyMat

    # Hermitian, so adjoint! hands the operator back
    @test GilaElectromagnetics.adjoint!(asyOpr) === asyOpr
    @test frbErr(dnsMat(adjoint(asyOpr)), asyDns) < 1e-12

    fld = discretize!(zerofield(Float64, mnyCvl), tstDns)
    out = asyOpr * fld
    @test out isa GlaFld
    @test out.cvol === mnyCvl
    @test norm(out.dat - asyDns * fld.dat) < 1e-12 * norm(out.dat)
    @test asyOpr * collect(fld.dat) == out.dat
    @test_throws ArgumentError asyOpr * zerofield(Float64, GlaCmpVol(mnyRef))
    @test_throws ArgumentError asyOpr * zeros(ComplexF64, 863)
    @test frbErr(asyOpr[1:4, 1:4], asyDns[1:4, 1:4]) < 1e-12

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
    triMat = dnsMat(triOpr)
    @test all(isfinite, triMat)
    triAgr = cmpAgrRef(triCvl, triRef, dnsMat(GlaOprVac{Float64}(triRef)))
    @test frbErr(triMat, triAgr) < 1e-13
    @test frbErr(triMat, transpose(triMat)) < 1e-13
    @test frbErr(dnsMat(adjoint(triOpr)), triMat') < 1e-13
    # Sandwich blocks in both orientations, plus a same-scale external pair
    @test frbErr(dnsMat(asym(triOpr)), asymMat(triMat)) < 1e-12
    @test frbErr(dnsMat(glaSym(triOpr)), symMat(triMat)) < 1e-12
end

@testset "Composite operator between two bodies" begin
    srcCvl = GlaCmpVol(GlaVol((2, 2, 2), scl16, (1//2, 0//1, 0//1)))
    opr = prxBld(() -> GlaCmpOprVac{Float64}(mnyCvl, srcCvl))
    @test isexternaloperator(opr)
    @test !isselfoperator(opr)
    @test size(opr) == (864, 24)
    # Apart and at the same scale, so both blocks take the ordinary external route
    @test cmpBlkCnt(opr) == (0, 2, 0, 0)
    mat = dnsMat(opr)
    @test all(isfinite, mat)
    srcRef = GlaVol((4, 4, 4), stdScl, (1//2, 0//1, 0//1))
    refMat = dnsMat(prxBld(() -> GlaOprVac{Float64}(mnyRef, srcRef)))
    @test frbErr(mat, cmpAgrRef(mnyCvl, mnyRef, srcCvl, srcRef, refMat)) < 1e-13
    @test frbErr(dnsMat(adjoint(opr)), mat') < 1e-13
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
    mat = dnsMat(opr)
    @test all(isfinite, mat)
    srcRef = GlaVol((4, 4, 4), stdScl, stdOrg)
    refMat = dnsMat(prxBld(() -> GlaOprVac{Float64}(regions(trgCvl)[1], srcRef)))
    #= The cross-scale expansion is exact in the pulse basis, so the composite
    matrix is the aggregated fine reference to the Float64 floor at any gap. =#
    @test frbErr(mat, cmpAgrRef(trgCvl, regions(trgCvl)[1], smlCvl, srcRef,
        refMat)) < 1e-12
    farCvl = GlaCmpVol(GlaVol((4, 4, 4), stdScl, (3//8, 0//1, 0//1)))
    farMat = dnsMat(prxBld(() -> GlaCmpOprVac{Float64}(farCvl, smlCvl)))
    farRef = dnsMat(prxBld(() -> GlaOprVac{Float64}(regions(farCvl)[1], srcRef)))
    @test frbErr(farMat, cmpAgrRef(farCvl, regions(farCvl)[1], smlCvl, srcRef,
        farRef)) < 1e-12
    # Moving the fine volume onto the coarse one switches the block over
    tchCvl = GlaCmpVol(GlaVol((4, 4, 4), stdScl, (1//8, 0//1, 0//1)))
    tchOpr = GlaCmpOprVac{Float64}(tchCvl, smlCvl)
    @test tchOpr.blkMat[1, 1] isa GlaSnd
    @test all(isfinite, dnsMat(tchOpr))
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
    @test frbErr(dnsMat(adjoint(opr)), mat') < 1e-13
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
    @test size(opr) == (24, 24)
    @test isselfoperator(opr)
    @test cmpBlkCnt(opr) == (1, 0, 0, 0)
    # One region means one cell volume, so the normalization is the identity
    @test dnsMat(opr) == dnsMat(GlaOprVac{Float64}(smlVol))
    # One region reproduces the plain anti-Hermitian operator
    @test frbErr(dnsMat(asym(opr)), dnsMat(AsyGlaOprVac{Float64}(smlVol))) < 1e-12
    @test slv(opr) isa GlaSlv
    @test get_backend(opr) isa CPU
    @test frbErr(dnsMat(adapt(CPU(), opr)), dnsMat(opr)) < 1e-12
    @test occursin("composite G₀", sprint(show, opr))
    @test !occursin("\n", sprint(show, opr))
    @test sprint(show, MIME"text/plain"(), opr) != sprint(show, opr)
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

@testset "Composite operator GPU" begin
    @testset "$(nameof(typeof(dev))) $T" for dev in gpuBackends, T in gpuPrcs(dev)
        oprGpu = GlaCmpOprVac{T}(mnyCvl; dev)
        @test get_backend(oprGpu) isa typeof(dev)
        fldGpu = discretize!(zerofield(T, mnyCvl; dev), tstDns)
        outGpu = oprGpu * fldGpu
        @test outGpu isa GlaFld
        @test get_backend(outGpu) isa typeof(dev)
        fldCpu = discretize!(zerofield(T, mnyCvl), tstDns)
        outCpu = GlaCmpOprVac{T}(mnyCvl) * fldCpu
        @test norm(Array(outGpu.dat) - outCpu.dat) < gpuTol(T) * norm(outCpu.dat)

        asyGpu = asym(oprGpu)
        @test get_backend(asyGpu) isa typeof(dev)
        asyOutGpu = asyGpu * fldGpu
        asyOutCpu = asym(GlaCmpOprVac{T}(mnyCvl)) * fldCpu
        @test norm(Array(asyOutGpu.dat) - asyOutCpu.dat) < gpuTol(T) * norm(asyOutCpu.dat)

        # adapt moves a composite operator to the device and back, matching the native build either way
        toGpu = adapt(dev, GlaCmpOprVac{T}(mnyCvl))
        @test get_backend(toGpu) isa typeof(dev)
        @test norm(Array((toGpu * fldGpu).dat) - outCpu.dat) < gpuTol(T) * norm(outCpu.dat)
        toCpu = adapt(CPU(), oprGpu)
        @test get_backend(toCpu) isa CPU
        @test norm((toCpu * fldCpu).dat - outCpu.dat) < gpuTol(T) * norm(outCpu.dat)
    end
end
