# Composite operator tests
# The reference for a composite operator is the same operator on a uniform mesh
# at the finest cell size of the tiling. A composite target row is the mean of
# the fine rows it covers, a composite source column injects unit density into
# the fine cells it covers, and the √ΔV basis puts sqrt(ΔVᵢ/ΔVⱼ) on block (i, j).
using Test, GilaElectromagnetics, LinearAlgebra, CUDA

const cmpScl16 = (1//16, 1//16, 1//16)
const cmpScl32 = (1//32, 1//32, 1//32)
const cmpOrg0 = (0//1, 0//1, 0//1)

cmpRelFro(matA, matB) = norm(matA - matB) / norm(matB)
cmpLwrEdg(vol::GlaVol) = Tuple(first.(vol.grd) .- (vol.scl .// 2))

# Linear indices in refVol of the cells under each cell of cvol, in layout order
function cmpChdIdx(cvol::GlaCmpVol, refVol::GlaVol)
    lin = LinearIndices(Tuple(refVol.cel))
    refLwr = cmpLwrEdg(refVol)
    chd = Vector{Vector{Int}}()
    for reg in regions(cvol)
        rat = ntuple(dir -> Int(reg.scl[dir] // refVol.scl[dir]), 3)
        bas = ntuple(dir ->
            Int((cmpLwrEdg(reg)[dir] - refLwr[dir]) // refVol.scl[dir]), 3)
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

const GlaSnd = GilaElectromagnetics.GilaOperators.GlaSndOprVac
import GilaElectromagnetics.GilaVacuum: arrTyp

# Self, external, masked union, and fine mesh blocks of a composite operator
cmpBlkCnt(opr) = (count(blk -> blk isa GlaOprVac && isselfoperator(blk), opr.blkMat),
    count(blk -> blk isa GlaOprVac && isexternaloperator(blk), opr.blkMat),
    count(blk -> blk isa GlaOprVac && isoverlappingoperator(blk), opr.blkMat),
    count(blk -> blk isa GlaSnd, opr.blkMat))

#= The money geometry: a (4,4,4) volume of 1/16 λ cells with its low x half
refined, so the tiling is one fine region face to face with one coarse one. =#
const mnyVol = GlaVol((4, 4, 4), cmpScl16, cmpOrg0)
const mnyCvl = refine(GlaCmpVol(mnyVol), ((-1//16, 0//1, 0//1), (1//8, 1//4, 1//4)))
const mnyRef = GlaVol((8, 8, 8), cmpScl32, cmpOrg0)
const mnyOpr = GlaCmpOprVac(mnyCvl)
const mnyMat = dnsMat(mnyOpr)
const mnyAgr = cmpAgrRef(mnyCvl, mnyRef, dnsMat(GlaOprVac(mnyRef)))

@testset "Composite operator geometry" begin
    @test nregions(mnyCvl) == 2
    @test regions(mnyCvl)[1].cel == (4, 8, 8)
    @test regions(mnyCvl)[1].scl == cmpScl32
    @test regions(mnyCvl)[2].cel == (2, 4, 4)
    @test regions(mnyCvl)[2].scl == cmpScl16
    @test size(mnyOpr) == (864, 864)
    @test size(mnyOpr, 1) == 864
    @test eltype(mnyOpr) == ComplexF64
    @test isselfoperator(mnyOpr)
    @test !isexternaloperator(mnyOpr)
    @test !isadjoint(mnyOpr)
    @test !isgpu(mnyOpr)
    @test glaSze(mnyOpr) == glaSze.(mnyOpr.blkMat)
    @test glaSze(mnyOpr, 2)[1, 1] == (4, 8, 8, 3)
    # The two cross-scale blocks touch, so they take the fine mesh route
    @test cmpBlkCnt(mnyOpr) == (2, 0, 0, 2)
    @test CompositeVacuumGreenOperator === GlaCmpOprVac
end

@testset "Composite operator against a uniform reference" begin
    @test all(isfinite, mnyMat)
    @test cmpRelFro(mnyMat, mnyAgr) < 1e-6
    # A composite target row is a mean, not a sum, so the wrong convention shows
    @test cmpRelFro(mnyMat, 8 .* mnyAgr) > 1e-3
end

@testset "Composite operator complex symmetry" begin
    @test cmpRelFro(mnyMat, transpose(mnyMat)) < 1e-8
end

@testset "Composite operator adjoint" begin
    adjOpr = adjoint(mnyOpr)
    @test adjOpr isa GlaCmpOprVac
    @test isadjoint(adjOpr)
    @test !isadjoint(mnyOpr)
    @test size(adjOpr) == reverse(size(mnyOpr))
    @test cmpRelFro(dnsMat(adjOpr), mnyMat') < 1e-13
    # The original operator is untouched
    @test dnsMat(mnyOpr) == mnyMat
end

@testset "Composite operator on a field" begin
    fld = discretize!(zerofield(mnyCvl), pos -> (exp(2im * pi * pos[1]), pos[2], 0))
    out = mnyOpr * fld
    @test out isa GlaFld
    @test out.cvol === mnyCvl
    @test length(out) == 864
    @test all(isfinite, out.dat)
    # The flat path and the field path are the same computation
    @test mnyOpr * collect(fld.dat) == out.dat
    @test norm(out.dat - mnyMat * fld.dat) < 1e-12 * norm(out.dat)
    # A field on another tiling does not fit
    @test_throws ArgumentError mnyOpr * zerofield(GlaCmpVol(mnyRef))
    @test_throws ArgumentError mnyOpr * zeros(ComplexF64, 863)
    # Five argument mul! comes from the generic fallback
    outMul = zerofield(mnyCvl)
    mul!(outMul, mnyOpr, fld, 2.0, 0.0)
    @test norm(outMul.dat - 2 .* out.dat) < 1e-12 * norm(out.dat)
    # Densification through getindex
    @test cmpRelFro(mnyOpr[1:4, 1:4], mnyMat[1:4, 1:4]) < 1e-12
end

const cmpSym = GilaElectromagnetics.GilaOperators.sym

#= The two Hermitian parts are read entry by entry off the composite matrix,
which the complex symmetry of the self operator makes exact. Both are built from
the same block matrix, so a matvec costs one application of G₀. =#
@testset "Composite operator Hermitian parts" begin
    asyOpr, symOpr = asym(mnyOpr), cmpSym(mnyOpr)
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
    @test !isgpu(asyOpr)
    @test arrTyp(asyOpr) <: Array
    @test useCpu!(asyOpr) === asyOpr
    @test occursin("composite Asym(G₀)", sprint(show, asyOpr))
    @test occursin("composite Sym(G₀)", sprint(show, symOpr))

    asyDns, symDns = dnsMat(asyOpr), dnsMat(symOpr)
    @test all(isfinite, asyDns)
    @test cmpRelFro(asyDns, asymMat(mnyMat)) < 1e-12
    @test cmpRelFro(symDns, symMat(mnyMat)) < 1e-12
    # Both parts are Hermitian, and together they rebuild the operator
    @test cmpRelFro(asyDns, asyDns') < 1e-12
    @test cmpRelFro(symDns, symDns') < 1e-12
    @test cmpRelFro(symDns + im .* asyDns, mnyMat) < 1e-12
    # The operator the part was taken from is untouched
    @test dnsMat(mnyOpr) == mnyMat

    # Hermitian, so adjoint! hands the operator back
    @test GilaElectromagnetics.adjoint!(asyOpr) === asyOpr
    @test cmpRelFro(dnsMat(adjoint(asyOpr)), asyDns) < 1e-12

    fld = discretize!(zerofield(mnyCvl), pos -> (exp(2im * pi * pos[1]), pos[2], 0))
    out = asyOpr * fld
    @test out isa GlaFld
    @test out.cvol === mnyCvl
    @test norm(out.dat - asyDns * fld.dat) < 1e-12 * norm(out.dat)
    @test asyOpr * collect(fld.dat) == out.dat
    @test_throws ArgumentError asyOpr * zerofield(GlaCmpVol(mnyRef))
    @test_throws ArgumentError asyOpr * zeros(ComplexF64, 863)
    @test cmpRelFro(asyOpr[1:4, 1:4], asyDns[1:4, 1:4]) < 1e-12

    # A part only makes sense for a self operator, and not in adjoint mode
    extOpr = GlaCmpOprVac(GlaCmpVol(GlaVol((2, 2, 2), cmpScl16, cmpOrg0)),
        GlaCmpVol(GlaVol((2, 2, 2), cmpScl16, (1//1, 0//1, 0//1))))
    @test_throws ArgumentError asym(extOpr)
    @test_throws ArgumentError cmpSym(extOpr)
    @test_throws ArgumentError asym(adjoint(mnyOpr))

    # The volume constructor builds the operator it needs
    @test AsyGlaCmpOprVac(GlaCmpVol(GlaVol((2, 2, 2), cmpScl16, cmpOrg0))) isa AsyGlaCmpOprVac
    @test SymGlaCmpOprVac(GlaCmpVol(GlaVol((2, 2, 2), cmpScl16, cmpOrg0))) isa SymGlaCmpOprVac
end

#= A coarse region on each side of a fine one, so a sandwich block appears in
both orientations and the two coarse regions see each other across a gap. =#
const triCvl = refine(GlaCmpVol(GlaVol((6, 4, 4), cmpScl16, cmpOrg0)),
    (cmpOrg0, (1//8, 1//4, 1//4)))
const triRef = GlaVol((12, 8, 8), cmpScl32, cmpOrg0)

@testset "Composite operator with three regions" begin
    @test nregions(triCvl) == 3
    @test regions(triCvl)[1].cel == (4, 8, 8)
    @test all(reg -> reg.cel == (2, 4, 4), regions(triCvl)[2:3])
    triOpr = GlaCmpOprVac(triCvl)
    @test size(triOpr) == (960, 960)
    # Three self blocks, the two coarse regions apart in x, four sandwiches
    @test cmpBlkCnt(triOpr) == (3, 2, 0, 4)
    triMat = dnsMat(triOpr)
    @test all(isfinite, triMat)
    triAgr = cmpAgrRef(triCvl, triRef, dnsMat(GlaOprVac(triRef)))
    @test cmpRelFro(triMat, triAgr) < 1e-6
    @test cmpRelFro(triMat, transpose(triMat)) < 1e-8
    @test cmpRelFro(dnsMat(adjoint(triOpr)), triMat') < 1e-13
    # Sandwich blocks in both orientations, plus a same-scale external pair
    @test cmpRelFro(dnsMat(asym(triOpr)), asymMat(triMat)) < 1e-12
    @test cmpRelFro(dnsMat(cmpSym(triOpr)), symMat(triMat)) < 1e-12
end

@testset "Composite operator between two bodies" begin
    srcCvl = GlaCmpVol(GlaVol((2, 2, 2), cmpScl16, (1//2, 0//1, 0//1)))
    opr = GlaCmpOprVac(mnyCvl, srcCvl)
    @test isexternaloperator(opr)
    @test !isselfoperator(opr)
    @test size(opr) == (864, 24)
    # Half a wavelength away, so both blocks take the ordinary external route
    @test cmpBlkCnt(opr) == (0, 2, 0, 0)
    mat = dnsMat(opr)
    @test all(isfinite, mat)
    srcRef = GlaVol((4, 4, 4), cmpScl32, (1//2, 0//1, 0//1))
    refMat = dnsMat(GlaOprVac(mnyRef, srcRef))
    @test cmpRelFro(mat, cmpAgrRef(mnyCvl, mnyRef, srcCvl, srcRef, refMat)) < 1e-6
    @test cmpRelFro(dnsMat(adjoint(opr)), mat') < 1e-13
    # The two argument constructor spelling
    @test GlaOprVac(mnyCvl, srcCvl) isa GlaCmpOprVac
end

@testset "Composite operator cross-scale near pair" begin
    # A coarse volume and a fine one, one coarse cell apart in x
    srcCvl = GlaCmpVol(GlaVol((2, 2, 2), cmpScl16, cmpOrg0))
    trgCvl = GlaCmpVol(GlaVol((4, 4, 4), cmpScl32, (3//16, 0//1, 0//1)))
    opr = GlaCmpOprVac(trgCvl, srcCvl)
    # A gap of one coarse cell keeps the pair off the contact quadrature
    @test opr.blkMat[1, 1] isa GlaOprVac
    mat = dnsMat(opr)
    @test all(isfinite, mat)
    srcRef = GlaVol((4, 4, 4), cmpScl32, cmpOrg0)
    refMat = dnsMat(GlaOprVac(regions(trgCvl)[1], srcRef))
    #= The cross-scale quadrature over a coarse cell is less accurate the closer
    the two volumes are. At one coarse cell of separation it agrees with the fine
    reference to a few times 1e-5, and the error falls by more than an order of
    magnitude for every doubling of the gap. =#
    @test cmpRelFro(mat, cmpAgrRef(trgCvl, regions(trgCvl)[1], srcCvl, srcRef,
        refMat)) < 1e-4
    farCvl = GlaCmpVol(GlaVol((4, 4, 4), cmpScl32, (3//8, 0//1, 0//1)))
    farMat = dnsMat(GlaCmpOprVac(farCvl, srcCvl))
    farRef = dnsMat(GlaOprVac(regions(farCvl)[1], srcRef))
    @test cmpRelFro(farMat, cmpAgrRef(farCvl, regions(farCvl)[1], srcCvl, srcRef,
        farRef)) < 1e-6
    # Moving the fine volume onto the coarse one switches the block over
    tchCvl = GlaCmpVol(GlaVol((4, 4, 4), cmpScl32, (1//8, 0//1, 0//1)))
    tchOpr = GlaCmpOprVac(tchCvl, srcCvl)
    @test tchOpr.blkMat[1, 1] isa GlaSnd
    @test all(isfinite, dnsMat(tchOpr))
end

@testset "Composite operator contact block" begin
    #= A small volume face to face with a taller one. The external construction
    corrects for the cells in contact, and the answer is the self operator of the
    union of the two, masked down. =#
    trgCvl = GlaCmpVol(GlaVol((2, 2, 2), cmpScl16, cmpOrg0))
    srcCvl = GlaCmpVol(GlaVol((2, 4, 4), cmpScl16, (1//8, 0//1, 0//1)))
    opr = GlaCmpOprVac(trgCvl, srcCvl)
    @test cmpBlkCnt(opr) == (0, 1, 0, 0)
    mat = dnsMat(opr)
    @test size(mat) == (24, 96)
    @test all(isfinite, mat)
    # The same entries read off the densified self operator of the union
    uniMat = dnsMat(GlaOprVac(GlaVol((4, 4, 4), cmpScl16, (1//16, 0//1, 0//1))))
    lin = LinearIndices((4, 4, 4))
    rowCel = vec([lin[xItr, yItr, zItr] for xItr in 1:2, yItr in 2:3, zItr in 2:3])
    colCel = vec([lin[xItr, yItr, zItr] for xItr in 3:4, yItr in 1:4, zItr in 1:4])
    rowDof = vcat([(dir - 1) * 64 .+ rowCel for dir in 1:3]...)
    colDof = vcat([(dir - 1) * 64 .+ colCel for dir in 1:3]...)
    @test cmpRelFro(mat, uniMat[rowDof, colDof]) < 1e-12
    @test cmpRelFro(dnsMat(adjoint(opr)), mat') < 1e-13
end

@testset "Composite operator overlapping bodies" begin
    cvolA = GlaCmpVol(GlaVol((2, 2, 2), cmpScl16, cmpOrg0))
    cvolB = GlaCmpVol(GlaVol((2, 2, 2), cmpScl16, (1//16, 0//1, 0//1)))
    @test_throws ArgumentError GlaCmpOprVac(cvolA, cvolB)
    # Two equal tilings are the same body, so the self operator is meant
    @test isselfoperator(GlaCmpOprVac(cvolA, GlaCmpVol(GlaVol((2, 2, 2), cmpScl16, cmpOrg0))))
end

@testset "Composite operator on a uniform mesh" begin
    vol = GlaVol((2, 2, 2), cmpScl16, cmpOrg0)
    opr = VacuumGreenOperator(GlaCmpVol(vol))
    @test opr isa GlaCmpOprVac
    @test size(opr) == (24, 24)
    @test isselfoperator(opr)
    @test cmpBlkCnt(opr) == (1, 0, 0, 0)
    # One region means one cell volume, so the normalization is the identity
    @test dnsMat(opr) == dnsMat(GlaOprVac(vol))
    # One region reproduces the plain anti-Hermitian operator
    @test cmpRelFro(dnsMat(asym(opr)), dnsMat(AsyGlaOprVac(vol))) < 1e-12
    @test slv(opr) isa GlaSlv
    @test arrTyp(opr) <: Array
    @test useCpu!(opr) === opr
    @test occursin("composite G₀", sprint(show, opr))
end

@testset "Plain operator on a field" begin
    #= A plain operator over one region is the one region composite, so the two
    have to agree on a field, including the sqrt(ΔV_trg / ΔV_src) of the basis. =#
    srcVol = GlaVol((2, 2, 2), cmpScl16, cmpOrg0)
    trgVol = GlaVol((4, 4, 4), cmpScl32, (3//8, 0//1, 0//1))
    fld = discretize!(zerofield(srcVol), pos -> (exp(2im * pi * pos[1]), pos[2], 0))
    cmpOut = GlaCmpOprVac(GlaCmpVol(trgVol), GlaCmpVol(srcVol)) * fld
    plnOut = GlaOprVac(trgVol, srcVol) * fld
    @test plnOut isa GlaFld
    @test regions(plnOut.cvol)[1] == trgVol
    @test norm(plnOut.dat - cmpOut.dat) < 1e-13 * norm(cmpOut.dat)
end

@testset "Composite operator GPU" begin
    if CUDA.functional()
        oprGpu = GlaCmpOprVac(mnyCvl; useGpu=true)
        @test isgpu(oprGpu)
        @test arrTyp(oprGpu) <: CuArray
        fldGpu = discretize!(zerofield(mnyCvl; useGpu=true),
            pos -> (exp(2im * pi * pos[1]), pos[2], 0))
        outGpu = oprGpu * fldGpu
        @test outGpu isa GlaFld
        @test parent(outGpu) isa CuVector{ComplexF64}
        fldCpu = discretize!(zerofield(mnyCvl),
            pos -> (exp(2im * pi * pos[1]), pos[2], 0))
        outCpu = mnyOpr * fldCpu
        @test norm(Array(outGpu.dat) - outCpu.dat) < 1e-10 * norm(outCpu.dat)

        asyGpu = asym(oprGpu)
        @test isgpu(asyGpu)
        @test arrTyp(asyGpu) <: CuArray
        asyOutGpu = asyGpu * fldGpu
        asyOutCpu = asym(mnyOpr) * fldCpu
        @test norm(Array(asyOutGpu.dat) - asyOutCpu.dat) < 1e-10 * norm(asyOutCpu.dat)
    end
end
