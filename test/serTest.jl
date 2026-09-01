# Serialization round-trips of the operator types
#= A round-trip has to reproduce the operator entry by entry, since nothing in
the written format is recomputed, and the copy has to apply, which only works if
the FFTW plans were rebuilt on load rather than read back as raw pointers. =#
using Test, GilaElectromagnetics, LinearAlgebra, Serialization

const serScl16 = (1//16, 1//16, 1//16)
const serOrg0 = (0//1, 0//1, 0//1)
const serSus = 0.5 + 0.05im
const serSnd = GilaElectromagnetics.GilaOperators.GlaSndOprVac
const serSym = GilaElectromagnetics.GilaOperators.sym

serRelFro(matA, matB) = norm(matA - matB) / norm(matB)

function serRnd(opr)
    buf = IOBuffer()
    serialize(buf, opr)
    seekstart(buf)
    return deserialize(buf, typeof(opr))
end

# tol is loosened for the operators whose application runs an iterative solve
function serChk(opr; tol=1e-12)
    desOpr = serRnd(opr)
    @test desOpr isa typeof(opr)
    @test serRelFro(dnsMat(desOpr), dnsMat(opr)) < tol
    innVec = rand(ComplexF64, size(opr, 2))
    @test norm(desOpr * innVec - opr * innVec) < tol * norm(opr * innVec)
    return desOpr
end

#= A coarse region face to face with a region refined in x alone, so both
cross-scale blocks run the contact quadrature and are a fine mesh block. =#
const serCvl = refine(GlaCmpVol(GlaVol((4, 2, 2), serScl16, serOrg0)),
    ((-1//16, 0//1, 0//1), (1//8, 1//8, 1//8)); factor=(2, 1, 1))
const serOpr = GlaCmpOprVac(serCvl)
const serFarVol = GlaVol((2, 2, 2), serScl16, (1//1, 0//1, 0//1))
#= Two volumes sharing interior, so construction folds them into their union and
the masks become the only record of the sub-volumes. =#
const serOvrA = GlaVol((2, 2, 2), serScl16, serOrg0)
const serOvrB = GlaVol((2, 2, 2), serScl16, (1//16, 1//16, 1//16))
serFld() = discretize!(zerofield(serCvl), pos -> (exp(2im * pi * pos[1]), pos[2], 0))

@testset "Vacuum operator serialization" begin
    serChk(_g0s())
    serChk(_gExt())
    # A block matrix, whose blocks go through the generic serializer
    serChk(MulRegGlaOprVac(reshape([_g0(), GlaOprVac(_vol4, _trgV4)], 1, 2)))
end

@testset "Overlapping operator serialization" begin
    ovrOpr = GlaOprVac(serOvrA, serOvrB)
    @test isoverlappingoperator(ovrOpr)
    # Without the masks the copy reads back as the self operator on the union
    @test size(serRnd(ovrOpr)) == size(ovrOpr)
    desOpr = serChk(ovrOpr)
    @test isoverlappingoperator(desOpr)
    @test (desOpr.srcMsk, desOpr.trgMsk) == (ovrOpr.srcMsk, ovrOpr.trgMsk)
end

@testset "Composite operator serialization" begin
    @test count(blk -> blk isa serSnd, serOpr.blkMat) == 2
    desOpr = serChk(serOpr)
    @test nregions(desOpr.srcCvl) == 2
    @test desOpr.srcCvl == serCvl
    @test isselfoperator(desOpr)
    @test count(blk -> blk isa serSnd, desOpr.blkMat) == 2
    # A field on the original tiling still applies, the tilings compare equal
    @test norm((desOpr * serFld()).dat - (serOpr * serFld()).dat) <
        1e-12 * norm((serOpr * serFld()).dat)
    # Two bodies, so the block matrix is not square
    serChk(GlaCmpOprVac(serCvl, GlaCmpVol(serFarVol)))
end

#= The parts hold the transformed Fourier coefficients of their blocks, so the
reader must not take the part again. =#
@testset "Composite Hermitian part serialization" begin
    for opr in (asym(serOpr), serSym(serOpr))
        desOpr = serChk(opr)
        @test GilaElectromagnetics.adjoint!(desOpr) === desOpr
        desDns = dnsMat(desOpr)
        @test serRelFro(desDns, desDns') < 1e-12
    end
    # The imaginary part taken twice is not the imaginary part
    @test serRelFro(dnsMat(serRnd(asym(serOpr))), asymMat(dnsMat(serOpr))) < 1e-12
end

# The kind of vacuum operator is tagged in the stream, so every kind reads back
@testset "Inverse scattering operator kinds" begin
    for invSct in (_invScts(), InvSctOpr(_asys(), _sus2s),
        InvSctOpr(serOpr, serSus))
        desInv = serChk(invSct)
        @test desInv.oprVac isa typeof(invSct.oprVac)
        @test desInv.sus == invSct.sus
    end
end

@testset "Scattering operator serialization" begin
    for invSct in (_invScts(), InvSctOpr(serOpr, serSus))
        serChk(SctOpr(invSct, BiCGStabSolver()); tol=1e-6)
        serChk(GlaOpr(SctOpr(invSct, BiCGStabSolver())); tol=1e-6)
    end
    # The composite application path, on a field and on the flat vector
    invSct = InvSctOpr(serOpr, serSus)
    desInv = serRnd(invSct)
    @test norm((desInv * serFld()).dat - (invSct * serFld()).dat) <
        1e-12 * norm((invSct * serFld()).dat)
    fldDat = collect(serFld().dat)
    @test norm(desInv * fldDat - invSct * fldDat) < 1e-12 * norm(invSct * fldDat)
end
