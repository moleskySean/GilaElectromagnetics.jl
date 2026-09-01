# Same-scale contact tests
# The reference for the same-scale geometries is the self operator on the union
# of the two volumes, masked down to the target and source cells by hand.
# Operators come from GlaVacOprMem, so what gets tested is the external
# construction rather than the routing in the GlaOprVac constructor.
import GilaElectromagnetics.GilaVolumes: uniVol
import GilaElectromagnetics.GilaOperators: mskRng

const cntScl = (1//16, 1//16, 1//16)
const cntOrg = (0//1, 0//1, 0//1)

cntRelFro(matA, matB) = norm(matA - matB) / norm(matB)

cntExtMat(trgVol::GlaVol, srcVol::GlaVol) =
    dnsMat(GlaVacOprMem(CPUKerOpt(), trgVol, srcVol))

function cntUniMat(trgVol::GlaVol, srcVol::GlaVol)
    uniVolume = uniVol(trgVol, srcVol)
    oprUni = GlaOprVac(uniVolume)
    innMsk, outMsk = mskRng(srcVol, uniVolume), mskRng(trgVol, uniVolume)
    colNum = prod(srcVol.cel) * 3
    mat = zeros(ComplexF64, prod(trgVol.cel) * 3, colNum)
    for colItr in 1:colNum
        srcVec = zeros(ComplexF64, srcVol.cel..., 3)
        srcVec[colItr] = one(ComplexF64)
        embVec = zeros(ComplexF64, uniVolume.cel..., 3)
        embVec[innMsk..., :] .= srcVec
        mat[:, colItr] .= vec((oprUni * embVec)[outMsk..., :])
    end
    return mat
end

# Agreement with the union reference in both orientations of the pair
cntBthOrn(volA::GlaVol, volB::GlaVol) =
    (cntRelFro(cntExtMat(volA, volB), cntUniMat(volA, volB)),
     cntRelFro(cntExtMat(volB, volA), cntUniMat(volB, volA)))

@testset "Contact, matching cross-sections" begin
    volA = GlaVol((2,2,2), cntScl, cntOrg)
    volB = GlaVol((2,2,2), cntScl, (2//16, 0//1, 0//1))
    @test all(cntBthOrn(volA, volB) .< 1e-12)
end

@testset "Contact, contained cross-section" begin
    # A box flush on the face of a wider slab, the shape carving produces
    slb = GlaVol((2,4,4), cntScl, cntOrg)
    box = GlaVol((2,2,2), cntScl, (2//16, 0//1, 0//1))
    @test all(cntBthOrn(slb, box) .< 1e-12)
end

@testset "Contact, overhanging cross-section" begin
    # Half the box hangs past the edge of the slab, so neither volume has a
    # corner inside the other
    slb = GlaVol((2,4,4), cntScl, cntOrg)
    box = GlaVol((2,2,2), cntScl, (2//16, 2//16, 0//1))
    @test all(cntBthOrn(slb, box) .< 1e-12)
end

@testset "Edge contact" begin
    volA = GlaVol((2,2,2), cntScl, cntOrg)
    volB = GlaVol((2,2,2), cntScl, (2//16, 2//16, 0//1))
    @test all(cntBthOrn(volA, volB) .< 1e-12)
end

@testset "Corner contact" begin
    volA = GlaVol((2,2,2), cntScl, cntOrg)
    volB = GlaVol((2,2,2), cntScl, (2//16, 2//16, 2//16))
    @test all(cntBthOrn(volA, volB) .< 1e-12)
end

@testset "Contact, partly meeting cross-sections" begin
    slb = GlaVol((2,4,4), cntScl, cntOrg)
    box = GlaVol((2,4,2), cntScl, (2//16, 2//16, 0//1))
    @test all(cntBthOrn(slb, box) .< 1e-12)
end

@testset "Separated same-scale pair" begin
    # The volume gate opens at a gap of one cell, but every cell pair is then
    # farther apart than the contact separation, so the quadrature is the
    # regular one and the widened gate changes nothing
    volA = GlaVol((2,2,2), cntScl, cntOrg)
    volB = GlaVol((2,2,2), cntScl, (3//16, 0//1, 0//1))
    @test all(cntBthOrn(volA, volB) .< 1e-12)
end

@testset "Near pair off the common lattice" begin
    # A sub-cell gap in x with the grids half a cell out of step in y. Contact
    # quadrature is not defined for this pair, so it falls through to the
    # regular quadrature. No accuracy claim, only that it builds and is finite.
    volA = GlaVol((2,2,2), cntScl, cntOrg)
    volB = GlaVol((2,2,2), cntScl, (9//64, 1//32, 0//1))
    @test all(isfinite, cntExtMat(volA, volB))
    @test all(isfinite, cntExtMat(volB, volA))
end

@testset "Cross-scale contact" begin
    #= A flush cross-scale pair reaches the contact quadrature through its
    partitioned sub-lattices. The values are finite, at the accuracy of the
    cross-scale quadrature rather than that of the same-scale contact path,
    which is what crsSclTest measures. The composite layer keeps the sandwich,
    which is exact. =#
    volCrs = GlaVol((2,2,2), cntScl, cntOrg)
    volFin = GlaVol((4,4,4), (1//32,1//32,1//32), (1//8, 0//1, 0//1))
    @test all(isfinite, cntExtMat(volCrs, volFin))
    @test all(isfinite, cntExtMat(volFin, volCrs))
    # A carved domain, where a fine region meets the coarse one it was cut from
    cvol = refine(GlaCmpVol(GlaVol((4,2,2), cntScl, cntOrg)),
        ((-1//16, 0//1, 0//1), (1//8, 1//8, 1//8)))
    @test nregions(cvol) == 2
    opr = GlaCmpOprVac(cvol)
    @test all(isfinite, dnsMat(opr))
end
