# Same-scale contact tests
# The reference for the same-scale geometries is the self operator on the union
# of the two volumes, masked down to the target and source cells by hand.
# Operators come from GlaVacOprMem, so what gets tested is the external
# construction rather than the routing in the GlaOprVac constructor.
cntExtMat(trgVol::GlaVol, srcVol::GlaVol) =
    dnsMat(GlaVacOprMem(GlaKerOpt{Float64}(), trgVol, srcVol))

# The topologies a pair of same-scale volumes can meet in: name, target, source
const cntTop = [
    ("Contact, matching cross-sections",
        GlaVol((2,2,2), scl16, stdOrg), GlaVol((2,2,2), scl16, (2//16, 0//1, 0//1))),
    # a box flush on the face of a wider slab, the shape carving produces
    ("Contact, contained cross-section",
        GlaVol((2,4,4), scl16, stdOrg), GlaVol((2,2,2), scl16, (2//16, 0//1, 0//1))),
    # half the box hangs past the edge of the slab, so neither volume has a
    # corner inside the other
    ("Contact, overhanging cross-section",
        GlaVol((2,4,4), scl16, stdOrg), GlaVol((2,2,2), scl16, (2//16, 2//16, 0//1))),
    ("Edge contact",
        GlaVol((2,2,2), scl16, stdOrg), GlaVol((2,2,2), scl16, (2//16, 2//16, 0//1))),
    ("Corner contact",
        GlaVol((2,2,2), scl16, stdOrg), GlaVol((2,2,2), scl16, (2//16, 2//16, 2//16))),
    ("Contact, partly meeting cross-sections",
        GlaVol((2,4,4), scl16, stdOrg), GlaVol((2,4,2), scl16, (2//16, 2//16, 0//1))),
    # the volume gate opens at a gap of one cell, but every cell pair is then
    # farther apart than the contact separation, so the quadrature is the regular
    # one and the widened gate changes nothing
    ("Separated same-scale pair",
        GlaVol((2,2,2), scl16, stdOrg), GlaVol((2,2,2), scl16, (3//16, 0//1, 0//1))),
]

# Agreement with the union reference in both orientations of the pair
@testset "$nam" for (nam, volA, volB) in cntTop
    @test all(frbErr(cntExtMat(trg, src), uniMskMat(trg, src)) < 1e-12
        for (trg, src) in ((volA, volB), (volB, volA)))
end

@testset "Near pair off the common lattice" begin
    # A sub-cell gap in x with the grids half a cell out of step in y. Neither
    # the contact block nor the shifted expansion covers such a pair, so it has
    # no exact route and throws. A gap of a cell or more is on the expansion
    # and builds at any shift.
    volA = GlaVol((2,2,2), scl16, stdOrg)
    volB = GlaVol((2,2,2), scl16, (9//64, 1//32, 0//1))
    @test_throws ErrorException cntExtMat(volA, volB)
    @test_throws ErrorException cntExtMat(volB, volA)
    volC = GlaVol((2,2,2), scl16, (1//4, 1//32, 0//1))
    @test all(isfinite, cntExtMat(volA, volC))
    @test all(isfinite, cntExtMat(volC, volA))
end

@testset "Cross-scale contact" begin
    #= A flush cross-scale pair reaches the contact quadrature through its
    partitioned sub-lattices. The values are finite, at the accuracy of the
    cross-scale quadrature rather than that of the same-scale contact path,
    which is what crsSclTest measures. The composite layer keeps the sandwich,
    which is exact. =#
    volCrs = GlaVol((2,2,2), scl16, stdOrg)
    volFin = GlaVol((4,4,4), stdScl, (1//8, 0//1, 0//1))
    @test all(isfinite, cntExtMat(volCrs, volFin))
    @test all(isfinite, cntExtMat(volFin, volCrs))
    # A carved domain, where a fine region meets the coarse one it was cut from
    cvol = refine(GlaCmpVol(GlaVol((4,2,2), scl16, stdOrg)),
        ((-1//16, 0//1, 0//1), (1//8, 1//8, 1//8)))
    @test nregions(cvol) == 2
    opr = GlaCmpOprVac{Float64}(cvol)
    @test all(isfinite, dnsMat(opr))
end
