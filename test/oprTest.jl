# GilaOperators tests
import GilaElectromagnetics.GilaOperators: mskRng, setSus!

const oprOvrOrg = ntuple(i -> Rational(4) * stdScl[i] // 2, 3)
const oprVolOvr = GlaVol((4,4,4), stdScl, oprOvrOrg)

@testset "Constructors & predicates" begin
    gOvr = GlaOprVac{Float64}(oprVolOvr, _vol4) # Shifted two cells, so it overlaps
    for (opr, slf, ext, ovr) in ((_g0(), true, false, false), (_gExt(), false, true, false),
        (gOvr, false, false, true), (_invSct(), true, false, false),
        (_sct(), true, false, false), (_gla(), true, false, false))
        @test isselfoperator(opr) == slf
        @test isexternaloperator(opr) == ext
        @test isoverlappingoperator(opr) == ovr
    end
end

@testset "Hermitian part constructors" begin
    gSelf = _g0()
    @test asym(gSelf) isa AsyGlaOprVac
    @test glaSym(gSelf) isa SymGlaOprVac
    @test isselfoperator(_asy()) && isselfoperator(_sym())
    # Neither part is defined for an operator between two different volumes
    @test_throws ArgumentError AsyGlaOprVac(_gExt())
    @test_throws ArgumentError SymGlaOprVac(_gExt())
end

@testset "Cross-constructors" begin
    invSct = _invSct()
    sct    = _sct()
    gla    = _gla()

    @test GlaOprVac(invSct) === invSct.oprVac
    @test GlaOprVac(sct)    === sct.invSctOpr.oprVac
    @test GlaOprVac(gla)    === gla.sctOpr.invSctOpr.oprVac
    @test SctOpr(gla)       === gla.sctOpr
    # Source bug: GlaOpr(::InvSctOpr) calls SctOpr(opr) without a solver,
    # but SctOpr(::InvSctOpr, ::GlaSlv) requires an explicit solver argument.
    @test_throws MethodError GlaOpr(sct.invSctOpr)

    @test InvSctOpr(sct)    === sct.invSctOpr
    @test InvSctOpr(gla)    === gla.sctOpr.invSctOpr
end

@testset "Composition identities" begin
    v = rand(ComplexF64, prod((4,4,4)) * 3)
    @test _invSct() * (_sct() * v) ≈ v
    @test _glaVac() * v ≈ _g0() * v
    @test _gla() * v ≈ _g0() * (_sct() * v)
end

#= The dot identity over one operator of each kind, and adjoint! there and back.
The bound leaves room for the √ε solves inside SctOpr and GlaOpr; a wrong adjoint
is wrong at order one. =#
const oprAdjXsc = GlaOprVac{Float64}(GlaVol((2,2,2), scl16, stdOrg), GlaVol((4,4,4), stdScl, (1//2, 0//1, 0//1)))
# Two touching regions at different scales, so the composite holds sandwich blocks in both orientations
const oprAdjCmp = GlaCmpOprVac{Float64}(refine(GlaCmpVol(GlaVol((4,2,2), scl16, stdOrg)),
    ((-1//16, 0//1, 0//1), (1//8, 1//8, 1//8)); factor=(2, 1, 1)))

@testset "adjoint" begin
    # A gyrotropic susceptibility: complex, not symmetric, coupling the three components
    gyr = SusOpr{Float64}(_vol2s, randn(ComplexF64, 3, 3))
    for opr in (_g0s(), _gExt(), _mskOpr, _asys(), SymGlaOprVac(_g0s()), oprAdjXsc, oprAdjCmp,
        gyr, _invScts(), _scts(), _glas())
        inp = randn(ComplexF64, size(opr, 2))
        out = opr * inp
        @test adjErr(opr, adjoint(opr)) < 1e-7
        @test !isadjoint(opr) && opr * inp == out
        @test norm(adjoint!(adjoint!(deepcopy(opr))) * inp - out) < 1e-7 * norm(out)
    end
    # Both Hermitian parts are their own adjoint
    for opr in (_asy(), _sym())
        @test adjoint!(opr) === opr
        @test adjErr(opr, opr) < 1e-12
    end
end

@testset "Touching with mismatched extents" begin
    # A small box flush against the face of a wider slab: the external contact
    # correction covers this shape, so both orientations take the external route
    # and reproduce the union of the two volumes
    box = GlaVol((2,2,2), scl16, (2//16, 0//1, 0//1))
    uni = uniVol(_mskSlb, box)
    uniMat = dnsMat(GlaOprVac{Float64}(uni))
    li = LinearIndices((uni.cel..., 3))
    dofIdx(v) = (r = mskRng(v, uni); vec([li[i,j,k,d] for i in r[1], j in r[2], k in r[3], d in 1:3]))
    slbDof, boxDof = dofIdx(_mskSlb), dofIdx(box)
    for (trg, src, rowDof, colDof) in ((_mskSlb, box, slbDof, boxDof), (box, _mskSlb, boxDof, slbDof))
        opr = GlaOprVac{Float64}(trg, src)
        @test isexternaloperator(opr)
        @test !isoverlappingoperator(opr)
        @test frbErr(dnsMat(opr), uniMat[rowDof, colDof]) < 1.5e-15
    end
    # A corner-fitting touching pair still takes the external route
    @test isexternaloperator(GlaOprVac{Float64}(_vol4, GlaVol((4,4,4), stdScl, (4//32, 0//1, 0//1))))
end

@testset "GlaOprVac on a field" begin
    opr = _g0()
    fld = discretize!(zerofield(Float64, opr.mem.srcVol), tstDns)
    out = opr * fld
    @test out isa GlaFld
    @test nregions(out.cvol) == 1
    @test regions(out.cvol)[1] == opr.mem.trgVol
    @test out.dat ≈ opr * fld.dat
    # A field over another volume, or over a tiling of several regions
    @test_throws ArgumentError opr * zerofield(Float64, GlaVol((2,2,2), stdScl, stdOrg))
    @test_throws ArgumentError opr * zerofield(Float64, GlaCmpVol(
        [GlaVol((2,2,2), stdScl, (-1//32, 0//1, 0//1)),
         GlaVol((2,2,2), stdScl, (1//32, 0//1, 0//1))]))
    # The masked route reads its input through a mask, so it takes no field
    @test_throws ArgumentError _mskOpr * zerofield(Float64, _mskBox)
end

@testset "setSus!" begin
    newSus = mkSus((4,4,4); val=1.0+0.1im)
    newDof = repeat(vec(newSus), 3)
    invSct = _invSct()
    setSus!(invSct, newSus)
    @test sus(invSct).sus == newDof
    # Wrong size throws
    @test_throws ArgumentError setSus!(invSct, mkSus((2,2,2)))
    # Propagates through SctOpr and GlaOpr
    sct = _sct()
    setSus!(sct, newSus)
    @test sus(sct).sus == newDof
    gla = _gla()
    setSus!(gla, newSus)
    @test sus(gla).sus == newDof
end

@testset "kernel option keywords, no hand-built GlaKerOpt" begin
    dflt = GlaOprVac{Float64}(_vol4)
    qss  = GlaOprVac{Float64}(_vol4; qssApx=true)
    frq  = GlaOprVac{Float64}(_vol4; frqPhz=2.0+0.5im)
    @test qss.mem.cmpInf.qssApx && !dflt.mem.cmpInf.qssApx
    @test frq.mem.cmpInf.frqPhz == 2.0+0.5im
    @test qss.mem.egoFur != dflt.mem.egoFur
    @test frq.mem.egoFur != dflt.mem.egoFur
    # shpCch only caches the far-field geometry table to disk; the operator itself is unchanged
    cch = GlaOprVac{Float64}(_vol4; shpCch=true)
    @test cch.mem.egoFur == dflt.mem.egoFur
end

@testset "show" begin
    for opr in (_g0(), _asy(), _sym(), _invSct(), _sct(), _gla())
        @test shwChk(opr)
    end
end

