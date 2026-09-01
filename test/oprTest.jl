# GilaOperators tests
import GilaElectromagnetics.GilaOperators: ovrChk, mskRng, rszSus, setSus!

# Shared helpers from tstHlp.jl; oprTest-specific:
const _ovrOrg4 = ntuple(i -> Rational(4) * stdScl[i] // 2, 3)
const _volOvr4 = GlaVol((4,4,4), stdScl, _ovrOrg4)

@testset "Constructors & predicates" begin
    vol = _vol4; trgV = _trgV4; sus = _sus4

    # Self GlaOprVac
    gSelf = _g0()
    @test isselfoperator(gSelf)
    @test !isexternaloperator(gSelf)
    @test !isoverlappingoperator(gSelf)
    @test !isadjoint(gSelf)
    @test !isgpu(gSelf)

    # External GlaOprVac
    gExt = _gExt()
    @test !isselfoperator(gExt)
    @test isexternaloperator(gExt)
    @test !isoverlappingoperator(gExt)

    # Overlapping GlaOprVac (shifted by 2 cells → overlap)
    gOvr = GlaOprVac(_volOvr4, _vol4)
    @test isoverlappingoperator(gOvr)
    @test !isselfoperator(gOvr)
    @test !isexternaloperator(gOvr)
    @test !all(==(0:0), gOvr.srcMsk)
    @test !all(==(0:0), gOvr.trgMsk)

    # GlaOprVac(mem) reproduces predicates
    gFromMem = _g0()
    @test isselfoperator(gFromMem)

    # InvSctOpr, SctOpr, GlaOpr predicates
    for opr in (_invSct(), _sct(), _gla())
        @test isselfoperator(opr)
        @test !isexternaloperator(opr)
        @test !isadjoint(opr)
        @test !isgpu(opr)
    end
end

@testset "isgpu follows cmpInf" begin
    # useGpu!/useCpu! swap mem.cmpInf between GPUKerOpt and CPUKerOpt, so isgpu
    # only needs to check the type of the options. Swapping the options by hand
    # avoids needing a GPU; nothing here applies the operator. Fresh mem so the
    # shared _selfMem4 is not mutated.
    mem = GlaVacOprMem(CPUKerOpt(), _vol4)
    opr = GlaOprVac(mem)
    @test !isgpu(opr)
    @test occursin("CPU", sprint(show, opr))

    mem.cmpInf = GPUKerOpt()
    @test isgpu(opr)
    @test occursin("GPU", sprint(show, opr))

    mem.cmpInf = CPUKerOpt()
    @test !isgpu(opr)
end

@testset "AsyGlaOprVac / SymGlaOprVac constructors" begin
    gSelf = _g0()
    gExt  = _gExt()

    asy1 = _asy()
    asy2 = AsyGlaOprVac(gSelf)
    @test isselfoperator(asy1)
    @test !isadjoint(asy1)
    @test isselfoperator(asy2)

    sym1 = _sym()
    sym2 = SymGlaOprVac(gSelf)
    @test isselfoperator(sym1)
    @test !isadjoint(sym1)

    @test (try AsyGlaOprVac(gExt); false catch e; isa(e, ArgumentError) end)
    @test (try SymGlaOprVac(gExt); false catch e; isa(e, ArgumentError) end)

    # asym() convenience
    @test asym(gSelf) isa AsyGlaOprVac
    # sym is not exported — call qualified
    @test GilaElectromagnetics.GilaOperators.sym(gSelf) isa SymGlaOprVac
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

@testset "size / glaSze / eltype" begin
    n = prod((4,4,4)) * 3

    gSelf = _g0()
    @test eltype(gSelf) == ComplexF64
    @test size(gSelf) == (n, n)
    @test size(gSelf, 1) == n
    @test size(gSelf, 2) == n
    @test glaSze(gSelf, 1) == ((4,4,4)..., 3)
    @test glaSze(gSelf, 2) == ((4,4,4)..., 3)

    gExt = _gExt()
    @test size(gExt, 1) == n
    @test size(gExt, 2) == n

    for opr in (_invSct(), _sct(), _gla())
        @test eltype(opr) == ComplexF64
        @test size(opr) == (n, n)
    end
end

@testset "matvec forms agree" begin
    for opr in (_g0(), _asy(), _sym(), _sct(), _gla(), _invSct())
        n  = size(opr, 2)
        vv = rand(ComplexF64, n)
        # Vector form matches 4D reshape → vector
        out_flat = opr * vv
        out_4d   = opr * reshape(vv, glaSze(opr, 2))
        @test out_flat ≈ vec(out_4d)
        # Matrix columns agree with vector products
        V = rand(ComplexF64, n, 3)
        outM = opr * V
        for j in 1:3
            @test outM[:, j] ≈ opr * V[:, j]
        end
    end
end

@testset "mul! 5-arg" begin
    for opr in (_g0(), _invSct(), _sct(), _gla())
        n = size(opr, 2); m = size(opr, 1)
        v   = rand(ComplexF64, n)
        out = rand(ComplexF64, m)
        α = (2.0 + 0.5im)
        β = (0.3 - 0.1im)
        out_old  = copy(out)
        expected = α .* (opr * v) .+ β .* out_old
        mul!(out, opr, v, α, β)
        @test out ≈ expected
    end
end

@testset "Non-mutation of input" begin
    for opr in (_g0(), _invSct(), _sct(), _gla())
        v      = rand(ComplexF64, size(opr, 2))
        v_copy = copy(v)
        _ = opr * v
        @test v == v_copy
    end
end

@testset "Composition identities" begin
    v      = rand(ComplexF64, prod((4,4,4)) * 3)
    invSct = _invSct()
    sct    = _sct()
    gla    = _gla()
    gVac   = _g0()
    glaVac = _glaVac()

    @test invSct * (sct * v) ≈ v
    @test glaVac * v ≈ gVac * v
end

@testset "adjoint" begin
    for opr in (_g0(), _asy(), _sym(),
                _invSct(), _sct(), _gla())
        mat    = dnsMat(opr)
        adjMat = dnsMat(adjoint(opr))
        @test adjMat ≈ mat'

        # Double adjoint
        v = rand(ComplexF64, size(opr, 2))
        @test adjoint(adjoint(opr)) * v ≈ opr * v
    end

    # adjoint! toggles and restores (fresh mutable operator)
    g = _g0()
    adjoint!(g)
    @test isadjoint(g)
    adjoint!(g)
    @test !isadjoint(g)

    # AsyGlaOprVac is Hermitian
    asy = _asy()
    @test adjoint!(asy) === asy
    asyMat = dnsMat(asy)
    @test asyMat ≈ asyMat'

    # SymGlaOprVac is Hermitian
    sym = _sym()
    @test adjoint!(sym) === sym
    symMat = dnsMat(sym)
    @test symMat ≈ symMat'
end

@testset "ovrChk / mskRng" begin
    v1 = _vol4
    # Non-overlapping
    v2 = mkVol((4,4,4); org=extOrg)
    @test !ovrChk(v1, v2)
    # Overlapping
    v3 = GlaVol((4,4,4), stdScl, (2//32, 2//32, 2//32))
    @test ovrChk(v1, v3)
    # Touching: edges meet but the volumes share no interior, so this is not
    # overlap. The external construction corrects for cell contact directly.
    v4 = GlaVol((4,4,4), stdScl, (4//32, 0//1, 0//1))
    @test !ovrChk(v1, v4)
    @test isexternaloperator(GlaOprVac(v1, v4))
    # mskRng
    bigVol = GlaVol((8,4,4), stdScl, (2//32, 0//1, 0//1))
    rng = mskRng(v1, bigVol)
    @test length(rng) == 3
    # Misaligned sub-volume throws AssertionError
    badSub = GlaVol((4,4,4), stdScl, (1//64, 0//1, 0//1))
    @test_throws AssertionError mskRng(badSub, bigVol)
end

@testset "Touching with mismatched extents" begin
    # A small box flush against the face of a wider slab: the external contact
    # correction covers this shape, so both orientations take the external route
    # and reproduce the union of the two volumes
    slb = GlaVol((2,4,4), (1//16,1//16,1//16), (0//1, 0//1, 0//1))
    box = GlaVol((2,2,2), (1//16,1//16,1//16), (2//16, 0//1, 0//1))
    uni = uniVol(slb, box)
    uniMat = dnsMat(GlaOprVac(uni))
    li = LinearIndices((uni.cel..., 3))
    dofIdx(v) = (r = mskRng(v, uni); vec([li[i,j,k,d] for i in r[1], j in r[2], k in r[3], d in 1:3]))
    slbDof, boxDof = dofIdx(slb), dofIdx(box)
    for (trg, src, rowDof, colDof) in ((slb, box, slbDof, boxDof), (box, slb, boxDof, slbDof))
        opr = GlaOprVac(trg, src)
        @test isexternaloperator(opr)
        @test !isoverlappingoperator(opr)
        @test norm(dnsMat(opr) - uniMat[rowDof, colDof]) / norm(uniMat[rowDof, colDof]) < 1e-12
    end
    # A corner-fitting touching pair still takes the external route
    @test isexternaloperator(GlaOprVac(_vol4, GlaVol((4,4,4), stdScl, (4//32, 0//1, 0//1))))
end

@testset "Masked operator adjoint" begin
    # A slab and a box sharing interior, which the constructor sends through the
    # union route, so the operator carries a source and a target mask
    slb = GlaVol((2,4,4), (1//16,1//16,1//16), (0//1, 0//1, 0//1))
    box = GlaVol((2,2,2), (1//16,1//16,1//16), (1//16, 0//1, 0//1))
    opr = GlaOprVac(slb, box)
    @test isoverlappingoperator(opr)
    fwdMat = dnsMat(opr)
    # The adjoint has to exchange the two masks along with the volumes
    adjMat = dnsMat(opr')
    @test size(adjMat) == reverse(size(fwdMat))
    @test norm(adjMat - fwdMat') / norm(fwdMat) < 1e-13
    # In place, and the round trip back
    adjOpr = adjoint!(deepcopy(opr))
    @test glaSze(adjOpr, 2) == glaSze(opr, 1)
    @test norm(dnsMat(adjOpr) - fwdMat') / norm(fwdMat) < 1e-13
    @test norm(dnsMat(adjoint!(adjOpr)) - fwdMat) / norm(fwdMat) < 1e-15
end

@testset "GlaOprVac on a field" begin
    opr = _g0()
    vol = opr.mem.srcVol
    fld = discretize!(zerofield(vol), pos -> (exp(2im * pi * pos[1]), pos[2], 0))
    out = opr * fld
    @test out isa GlaFld
    @test nregions(out.cvol) == 1
    @test regions(out.cvol)[1] == opr.mem.trgVol
    @test out.dat ≈ opr * fld.dat
    # A field over another volume, or over a tiling of several regions
    @test_throws ArgumentError opr * zerofield(GlaVol((2,2,2), stdScl, stdOrg))
    @test_throws ArgumentError opr * zerofield(GlaCmpVol(
        [GlaVol((2,2,2), stdScl, (-1//32, 0//1, 0//1)),
         GlaVol((2,2,2), stdScl, (1//32, 0//1, 0//1))]))
    # The masked route reads its input through a mask, so it takes no field
    slb = GlaVol((2,4,4), (1//16,1//16,1//16), (0//1, 0//1, 0//1))
    box = GlaVol((2,2,2), (1//16,1//16,1//16), (1//16, 0//1, 0//1))
    @test_throws ArgumentError GlaOprVac(slb, box) * zerofield(box)
end

@testset "rszSus" begin
    cel   = (4,4,4)
    sus3d = rand(ComplexF64, cel...)
    sus1d = vec(sus3d)
    # 3D passthrough
    @test rszSus(sus3d, cel) === sus3d
    # 1D reshape
    @test rszSus(sus1d, cel) == sus3d
    # Wrong length throws
    @test_throws ArgumentError rszSus(zeros(ComplexF64, 5), cel)
    # 2D throws
    @test_throws ArgumentError rszSus(rand(ComplexF64, 4, 16), cel)
end

@testset "setSus!" begin
    newSus = mkSus((4,4,4); val=1.0+0.1im)
    invSct = _invSct()
    setSus!(invSct, newSus)
    @test invSct.sus == newSus
    # Wrong size throws
    @test_throws ArgumentError setSus!(invSct, mkSus((2,2,2)))
    # Propagates through SctOpr
    sct = _sct()
    setSus!(sct, newSus)
    @test sct.invSctOpr.sus == newSus
    # Propagates through GlaOpr
    gla = _gla()
    setSus!(gla, newSus)
    @test gla.sctOpr.invSctOpr.sus == newSus
end

@testset "MulRegGlaOprVac" begin
    vol1 = _vol4
    vol2 = _trgV4
    vols = [vol1, vol2]
    op   = MulRegGlaOprVac(vols, vols)
    n    = prod((4,4,4)) * 3

    # oprMat structure
    @test size(op.oprMat) == (2, 2)
    @test isselfoperator(op.oprMat[1,1])
    @test isselfoperator(op.oprMat[2,2])
    @test isexternaloperator(op.oprMat[1,2])
    @test isexternaloperator(op.oprMat[2,1])

    # size
    @test size(op) == (2n, 2n)
    @test size(op, 1) == 2n
    @test size(op, 2) == 2n

    # flat-vector matvec
    x  = rand(ComplexF64, 2n)
    y  = op * x
    x1 = x[1:n]; x2 = x[n+1:2n]
    y_man = vcat(op.oprMat[1,1] * x1 + op.oprMat[1,2] * x2,
                 op.oprMat[2,1] * x1 + op.oprMat[2,2] * x2)
    @test y ≈ y_man

    # block-vector matvec
    xBlk = [reshape(x1, (4,4,4,3)), reshape(x2, (4,4,4,3))]
    yBlk = op * xBlk
    @test vec(yBlk[1]) ≈ op.oprMat[1,1] * x1 + op.oprMat[1,2] * x2
    @test vec(yBlk[2]) ≈ op.oprMat[2,1] * x1 + op.oprMat[2,2] * x2

    # Dense matrix consistency
    D = dnsMat(op)
    @test D * x ≈ y

    # adjoint
    adjD = dnsMat(adjoint(op))
    @test adjD ≈ D'
    @test size(adjoint(op)) == size(op)

    # show contains "multi-region" and counts
    str = sprint(show, op)
    @test occursin("multi-region", str)
    @test occursin("2", str)
end

@testset "show" begin
    strs = [
        (_g0(),                        ["Self",     "CPU", "G₀"]),
        (_gExt(),                      ["External", "CPU", "G₀"]),
        (_asy(),                       ["Self",     "CPU", "Asym(G₀)"]),
        (_sym(),                       ["Self",     "CPU", "Sym(G₀)"]),
        (_invSct(),                    ["Self",     "CPU", "(I - XG₀)"]),
        (_sct(),                       ["Self",     "CPU", "(I - XG₀)⁻¹"]),
        (_gla(),                       ["Self",     "CPU", "G₀(I - XG₀)⁻¹"]),
    ]
    for (opr, keys) in strs
        s = sprint(show, opr)
        for k in keys
            @test occursin(k, s)
        end
    end
    # Adjoint adds "Adjoint"
    adjStr = sprint(show, adjoint(_g0()))
    @test occursin("Adjoint", adjStr)
end

@testset "Operator serialization" begin
    v    = rand(ComplexF64, size(_g0(), 2))

    oprs = [
        _g0(),
        _asy(),
        _sym(),
        _invSct(),
        _sct(),
        _gla(),
    ]
    for opr in oprs
        tmpFil = tempname()
        try
            open(tmpFil, "w") do io; serialize(io, opr); end
            T = typeof(opr)
            desOpr = open(tmpFil, "r") do io; deserialize(io, T); end
            # Action round-trip where size matches
            if size(opr, 2) == length(v)
                @test opr * v ≈ desOpr * v
            end
        finally
            isfile(tmpFil) && rm(tmpFil)
        end
    end

    # Multi-region blocks go through the generic serializer, which must rebuild
    # the FFTW plans on load rather than restore the written pointers
    tmpFil = tempname()
    mr = MulRegGlaOprVac([_vol4, _trgV4], [_vol4, _trgV4])
    vMr = rand(ComplexF64, size(mr, 2))
    try
        open(tmpFil, "w") do io; serialize(io, mr); end
        desMr = open(tmpFil, "r") do io; deserialize(io, MulRegGlaOprVac); end
        @test desMr * vMr ≈ mr * vMr
    finally
        isfile(tmpFil) && rm(tmpFil)
    end

    # Same path for operators nested in a container
    tmpFil = tempname()
    opr = _g0()
    try
        open(tmpFil, "w") do io; serialize(io, [opr]); end
        @test isnothing(findfirst(codeunits("FFTW"), read(tmpFil)))
        desOpr = only(open(deserialize, tmpFil))
        @test desOpr * v ≈ opr * v
    finally
        isfile(tmpFil) && rm(tmpFil)
    end
end

@testset "Operator CPU/GPU parity" begin
    if CUDA.functional()
        sus  = _sus4
        susg = CuArray(sus)
        vcpu = rand(ComplexF64, size(_g0(), 2))
        vgpu = CuArray(vcpu)

        pairs = [
            (_g0(),                        GlaOprVac(_vol4; useGpu=true)),
            (InvSctOpr(_vol4, sus),        InvSctOpr(_vol4, susg; useGpu=true)),
            (SctOpr(_vol4, sus),           SctOpr(_vol4, susg; useGpu=true)),
            (GlaOpr(_vol4, sus),           GlaOpr(_vol4, susg; useGpu=true)),
        ]
        for (cOpr, gOpr) in pairs
            @test cOpr * vcpu ≈ Array(gOpr * vgpu)
            # useGpu! on cpu opr
            useGpu!(cOpr)
            @test isgpu(cOpr)
            @test cOpr * vgpu ≈ gOpr * vgpu
            # useCpu! on gpu opr
            useCpu!(gOpr)
            @test !isgpu(gOpr)
            # No-ops
            useCpu!(gOpr)
            @test !isgpu(gOpr)
            useCpu!(cOpr)  # after useGpu!
            useGpu!(gOpr)
        end
    end
end
