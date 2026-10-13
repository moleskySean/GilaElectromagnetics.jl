# Float32 storage precision — the fp64 numerics live in the other test files
#= Generation is always fp64 and narrowed once, so the fp32 operator is the
correctly rounded image of the fp64 one and every accuracy bound below is a
statement about that single rounding, not about fp32 arithmetic in the build. =#
using Test, GilaElectromagnetics, LinearAlgebra, LinearMaps, Serialization, CUDA
import GilaElectromagnetics.GilaOperators: setSus!, invMul!, invMulAdj!
import GilaElectromagnetics.GilaSolvers: slvPrm

const prcVol   = mkVol((4,4,4))
const prcTrg   = mkVol((4,4,4); org=extOrg)
const prcVol2  = mkVol((2,2,2))
const prcTrg2  = mkVol((2,2,2); org=extOrg)
const prcSus32 = mkSus(Float32, (4,4,4))
const prcSus64 = mkSus((4,4,4))

# The fp32 twins of tstHlp's _selfMem4 / _extMem4, generated independently
const prcSlfMem32 = GlaVacOprMem(GlaKerOpt{Float32}(), prcVol)
const prcExtMem32 = GlaVacOprMem(GlaKerOpt{Float32}(), prcTrg, prcVol)

prcG032() = GlaOprVac(prcSlfMem32)

const prcDns32 = dnsMat(prcG032())

# Two touching regions, so both cross-scale blocks take the fine mesh route
const prcCvl = refine(GlaCmpVol(GlaVol((4,2,2), scl16, stdOrg)),
    ((-1//16, 0//1, 0//1), (1//8, 1//8, 1//8)); factor=(2, 1, 1))
# A coarse tiling and a fine one half a wavelength apart, the cross-scale pair
const prcCrsCvl = GlaCmpVol(GlaVol((2,2,2), scl16, stdOrg))
const prcFinCvl = GlaCmpVol(GlaVol((4,4,4), stdScl, (1//2, 0//1, 0//1)))

# Every operator below costs a numerical integration, so each pair is built once
const prcCmp32 = GlaCmpOprVac{Float32}(prcCvl)
const prcCmp64 = GlaCmpOprVac{Float64}(prcCvl)
const prcXsc32 = GlaCmpOprVac{Float32}(prcFinCvl, prcCrsCvl)
const prcXsc64 = GlaCmpOprVac{Float64}(prcFinCvl, prcCrsCvl)
# Two separated regions, so the tiling has a gap and both blocks are external
const prcGapCvl = GlaCmpVol([prcVol2, prcTrg2])
const prcGap32 = GlaCmpOprVac{Float32}(prcGapCvl)
const prcGap64 = GlaCmpOprVac{Float64}(prcGapCvl)
const prcCmpDns32 = dnsMat(prcCmp32)
const prcDflOpr = GlaOprVac(prcVol2)

# Entrywise error against the largest entry, so one bad entry cannot hide in a norm
prcEntErr(m32, m64) = maximum(abs.(ComplexF64.(m32) .- m64)) / maximum(abs.(m64))

# Both bounds on the same pair of dense forms
function prcChkDns(opr32, opr64, label)
    m32, m64 = dnsMat(opr32), dnsMat(opr64)
    nrmErr, entErr = relErr(ComplexF64.(m32), m64), prcEntErr(m32, m64)
    @info "$label: dense relErr = $nrmErr, entrywise = $entErr"
    @test nrmErr < 1e-5
    @test entErr < 1e-6
end

@testset "Default precision" begin
    @test dflPrc === Float32
    @test GlaKerOpt() isa GlaKerOpt{Float32, CPU}
    @test prcDflOpr isa GlaOprVac{Float32}
    @test zerofield(GlaCmpVol(prcVol2)) isa GlaFld{Float32}
    @test zerofield(prcVol2) isa GlaFld{Float32}
    # An fp64 susceptibility is converted on construction, the one allowed conversion
    @test eltype(GlaOpr(prcDflOpr, mkSus((2,2,2)))) == ComplexF32
    @test InvSctOpr(prcDflOpr, mkSus((2,2,2))) isa InvSctOpr{Float32}
    @test InvSctOpr(prcVol2, mkSus((2,2,2))) isa InvSctOpr{Float32}
end

@testset "Construction and traits" begin
    @test GlaKerOpt{Float32}() isa GlaKerOpt{Float32, CPU}
    @test GlaKerOpt{Float32}(GlaKerOpt{Float64}()) isa GlaKerOpt{Float32, CPU}
    @test prcSlfMem32 isa GlaVacOprMem{Float32}
    @test eltype(first(prcSlfMem32.egoFur)) == ComplexF32
    @test eltype(first(prcSlfMem32.phzInf)) == ComplexF32

    # generation at Float32 returns NaN, and the options are mutable, so the guard has
    # to sit on the assignment as well as on the constructor
    prcOptGen = GlaKerOpt{Float64}()
    @test_throws ArgumentError GlaKerOpt{Float64}(1.0+0.0im, Float32, false, false, prcOptGen.dev)
    @test_throws ArgumentError (prcOptGen.genPrc = Float32)
    prcOptGen.genPrc = Float64
    prcOptGen.frqPhz = 1.0 + 0.1im
    @test prcOptGen.genPrc === Float64
    @test prcOptGen.frqPhz == 1.0 + 0.1im

    # A backend without Float64 (Metal, OpenCL per KernelAbstractions' trait) refuses fp64 storage
    @testset "$(nameof(typeof(dev)))" for dev in gpuBackends
        supports_float64(dev) || @test_throws ArgumentError GlaKerOpt{Float64}(dev)
    end

    opr32 = prcG032()
    @test eltype(opr32) == ComplexF32
    @test eltype(typeof(opr32)) == ComplexF32
    @test eltype(similar(opr32, (4,))) == ComplexF32
    @test size(opr32) == (192, 192)
    @test get_backend(opr32) isa CPU
    @test isselfoperator(opr32)
    for opr in (AsyGlaOprVac(opr32), SymGlaOprVac(opr32), GlaOprVac(prcExtMem32),
        InvSctOpr(opr32, prcSus32), SctOpr(opr32, prcSus32), GlaOpr(opr32, prcSus32))
        @test eltype(opr) == ComplexF32
    end
    @test AsyGlaOprVac{Float32}(prcVol2) isa AsyGlaOprVac{Float32}
    @test SymGlaOprVac{Float32}(prcVol2) isa SymGlaOprVac{Float32}
    @test prcCmp32 isa GlaCmpOprVac{Float32}
    @test prcGap32 isa GlaCmpOprVac{Float32}
end

@testset "Fourier coefficient rounding" begin
    #= A correctly rounded fp64 coefficient is within eps(Float32) relative of
    the fp64 one, so this bound fails if any fp32 arithmetic ran upstream. =#
    for (fur32, fur64) in zip(prcSlfMem32.egoFur, _selfMem4.egoFur)
        @test eltype(fur32) == ComplexF32
        nzr = abs.(fur64) .> 0
        @test maximum(abs.(ComplexF64.(fur32[nzr]) .- fur64[nzr]) ./ abs.(fur64[nzr])) <=
            eps(Float32)
    end
    furErr = maximum(maximum(abs.(ComplexF64.(f32[abs.(f64) .> 0]) .- f64[abs.(f64) .> 0]) ./
        abs.(f64[abs.(f64) .> 0])) for (f32, f64) in zip(prcSlfMem32.egoFur, _selfMem4.egoFur))
    @info "egoFur worst relative rounding = $furErr, eps(Float32) = $(eps(Float32))"
    for (fur32, fur64) in zip(prcExtMem32.egoFur, _extMem4.egoFur)
        nzr = abs.(fur64) .> 0
        @test maximum(abs.(ComplexF64.(fur32[nzr]) .- fur64[nzr]) ./ abs.(fur64[nzr])) <=
            eps(Float32)
    end
end

@testset "Dense truncation accuracy" begin
    prcChkDns(prcG032(), _g0(), "self GlaOprVac")
    prcChkDns(GlaOprVac(prcExtMem32), _gExt(), "external GlaOprVac")
    prcChkDns(AsyGlaOprVac(prcG032()), _asy(), "AsyGlaOprVac")
    prcChkDns(SymGlaOprVac(prcG032()), _sym(), "SymGlaOprVac")
    prcChkDns(prcGap32, prcGap64, "GlaCmpOprVac gapped")
    prcChkDns(prcCmp32, prcCmp64, "GlaCmpOprVac two regions")
    prcChkDns(prcXsc32, prcXsc64, "GlaCmpOprVac cross-scale")
    prcChkDns(InvSctOpr(prcG032(), prcSus32), InvSctOpr(_g0(), prcSus64), "InvSctOpr")
end

@testset "Matvec and adjoint truncation accuracy" begin
    opr32, opr64 = prcG032(), _g0()
    v32 = ComplexF32.(rand(ComplexF64, 192))
    v32 ./= norm(v32)
    v64 = ComplexF64.(v32)
    out32, out64 = opr32 * copy(v32), opr64 * copy(v64)
    mvErr = norm(ComplexF64.(out32) .- out64) / norm(out64)
    @info "matvec relErr = $mvErr"
    @test mvErr < 1e-5

    u32 = ComplexF32.(rand(ComplexF64, 192))
    u32 ./= norm(u32)
    adjOpr = adjoint(opr32)
    lhs = dot(adjOpr * copy(u32), v32)
    rhs = dot(u32, opr32 * copy(v32))
    adjErr = abs(lhs - rhs) / abs(rhs)
    @info "adjoint dot identity relErr = $adjErr"
    @test adjErr < 1e-5
end

@testset "Asym(G₀) PSD survives truncation" begin
    #= physTest asserts this at 1e-9 in fp64; rounding the coefficients to fp32
    perturbs the spectrum by the truncation error, hence the looser bound. =#
    mat = asymMat(ComplexF64.(prcDns32))
    nrm = opnorm(mat)
    wrs = minimum(eigvals(Hermitian((mat + mat') / 2)))
    @info "fp32 Asym(G₀) worst normalized eigenvalue = $(wrs / nrm)"
    @test wrs >= -1e-5 * nrm
end

@testset "Precision conversion" begin
    memCnv = GlaVacOprMem{Float32}(_selfMem4)
    @test memCnv isa GlaVacOprMem{Float32}
    # Generation is deterministic fp64, so narrowing reproduces a fresh build
    for (furCnv, fur32) in zip(memCnv.egoFur, prcSlfMem32.egoFur)
        @test furCnv == fur32
    end
    @test GlaVacOprMem{Float32}(prcSlfMem32) === prcSlfMem32
    @test dnsMat(GlaOprVac{Float32}(_g0())) == prcDns32

    # Same precision hands back the same object rather than a copy
    opr32 = prcG032()
    @test GlaOprVac{Float32}(opr32) === opr32
    for opr in (AsyGlaOprVac(opr32), SymGlaOprVac(opr32),
        InvSctOpr(opr32, prcSus32), SctOpr(opr32, prcSus32), GlaOpr(opr32, prcSus32),
        prcCmp32)
        @test Base.typename(typeof(opr)).wrapper{Float32}(opr) === opr
    end

    # A round trip through fp32 costs one rounding of the coefficients, nothing more
    gla64 = _glas()
    gla32 = GlaOpr{Float32}(gla64)
    @test gla32 isa GlaOpr{Float32}
    rndErr = relErr(dnsMat(GlaOpr{Float64}(gla32)), dnsMat(gla64))
    @info "GlaOpr fp64 -> fp32 -> fp64 relErr = $rndErr"
    @test rndErr < 1e-6

    # A converted composite is the composite that would have been built directly
    @test dnsMat(GlaCmpOprVac{Float32}(prcCmp64)) == prcCmpDns32
    @test sus(InvSctOpr{Float32}(InvSctOpr(_g0(), prcSus64))).sus == repeat(vec(prcSus32), 3)

    # An adjoint between unequal volumes converts, copies and moves with its plans intact
    adj = adjoint!(GlaOprVac{Float64}(mkVol((6,4,8); org=extOrg), mkVol((4,4,4))))
    x = rand(ComplexF64, size(adj, 2))
    ref = adj * x
    @test norm(GlaOprVac(GlaVacOprMem{Float32}(adj.mem)) * ComplexF32.(x) - ref) < 1e-6 * norm(ref)
    @test deepcopy(adj) * x == ref
    @test adapt(CPU(), adj) * x == ref
end

@testset "Mixed precision throws" begin
    opr32, opr64 = prcG032(), _g0()
    v32, v64 = zeros(ComplexF32, 192), zeros(ComplexF64, 192)
    out32, out64 = similar(v32), similar(v64)
    ten32 = zeros(ComplexF32, 4, 4, 4, 3)
    ten64 = zeros(ComplexF64, 4, 4, 4, 3)
    fld64 = zerofield(Float64, prcVol)
    fld32 = zerofield(Float32, prcVol)
    #= Every one of these has to be an explicit throwing method: absent, the
    caller gets a method error rather than the conversion to make, and nothing
    may reach the scalar getindex on the way there, so no warning may appear. =#
    @test_logs begin
        @test_throws ArgumentError opr32 * v64
        @test_throws ArgumentError opr64 * v32
        @test_throws ArgumentError mul!(out64, opr32, v32)
        @test_throws ArgumentError mul!(out32, opr32, v64)
        @test_throws ArgumentError opr32 * zeros(Float64, 192)
        @test_throws ArgumentError opr32 * ten64
        @test_throws ArgumentError mul!(ten64, opr32, ten32, 1.0, 0.0)
        @test_throws ArgumentError opr32 * zeros(ComplexF64, 192, 2)
        @test_throws ArgumentError opr32 * fld64
        @test_throws ArgumentError mul!(fld64, opr32, fld32, 1.0, 0.0)
        @test_throws ArgumentError GilaElectromagnetics.GilaOperators.mulAct!(opr32, v64)
    end
end

@testset "GlaFld at Float32" begin
    cvol = GlaCmpVol(prcVol)
    fld = zerofield(Float32, cvol)
    @test fld isa GlaFld{Float32}
    @test eltype(fld) == ComplexF32
    @test eltype(parent(fld)) == ComplexF32
    @test eltype(similar(fld)) == ComplexF32

    dsc = discretize!(zerofield(Float32, cvol), tstDns)
    @test eltype(dsc.dat) == ComplexF32
    # Broadcast keeps the wrapper and the precision
    bct = 2 .* dsc .+ dsc
    @test bct isa GlaFld{Float32}
    @test eltype(bct) == ComplexF32
    @test norm(bct.dat .- 3 .* dsc.dat) < 1e-6 * norm(dsc.dat)
    @test eltype(regrid(dsc)) == ComplexF32

    opr32 = prcG032()
    out = opr32 * dsc
    @test out isa GlaFld{Float32}
    outMul = zerofield(Float32, cvol)
    mul!(outMul, opr32, dsc, 2.0, 0.0)
    @test norm(outMul.dat .- 2 .* out.dat) < 1e-5 * norm(out.dat)

    # The plain vector space operations go through the wrapper unchanged
    @test dot(dsc, dsc) ≈ norm(dsc)^2 rtol=1e-5
    axpy!(ComplexF32(2), dsc, outMul)
    @test eltype(outMul.dat) == ComplexF32
end

@testset "Solvers at Float32" begin
    opr32 = prcG032()
    invSct32 = InvSctOpr(opr32, prcSus32)
    invSct64 = InvSctOpr(_g0(), prcSus64)
    b32 = ComplexF32.(rand(ComplexF64, 192))
    b32 ./= norm(b32)
    b64 = ComplexF64.(b32)

    @test slvPrm(GMRESSolver(), b32).relTol == sqrt(eps(Float32))
    @test slvPrm(BiCGStabSolver(), b32).relTol == sqrt(eps(Float32))

    for slv in (GMRESSolver(), BiCGStabSolver())
        sol = solve(invSct32, copy(b32), slv)
        @test eltype(sol) == ComplexF32
        res32 = norm(invSct32 * copy(sol) - b32) / norm(b32)
        #= The same solution measured with the fp64 operator stalls at the
        truncation error of the operator, which is the floor MixPrcRfn beats. =#
        res64 = norm(invSct64 * ComplexF64.(sol) - b64) / norm(b64)
        @info "$(typeof(slv)) fp32 residual = $res32, measured in fp64 = $res64"
        @test res32 < 1e-3
        @test res64 > 1e-8
    end
end

@testset "Mixed precision refinement" begin
    #= Three routes on the same fp64 system: fp64 GMRES as the reference, fp32
    corrections refined against the fp64 residual, and fp32 GMRES as the
    control that shows the floor the refinement is there to beat. =#
    prcTgt = 1e-10 # Tight enough that neither route stops at its own relTol
    prcRef = GMRESSolver(nothing, nothing, nothing, prcTgt)
    opr32 = InvSctOpr(prcG032(), prcSus32)
    b64 = rand(ComplexF64, 192)
    b64 ./= norm(b64)
    b32 = ComplexF32.(b64)

    solRef = solve(_invSct(), copy(b64), prcRef)
    solIr = solve(_invSct(), copy(b64), MixPrcRfn(Float32; relTol=prcTgt))
    sol32 = ComplexF64.(solve(opr32, copy(b32), GMRESSolver(nothing, nothing, nothing, prcTgt)))

    resIr = norm(_invSct() * copy(solIr) - b64) / norm(b64)
    res32 = norm(_invSct() * copy(sol32) - b64) / norm(b64)
    errIr = frbErr(solIr, solRef)
    err32 = frbErr(sol32, solRef)
    @info "IR fp64 residual = $resIr, relErr = $errIr; fp32 residual in fp64 = $res32, relErr = $err32"
    @test resIr < 1e-9
    @test errIr < 1e-7
    #= The fp32 route stalls at the truncation error of the operator, measured
    at 5e-8 here, three orders above the refined solution. =#
    @test err32 > 1e-8
    @test err32 < 1e-2
    @test res32 > 1e-8

    # Three outer steps suffice, so ten never warns
    @test_nowarn solve(_invSct(), copy(b64), MixPrcRfn(Float32; maxItr=10, relTol=prcTgt))

    # A susceptibility change on the high operator reaches the cached low copy
    rfn = MixPrcRfn(Float32; relTol=prcTgt)
    invSct = _invSct()
    solve(invSct, copy(b64), rfn)
    susNew = mkSus(Float64, (4,4,4); val=0.8+0.1im)
    setSus!(invSct, susNew)
    solNew = solve(invSct, copy(b64), rfn)
    invNew = InvSctOpr(_g0(), susNew)
    @test frbErr(solNew, solve(invNew, copy(b64), prcRef)) < 1e-7
    @test norm(invNew * copy(solNew) - b64) / norm(b64) < 1e-9

    # The refinement is the solver of a scattering operator, used through mul!
    for opr in (SctOpr, GlaOpr)
        act = opr(_g0(), prcSus64; slv=MixPrcRfn(Float32; relTol=prcTgt)) * copy(b64)
        ref = opr(_g0(), prcSus64; slv=prcRef) * copy(b64)
        @test frbErr(act, ref) < 1e-7
    end

    # The inner solver is pluggable
    solBcg = solve(_invSct(), copy(b64), MixPrcRfn(Float32; innSlv=BiCGStabSolver(), relTol=prcTgt))
    @test norm(_invSct() * copy(solBcg) - b64) / norm(b64) < 1e-9
    @test frbErr(solBcg, solRef) < 1e-7

    #= The inverse paths of the wrappers route their solve through the vacuum
    default rather than the operator's own solver, so this checks that a
    refining operator still works there, not that it refines. =#
    gla = GlaOpr(_g0(), prcSus64; slv=MixPrcRfn(Float32; relTol=prcTgt))
    glaRef = GlaOpr(_g0(), prcSus64; slv=prcRef)
    @test LinearMap(gla) * copy(b64) ≈ LinearMap(glaRef) * copy(b64) rtol=1e-7
    w, wRef = zeros(ComplexF64, 192), zeros(ComplexF64, 192)
    invMul!(w, gla, copy(b64), one(ComplexF64), zero(ComplexF64))
    invMul!(wRef, glaRef, copy(b64), one(ComplexF64), zero(ComplexF64))
    @test frbErr(w, wRef) < 1e-7
    invMulAdj!(w, gla, copy(b64), one(ComplexF64), zero(ComplexF64))
    invMulAdj!(wRef, glaRef, copy(b64), one(ComplexF64), zero(ComplexF64))
    @test frbErr(w, wRef) < 1e-7
    @test !isadjoint(gla)

    # An adjoint! on the high operator also has to reach the cached copy
    rfnAdj = MixPrcRfn(Float32; relTol=prcTgt)
    solve(_invSct(), copy(b64), rfnAdj)
    invAdj = adjoint(_invSct())
    solAdj = solve(invAdj, copy(b64), rfnAdj)
    @test norm(invAdj * copy(solAdj) - b64) / norm(b64) < 1e-9
end

@testset "Mixed precision refinement GPU" begin
    @testset "$(nameof(typeof(dev)))" for dev in gpuBackends
        supports_float64(dev) || continue
        vol = prcVol
        sus = adapt(dev, prcSus64)
        opr = InvSctOpr(GlaOprVac{Float64}(vol; dev), sus)
        b = adapt(dev, rand(ComplexF64, 192))
        b ./= norm(b)
        sol = solve(opr, copy(b), MixPrcRfn(Float32; relTol=1e-10))
        @test eltype(sol) == ComplexF64
        @test norm(opr * copy(sol) - b) / norm(b) < 1e-9
    end
end

@testset "Serialization at Float32" begin
    buf = IOBuffer()
    serialize(buf, prcSlfMem32)
    seekstart(buf)
    memDes = deserialize(buf)
    @test memDes isa GlaVacOprMem{Float32}
    @test memDes.cmpInf isa GlaKerOpt{Float32}
    for (furDes, fur32) in zip(memDes.egoFur, prcSlfMem32.egoFur)
        @test furDes == fur32
    end

    gla32 = GlaOpr{Float32}(_glas())
    buf = IOBuffer()
    serialize(buf, gla32)
    seekstart(buf)
    glaDes = deserialize(buf)
    @test glaDes isa GlaOpr{Float32}
    @test glaDes.sctOpr.invSctOpr.oprVac.mem.egoFur ==
        gla32.sctOpr.invSctOpr.oprVac.mem.egoFur
    @test relErr(dnsMat(glaDes), dnsMat(gla32)) < 1e-5
end

@testset "LinearMaps at Float32" begin
    opr32 = prcG032()
    lmp = LinearMap(opr32)
    @test eltype(lmp) == ComplexF32
    @test size(lmp) == (192, 192)
    @test Matrix(lmp) == prcDns32
end

@testset "Float32 GPU" begin
    @testset "$(nameof(typeof(dev)))" for dev in gpuBackends
        opr32 = GlaOprVac{Float32}(prcVol; dev)
        @test get_backend(opr32) isa typeof(dev)
        @test eltype(opr32) == ComplexF32
        v32 = adapt(dev, rand(ComplexF32, 192))
        out32 = opr32 * copy(v32)
        @test get_backend(out32) isa typeof(dev)
        outCpu = prcG032() * Array(v32)
        @test norm(Array(out32) - outCpu) < gpuTol(Float32) * norm(outCpu)
        supports_float64(dev) && @test_throws(ArgumentError, opr32 * adapt(dev, zeros(ComplexF64, 192)))
        supports_float64(dev) && @test(GlaOprVac{Float64}(opr32) isa GlaOprVac{Float64})
        fld32 = zerofield(Float32, GlaCmpVol(prcVol); dev)
        @test get_backend(fld32) isa typeof(dev)
        sol = solve(InvSctOpr(opr32, adapt(dev, prcSus32)), copy(v32), GMRESSolver())
        @test eltype(sol) == ComplexF32
    end
end
