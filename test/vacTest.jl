# GilaVacuum tests
# GlaVacOprMem construction is expensive (numerical integration). We compute each
# mem once and share the precomputed egoFur arrays across testsets to avoid redundant work.

# Reuse _mem2s from tstHlp.jl for (2,2,2); build external (2,2,2) separately
const _vacMem2    = _mem2s   # alias — no extra integration
const _vacExtMem2 = GlaVacOprMem(CPUKerOpt(), mkVol((2,2,2); org=extOrg), mkVol((2,2,2)))
const _vacDims    = [(2,2,2), (4,4,4)]

# Return a fresh egoFur-based copy for given dim (cheap — no integration)
function _freshVacMem(dim)
    v   = mkVol(dim)
    mem = dim == (4,4,4) ? _selfMem4 : _mem2s
    GlaVacOprMem(CPUKerOpt(), mem.egoFur, v, v)
end

@testset "GlaVacOprMem self" begin
    for (dim, mem) in (((2,2,2), _vacMem2), ((4,4,4), _selfMem4))
        vol = mkVol(dim)
        @test length(mem.egoFur) == 8
        @test all(all(isfinite, fur) for fur in mem.egoFur)
        @test mem.trgVol == vol
        @test mem.srcVol == vol
        @test mem.cmpInf isa CPUKerOpt
    end
end

@testset "GlaVacOprMem external" begin
    for (dim, mem) in (((2,2,2), _vacExtMem2), ((4,4,4), _extMem4))
        srcVol = mkVol(dim)
        trgVol = mkVol(dim; org=extOrg)
        @test length(mem.egoFur) == 8
        @test all(all(isfinite, fur) for fur in mem.egoFur)
        @test mem.trgVol == trgVol
        @test mem.srcVol == srcVol
        @test mem.cmpInf isa CPUKerOpt
    end
end

@testset "GlaVacOprMem precomputed-egoFur constructor" begin
    vol  = mkVol((4,4,4))
    mem2 = GlaVacOprMem(CPUKerOpt(), _selfMem4.egoFur, vol, vol)
    v    = rand(ComplexF64, vol.cel..., 3)
    @test egoOpr!(_freshVacMem((4,4,4)), deepcopy(v)) ≈ egoOpr!(mem2, deepcopy(v))
end

@testset "GlaVacOprMem overlap at memory layer" begin
    # GlaVacOprMem with overlapping volumes would eventually throw ArgumentError due to
    # NaN/Inf in the circulant (zero-separation singularity), but the computation hangs
    # before reaching the check (integrals for coincident cells take indefinitely long).
    # There is no explicit pre-computation overlap guard at the memory layer.
    # Use GlaOprVac(trgVol, srcVol) for overlapping volumes: it builds a union and masks.
    @test_skip "overlap guard not practically testable: hangs during integration"
end

@testset "egoOpr! action" begin
    vol = mkVol((4,4,4))
    v1  = rand(ComplexF64, vol.cel..., 3)
    v2  = rand(ComplexF64, vol.cel..., 3)
    a   = (1.3 + 0.7im)
    # Linearity: egoOpr!(a*v1 + v2) ≈ a*egoOpr!(v1) + egoOpr!(v2)
    lhs = egoOpr!(_freshVacMem((4,4,4)), a .* deepcopy(v1) .+ deepcopy(v2))
    r1  = egoOpr!(_freshVacMem((4,4,4)), deepcopy(v1))
    r2  = egoOpr!(_freshVacMem((4,4,4)), deepcopy(v2))
    @test lhs ≈ a .* r1 .+ r2
    # Determinism
    @test egoOpr!(_freshVacMem((4,4,4)), deepcopy(v1)) ≈ egoOpr!(_freshVacMem((4,4,4)), deepcopy(v1))
    # Output shape
    @test size(egoOpr!(_freshVacMem((4,4,4)), deepcopy(v1))) == (vol.cel..., 3)
end

@testset "egoOpr! external action" begin
    srcVol = mkVol((4,4,4))
    trgVol = mkVol((4,4,4); org=extOrg)
    v      = rand(ComplexF64, srcVol.cel..., 3)
    mem2   = GlaVacOprMem(CPUKerOpt(), _extMem4.egoFur, trgVol, srcVol)
    out  = egoOpr!(mem2, deepcopy(v))
    @test size(out) == (trgVol.cel..., 3)
end

@testset "Integration convergence" begin
    function wekIntChk(scl, glOrd)
        opts = CPUKerOpt()
        ws1 = GilaElectromagnetics.GilaVacuum.wekS(scl, GilaElectromagnetics.GilaVacuum.gauQud(glOrd), opts)
        we1 = GilaElectromagnetics.GilaVacuum.wekE(scl, GilaElectromagnetics.GilaVacuum.gauQud(glOrd), opts)
        wv1 = GilaElectromagnetics.GilaVacuum.wekV(scl, GilaElectromagnetics.GilaVacuum.gauQud(glOrd), opts)
        ws2 = GilaElectromagnetics.GilaVacuum.wekS(scl, GilaElectromagnetics.GilaVacuum.gauQud(glOrd + 8), opts)
        we2 = GilaElectromagnetics.GilaVacuum.wekE(scl, GilaElectromagnetics.GilaVacuum.gauQud(glOrd + 8), opts)
        wv2 = GilaElectromagnetics.GilaVacuum.wekV(scl, GilaElectromagnetics.GilaVacuum.gauQud(glOrd + 8), opts)
        intDifS = real(maximum(abs.(ws1 .- ws2) ./ abs.(ws2)))
        intDifE = real(maximum(abs.(we1 .- we2) ./ abs.(we2)))
        intDifV = real(maximum(abs.(wv1 .- wv2) ./ abs.(wv2)))
        return [intDifS, intDifE, intDifV]
    end
    # Use only (2,2,2) — (4,4,4) re-integration takes 14+ minutes in isolation.
    # In the full suite context GC is active enough that this completes safely (~3.5 min).
    intDif = wekIntChk(_vacMem2.srcVol.scl, _vacMem2.cmpInf.intOrd)
    @test maximum(abs.(intDif)) < 1e-8
    @test length(intDif) == 3
end

@testset "GlaVacOprMem serialization CPU" begin
    vol    = mkVol((4,4,4))
    mem    = _selfMem4
    tmpFil = tempname()
    try
        open(tmpFil, "w") do io; serialize(io, mem); end
        desMem = open(tmpFil, "r") do io; deserialize(io, GlaVacOprMem); end
        useCpu!(desMem)
        @test desMem.cmpInf.intOrd == mem.cmpInf.intOrd
        @test desMem.cmpInf.frqPhz == mem.cmpInf.frqPhz
        @test desMem.cmpInf.adjMod == mem.cmpInf.adjMod
        @test typeof(desMem.cmpInf.bckEnd) == typeof(mem.cmpInf.bckEnd)
        @test desMem.trgVol == mem.trgVol
        @test desMem.srcVol == mem.srcVol
        @test desMem.mixInf.minScl == mem.mixInf.minScl
        @test desMem.mixInf.trgDiv == mem.mixInf.trgDiv
        @test desMem.mixInf.srcDiv == mem.mixInf.srcDiv
        @test desMem.mixInf.trgCel == mem.mixInf.trgCel
        @test desMem.mixInf.srcCel == mem.mixInf.srcCel
        @test desMem.mixInf.trgPar == mem.mixInf.trgPar
        @test desMem.mixInf.srcPar == mem.mixInf.srcPar
        @test desMem.egoFur == mem.egoFur
        @test desMem.phzInf == mem.phzInf
        # Action round-trip (using egoFur copies — no reintegration)
        v    = rand(ComplexF64, vol.cel..., 3)
        memA = GlaVacOprMem(CPUKerOpt(), mem.egoFur, vol, vol)
        memB = GlaVacOprMem(CPUKerOpt(), desMem.egoFur, vol, vol)
        @test egoOpr!(memA, deepcopy(v)) ≈ egoOpr!(memB, deepcopy(v))
    finally
        isfile(tmpFil) && rm(tmpFil)
    end
end

@testset "GlaVacOprMem CPU/GPU parity" begin
    if CUDA.functional()
        for (dim, memCpu) in (((2,2,2), _vacMem2), ((4,4,4), _selfMem4))
            vol    = mkVol(dim)
            memGpu = GlaVacOprMem(GPUKerOpt(), vol)
            vCpu   = rand(ComplexF64, vol.cel..., 3)
            vGpu   = CuArray(vCpu)
            outCpu = egoOpr!(GlaVacOprMem(CPUKerOpt(), memCpu.egoFur, vol, vol), deepcopy(vCpu))
            outGpu = egoOpr!(memGpu, deepcopy(vGpu))
            @test maximum(abs.(outCpu .- Array(outGpu))) < 1e-6
        end
        # useGpu!/useCpu! round-trip (on a fresh copy to avoid mutating cached mem)
        vol    = mkVol((4,4,4))
        mem    = _selfMem4
        v      = rand(ComplexF64, vol.cel..., 3)
        outCpu = egoOpr!(GlaVacOprMem(CPUKerOpt(), mem.egoFur, vol, vol), deepcopy(v))
        tmpMem = GlaVacOprMem(CPUKerOpt(), mem.egoFur, vol, vol)
        useGpu!(tmpMem)
        @test tmpMem.cmpInf isa GPUKerOpt
        useCpu!(tmpMem)
        @test tmpMem.cmpInf isa CPUKerOpt
        @test egoOpr!(GlaVacOprMem(CPUKerOpt(), mem.egoFur, vol, vol), deepcopy(v)) ≈ outCpu
    end
end
