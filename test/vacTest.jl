# GlaVacOprMem: construction, action, and the volume pairings it accepts.
# Construction integrates, so the shared mems of tstHlp.jl are reused and fresh
# copies are built from their egoFur rather than integrated again.

const _vacExtMem2 = GlaVacOprMem(GlaKerOpt{Float64}(), mkVol((2,2,2); org=extOrg),
    mkVol((2,2,2)))

# a fresh (4,4,4) mem sharing the precomputed egoFur — no integration
_freshVacMem() = GlaVacOprMem(GlaKerOpt{Float64}(), _selfMem4.egoFur, _vol4, _vol4)

@testset "GlaVacOprMem self" begin
    for (dim, mem) in (((2,2,2), _mem2s), ((4,4,4), _selfMem4))
        @test length(mem.egoFur) == 8
        @test all(all(isfinite, fur) for fur in mem.egoFur)
        @test mem.trgVol == mem.srcVol == mkVol(dim)
        @test mem.cmpInf isa GlaKerOpt
    end
end

@testset "GlaVacOprMem external" begin
    for (dim, mem) in (((2,2,2), _vacExtMem2), ((4,4,4), _extMem4))
        @test length(mem.egoFur) == 8
        @test all(all(isfinite, fur) for fur in mem.egoFur)
        @test mem.trgVol == mkVol(dim; org=extOrg)
        @test mem.srcVol == mkVol(dim)
        @test mem.cmpInf isa GlaKerOpt
    end
end

@testset "Self, external and overlap at (8,8,8)" begin
    volSrc = mkVol((8,8,8))
    @test all(isfinite, GlaVacOprMem(GlaKerOpt{Float64}(), volSrc).egoFur[1])
    @test all(isfinite, GlaVacOprMem(GlaKerOpt{Float64}(),
        mkVol((8,8,8); org=extOrg), volSrc).egoFur[1])
    # A half cell shift overlaps, which the memory layer throws on rather than
    # integrating through the singularity. GlaOprVac unions and masks instead.
    @test_throws ArgumentError GlaVacOprMem(GlaKerOpt{Float64}(),
        GlaVol((8,8,8), stdScl, (1//64, 0//1, 0//1)), volSrc)
end

@testset "GlaVacOprMem from a precomputed egoFur" begin
    v = rand(ComplexF64, _vol4.cel..., 3)
    @test egoOpr!(_freshVacMem(), deepcopy(v)) ≈ egoOpr!(_freshVacMem(), deepcopy(v))
end

@testset "egoOpr! action" begin
    v1  = rand(ComplexF64, _vol4.cel..., 3)
    v2  = rand(ComplexF64, _vol4.cel..., 3)
    a   = 1.3 + 0.7im
    lhs = egoOpr!(_freshVacMem(), a .* deepcopy(v1) .+ deepcopy(v2))
    r1  = egoOpr!(_freshVacMem(), deepcopy(v1))
    r2  = egoOpr!(_freshVacMem(), deepcopy(v2))
    @test lhs ≈ a .* r1 .+ r2
    @test egoOpr!(_freshVacMem(), deepcopy(v1)) ≈ r1
    @test size(r1) == (_vol4.cel..., 3)
end

@testset "egoOpr! external action" begin
    srcVol = mkVol((4,4,4))
    trgVol = mkVol((4,4,4); org=extOrg)
    v      = rand(ComplexF64, srcVol.cel..., 3)
    mem    = GlaVacOprMem(GlaKerOpt{Float64}(), _extMem4.egoFur, trgVol, srcVol)
    @test size(egoOpr!(mem, deepcopy(v))) == (trgVol.cel..., 3)
end
