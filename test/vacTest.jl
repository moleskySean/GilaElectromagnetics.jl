# GlaVacOprMem: the volume pairings it accepts, and construction from given Fourier data

@testset "GlaVacOprMem from a precomputed egoFur" begin
    mem = GlaVacOprMem(GlaKerOpt{Float64}(), _selfMem4.egoFur, _vol4, _vol4)
    v = rand(ComplexF64, size(_g0(), 2))
    @test GlaOprVac(mem) * v == _g0() * v
end

@testset "Overlapping volumes" begin
    # A half cell shift overlaps, which the memory layer throws on rather than
    # integrating through the singularity. GlaOprVac unions and masks instead.
    @test_throws ArgumentError GlaVacOprMem(GlaKerOpt{Float64}(),
        GlaVol((4,4,4), stdScl, (1//64, 0//1, 0//1)), _vol4)
end
