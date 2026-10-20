using Test, Logging, GilaElectromagnetics

prxVol(xOrg) = GlaVol((4, 4, 4), scl16, (xOrg, 0//1, 0//1))
prxMem(opt, xOrg) = GlaVacOprMem(opt, prxVol(0//1), prxVol(xOrg))
prxLsy() = GlaKerOpt{Float64}(; frqPhz = 1.0 + 0.3im)

@testset "Proximity warns under a third of a wavelength" begin
    @test_logs (:warn, r"third of a wavelength") prxMem(GlaKerOpt{Float64}(), 6//16)
    @test_logs min_level = Logging.Warn prxMem(GlaKerOpt{Float64}(), 10//16)
end

@testset "Proximity says to measure when lossy" begin
    @test_logs (:warn, r"lossy") prxMem(prxLsy(), 6//16)
    @test_logs min_level = Logging.Warn prxMem(prxLsy(), 10//16)
end

@testset "Proximity skips touching, self and suppressed" begin
    @test_logs min_level = Logging.Warn prxMem(GlaKerOpt{Float64}(), 4//16)
    @test_logs min_level = Logging.Warn GlaVacOprMem(GlaKerOpt{Float64}(), prxVol(0//1))
    @test_logs min_level = Logging.Warn GlaVacOprMem(GlaKerOpt{Float64}(),
        prxVol(0//1), prxVol(6//16); prxWrn = false)
end

@testset "Proximity warns once per composite pair" begin
    cvlA = GlaCmpVol([prxVol(0//1)])
    @test_logs (:warn, r"third of a wavelength") GlaCmpOprVac{Float64}(cvlA,
        GlaCmpVol([prxVol(6//16)]))
    @test_logs min_level = Logging.Warn GlaCmpOprVac{Float64}(cvlA,
        GlaCmpVol([prxVol(10//16)]))
    @test_logs min_level = Logging.Warn GlaCmpOprVac{Float64}(cvlA, cvlA)
    @test_logs min_level = Logging.Warn GlaOprVac{Float64}(cvlA,
        GlaCmpVol([prxVol(6//16)]); prxWrn = false)
end
