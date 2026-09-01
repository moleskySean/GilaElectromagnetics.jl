using Test, GilaElectromagnetics, CUDA

@testset "External and Self Green Function Tests" begin
    volSrc = GlaVol((8, 8, 8), (1//32, 1//32, 1//32), (0//1, 0//1, 0//1))
    volTrg = GlaVol((8, 8, 8), (1//32, 1//32, 1//32), (1//1, 1//1, 1//1)) # Non-overlapping

    # Test self Green function
    oprMemSelf = GlaVacOprMem(CPUKerOpt(), volSrc)
    @test all(isfinite.(oprMemSelf.egoFur[1]))

    # Test external Green function
    oprMemExt = GlaVacOprMem(CPUKerOpt(), volTrg, volSrc)
    @test all(isfinite.(oprMemExt.egoFur[1]))

    # Test overlapping volumes (should throw error)
    volOverlap = GlaVol((8, 8, 8), (1//32, 1//32, 1//32), (1//64, 0//1, 0//1))
    @test_throws ArgumentError GlaVacOprMem(CPUKerOpt(), volOverlap, volSrc)

    # Test that an explicitly created self Green function operator does not throw
    @test all(isfinite.(GlaVacOprMem(CPUKerOpt(), volSrc).egoFur[1]))
end