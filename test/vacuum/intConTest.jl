using Test, GilaElectromagnetics

const cubRelTol = 1e-8
const cubAbsTol = 1e-12

#=
Verify that increasing quadrature order does not change integral values.
=#
function wekIntChk(scl::NTuple{3,<:Rational}, glOrd::Integer)
    opts = CPUKerOpt()
    # Weak integral values for internally set quadrature order
    ws1 = GilaElectromagnetics.GilaVacuum.wekS(scl, GilaElectromagnetics.GilaVacuum.gauQud(glOrd), opts)
    we1 = GilaElectromagnetics.GilaVacuum.wekE(scl, GilaElectromagnetics.GilaVacuum.gauQud(glOrd), opts)
    wv1 = GilaElectromagnetics.GilaVacuum.wekV(scl, GilaElectromagnetics.GilaVacuum.gauQud(glOrd), opts)
    # Weak integral values for additional quadrature points
    ws2 = GilaElectromagnetics.GilaVacuum.wekS(scl, GilaElectromagnetics.GilaVacuum.gauQud(glOrd + 8), opts)
    we2 = GilaElectromagnetics.GilaVacuum.wekE(scl, GilaElectromagnetics.GilaVacuum.gauQud(glOrd + 8), opts)
    wv2 = GilaElectromagnetics.GilaVacuum.wekV(scl, GilaElectromagnetics.GilaVacuum.gauQud(glOrd + 8), opts)
    # Differences
    intDifS = real(maximum(abs.(ws1 .- ws2) ./ abs.(ws2)))
    intDifE = real(maximum(abs.(we1 .- we2) ./ abs.(we2)))
    intDifV = real(maximum(abs.(wv1 .- wv2) ./ abs.(wv2)))
    return [intDifS, intDifE, intDifV]
end

@testset "Integration Convergence Tests" begin
    volSizes = [
        (2, 2, 2),
        (4, 4, 4),
        (6, 6, 6),
        (8, 8, 8),
        (6, 4, 8),
        (8, 2, 10)
    ]
    sclArr = (1//32, 1//32, 1//32)

    for volDim in volSizes
        # println("Testing volume size: ", volDim)
        volObj = GlaVol(volDim, sclArr, (0//1, 0//1, 0//1))
        oprMem = GlaVacOprMem(CPUKerOpt(), volObj)

        # Perform integration convergence test
        intDif = wekIntChk(oprMem.srcVol.scl, oprMem.cmpInf.intOrd)
        # println("Integration differences: ", intDif)

        # if maximum(abs.(intDif)) < cubRelTol
        #     println("Integration convergence test passed for volume size ", volDim)
        # else
        #     println("Integration convergence test failed for volume size ", volDim, ".")
        #     println("Default GlaKerOpt integration order may be insufficient, consult glaMem.jl.")
        # end

        # Assert that the maximum difference is within the relative tolerance
        @test maximum(abs.(intDif)) < cubRelTol
    end
end
