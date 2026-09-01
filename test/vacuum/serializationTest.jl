using Test
using Serialization
using GilaElectromagnetics
using CUDA

@testset "Serialization Tests" begin
    # Test CPU-based GlaVacOprMem
    cpuOpt = CPUKerOpt()
    trgVol = GlaVol((4, 4, 4), (1//32, 1//32, 1//32), (0//1, 0//1, 0//1))
    srcVol = trgVol
    oprMemCpu = GlaVacOprMem(cpuOpt, trgVol, srcVol)

    tmpFilCpu = tempname()
    open(tmpFilCpu, "w") do io
        serialize(io, oprMemCpu)
    end

    desOprCpu = open(tmpFilCpu, "r") do io
        deserialize(io, GlaVacOprMem)
    end
    useCpu!(desOprCpu)

    # Verify all fields for CPU-based object
    @test desOprCpu.cmpInf.intOrd == oprMemCpu.cmpInf.intOrd
    @test desOprCpu.cmpInf.frqPhz == oprMemCpu.cmpInf.frqPhz
    @test desOprCpu.cmpInf.adjMod == oprMemCpu.cmpInf.adjMod
    @test typeof(desOprCpu.cmpInf.bckEnd) == typeof(oprMemCpu.cmpInf.bckEnd)
    @test desOprCpu.trgVol.cel == oprMemCpu.trgVol.cel
    @test desOprCpu.trgVol.scl == oprMemCpu.trgVol.scl
    @test desOprCpu.trgVol.org == oprMemCpu.trgVol.org
    @test all(desOprCpu.trgVol.grd .== oprMemCpu.trgVol.grd)
    @test desOprCpu.srcVol.cel == oprMemCpu.srcVol.cel
    @test desOprCpu.srcVol.scl == oprMemCpu.srcVol.scl
    @test desOprCpu.srcVol.org == oprMemCpu.srcVol.org
    @test all(desOprCpu.srcVol.grd .== oprMemCpu.srcVol.grd)
    @test desOprCpu.mixInf.minScl == oprMemCpu.mixInf.minScl
    @test desOprCpu.mixInf.trgDiv == oprMemCpu.mixInf.trgDiv
    @test desOprCpu.mixInf.srcDiv == oprMemCpu.mixInf.srcDiv
    @test desOprCpu.mixInf.trgCel == oprMemCpu.mixInf.trgCel
    @test desOprCpu.mixInf.srcCel == oprMemCpu.mixInf.srcCel
    @test desOprCpu.mixInf.trgPar == oprMemCpu.mixInf.trgPar
    @test desOprCpu.mixInf.srcPar == oprMemCpu.mixInf.srcPar
    @test desOprCpu.egoFur == oprMemCpu.egoFur
    @test desOprCpu.phzInf == oprMemCpu.phzInf

    # Verify operator action produces the same result
    actVec = ones(ComplexF64, (4, 4, 4, 3))
    resOrig = egoOpr!(oprMemCpu, deepcopy(actVec))
    resDeser = egoOpr!(desOprCpu, deepcopy(actVec))
    @test resOrig ≈ resDeser

    rm(tmpFilCpu)

    # Test GPU-based GlaVacOprMem (only if CUDA is available)
    if CUDA.has_cuda()
        gpuOpt = GPUKerOpt()
        oprMemGpu = GlaVacOprMem(gpuOpt, trgVol, srcVol)

        tmpFilGpu = tempname()
        open(tmpFilGpu, "w") do io
            serialize(io, oprMemGpu)
        end

        desOprGpu = open(tmpFilGpu, "r") do io
            deserialize(io, GlaVacOprMem)
        end
        useGpu!(desOprGpu)

        # Verify all fields for GPU-based object
        @test desOprGpu.cmpInf.intOrd == oprMemGpu.cmpInf.intOrd
        @test desOprGpu.cmpInf.frqPhz == oprMemGpu.cmpInf.frqPhz
        @test desOprGpu.cmpInf.adjMod == oprMemGpu.cmpInf.adjMod
        @test typeof(desOprGpu.cmpInf.bckEnd) == typeof(oprMemGpu.cmpInf.bckEnd)
        @test desOprGpu.trgVol.cel == oprMemGpu.trgVol.cel
        @test desOprGpu.trgVol.scl == oprMemGpu.trgVol.scl
        @test desOprGpu.trgVol.org == oprMemGpu.trgVol.org
        @test all(desOprGpu.trgVol.grd .== oprMemGpu.trgVol.grd)
        @test desOprGpu.srcVol.cel == oprMemGpu.srcVol.cel
        @test desOprGpu.srcVol.scl == oprMemGpu.srcVol.scl
        @test desOprGpu.srcVol.org == oprMemGpu.srcVol.org
        @test all(desOprGpu.srcVol.grd .== oprMemGpu.srcVol.grd)
        @test desOprGpu.mixInf.minScl == oprMemGpu.mixInf.minScl
        @test desOprGpu.mixInf.trgDiv == oprMemGpu.mixInf.trgDiv
        @test desOprGpu.mixInf.srcDiv == oprMemGpu.mixInf.srcDiv
        @test desOprGpu.mixInf.trgCel == oprMemGpu.mixInf.trgCel
        @test desOprGpu.mixInf.srcCel == oprMemGpu.mixInf.srcCel
        @test desOprGpu.mixInf.trgPar == oprMemGpu.mixInf.trgPar
        @test desOprGpu.mixInf.srcPar == oprMemGpu.mixInf.srcPar
        @test desOprGpu.egoFur == oprMemGpu.egoFur
        @test desOprGpu.phzInf == oprMemGpu.phzInf

        # Verify operator action produces the same result
        actVecGpu = CUDA.ones(ComplexF64, (4, 4, 4, 3))
        resOrigGpu = egoOpr!(oprMemGpu, deepcopy(actVecGpu))
        resDeserGpu = egoOpr!(desOprGpu, deepcopy(actVecGpu))
        @test resOrigGpu ≈ resDeserGpu

        rm(tmpFilGpu)
    end
end
