# The CPU and GPU builds of an operator must agree, and adapt must move a mem
# between backends. Skipped backend by backend when its device is not functional.
using Test, GilaElectromagnetics

# widest difference between the two backends applied to the same vector, relative to the largest entry
function gpuDif(mkMem, dev, T, celSrc)
    innVec = rand(Complex{T}, celSrc..., 3)
    outCpu = egoOpr!(mkMem(GlaKerOpt{T}()), deepcopy(innVec))
    outGpu = egoOpr!(mkMem(GlaKerOpt{T}(dev)), adapt(dev, innVec))
    return maximum(abs, outCpu .- Array(outGpu)) / maximum(abs, outCpu)
end

@testset "CPU-GPU consistency" begin
    @testset "$(nameof(typeof(dev))) $T" for dev in gpuBackends, T in gpuPrcs(dev)
        for celDim in volSizes
            srcVol = mkVol(celDim)
            trgVol = mkVol(celDim; org=extOrg)
            @test gpuDif(opt -> GlaVacOprMem(opt, srcVol), dev, T, celDim) < gpuTol(T)
            @test gpuDif(opt -> GlaVacOprMem(opt, trgVol, srcVol), dev, T, celDim) < gpuTol(T)
        end
    end
end

@testset "CPU-GPU conversion" begin
    @testset "$(nameof(typeof(dev))) $T" for dev in gpuBackends, T in gpuPrcs(dev)
        vol = mkVol((4,4,4))
        memCpu = GlaVacOprMem(GlaKerOpt{T}(), vol)
        memGpu = GlaVacOprMem(GlaKerOpt{T}(dev), vol)
        cpuVec = ones(Complex{T}, vol.cel..., 3)
        gpuVec = adapt(dev, cpuVec)
        outCpu = egoOpr!(memCpu, deepcopy(cpuVec))
        tol = gpuTol(T) * maximum(abs, outCpu)
        @test maximum(abs, outCpu .- Array(egoOpr!(memGpu, deepcopy(gpuVec)))) < tol
        # adapt moves a mem to another backend as a new object; a round trip still applies
        movedGpu, movedCpu = adapt(dev, memCpu), adapt(CPU(), memGpu)
        @test get_backend(movedGpu) isa typeof(dev)
        @test get_backend(movedCpu) isa CPU
        @test maximum(abs, egoOpr!(movedCpu, deepcopy(cpuVec)) .-
            Array(egoOpr!(movedGpu, deepcopy(gpuVec)))) < tol
    end
end
