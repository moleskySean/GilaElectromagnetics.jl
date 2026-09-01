using Test, GilaElectromagnetics, CUDA

@testset "CPU-GPU Consistency Tests for Green Operator" begin
    if !CUDA.functional()
        # println("CUDA is not functional. Skipping GPU consistency tests.")
        return
    end
    volSizes = [
        (2, 2, 2),
        (4, 4, 4),
        (6, 6, 6),
        (8, 8, 8),
        (6, 4, 8),
        (8, 2, 10)
    ]
    sclArr = (1//32, 1//32, 1//32)
    orgSrc = (0//1, 0//1, 0//1)
    orgTrg = (1//1, 1//1, 1//1)  # Ensure non-overlapping target volume

    for volDim in volSizes
        # Self Green function
        volObj = GlaVol(volDim, sclArr, orgSrc)
        oprMemCpu = GlaVacOprMem(CPUKerOpt(), volObj)
        randVecCpu = rand(ComplexF64, oprMemCpu.srcVol.cel..., 3)
        outVecCpu = egoOpr!(oprMemCpu, deepcopy(randVecCpu))

        oprMemGpu = GlaVacOprMem(GPUKerOpt(), volObj)
        randVecGpu = CUDA.zeros(ComplexF64, oprMemGpu.srcVol.cel..., 3)
        copyto!(randVecGpu, randVecCpu)
        outVecGpu = egoOpr!(oprMemGpu, randVecGpu)

        # Transfer GPU result back to CPU for comparison
        outVecGpuCpu = Array(outVecGpu)

        # Compute the maximum difference between CPU and GPU results
        diff = maximum(abs.(outVecCpu .- outVecGpuCpu))
        @test diff < 1e-6

        # External Green function
        volTrg = GlaVol(volDim, sclArr, orgTrg)
        oprMemExtCpu = GlaVacOprMem(CPUKerOpt(), volTrg, volObj)
        randVecExtCpu = rand(ComplexF64, oprMemExtCpu.srcVol.cel..., 3)
        outVecExtCpu = egoOpr!(oprMemExtCpu, deepcopy(randVecExtCpu))

        oprMemExtGpu = GlaVacOprMem(GPUKerOpt(), volTrg, volObj)
        randVecExtGpu = CUDA.zeros(ComplexF64, oprMemExtGpu.srcVol.cel..., 3)
        copyto!(randVecExtGpu, randVecExtCpu)
        outVecExtGpu = egoOpr!(oprMemExtGpu, randVecExtGpu)

        # Transfer GPU result back to CPU for comparison
        outVecExtGpuCpu = Array(outVecExtGpu)

        # Compute the maximum difference between CPU and GPU results
        diffExt = maximum(abs.(outVecExtCpu .- outVecExtGpuCpu))
        @test diffExt < 1e-6
    end
end

@testset "CPU-GPU Conversion Tests" begin
    if !CUDA.functional()
        # println("CUDA is not functional. Skipping CPU-GPU conversion tests.")
        return
    end
    volSize = (4, 4, 4)
    sclArr = (1//32, 1//32, 1//32)
    org = (0//1, 0//1, 0//1)
    orgTrg = (1//1, 1//1, 1//1)  # Ensure non-overlapping target volume

    volObj = GlaVol(volSize, sclArr, org)
    oprMemCpu = GlaVacOprMem(CPUKerOpt(), volObj)
    oprMemGpu = GlaVacOprMem(GPUKerOpt(), volObj)

    cpuVec = ones(ComplexF64, oprMemCpu.srcVol.cel..., 3)
    gpuVec = CUDA.ones(ComplexF64, oprMemGpu.srcVol.cel..., 3)

    # Make sure the CPU and GPU computations are consistent
    cpuOut = egoOpr!(oprMemCpu, deepcopy(cpuVec))
    gpuOut = egoOpr!(oprMemGpu, deepcopy(gpuVec))
    @test cpuOut ≈ Array(gpuOut)

    # Test GPU to CPU conversion
    useCpu!(oprMemGpu)
    useCpu!(oprMemCpu) # Should be a no-op
    cpuOut = egoOpr!(oprMemCpu, deepcopy(cpuVec))
    gpuOut = egoOpr!(oprMemGpu, deepcopy(cpuVec))
    @test cpuOut ≈ gpuOut

    # Test CPU to GPU conversion
    useGpu!(oprMemCpu)
    useGpu!(oprMemGpu) # Should be a no-op
    cpuOut = egoOpr!(oprMemCpu, deepcopy(gpuVec))
    gpuOut = egoOpr!(oprMemGpu, deepcopy(gpuVec))
    @test cpuOut ≈ gpuOut

end
