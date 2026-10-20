#= Every GPU code path once per backend: the device FFT and plans, the action and
partition kernels, adapt, fields, the susceptibility and serialization. Storage is
Float32: generation is Float64 either way and the kernels are generic in eltype, so
Float64 storage only appears where the code branches on it. =#
# pocl_jll before OpenCL activates its platform. Metal compiles in the world of its __init__,
# so loading it last lets it see the GPUCompiler methods OpenCL adds
using CUDA, pocl_jll, OpenCL, Metal
using KernelAbstractions: functional, supports_float64

# OpenCL.jl leaves functional unimplemented (missing); pocl always provides a device
const gpuBackends = filter(dev -> coalesce(functional(dev), true), [CUDABackend(), MetalBackend(), OpenCLBackend()])

gpuErr(cpu, gpu) = norm(Array(gpu) - cpu) / norm(cpu)

# A coarse and a fine volume half a wavelength apart: the finer of a pair is partitioned
const gpuCrs = GlaVol((2,2,2), scl16, stdOrg)
const gpuFin = GlaVol((4,4,4), stdScl, (1//2, 0//1, 0//1))
# Two touching regions at different scales, so both cross-scale blocks take the fine mesh route
const gpuCvl = refine(GlaCmpVol(GlaVol((4,2,2), scl16, stdOrg)),
    ((-1//16, 0//1, 0//1), (1//8, 1//8, 1//8)); factor=(2, 1, 1))

@testset "$(nameof(typeof(dev)))" for dev in gpuBackends
    @testset "vacuum operator" begin
        # Self and external pairs at two shapes, and the cross-scale pair both ways round
        for (trg, src) in ((_vol4, _vol4), (_trgV4, _vol4), (mkVol((8,2,10)), mkVol((8,2,10))),
            (mkVol((8,2,10); org=extOrg), mkVol((8,2,10))), (gpuFin, gpuCrs), (gpuCrs, gpuFin))
            oprCpu = GlaOprVac{Float32}(trg, src)
            oprGpu = GlaOprVac{Float32}(trg, src; dev)
            inp = rand(ComplexF32, size(oprCpu, 2))
            out = oprCpu * inp
            outGpu = oprGpu * adapt(dev, inp)
            @test get_backend(outGpu) isa typeof(dev)
            @test gpuErr(out, outGpu) < 1e-5
            toGpu = adapt(dev, oprCpu)
            @test get_backend(toGpu) isa typeof(dev)
            @test gpuErr(out, toGpu * adapt(dev, inp)) < 1e-5
            @test gpuErr(out, adapt(CPU(), oprGpu) * inp) < 1e-5
        end
        opr = GlaOprVac{Float32}(_vol2s; dev)
        @test get_backend(similar(opr, (5,))) isa typeof(dev)
        @test_logs (:warn, r"not on the operator's") opr * zeros(ComplexF32, size(opr, 2))
    end

    @testset "composite operator" begin
        oprCpu = GlaCmpOprVac{Float32}(gpuCvl)
        oprGpu = GlaCmpOprVac{Float32}(gpuCvl; dev)
        fldCpu = discretize!(zeros(ComplexF32, gpuCvl), tstDns)
        fldGpu = discretize!(zeros(ComplexF32, gpuCvl; dev), tstDns)
        out = (oprCpu * fldCpu).dat
        outGpu = oprGpu * fldGpu
        @test get_backend(outGpu) isa typeof(dev)
        @test gpuErr(out, outGpu.dat) < 1e-5
        @test gpuErr((asym(oprCpu) * fldCpu).dat, (asym(oprGpu) * fldGpu).dat) < 1e-5
        @test gpuErr(out, (adapt(dev, oprCpu) * fldGpu).dat) < 1e-5
        @test gpuErr(out, (adapt(CPU(), oprGpu) * fldCpu).dat) < 1e-5
    end

    @testset "scattering solve" begin
        chi = mkSus(Float32, (4,4,4))
        oprCpu = GlaOpr{Float32}(_vol4, chi)
        oprGpu = GlaOpr{Float32}(_vol4, adapt(dev, chi); dev)
        @test get_backend(sus(oprGpu).sus) isa typeof(dev)
        inp = rand(ComplexF32, size(oprCpu, 2))
        @test gpuErr(oprCpu * inp, oprGpu * adapt(dev, inp)) < 1e-4
    end

    @testset "Float64 storage" begin
        if supports_float64(dev)
            opr = InvSctOpr(GlaOprVac{Float64}(_vol4; dev), adapt(dev, _sus4))
            rhs = adapt(dev, normalize(rand(ComplexF64, size(opr, 2))))
            sol = solve(opr, copy(rhs), MixPrcRfn(Float32; relTol=1e-10))
            @test eltype(sol) == ComplexF64
            @test norm(opr * sol - rhs) < 1e-9
        else
            @test_throws ArgumentError GlaKerOpt{Float64}(dev)
        end
    end

    @testset "field" begin
        fldCpu = discretize!(zeros(ComplexF32, gpuCvl), tstDns)
        fldGpu = discretize!(zeros(ComplexF32, gpuCvl; dev), tstDns)
        @test get_backend(fldGpu) isa typeof(dev)
        @test norm(fldGpu) ≈ norm(fldCpu)
        @test dot(fldGpu, fldGpu) ≈ dot(fldCpu, fldCpu)
        # Broadcasting stays on the device
        fldTwo = 2 .* fldGpu
        @test fldTwo isa GlaFld && get_backend(fldTwo) isa typeof(dev)
        fldTwo .= fldGpu .+ fldGpu
        @test norm(fldTwo) ≈ 2 * norm(fldCpu)
        @test norm(axpy!(one(Float32), fldGpu, copy(fldGpu))) ≈ 2 * norm(fldCpu)
        @test maximum(abs, regrid(fldGpu) .- regrid(fldCpu)) < 1e-5
        @test adapt(CPU(), fldGpu) ≈ fldCpu
    end

    @testset "serialization from device memory" begin
        opr = GlaOprVac{Float32}(_vol4; dev)
        desOpr = serRnd(opr)
        @test get_backend(desOpr) isa CPU
        @test desOpr.mem.egoFur == Array.(opr.mem.egoFur)
        inp = rand(ComplexF32, size(opr, 2))
        @test gpuErr(desOpr * inp, opr * adapt(dev, inp)) < 1e-5
    end
end
