# glaMatFreExtOps tests
# Uses the (2,2,2) builders from tstHlp.jl
import GilaElectromagnetics: _witAdj
import VoxelShapes

# Each wrapper is a closure over mul! and adjoint!, so one operator apiece covers it
@testset "matrix-free wrappers" begin
    for (wrp, opr) in ((FunctionOperator, _g0s()), (LinearOperator, _invScts()), (LinearMap, _glas()))
        wrpOpr = wrp(opr)
        n, m = size(opr, 2), size(opr, 1)
        inp, out = rand(ComplexF64, n), rand(ComplexF64, m)
        @test size(wrpOpr) == (m, n)
        @test eltype(wrpOpr) == ComplexF64

        fwd = similar(out)
        mul!(fwd, wrpOpr, inp)
        @test fwd ≈ opr * inp

        adj = similar(inp)
        mul!(adj, wrpOpr', out)
        @test adj ≈ adjoint(opr) * out
    end
    # Per-wrapper extras
    lnOpr = LinearOperator(_g0s())
    @test !lnOpr.symmetric
    @test !lnOpr.hermitian
    @test Matrix(LinearMap(_g0s())) ≈ dnsMat(_g0s())
end

@testset "ldiv! and the adjoint solve" begin
    v = rand(ComplexF64, size(_invScts(), 2))

    # SctOpr: ldiv! solves sctOpr * w = v → checks opr*w ≈ v
    sct = _scts()
    w   = zeros(ComplexF64, length(v))
    ldiv!(w, sct, v)
    @test sct * w ≈ v

    # Neither inverse path leaves its operator adjoint
    sct2 = _scts()
    _witAdj(adjOpr -> ldiv!(zeros(ComplexF64, length(v)), adjOpr, v), sct2)
    @test !isadjoint(sct2)

    gla = _glas()
    ldiv!(zeros(ComplexF64, length(v)), gla, v)
    @test !isadjoint(gla)
end

@testset "VoxelShapes geometry" begin
    ctr = (1//2, 0//1, -1//4)
    reg = VoxelShapes.Region((16, 16, 16), (1//4, 1//4, 1//4), ctr)
    @test GlaVol(reg) == GlaVol((16, 16, 16), (1//4, 1//4, 1//4), ctr)

    shp = VoxelShapes.FillableSphere(Float64.(ctr), 0.6, 2.0)
    cvol = GlaCmpVol(VoxelShapes.refine(reg, shp; factor=2))
    @test nregions(cvol) > 1
    @test SusOpr(cvol, VoxelShapes.Geometry([shp], 0.0)).sus == SusOpr(cvol, pos -> norm(pos .- ctr) < 0.6 ? 2.0 : 0.0).sus
end
