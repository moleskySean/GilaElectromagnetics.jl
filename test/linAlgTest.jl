# glaLinAlg AbstractMatrix interface tests
# Uses _g0s(), _asys(), _invScts() (2,2,2) from tstHlp.jl for cheap loop iterations

@testset "similar" begin
    opr  = _g0s()
    n, m = size(opr, 1), size(opr, 2)

    s1 = similar(opr)
    @test s1 isa Array
    @test size(s1) == (n, m)
    @test eltype(s1) == ComplexF64

    s2 = similar(opr, Float64)
    @test eltype(s2) == Float64
    @test size(s2) == (n, m)

    s3 = similar(opr, (5, 7))
    @test size(s3) == (5, 7)
    @test eltype(s3) == ComplexF64

    s4 = similar(opr, Float32, (3, 4))
    @test size(s4) == (3, 4)
    @test eltype(s4) == Float32

    if CUDA.functional()
        gOpr = GlaOprVac(_vol2s; useGpu=true)
        @test similar(gOpr) isa CuArray
        @test similar(gOpr, Float32) isa CuArray{Float32}
        @test similar(gOpr, (5,)) isa CuArray
    end
end

@testset "getindex batched" begin
    # (2,2,2) operators → 24 columns each, manageable mul! count
    for opr in (_g0s(), _asys(), _invScts())
        D    = dnsMat(opr)
        n, m = size(opr, 1), size(opr, 2)

        # Full slice
        @test opr[:, :] ≈ D

        # Row slice
        @test opr[1:3, :] ≈ D[1:3, :]

        # Column slice
        @test opr[:, 1:3] ≈ D[:, 1:3]

        # Sub-block
        r = 1:min(4, n)
        c = 1:min(4, m)
        @test opr[r, c] ≈ D[r, c]
    end
end

@testset "scalar indexing warns" begin
    opr = _g0s()
    D   = dnsMat(opr)
    val = @test_logs (:warn, r"Scalar indexing") opr[1, 1]
    @test val ≈ D[1, 1]
end

@testset "setindex! / IndexStyle" begin
    opr = _g0s()
    @test_throws ArgumentError (opr[1, 1] = zero(ComplexF64))
    @test IndexStyle(typeof(opr)) == IndexCartesian()
end

#= The adjoint branch of getindex runs when more rows than columns are asked for.
Every operator whose adjoint! hands back a new wrapper instead of the argument has
to survive it, in the answer and in the state it leaves behind. =#
@testset "getindex adjoint branch" begin
    lnaScl = (1//16, 1//16, 1//16)
    lnaOrg = (0//1, 0//1, 0//1)
    # The masked union route, a block matrix, and a composite operator
    mskOpr = GlaOprVac(GlaVol((2,4,4), lnaScl, lnaOrg),
        GlaVol((2,2,2), lnaScl, (2//16, 0//1, 0//1)))
    mulOpr = MulRegGlaOprVac(reshape(
        [GlaOprVac(deepcopy(_selfMem4)), GlaOprVac(deepcopy(_extMem4))], 2, 1))
    cmpOpr = GlaCmpOprVac(GlaCmpVol([GlaVol((2,2,2), lnaScl, lnaOrg),
        GlaVol((2,2,2), lnaScl, (1//8, 0//1, 0//1))]))
    for opr in (mskOpr, mulOpr, cmpOpr)
        dns = dnsMat(opr)
        numRow, numCol = size(opr)
        @test numRow >= numCol
        inp = randn(ComplexF64, numCol)
        ref = opr * inp
        # More rows than columns, so the adjoint branch runs
        @test opr[1:5, 1:2] ≈ dns[1:5, 1:2]
        # The forward branch, for comparison
        @test opr[1:2, 1:5] ≈ dns[1:2, 1:5]
        # The operator is left in the state it was found in
        @test opr * inp ≈ ref
        @test !isadjoint(opr)
    end
end
