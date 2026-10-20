# glaLinAlg matrix-free interface tests
# Uses the (2,2,2) builders from tstHlp.jl, for cheap loop iterations

@testset "similar" begin
    opr = _g0s()

    s3 = similar(opr, (5, 7))
    @test size(s3) == (5, 7)
    @test eltype(s3) == ComplexF64

    s4 = similar(opr, Float32, (3, 4))
    @test size(s4) == (3, 4)
    @test eltype(s4) == Float32

    # The dense shape of the operator is not an allocation anyone asks for by accident
    @test_throws MethodError similar(opr)
    @test_throws MethodError similar(opr, Float64)
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

@testset "setindex!" begin
    opr = _g0s()
    @test_throws ArgumentError (opr[1, 1] = zero(ComplexF64))
end

#= The adjoint branch of getindex runs when more rows than columns are asked for.
Every operator whose adjoint! hands back a new wrapper instead of the argument has
to survive it, in the answer and in the state it leaves behind. =#
@testset "getindex adjoint branch" begin
    # The masked union route and a composite operator
    for opr in (_mskOpr, _cmp2s)
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

#= The operators are not AbstractArrays, so the shape methods the fallbacks used
to supply are their own, and the fallbacks that densified by matvec are gone. =#
@testset "matrix free shape" begin
    opr = _g0s()
    n, m = size(opr)

    @test axes(opr) == (Base.OneTo(n), Base.OneTo(m))
    @test axes(opr, 2) == Base.OneTo(m)
    @test CartesianIndices(opr) == CartesianIndices((n, m))
    @test length(collect(CartesianIndices(opr))) == n * m
    @test Matrix(opr) ≈ dnsMat(opr)

    # A linear index goes through CartesianIndices, and still warns on the way
    dns = dnsMat(opr)
    @test (@test_logs (:warn, r"Scalar indexing") opr[1]) ≈ dns[1]
    @test (@test_logs (:warn, r"Scalar indexing") opr[n + 2]) ≈ dns[n + 2]

    # The LinearAlgebra fallbacks that ran a full matvec per entry are gone
    @test_throws MethodError opnorm(opr)
    @test_throws MethodError tr(opr)
    @test_throws MethodError inv(opr)
end

@testset "backslash" begin
    for opr in (_g0s(), _invScts(), _scts(), _glas())
        inp = randn(ComplexF64, size(opr, 2))
        @test opr \ inp ≈ solve(opr, inp, slv(opr)) rtol=1e-6
    end

    sct = _scts()
    inp = randn(ComplexF64, size(sct, 2))
    @test sct * (sct \ inp) ≈ inp rtol=1e-6

    # ldiv! writes the same answer into the output it is handed
    out = similar(inp)
    @test ldiv!(out, sct, inp) === out
    @test out ≈ sct \ inp rtol=1e-6
end

#= The inverse actions the solve route through, against the operator they invert:
the factored ones are the easy place to invert the wrong half. =#
@testset "inverse action" begin
    _witAdj = GilaElectromagnetics._witAdj
    for opr in (_invScts(), _scts(), _glas())
        inp = randn(ComplexF64, size(opr, 2))
        @test opr * (opr \ inp) ≈ inp rtol=1e-6

        out = zeros(ComplexF64, length(inp))
        _witAdj(adjOpr -> ldiv!(out, adjOpr, inp), opr)
        @test adjoint(opr) * out ≈ inp rtol=1e-6
        @test !isadjoint(opr)
    end
end

#= A field has to come back as a field on the tiling it went in on: a solver
allocates its Krylov basis with a matrix shaped similar, which would strip it. =#
@testset "backslash field" begin
    sct = _scts()
    fld = GlaFld(randn(ComplexF64, size(sct, 2)), GlaCmpVol(_vol2s))
    sol = sct \ (sct * fld)
    @test sol isa GlaFld
    @test sol.cvol == fld.cvol
    @test sol.dat ≈ fld.dat rtol=1e-6

    out = zeros(ComplexF64, GlaCmpVol(_vol2s))
    @test ldiv!(out, sct, sct * fld) === out
    @test out.dat ≈ fld.dat rtol=1e-6

    # A field on another tiling is refused rather than solved on
    @test_throws ArgumentError sct \ zeros(ComplexF64, GlaCmpVol(_vol4))
end
