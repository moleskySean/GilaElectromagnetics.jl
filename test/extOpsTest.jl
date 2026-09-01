# glaMatFreExtOps tests
# Uses _g0s(), _invScts(), _scts(), _glas() (2,2,2) from tstHlp.jl for cheap loops
import GilaElectromagnetics.GilaOperators: invMul!, invMulAdj!

@testset "FunctionOperator" begin
    for opr in (_g0s(), _invScts(), _scts(), _glas())
        fo = FunctionOperator(opr)
        n  = size(opr, 2)
        m  = size(opr, 1)
        v  = rand(ComplexF64, n)
        w  = rand(ComplexF64, m)

        # Forward: mul! via FunctionOperator
        w_fwd = similar(w)
        mul!(w_fwd, fo, v)
        @test w_fwd ≈ opr * v

        # Adjoint
        w_adj = similar(v)
        mul!(w_adj, fo', w)
        @test w_adj ≈ adjoint(opr) * w
    end
end

@testset "LinearOperator" begin
    for opr in (_g0s(), _invScts(), _scts(), _glas())
        lo   = LinearOperator(opr)
        n, m = size(opr, 2), size(opr, 1)
        v    = rand(ComplexF64, n)
        w    = rand(ComplexF64, m)

        w_fwd = similar(w)
        mul!(w_fwd, lo, v)
        @test w_fwd ≈ opr * v

        @test size(lo) == (m, n)
        @test eltype(lo) == ComplexF64
        @test !lo.symmetric
        @test !lo.hermitian

        # Adjoint
        w_adj = similar(v)
        mul!(w_adj, lo', w)
        @test w_adj ≈ adjoint(opr) * w
    end
end

@testset "LinearMap" begin
    for opr in (_g0s(), _invScts(), _glas())
        lm   = LinearMap(opr)
        n, m = size(opr, 2), size(opr, 1)
        v    = rand(ComplexF64, n)
        w    = rand(ComplexF64, m)
        @test lm * v ≈ opr * v
        @test lm' * w ≈ adjoint(opr) * w
        @test Matrix(LinearMap(opr)) ≈ dnsMat(opr)
    end
end

@testset "invMul! and invMulAdj!" begin
    v = rand(ComplexF64, size(_invScts(), 2))

    # SctOpr: invMul! solves sctOpr * w = v → checks opr*w ≈ v
    sct = _scts()
    w   = zeros(ComplexF64, length(v))
    invMul!(w, sct, v, one(ComplexF64), zero(ComplexF64))
    @test sct * w ≈ v

    # Operator restored to non-adjoint after invMulAdj!
    sct2 = _scts()
    w2   = zeros(ComplexF64, length(v))
    invMulAdj!(w2, sct2, v, one(ComplexF64), zero(ComplexF64))
    @test !isadjoint(sct2)

    # GlaOpr: invMul! should not leave operator in adjoint state
    gla = _glas()
    w3  = zeros(ComplexF64, length(v))
    invMul!(w3, gla, v, one(ComplexF64), zero(ComplexF64))
    @test !isadjoint(gla)
end
