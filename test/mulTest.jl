# mul! and * across every operator type
# Uses the cheap (2,2,2) and (4,4,4) builders from tstHlp.jl

const mulExt = GlaOprVac{Float64}(GlaVol((2,2,2), stdScl, (4//32, 0//1, 0//1)), _vol2s)
# Two regions a wavelength apart, so the tiling has a gap
const mulGap = GlaCmpOprVac{Float64}(GlaCmpVol([_vol2s,
    GlaVol((2,2,2), stdScl, extOrg)]))

const mulOprs = (_g0s(), _asys(), SymGlaOprVac(_g0s()), mulExt, _invScts(),
    _scts(), _glas(), _cmp2s, mulGap)

const mulNaN = ComplexF64(NaN, NaN)
const mulScls = (0.0, 1.0, 2.0 + 1.0im)

# One generic method combines for every operator, so one operator covers it
@testset "mul!" begin
    opr = mulExt
    inp = randn(ComplexF64, size(opr, 2))
    ref = opr * inp
    out0 = randn(ComplexF64, size(opr, 1))
    for α in mulScls, β in mulScls
        # A β of zero has to overwrite, so poisoned memory proves out is unread
        out = iszero(β) ? fill!(similar(out0), mulNaN) : copy(out0)
        @test mul!(out, opr, inp, α, β) === out
        tst = iszero(β) ? α .* ref : α .* ref .+ β .* out0
        @test norm(out - tst) <= 1e-12 * max(norm(tst), norm(ref))
    end
    @test_throws ArgumentError mul!(similar(out0), opr, inp[1:end-1], 1.0, 0.0)
    @test_throws ArgumentError mul!(similar(out0, size(opr, 1) - 1), opr, inp, 1.0, 0.0)
    # The three argument form comes from the LinearAlgebra fallback
    out = fill!(similar(out0), mulNaN)
    @test mul!(out, opr, inp) === out
    @test out == ref
end

@testset "input form parity" begin
    for opr in mulOprs
        inp = randn(ComplexF64, size(opr, 2))
        ref = opr * inp
        @test ref isa Vector{ComplexF64}
        @test length(ref) == size(opr, 1)

        # A matrix goes column by column, and comes back a matrix
        mat = hcat(inp, randn(ComplexF64, size(opr, 2)))
        col2 = opr * mat[:, 2]
        out = opr * mat
        @test out isa Matrix{ComplexF64}
        @test size(out) == (size(opr, 1), 2)
        @test norm(out[:, 1] - ref) <= 1e-12 * norm(ref)
        @test norm(out[:, 2] - col2) <= 1e-12 * norm(col2)

        # The tensor form only exists where each side is a single volume
        if glaSze(opr, 2) isa NTuple{4, Int}
            ten = reshape(copy(inp), glaSze(opr, 2))
            outTen = opr * ten
            @test outTen isa Array{ComplexF64, 4}
            @test size(outTen) == glaSze(opr, 1)
            @test norm(vec(outTen) - ref) <= 1e-12 * norm(ref)
            mulTen = fill!(similar(outTen), mulNaN)
            @test mul!(mulTen, opr, ten, 1.0, 0.0) === mulTen
            @test norm(vec(mulTen) - ref) <= 1e-12 * norm(ref)
        end
    end
end

#= The kernel consumes what it is handed, so every path owes the caller a
defensive copy. The composite block loop is the one most likely to forget. =#
@testset "input untouched" begin
    for opr in (_g0s(), _invScts(), _scts(), _glas(), _cmp2s)
        inp = randn(ComplexF64, size(opr, 2))
        sav = copy(inp)
        opr * inp
        @test inp == sav
        mul!(fill!(similar(inp, size(opr, 1)), mulNaN), opr, inp, 2.0 + 1.0im, 0.0)
        @test inp == sav
        mul!(randn(ComplexF64, size(opr, 1)), opr, inp, 1.0, 1.0)
        @test inp == sav
        mat = hcat(inp, inp)
        matSav = copy(mat)
        opr * mat
        @test mat == matSav
    end
end

@testset "GlaFld mul!" begin
    for (opr, cvl) in ((_cmp2s, _cvl2s), (_g0s(), GlaCmpVol(_vol2s)))
        inp = discretize!(zerofield(Float64, cvl), tstDns)
        sav = copy(inp.dat)
        ref = opr * inp
        @test ref isa GlaFld
        out0 = GlaFld(randn(ComplexF64, length(ref)), ref.cvol)
        for α in mulScls, β in mulScls
            out = iszero(β) ? GlaFld(fill(mulNaN, length(ref)), ref.cvol) : copy(out0)
            @test mul!(out, opr, inp, α, β) === out
            tst = iszero(β) ? α .* ref.dat : α .* ref.dat .+ β .* out0.dat
            @test norm(out.dat - tst) <= 1e-12 * max(norm(tst), norm(ref.dat))
        end
        @test inp.dat == sav
    end
end

# Counts the matvecs a solve takes, so the allocation cap can be read in matvecs
mutable struct MulCntOpr{OT} <: AbstractGlaOpr{Float64}
    opr::OT
    cnt::Int
end
LinearAlgebra.mul!(out::AbstractVector{ComplexF64}, opr::MulCntOpr,
    inp::AbstractVector{ComplexF64}) = (opr.cnt += 1; mul!(out, opr.opr, inp))
Base.size(opr::MulCntOpr, dim::Int) = size(opr.opr, dim)

# A matvec that allocates nothing, so a solve's own allocation is all that is left
struct MulDiaOpr <: AbstractGlaOpr{Float64}
    dia::Vector{ComplexF64}
end
LinearAlgebra.mul!(out::AbstractVector{ComplexF64}, opr::MulDiaOpr,
    inp::AbstractVector{ComplexF64}) = (out .= opr.dia .* inp)
Base.size(opr::MulDiaOpr, ::Int) = length(opr.dia)

#= The solvers preallocate their work vectors, so a solve costs its matvecs plus
a fixed handful of buffers. Both caps are wide enough that ordinary noise cannot
reach them and narrow enough that a return to per-iteration temporaries does. =#
@testset "solve allocs" begin
    numDof = 1536
    dia = ComplexF64[1 + 0.9 * cospi(k / numDof) + 0.5im * sinpi(3k / numDof)
        for k in 1:numDof]
    rhs = randn(ComplexF64, numDof)
    vecByt = numDof * sizeof(ComplexF64)

    for (slv, vecCap) in ((BiCGStabSolver(), 20), (GMRESSolver(), 40))
        cnt = MulCntOpr(MulDiaOpr(dia), 0)
        sol = solve(cnt, rhs, slv)
        @test norm(dia .* sol - rhs) < 1e-6 * norm(rhs)
        # Enough matvecs that per-iteration temporaries would blow the cap
        @test cnt.cnt >= 10
        @test (@allocated solve(cnt, rhs, slv)) < vecCap * vecByt
    end

    # The same bound on the real thing, in units of the matvec it is made of
    sctOpr = _scts()
    invOpr = sctOpr.invSctOpr
    inp = randn(ComplexF64, size(sctOpr, 2))
    mulBuf = similar(inp)
    mul!(mulBuf, invOpr, inp)
    mulByt = @allocated mul!(mulBuf, invOpr, inp)
    cntOpr = MulCntOpr(invOpr, 0)
    solve(cntOpr, inp, sctOpr.slv)
    @test cntOpr.cnt >= 4
    sctOpr * inp
    @test (@allocated sctOpr * inp) <
        2 * cntOpr.cnt * mulByt + 64 * length(inp) * sizeof(ComplexF64)
end
