# Preconditioned solves, the solve log, and the operator interface solve needs

# A dense operator that remembers whether it was ever handed a non-finite
# vector: a Krylov basis column that is not a direction reaches it as an input
mutable struct FinOpr
    mat::Matrix{ComplexF64}
    nan::Bool
end
FinOpr(mat) = FinOpr(mat, false)
Base.size(opr::FinOpr, dim) = size(opr.mat, dim)
LinearAlgebra.mul!(out, opr::FinOpr, inp) =
    (opr.nan |= any(!isfinite, inp); mul!(out, opr.mat, inp))

const prcRng = MersenneTwister(0x7072636e)
const prcMat = I + 0.5 .* randn(prcRng, ComplexF64, 60, 60) ./ sqrt(60)
const prcRhs = randn(prcRng, ComplexF64, 60)

@testset "preCon dense" begin
    ref = prcMat \ prcRhs
    for pcn in (nothing, Diagonal(diag(prcMat)))
        gmr = GMRESSolver(20, 500, 0.0, 1e-10; preCon=pcn)
        bcg = BiCGStabSolver(500, 0.0, 1e-10; preCon=pcn)
        @test solve(prcMat, copy(prcRhs), gmr) ≈ ref rtol=1e-7
        @test solve(prcMat, copy(prcRhs), bcg) ≈ ref rtol=1e-7
    end
end

#= A nearby operator is the preconditioner anyone reaches for first, and it is
not admissible here: its ldiv! is an inner Krylov solve, so the map it applies
is chosen against the vector it is handed and differs from one application to
the next. Both solvers assume a fixed map and refuse it. =#
@testset "preCon Gila" begin
    opr = _invSct()
    pcn = InvSctOpr(_g0(), mkSus((4,4,4); val=0.4+0.05im))
    rhs = ones(ComplexF64, size(opr, 1))
    @test isvarying(pcn)
    @test !isvarying(Diagonal(ones(ComplexF64, size(opr, 1))))
    @test !isvarying(nothing)
    # The two operators whose inverse action is a matvec stay admissible
    @test !isvarying(SusOpr(mkVol((4,4,4)), _sus4))
    @test !isvarying(_sct())
    # An exact guess has no residual left to build a Krylov direction from
    lg = SlvLog()
    @test solve(prcMat, copy(prcRhs), GMRESSolver(); x0=prcMat \ prcRhs, log=lg) ≈ prcMat \ prcRhs
    @test lg.numItr == 0
    @test_throws ArgumentError solve(opr, copy(rhs), GMRESSolver(20, 200, 0.0, 1e-6; preCon=pcn))
    @test_throws ArgumentError solve(opr, copy(rhs), BiCGStabSolver(200, 0.0, 1e-6; preCon=pcn))
end

@testset "preCon side" begin
    ref = prcMat \ prcRhs
    dgn = Diagonal(diag(prcMat))
    @test_throws ArgumentError GMRESSolver(; side=:both)
    # Both sides reduce to the same algorithm with nothing to apply
    @test solve(prcMat, copy(prcRhs), GMRESSolver(20, 500, 0.0, 1e-10; side=:right)) ==
        solve(prcMat, copy(prcRhs), GMRESSolver(20, 500, 0.0, 1e-10))
    for sid in (:left, :right)
        @test solve(prcMat, copy(prcRhs), GMRESSolver(20, 500, 0.0, 1e-10; preCon=dgn, side=sid)) ≈
            ref rtol=1e-8
    end

    #= Left preconditioning stops on ‖M⁻¹(b - Ax)‖, which a cond(M) = 1e3
    preconditioner puts two orders of magnitude below the residual it is taken
    for. Right preconditioning iterates on b - Ax itself and stops where it
    says it does. =#
    ill = Diagonal(ComplexF64.(exp.(range(0, log(1e3), size(prcMat, 1)))))
    lft, rgt = SlvLog(), SlvLog()
    solve(prcMat, copy(prcRhs), GMRESSolver(20, 500, 0.0, 1e-10; preCon=ill, side=:left); log=lft)
    solve(prcMat, copy(prcRhs), GMRESSolver(20, 500, 0.0, 1e-10; preCon=ill, side=:right); log=rgt)
    @test lft.status == rgt.status == :converged
    @test rgt.resTru[end] < 1.5e-10
    @test lft.resTru[end] > 100 * rgt.resTru[end]
end

# Anchoring relTol to ‖inp‖ rather than to ‖res₀‖ is what makes this hold: a
# guess good to three digits otherwise buys three digits of tighter threshold
@testset "warm start" begin
    ref = prcMat \ prcRhs
    dgn = Diagonal(diag(prcMat))
    x0 = solve(prcMat, copy(prcRhs), GMRESSolver(20, 500, 0.0, 1e-3))
    for slv in (GMRESSolver(20, 500, 0.0, 1e-10),
        GMRESSolver(20, 500, 0.0, 1e-10; preCon=dgn),
        GMRESSolver(20, 500, 0.0, 1e-10; preCon=dgn, side=:right),
        BiCGStabSolver(500, 0.0, 1e-10),
        BiCGStabSolver(500, 0.0, 1e-10; preCon=dgn))
        cld, wrm = SlvLog(), SlvLog()
        solve(prcMat, copy(prcRhs), slv; log=cld)
        out = solve(prcMat, copy(prcRhs), slv; x0=copy(x0), log=wrm)
        @test wrm.status == :converged
        @test wrm.numItr < cld.numItr
        @test wrm.resTru[end] < 2e-10
        @test out ≈ ref rtol=1e-8
    end
end

@testset "SlvLog" begin
    @test SlvLog().status == :none
    for slv in (GMRESSolver(20, 500, 0.0, 1e-10), BiCGStabSolver(500, 0.0, 1e-10))
        lg = SlvLog()
        out = solve(prcMat, copy(prcRhs), slv; log=lg)
        @test lg.status == :converged
        @test lg.numItr > 0
        @test lg.resTru[end] ≈ norm(prcRhs - prcMat * out) / norm(prcRhs)
        @test lg.resTru[end] < 1e-9
    end
    # A cap below what the problem needs is reported, not passed off as a solution
    for slv in (GMRESSolver(20, 2, 0.0, 1e-12), BiCGStabSolver(2, 0.0, 1e-12))
        lg = SlvLog()
        @test_logs (:warn,) solve(prcMat, copy(prcRhs), slv; log=lg)
        @test lg.status == :maxiter
        @test lg.numItr == 2
        @test lg.resTru[end] > 1e-6
    end
end

@testset "happy breakdown" begin
    n = 30
    opr = FinOpr(Matrix(Diagonal(ComplexF64.(1:n))))
    rhs = zeros(ComplexF64, n)
    rhs[1:3] .= 1 # spans a three dimensional invariant subspace of opr
    lg = SlvLog()
    out = solve(opr, copy(rhs), GMRESSolver(n, 100, 0.0, 1e-12); log=lg)
    @test !opr.nan
    @test lg.status == :converged
    @test lg.numItr == 3
    @test lg.resTru[end] < 1e-14
    @test out ≈ opr.mat \ rhs
end

# A diagonal perturbed afresh on every application, so the map it applies is
# never twice the same. Whether it owns up to that is a field, which lets the
# same behaviour be run past the solvers declared and undeclared.
struct JtrPcn
    dgn::Vector{ComplexF64}
    jtr::Float64
    vry::Bool
    rng::MersenneTwister
end
JtrPcn(dgn, jtr, vry) = JtrPcn(dgn, jtr, vry, MersenneTwister(0x6a747472))
GilaElectromagnetics.isvarying(pcn::JtrPcn) = pcn.vry
LinearAlgebra.ldiv!(out, pcn::JtrPcn, inp) =
    (out .= inp ./ (pcn.dgn .* (1 .+ pcn.jtr .* randn(pcn.rng, length(inp)))); out)

@testset "flexible" begin
    flxRng = MersenneTwister(0x666c7865)
    n = 300
    mat = I + 0.9 .* randn(flxRng, ComplexF64, n, n) ./ sqrt(n)
    rhs = randn(flxRng, ComplexF64, n)
    dgn = ComplexF64.(diag(mat))
    mkPcn(vry) = JtrPcn(dgn, 0.1, vry) # 10% relative jitter

    @test_throws ArgumentError GMRESSolver(; flexible=true, side=:left)
    # Storing zₖ per step is what lets the applications disagree
    lg = SlvLog()
    out = solve(mat, copy(rhs),
        GMRESSolver(n, 2000, 0.0, 1e-10; preCon=mkPcn(true), flexible=true); log=lg)
    @test lg.status == :converged
    @test lg.resTru[end] < 2e-10
    @test out ≈ mat \ rhs rtol=1e-8

    # Restarting rebuilds Z from scratch each cycle, so the columns have to
    # line up with the basis the cycle is on and not the one before it
    rst = SlvLog()
    out = solve(mat, copy(rhs),
        GMRESSolver(20, 2000, 0.0, 1e-10; preCon=mkPcn(true), flexible=true); log=rst)
    @test rst.status == :converged
    @test rst.resTru[end] < 2e-10
    @test out ≈ mat \ rhs rtol=1e-8
    @test rst.numItr > 20 # several cycles, not one

    # Nothing else takes one
    @test_throws ArgumentError solve(mat, copy(rhs),
        GMRESSolver(n, 2000, 0.0, 1e-10; preCon=mkPcn(true)))
    @test_throws ArgumentError solve(mat, copy(rhs),
        BiCGStabSolver(2000, 0.0, 1e-10; preCon=mkPcn(true)))

    #= Undeclared, standard GMRES reports convergence on an answer with no
    correct digits: un-restarted the Arnoldi relation drifts the whole way, and
    the residual it minimises stops being the residual of anything. This is the
    number the trait exists to keep out of reach. =#
    und = SlvLog()
    solve(mat, copy(rhs), GMRESSolver(n, 2000, 0.0, 1e-10; preCon=mkPcn(false)); log=und)
    @test und.status == :converged
    @test und.resTru[end] > 1e-2

    #= Each restart re-forms inp - opr * out from the operator, discarding the
    drift of the cycle before it, so at the default depth the same
    preconditioner costs iterations and nothing else. =#
    shl = SlvLog()
    solve(mat, copy(rhs), GMRESSolver(20, 2000, 0.0, 1e-10; preCon=mkPcn(false)); log=shl)
    @test shl.status == :converged
    @test shl.resTru[end] < 2e-9
end

# A zero first row puts the range orthogonal to the shadow residual exactly, so
# BiCGStab's first denominator vanishes instead of merely getting small
@testset "breakdown" begin
    sng = copy(prcMat); sng[1, :] .= 0
    rhs = zeros(ComplexF64, size(sng, 1)); rhs[1] = 1
    lg = SlvLog()
    solve(sng, rhs, BiCGStabSolver(100, 0.0, 1e-10); log=lg)
    @test lg.status == :breakdown
    @test lg.numItr == 1
    @test lg.resTru[end] ≈ 1
end

@testset "log counters" begin
    dgn = Diagonal(diag(prcMat))
    # A cap below what the problem needs pins the iteration count exactly
    for (slv, pcn) in ((GMRESSolver(20, 7, 0.0, 1e-14), 0),
        (GMRESSolver(20, 7, 0.0, 1e-14; preCon=dgn), 1 + 7),
        (GMRESSolver(20, 7, 0.0, 1e-14; preCon=dgn, side=:right), 7 + 1))
        lg = SlvLog()
        @test_logs (:warn,) solve(prcMat, copy(prcRhs), slv; log=lg)
        @test lg.numItr == 7
        @test lg.oprApp == 7 # one Arnoldi step apiece, no restart at depth 20
        @test lg.pcnApp == pcn
        @test lg.oprAppLog == 1 # the exit residual, and nothing else
        @test lg.oprAppPcn == 0 # schema only, for a harness to fill
        @test length(lg.resRec) == 8
        @test length(lg.resTru) == 1
        @test isempty(lg.prm.rstIdx)
        @test length(lg.hss) == 1
    end

    # Two restarts inside twelve iterations, so the extra applications show up
    for (slv, opr, pcn) in ((GMRESSolver(5, 12, 0.0, 1e-14), 14, 0),
        (GMRESSolver(5, 12, 0.0, 1e-14; preCon=dgn), 14, 1 + 12 + 2),
        (GMRESSolver(5, 12, 0.0, 1e-14; preCon=dgn, side=:right), 14, 12 + 3))
        lg = SlvLog()
        @test_logs (:warn,) solve(prcMat, copy(prcRhs), slv; log=lg)
        @test lg.prm.rstIdx == [5, 10]
        @test lg.oprApp == opr # twelve Arnoldi steps and two re-formed residuals
        @test lg.pcnApp == pcn
        @test length(lg.hss) == 3 # two restarts and the final update
        @test length(lg.resTru) == 3 # a sample per restart, then the exit
    end

    # BiCGStab applies both twice per iteration and tracks two residuals
    lg = SlvLog()
    @test_logs (:warn,) solve(prcMat, copy(prcRhs), BiCGStabSolver(4, 0.0, 1e-14;
        preCon=Diagonal(diag(prcMat))); log=lg)
    @test lg.oprApp == 8
    @test lg.pcnApp == 8
    @test lg.oprAppLog == 1
    @test length(lg.resRec) == 8
    @test length(lg.ρ) == length(lg.α) == length(lg.ω) == length(lg.dnm) == 4
    @test isempty(lg.hss) # a Hessenberg is GMRES's to build
    @test lg.prm.side == :right

    lg = SlvLog(; truSmp=:never, hssSmp=:first)
    solve(prcMat, copy(prcRhs), GMRESSolver(5, 500, 0.0, 1e-10); log=lg)
    @test length(lg.resTru) == 1 # the exit value is not a policy
    @test length(lg.hss) == 1
    @test length(lg.prm.rstIdx) > 1
    @test_throws ArgumentError SlvLog(; truSmp=:always)
    @test_throws ArgumentError SlvLog(; hssSmp=:every)

    # A zero right hand side is solved exactly and has no scale to divide by
    for slv in (GMRESSolver(), BiCGStabSolver())
        lg = SlvLog()
        @test iszero(solve(prcMat, zeros(ComplexF64, size(prcMat, 1)), slv; log=lg))
        @test lg.resRec == [0.0] && lg.resTru == [0.0]
    end

    @test SlvLog().prm == NamedTuple()
    prm = SlvLog()
    solve(prcMat, copy(prcRhs), GMRESSolver(20, 500, 1e-12, 1e-10; preCon=dgn); log=prm)
    @test prm.prm.rstItr == 20 && prm.prm.maxItr == 500
    @test prm.prm.absTol == max(1e-12, 1e-10 * norm(dgn \ prcRhs)) # after the rescaling
    @test prm.prm.relTol == 1e-10
    @test prm.prm.side == :left && prm.prm.flexible == false
    @test prm.prm.elmTyp == ComplexF64 && prm.prm.arrTyp == Array
end

#= The recursive residual is the one the solver stops on. On the right that is
inp - opr * out itself, so the two curves are one curve; on the left it lives in
preCon⁻¹'s image and sits cond(preCon) below the residual it is taken for. =#
@testset "resRec resTru" begin
    ill = Diagonal(ComplexF64.(exp.(range(0, log(1e3), size(prcMat, 1)))))
    rgt = SlvLog()
    solve(prcMat, copy(prcRhs), GMRESSolver(20, 500, 0.0, 1e-10; preCon=ill, side=:right);
        log=rgt)
    @test rgt.resRec[end] ≈ rgt.resTru[end] rtol=1e-6
    # A restart sample is the true residual of the iterate the cycle ended on
    for (i, itr) in enumerate(rgt.prm.rstIdx)
        @test rgt.resRec[itr + 1] ≈ rgt.resTru[i] rtol=1e-6
    end

    lft = SlvLog()
    solve(prcMat, copy(prcRhs), GMRESSolver(20, 500, 0.0, 1e-10; preCon=ill, side=:left);
        log=lft)
    @test lft.resTru[end] > 100 * lft.resRec[end]
    @test all(lft.resTru[1:end-1] .> 50 .* lft.resRec[lft.prm.rstIdx .+ 1])
end

# lstSqrHss triangularizes the Hessenberg in place, which would leave the R
# factor's diagonal where the Ritz values should be
@testset "hss capture" begin
    n = 10
    hssRng = MersenneTwister(0x68737363)
    mat = I + randn(hssRng, ComplexF64, n, n) ./ sqrt(n)
    rhs = randn(hssRng, ComplexF64, n)
    lg = SlvLog()
    solve(mat, copy(rhs), GMRESSolver(n, n, 0.0, 1e-14); log=lg)
    hss = only(lg.hss)
    @test size(hss) == (n + 1, n)
    @test all(i -> !iszero(hss[i + 1, i]), 1:n-1) # a Givens pass zeroes the subdiagonal
    ritz = eigvals(hss[1:n, 1:n])
    @test sort(ritz; by=reim) ≈ sort(eigvals(Matrix(mat)); by=reim) rtol=1e-6

    # Every cycle's matrix, not only the first, is the Hessenberg it is read as
    lg = SlvLog()
    @test_logs (:warn,) solve(prcMat, copy(prcRhs), GMRESSolver(5, 12, 0.0, 1e-14); log=lg)
    @test length(lg.hss) == 3
    @test all(h -> iszero(tril(h, -2)), lg.hss)
end

# Every refinement step runs its own inner solve, and each leaves a record
@testset "nested log" begin
    lg = SlvLog()
    out = solve(_invSct(), ComplexF64.(range(0.1, 1.0, size(_invSct(), 1))),
        MixPrcRfn(Float32; relTol=1e-10); log=lg)
    @test lg.status == :converged
    @test lg.numItr > 1
    @test length(lg.inner) == lg.numItr
    @test all(inn -> inn.status == :converged, lg.inner)
    @test all(inn -> inn.numItr > 0, lg.inner)
    @test all(inn -> inn.prm.elmTyp == ComplexF32, lg.inner)
    @test length(lg.resRec) == lg.numItr + 1 # the starting residual, then one per step
    @test lg.resTru[end] < 1e-10
    @test lg.oprApp == lg.numItr # one high precision matvec per step
end
