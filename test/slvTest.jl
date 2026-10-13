# GilaSolvers tests
import GilaElectromagnetics.GilaSolvers: slvPrm

@testset "Solver constructors" begin
    gmr = GMRESSolver()
    @test isnothing(gmr.rstItr)
    @test isnothing(gmr.maxItr)
    @test isnothing(gmr.absTol)
    @test isnothing(gmr.relTol)

    bcg = BiCGStabSolver()
    @test isnothing(bcg.maxItr)
    @test isnothing(bcg.absTol)
    @test isnothing(bcg.relTol)

    rfn = MixPrcRfn(Float32)
    @test rfn isa MixPrcRfn{Float32}
    @test rfn.innSlv isa GMRESSolver
    @test rfn.cch[] == (nothing, nothing)
    @test isnothing(rfn.maxItr)
    @test isnothing(rfn.absTol)
    @test isnothing(rfn.relTol)
    @test MixedPrecisionRefinement === MixPrcRfn
    @test MixPrcRfn(Float32; innSlv=BiCGStabSolver()).innSlv isa BiCGStabSolver

    gcr = GCRODRSolver()
    @test isnothing(gcr.rstItr)
    @test gcr.rcyDim == 10 && gcr.lrgFrc == 0.0 && gcr.side == :left
    @test RcySpc().bas === RcySpc().img === nothing
    @test_throws ArgumentError GCRODRSolver(; side=:both)
    @test_throws ArgumentError GCRODRSolver(; lrgFrc=1.5)
    @test_throws ArgumentError GCRODRSolver(; rcyDim=-1)
    @test_throws ArgumentError GCRODRSolver(5; rcyDim=5) # no Arnoldi step left in a cycle
end

@testset "slvPrm defaults" begin
    n = 100
    v = zeros(ComplexF64, n)

    @test slvPrm(GMRESSolver(), v) == (rstItr=min(20, n), maxItr=max(5000, n),
        absTol=zero(Float64), relTol=sqrt(eps(Float64)))
    @test slvPrm(BiCGStabSolver(), v) ==
        (maxItr=n, absTol=zero(Float64), relTol=sqrt(eps(Float64)))
    @test slvPrm(MixPrcRfn(Float32), v) ==
        (maxItr=20, absTol=zero(Float64), relTol=sqrt(eps(Float64)))

    # Set fields are passed through
    @test slvPrm(GMRESSolver(5, 1000, 1e-10, 1e-8), v) ==
        (rstItr=5, maxItr=1000, absTol=1e-10, relTol=1e-8)
    @test slvPrm(BiCGStabSolver(500, 1e-9, 1e-7), v) ==
        (maxItr=500, absTol=1e-9, relTol=1e-7)
    @test slvPrm(MixPrcRfn(Float32; maxItr=7, absTol=1e-9, relTol=1e-10), v) ==
        (maxItr=7, absTol=1e-9, relTol=1e-10)

    @test slvPrm(GCRODRSolver(), v) == (rstItr=min(20, n), maxItr=max(5000, n),
        absTol=zero(Float64), relTol=sqrt(eps(Float64)), rcyDim=10)
    # A right hand side too short for the asked cycle clamps rcyDim, not rstItr
    @test slvPrm(GCRODRSolver(; rcyDim=10), zeros(ComplexF64, 6)).rcyDim == 5
end

#= A solver instance carries no problem: the settings used to be written back
into the struct, and MixPrcRfn's low precision copy used to be kept across a
change of operator, so a reused solver answered the second problem with the
first one's tolerances or the first one's Green function. =#
@testset "solver reuse" begin
    opr32 = InvSctOpr(GlaOprVac{Float32}(_g0s()), ComplexF32.(_sus2s))
    opr64 = _invSct()
    rhs32 = normalize(ones(ComplexF32, size(opr32, 1)))
    rhs64 = normalize(ones(ComplexF64, size(opr64, 1)))
    for mkSlv in (BiCGStabSolver, GMRESSolver)
        slv = mkSlv()
        solve(opr32, copy(rhs32), slv)
        @test solve(opr64, copy(rhs64), slv) ≈ solve(opr64, copy(rhs64), mkSlv()) rtol=1e-12
    end

    oprAsy = InvSctOpr(_asy(), _sus4) # Same size and precision, different kernel
    rfn = MixPrcRfn(Float32)
    solve(opr64, copy(rhs64), rfn)
    @test solve(oprAsy, copy(rhs64), rfn) ≈
        solve(oprAsy, copy(rhs64), MixPrcRfn(Float32)) rtol=1e-12
end

@testset "MixPrcRfn precision guards" begin
    opr32 = GlaOprVac{Float32}(_g0s()) # Converted, so no integration cost
    n = size(opr32, 2)
    # Refining fp32 with fp32 throws; an operator/right hand side mismatch has no method
    @test_throws ArgumentError solve(opr32, zeros(ComplexF32, n), MixPrcRfn(Float32))
    @test_throws MethodError solve(opr32, zeros(ComplexF64, n), MixPrcRfn(Float32))
end

# G₀ is indefinite where (I - XG₀) is not, so both are worth a residual
@testset "solve residuals" begin
    for opr in (_g0(), _invSct())
        rhs = ones(ComplexF64, size(opr, 1))
        for slv in (BiCGStabSolver(), GMRESSolver())
            sol = solve(opr, rhs, slv)
            @test size(sol) == (size(opr, 2),)
            @test norm(opr * sol - rhs) < sqrt(eps(Float64)) * norm(rhs)
        end
    end
end

@testset "lstSqrHss" begin
    # Build a random upper-Hessenberg (n+1)×n matrix and rhs = β*e₁
    n = 5
    H = zeros(ComplexF64, n+1, n)
    for j in 1:n, i in 1:(j+1)
        H[i, j] = randn(ComplexF64)
    end
    β = randn(ComplexF64)
    rhs = zeros(ComplexF64, n+1)
    rhs[1] = β

    H_ref  = copy(H)
    rhs_ref = copy(rhs)
    GilaElectromagnetics.GilaSolvers.lstSqrHss(H, rhs)
    y_test = rhs[1:n]

    # Reference: dense least squares via QR
    y_ref = H_ref \ rhs_ref
    @test y_test ≈ y_ref[1:n] atol=1e-10
end

const gcrRng = MersenneTwister(0x67637264)
const gcrMat = I + 0.9 .* randn(gcrRng, ComplexF64, 80, 80) ./ sqrt(80)
const gcrRhs = normalize(randn(gcrRng, ComplexF64, 80))
const gcrRef = gcrMat \ gcrRhs
# Two eigenvalues a few hundred times smaller than the rest: what deflation is for
const gcrDgn = Diagonal(ComplexF64[1e-2, 2e-2, 3:60...])
const gcrOne = normalize(ones(ComplexF64, 60))

# rcyDim = 0 leaves nothing to recycle, so the two must agree step for step
@testset "GCRODR rcyDim 0" begin
    for pcn in (nothing, Diagonal(diag(gcrMat))), sid in (:left, :right)
        gmr, gcr = SlvLog(), SlvLog()
        og = solve(gcrMat, copy(gcrRhs), GMRESSolver(10, 2000, 0.0, 1e-10; preCon=pcn, side=sid); log=gmr)
        oc = solve(gcrMat, copy(gcrRhs), GCRODRSolver(10, 2000, 0.0, 1e-10; preCon=pcn, side=sid, rcyDim=0); log=gcr)
        @test og == oc
        @test gmr.resRec == gcr.resRec
        @test gmr.hss == gcr.hss
        @test gmr.prm.rstIdx == gcr.prm.rstIdx
        @test gcr.prm.rcyDim == 0 && gcr.prm.rcyApp == 0
    end
end

#= C is rebuilt from the previous C forever, so a drift in either half of the
invariant compounds cycle after cycle rather than being discarded at a restart. =#
@testset "GCRODR invariant" begin
    for mx in (6, 12, 18, 24)
        rcy, lg = RcySpc(), SlvLog()
        @test_logs (:warn,) solve(gcrDgn, copy(gcrOne), GCRODRSolver(8, mx, 0.0, 1e-12; rcyDim=4); rcy, log=lg)
        @test size(rcy.bas) == (60, 4)
        @test norm(gcrDgn * rcy.bas - rcy.img) < 1e-13 * norm(rcy.img)
        @test length(lg.rcyErr) == length(lg.hss) # one rebuild per cycle
        @test all(<(1e-13), lg.rcyErr)
    end
end

# Deflating the small end is the whole point, and restarted GMRES is the baseline
@testset "GCRODR deflation" begin
    gmr, gcr = SlvLog(), SlvLog()
    solve(gcrDgn, copy(gcrOne), GMRESSolver(10, 3000, 0.0, 1e-10); log=gmr)
    out = solve(gcrDgn, copy(gcrOne), GCRODRSolver(10, 3000, 0.0, 1e-10; rcyDim=3); log=gcr)
    @test gmr.status == gcr.status == :converged
    @test gcr.numItr < gmr.numItr
    @test out ≈ gcrDgn \ gcrOne rtol=1e-9
end

@testset "GCRODR carry" begin
    prtRng = MersenneTwister(0x70727462)
    prt = Diagonal(diag(gcrDgn) .* (1 .+ 1e-4 .* randn(prtRng, 60)))
    rhs = normalize(randn(prtRng, ComplexF64, 60))
    rcy = RcySpc()
    solve(gcrDgn, copy(gcrOne), GCRODRSolver(10, 3000, 0.0, 1e-10; rcyDim=3); rcy)
    cld, wrm = SlvLog(), SlvLog()
    solve(prt, copy(rhs), GCRODRSolver(10, 3000, 0.0, 1e-10; rcyDim=3); log=cld)
    out = solve(prt, copy(rhs), GCRODRSolver(10, 3000, 0.0, 1e-10; rcyDim=3); rcy, log=wrm)
    @test wrm.numItr < cld.numItr
    @test out ≈ prt \ rhs rtol=1e-9
end

#= A carried C that no longer matches A * U deflates directions the operator does
not have, which converges quickly to a wrong answer rather than failing. =#
@testset "GCRODR probe" begin
    rcy = RcySpc()
    solve(gcrMat, copy(gcrRhs), GCRODRSolver(10, 400, 0.0, 1e-10; rcyDim=6); rcy)
    bad = RcySpc(Array(rcy.bas), 3 .* Array(rcy.img)) # invariant off by a factor of three
    lg = SlvLog()
    out = solve(gcrMat, copy(gcrRhs), GCRODRSolver(10, 400, 0.0, 1e-10; rcyDim=6); rcy=bad, log=lg)
    @test lg.prm.rcyApp == 7 # the probe, then six applications rebuilding C
    @test norm(gcrMat * bad.bas - bad.img) < 1e-13 * norm(bad.img)
    @test out ≈ gcrRef rtol=1e-8

    # A subspace of the wrong shape is a cold start, not an error and not a rebuild
    sml = RcySpc()
    solve(gcrMat[1:40, 1:40], normalize(gcrRhs[1:40]), GCRODRSolver(10, 400, 0.0, 1e-10; rcyDim=4); rcy=sml)
    lg = SlvLog()
    out = solve(gcrMat, copy(gcrRhs), GCRODRSolver(10, 400, 0.0, 1e-10; rcyDim=6); rcy=sml, log=lg)
    @test lg.prm.rcyApp == 0
    @test out ≈ gcrRef rtol=1e-8
end

#= fp32 cannot hold the invariant to a tolerance anyone would ask of it, so the
probe floors at eps^(3/4). Tied to relTol alone this subspace is rejected and
rebuilt on every solve, which costs rcyDim applications and recycles nothing. =#
@testset "GCRODR probe fp32" begin
    rng = MersenneTwister(0x66703332)
    mat = ComplexF32.(I + 0.8 .* randn(rng, ComplexF64, 200, 200) ./ sqrt(200))
    rhs = ComplexF32.(normalize(randn(rng, ComplexF64, 200)))
    rcy = RcySpc()
    solve(mat, copy(rhs), GCRODRSolver(20, 2000, 0.0f0, 1f-6; rcyDim=8); rcy)
    zRnd = randn(MersenneTwister(0x7a726e64), ComplexF32, 8)
    @test norm(mat * (rcy.bas * zRnd) - rcy.img * zRnd) > 1e-6 * norm(rcy.img * zRnd)
    lg = SlvLog()
    out = solve(mat, copy(rhs), GCRODRSolver(20, 2000, 0.0f0, 1f-6; rcyDim=8); rcy, log=lg)
    @test lg.prm.rcyApp == 1 # the probe alone
    # fp32 overshoots a 1e-6 stopping test by a tenth, GMRES by rather more
    @test lg.resTru[end] < 2e-6
    @test out ≈ ComplexF64.(mat) \ ComplexF64.(rhs) rtol=1e-4
end

# Deflation that has drifted converges fast onto the wrong vector, so the answer
# is checked against the factorization and not only against the residual
@testset "GCRODR residual" begin
    dgn = Diagonal(diag(gcrMat))
    for (rcyDim, lrgFrc, pcn, sid) in ((0, 0.0, nothing, :left), (6, 0.0, nothing, :left),
        (6, 0.5, nothing, :left), (6, 1.0, nothing, :left),
        (6, 0.0, dgn, :right), (6, 0.0, dgn, :left))
        lg = SlvLog()
        out = solve(gcrMat, copy(gcrRhs),
            GCRODRSolver(10, 2000, 0.0, 1e-10; rcyDim, lrgFrc, preCon=pcn, side=sid); log=lg)
        @test lg.status == :converged
        @test length(lg.resRec) == lg.numItr + 1
        @test lg.resTru[end] < 1e-10
        @test out ≈ gcrRef rtol=1e-8
    end
end

#= The inner operator is the same in every refinement step and only the right
hand side moves, so the probe passes and the subspace is never rebuilt. =#
@testset "GCRODR refinement" begin
    opr = InvSctOpr(_g0(), mkSus((4,4,4); val=6.0+0.5im))
    rhs = ComplexF64.(range(0.1, 1.0, size(opr, 1)))
    gmr, gcr = SlvLog(), SlvLog()
    solve(opr, copy(rhs), MixPrcRfn(Float32; relTol=1e-10); log=gmr)
    out = solve(opr, copy(rhs),
        MixPrcRfn(Float32; innSlv=GCRODRSolver(; rcyDim=10), relTol=1e-10); log=gcr)
    @test gcr.status == :converged
    @test gcr.resTru[end] < 1e-10
    @test [inn.prm.rcyApp for inn in gcr.inner] == [0; fill(1, length(gcr.inner) - 1)]
    @test all(inn -> inn.numItr <= gcr.inner[1].numItr, gcr.inner)
    # Refinement steps cost GMRES more as they go and GCRO-DR does not pay it
    @test sum(inn -> inn.numItr, gcr.inner) < sum(inn -> inn.numItr, gmr.inner)
end

@testset "solve GPU" begin
    vol = mkVol((4,4,4))
    @testset "$(nameof(typeof(dev))) $T" for dev in gpuBackends, T in gpuPrcs(dev)
        opr = GlaOprVac{T}(vol; dev)
        rhs = adapt(dev, ones(Complex{T}, size(opr, 1)))

        sol_bcg = solve(opr, rhs, BiCGStabSolver())
        @test size(sol_bcg) == (size(opr, 2),)
        @test norm(opr * sol_bcg - rhs) < sqrt(eps(T)) * norm(rhs)

        sol_gmr = solve(opr, rhs, GMRESSolver())
        @test size(sol_gmr) == (size(opr, 2),)
        @test norm(opr * sol_gmr - rhs) < sqrt(eps(T)) * norm(rhs)
    end
end
