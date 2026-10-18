#= The quasistatic operator against physics it does not itself contain: the cell's demagnetizing
tensor, the depolarization sum rule, tracelessness away from the origin, the point-dipole far field,
definiteness, and the 1/f^2 that carries the whole frequency dependence. =#
using Test, GilaElectromagnetics, DoubleFloats, LinearAlgebra
const GVQ = GilaElectromagnetics

# cell shapes as full edge lengths; S sees the cell only through its aspect ratio
const qssCub = (1//1, 1//1, 1//1)
const qssAni = (1//1, 1//1, 4//1)
const qssSkw = (7//10, 13//1000, 19//10)

# The signed 36-pair assembly amplifies roundoff by Lambda(D, s) = 4 (l / s_min)^2, with
# l = max(|D .* s|, max_i s_i).  Every roundoff-limited bar below is that law at a factor 100 of
# headroom; a flat bar cannot follow the cell aspect, which here spans a cube to an aspect of 146.
qssAmp(D, s) = 4 * (max(sqrt(sum(abs2, float.(D) .* float.(s))), maximum(float.(s))) /
    minimum(float.(s)))^2
qssTol(D, s, ::Type{T} = Float64) where {T} = 100 * eps(T) * qssAmp(D, s)

# Aharoni's closed form for the demagnetizing factor of a prism of half edges (a, b, c) along the
# axis of its THIRD argument; A. Aharoni, J. Appl. Phys. 83, 3432 (1998) eq. 1.
function qssAhr(a::T, b::T, c::T) where {T}
    r = sqrt(a^2 + b^2 + c^2); rab = sqrt(a^2 + b^2); rbc = sqrt(b^2 + c^2); rac = sqrt(a^2 + c^2)
    t = (b^2 - c^2) / (2b * c) * log((r - a) / (r + a)) + b / (2c) * log((rab + a) / (rab - a)) +
        (a^2 - c^2) / (2a * c) * log((r - b) / (r + b)) + a / (2c) * log((rab + b) / (rab - b)) +
        c / (2a) * log((rbc - b) / (rbc + b)) + c / (2b) * log((rac - a) / (rac + a)) +
        2 * atan(a * b / (c * r)) + (a^3 + b^3 - 2c^3) / (3a * b * c) + c / (a * b) * (rac + rbc) +
        (a^2 + b^2 - 2c^2) * r / (3a * b * c) - (rab^3 + rbc^3 + rac^3) / (3a * b * c)
    return t / T(pi)
end

# the three factors of a cuboid of full edges s, in axis order, at Double64 so it carries no roundoff
qssDem(s) = (h = Double64.(s) ./ 2;
    Float64.((qssAhr(h[2], h[3], h[1]), qssAhr(h[3], h[1], h[2]), qssAhr(h[1], h[2], h[3]))))

# the point dipole the cell looks like from afar: V (3 n_a n_b - delta_ab) / (4 pi r^3)
qssDip(D, s) = (R = float.(D) .* float.(s); r = sqrt(sum(abs2, R)); V = prod(float.(s));
    [V * (3R[a] * R[b] / r^2 - (a == b)) / (4pi * r^3) for a ∈ 1:3, b ∈ 1:3])

# S at every offset of a cel block, as genEgoSlf! assembles it under qssApx: contact offsets and the
# near band the multipole refuses from the closed-form moments, the rest from the multipole.  The
# identity term and the 1/f^2 genEgoSlf! adds after are not here.
function qssToe(cel::NTuple{3,Int}, scl::NTuple{3,Rational{Int}})
    vol = mkVol(cel; scl = scl)
    toe = fill(ComplexF64(NaN), 3, 3, cel...)
    GVQ.momBlkQss!(toe, vol, [Tuple(itr) .- 1 for itr ∈ vec(CartesianIndices(min.(cel, 2)))])
    GVQ.momBlkQss!(toe, vol, GVQ.farBlkQss!(toe, scl; tol = GVQ.FARTOL, disk = false))
    @test all(isfinite, toe)
    return real.(toe)
end

# the assembled quasistatic operator, the object GlaOprVac is handed
qssMem(cel, scl, frq = 1.0 + 0.0im) =
    GlaVacOprMem(GlaKerOpt{Float64}(; qssApx = true, frqPhz = frq), mkVol(cel; scl = scl))

@testset "Quasistatic Tests" begin
    # the self block is minus the cell's own demagnetizing tensor, of trace one
    @testset "self block" begin
        for s ∈ (qssCub, qssAni, qssSkw)
            S = qssToe((1, 1, 1), s)[:, :, 1, 1, 1] - I
            frc = maximum(abs, S + Diagonal(collect(qssDem(s)))) / qssTol((0, 0, 0), s)
            @test frc < 1
            @test abs(tr(S) + 1) < qssTol((0, 0, 0), s)
        end
    end

    # The volume average of the field of a uniformly polarized cuboid body is -L(body) . P, L the
    # Aharoni factors of the WHOLE body: the only test here that knows the touching blocks are not
    # point dipoles.
    @testset "depolarization sum rule" begin
        for (cel, s) ∈ (((3, 2, 5), qssCub), ((4, 4, 4), qssAni), ((8, 8, 2), qssCub),
                ((4, 4, 4), (1//1, 2//1, 1//4)))
            n = prod(cel)
            O = real.(dnsMat(qssMem(cel, s)))
            A = [sum(view(O, (a-1) * n .+ (1:n), (b-1) * n .+ (1:n))) for a ∈ 1:3, b ∈ 1:3] ./ n
            L = qssDem(cel .* s)
            tol = qssTol(cel, s)
            frc = maximum(abs((A[a, a] + L[a]) / L[a]) for a ∈ 1:3) / tol
            @test frc < 1
            @test maximum(abs, A - Diagonal(diag(A))) < tol
            @test abs(tr(A) + 1) < tol
        end
    end

    # away from the origin S solves Laplace's equation, so it is traceless
    @testset "trace free away from the origin" begin
        for s ∈ (qssCub, qssAni, qssSkw)
            toe = qssToe((4, 4, 4), s)
            for i ∈ 0:3, j ∈ 0:3, k ∈ 0:3
                (i, j, k) == (0, 0, 0) && continue
                S = view(toe, :, :, i + 1, j + 1, k + 1)
                frc = abs(tr(S)) / maximum(abs, S) / qssTol((i, j, k), s)
                @test frc < 1
            end
        end
    end

    # The cell radiates as a point dipole of moment V P, its own second moments the leading
    # correction.  These bars are asymptotic, not roundoff, so they do not follow Lambda.
    @testset "far field is a point dipole" begin
        # a cubic cell has no quadrupole correction, so the deviation is exactly -(7/16)(h/r)^4
        toe = qssToe((129, 1, 1), qssCub)
        for n ∈ (32, 64, 128)
            rat = toe[1, 1, n + 1, 1, 1] / qssDip((n, 0, 0), qssCub)[1, 1]
            @test abs((rat - 1) * n^4 + 7 / 16) / (7 / 16) < 4e-3
        end
        for s ∈ (qssAni, qssSkw)  # otherwise the correction is (2 s1^2 - s2^2 - s3^2) / (2 r^2)
            toe = qssToe((129, 1, 1), s)
            tgt = (2 * float(s[1])^2 - float(s[2])^2 - float(s[3])^2) / 2
            for n ∈ (64, 128)
                rat = toe[1, 1, n + 1, 1, 1] / qssDip((n, 0, 0), s)[1, 1]
                @test abs((rat - 1) * (n * float(s[1]))^2 - tgt) / abs(tgt) < 6e-2
            end
        end
        # a bound holding at every separated offset, not only along an axis
        for (s, p, c) ∈ ((qssCub, 4, 1.0), (qssAni, 2, 3.0), (qssSkw, 2, 15.0),
                ((1//1, 2//1, 1//4), 2, 3.0))
            toe = qssToe((7, 7, 7), s)
            for i ∈ 0:6, j ∈ 0:6, k ∈ 0:6
                max(i, j, k) >= 2 || continue
                P = qssDip((i, j, k), s)
                @test maximum(abs, view(toe, :, :, i + 1, j + 1, k + 1) - P) / maximum(abs, P) <
                    c * (maximum(float.(s)) / sqrt(sum(abs2, float.((i, j, k)) .* float.(s))))^p
            end
        end
    end

    # Electrostatics of a passive body: 0 < integral |E|^2 <= integral |P|^2 puts every eigenvalue
    # of the operator strictly inside (-1, 0).  Bars are 20x the measured extremes.
    @testset "definiteness and energy" begin
        for (cel, s) ∈ (((3, 3, 3), qssCub), ((3, 3, 3), qssAni), ((2, 2, 6), qssCub),
                ((4, 2, 1), qssSkw))
            ev = eigvals(Symmetric(real.(dnsMat(qssMem(cel, s)))))
            @test ev[end] < -1e-4
            @test ev[1] > -1 + 1e-4
        end
    end

    # The static kernel is frequency free, so every f in the operator is the single 1/f^2 genEgoSlf!
    # applies last: f^2 egoFur is f independent, and Helmholtz converges onto it as (k r)^2.
    @testset "homogeneity in f" begin
        cel = (3, 3, 3)
        qss = qssMem(cel, stdScl)
        ref = maximum(maximum(abs, blk) for blk ∈ qss.egoFur)
        for f ∈ (1.0 + 0.1im, 2.0 + 0.7im, 1e-8 + 0.0im, 1e5 - 3e4im, 0.1 + 1.0im)
            mem = qssMem(cel, stdScl, f)
            dev = maximum(maximum(abs, f^2 .* b .- a) for (a, b) ∈ zip(qss.egoFur, mem.egoFur))
            @test dev < 8 * eps() * ref
        end
        err = map((1e-2, 1e-3)) do f
            opt = GlaKerOpt{Float64}()
            opt.frqPhz = complex(f)
            mem = GlaVacOprMem(opt, mkVol(cel))
            sqrt(sum(sum(abs2, f^2 .* b .- a) for (a, b) ∈ zip(qss.egoFur, mem.egoFur)) /
                sum(sum(abs2, a) for a ∈ qss.egoFur))
        end
        @test err[2] < 1e-7
        @test abs(log10(err[1] / err[2]) - 2) < 0.05
    end
end
