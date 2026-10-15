#= The closed-form contact moments and the contact block they assemble. The
sub-geometry constructor and the exact even-moment polynomial live here rather
than in src: parMom never needs a sub-geometry name, only these tests do. =#
using Test, GilaElectromagnetics, DoubleFloats
const GVM = GilaElectromagnetics

# the 15 canonical ways two axis-aligned rectangular panels can touch
function canPnl(cod::AbstractString, a::T, b::T, c::T) where {T}
    z = zero(T)
    A = ((z, a), (z, b), (z, z))
    cod == "S"     && return (A, A)
    cod == "Ef"    && return (A, ((a, 2a), (z, b), (z, z)))
    cod == "Ec"    && return (A, ((z, a), (b, b), (z, c)))
    cod == "Vc"    && return (A, ((a, 2a), (b, 2b), (z, z)))
    cod == "Vp"    && return (A, ((a, a), (b, 2b), (z, c)))
    cod == "P1"    && return (A, ((z, a), (z, b), (c, c)))
    cod == "P2"    && return (A, ((z, a), (z, b), (2c, 2c)))
    cod == "P1s"   && return (A, ((a, 2a), (z, b), (c, c)))
    cod == "P2s"   && return (A, ((a, 2a), (z, b), (2c, 2c)))
    cod == "Xg"    && return (A, ((z, a), (2b, 2b), (z, c)))
    cod == "Xg-b"  && return (((2b, 2b), (a, 2a), (z, c)), ((z, b), (z, a), (z, z)))
    cod == "Xg-c"  && return (((2b, 2b), (c, 2c), (z, a)), ((z, b), (z, z), (z, a)))
    cod == "Xg-d"  && return (((2b, 2b), (c, 2c), (a, 2a)), ((z, b), (z, z), (z, a)))
    cod == "P1s-b" && return (((a, 2a), (b, 2b), (z, z)), ((z, a), (z, b), (c, c)))
    cod == "P2s-b" && return (((2c, 2c), (a, 2a), (b, 2b)), ((z, z), (z, a), (z, b)))
    error("canPnl: unknown code $cod")
end

const momCod = ["S", "Ef", "Ec", "Vc", "Vp", "P1", "P2", "P1s", "P2s", "Xg",
    "Xg-b", "Xg-c", "Xg-d", "P1s-b", "P2s-b"]

#= Even moments in closed rational form: r^2k expands multinomially into
per-axis polynomial integrals, each exact, so this is independent of the box
decomposition parMom uses. =#
function axsPow(A, B, n::Integer)
    p, q = A; r, s = B
    p == q && r == s && return (p - r)^n
    p == q && return ((p - r)^(n + 1) - (p - s)^(n + 1)) // (n + 1)
    r == s && return ((q - r)^(n + 1) - (p - r)^(n + 1)) // (n + 1)
    H(u) = u^(n + 2) // ((n + 1) * (n + 2))
    return H(q - r) - H(q - s) - H(p - r) + H(p - s)
end

function momPol(pA, pB, m::Integer)
    Q = Rational{BigInt}
    qA = ntuple(d -> (Q(pA[d][1]), Q(pA[d][2])), 3)
    qB = ntuple(d -> (Q(pB[d][1]), Q(pB[d][2])), 3)
    k = m ÷ 2
    fac(n) = factorial(big(n))
    acc = zero(Q)
    for i ∈ 0:k, j ∈ 0:(k - i)
        l = k - i - j
        acc += (fac(k) ÷ (fac(i) * fac(j) * fac(l))) *
            axsPow(qA[1], qB[1], 2i) * axsPow(qA[2], qB[2], 2j) *
            axsPow(qA[3], qB[3], 2l)
    end
    return acc
end

# test/ref/mom.txt: sub-code, edge lengths, m, I_m at 220 bits
const momRef = setprecision(BigFloat, 256) do
    ref = Dict{Tuple{String,Float64,Float64,Float64,Int},BigFloat}()
    for lin ∈ eachline(joinpath(@__DIR__, "..", "ref", "mom.txt"))
        startswith(lin, "#") && continue
        f = split(lin)
        ref[(f[1], parse(Float64, f[2]), parse(Float64, f[3]),
            parse(Float64, f[4]), parse(Int, f[5]))] = parse(BigFloat, f[6])
    end
    ref
end

const momShp = [(1.0, 1.0, 1.0), (1 / 32, 1 / 32, 1 / 32),
    (1 / 32, 1 / 32, 1 / 512), (1.0, 1.0, 1e-3), (1.0, 1.0, 1e3)]
const momSlv = (1//32, 1//32, 1//512)

@testset "Contact moments" begin
    @testset "even moments against exact rationals" begin
        wst = 0.0
        for cod ∈ momCod, s ∈ momShp[1:3]
            pA, pB = canPnl(cod, s...)
            v = GVM.parMom(pA, pB, 12)
            for m ∈ 0:2:12
                rf = Float64(momPol(pA, pB, m))
                wst = max(wst, abs(v[m + 2] - rf) / abs(rf))
            end
        end
        @test wst < 1e-13
    end

    @testset "220-bit reference moments" begin
        wst = 0.0
        hit = 0
        setprecision(BigFloat, 256) do
            for cod ∈ momCod, s ∈ momShp
                v = GVM.parMom(canPnl(cod, s...)..., 11)
                for m ∈ (-1, 1, 5, 11)
                    ky = (cod, s..., m)
                    haskey(momRef, ky) || continue
                    hit += 1
                    wst = max(wst, Float64(abs(BigFloat(v[m + 2]) - momRef[ky]) /
                        abs(momRef[ky])))
                end
            end
        end
        @test hit == length(momRef) == 300
        @test wst < 1e-13
    end

    #= G(lam s, f / lam) = lam^5 G(s, f): the moments carry I_m(lam s) =
    lam^(m+4) I_m(s) and the series carries the rest. =#
    @testset "homogeneity in f" begin
        wst = 0.0
        for D ∈ ((0, 0, 0), (1, 0, 0), (1, 1, 1)), F ∈ 1:6, Fp ∈ 1:6,
                lam ∈ (3.0, 1 / 7), f ∈ (1.0 + 0.0im, 1.0 + 0.1im)
            s = (1 / 32, 1 / 32, 1 / 512)
            v = first(GVM.momSer(GVM.parFac(D, F, Fp, s)..., f, 16))
            w = first(GVM.momSer(GVM.parFac(D, F, Fp, lam .* s)..., f / lam, 16))
            wst = max(wst, abs(w - lam^5 * v) / abs(w))
        end
        @test wst < 1e-12
    end

    #= I_m(lam s) = lam^(m+4) I_m(s).  A dyadic lam multiplies every moment by an exact
    power of two, so the whole box decomposition has to reproduce it bit for bit. =#
    @testset "homogeneity in scale" begin
        mMx = 10
        cnt = 0
        bad = 0
        wst = 0.0
        for s ∈ ((1 / 32, 1 / 32, 1 / 32), (1 / 32, 1 / 32, 1 / 512), (1 / 4, 1 / 4, 1 / 4),
                    (1 / 32, 1 / 16, 1 / 8)),
                D ∈ ((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)), F ∈ 1:6, Fp ∈ 1:6
            v = GVM.parMom(GVM.parFac(D, F, Fp, s)..., mMx)
            for lam ∈ (2.0, 0.5, 0.125)
                w = GVM.parMom(GVM.parFac(D, F, Fp, lam .* s)..., mMx)
                for m ∈ -1:mMx
                    cnt += 1
                    w[m + 2] === lam^(m + 4) * v[m + 2] || (bad += 1)
                end
            end
            for lam ∈ (3.0, 1 / 10)
                w = GVM.parMom(GVM.parFac(D, F, Fp, lam .* s)..., mMx)
                for m ∈ -1:mMx
                    iszero(w[m + 2]) && continue
                    wst = max(wst, abs(w[m + 2] - lam^(m + 4) * v[m + 2]) / abs(w[m + 2]))
                end
            end
        end
        @test bad == 0
        @test wst < 1e-13
    end

    @testset "sub-geometry symmetries" begin
        a, b, c = 7 / 3, 5 / 11, 13 / 6
        for (cod, sw) ∈ (("Ec", (a, c, b)), ("Vp", (c, b, a)))
            v = GVM.parMom(canPnl(cod, a, b, c)..., 12)
            w = GVM.parMom(canPnl(cod, sw...)..., 12)
            e = maximum(abs.(w .- v) ./ abs.(v))
            @test e < 1e-13
        end
        # every other length swap is either a symmetry or absent, never wrong
        for cod ∈ momCod
            v = GVM.parMom(canPnl(cod, a, b, c)..., 6)
            @test all(isfinite, v)
            # S, Ef and Vc have no c edge, so a swap touching it tests nothing
            cod ∈ ("S", "Ef", "Vc") ||
                @test all(isfinite, GVM.parMom(canPnl(cod, a, c, b)..., 6))
        end
    end

    #= The slender cell is where the deleted order-9 shell rule was 117 % wrong
    at (0,0,1) and 12.7 % at (0,0,0). The block is re-assembled here at 192 bits
    to check the Double64 assembly, the memo and the srfSum! signs; the moments
    themselves are checked against the 220-bit reference above. =#
    @testset "slender contact block" begin
        vol = mkVol((4, 4, 4); scl=momSlv)
        frq = 1.0 + 0.1im
        opt = GlaKerOpt{Float64}()
        opt.frqPhz = frq
        toe = zeros(ComplexF64, 3, 3, 2, 2, 2)
        GVM.cntBlk!(toe, vol, opt)
        @test all(isfinite, toe)
        wst = 0.0
        setprecision(BigFloat, 192) do
            sB = BigFloat.(momSlv)
            celInv = inv(prod(sB))
            mM = GVM.serOrd(momSlv, momSlv, frq)
            srf = zeros(Complex{BigFloat}, 36)
            ref = zeros(Complex{BigFloat}, 3, 3)
            for D ∈ ((0, 0, 0), (0, 0, 1))
                for fp ∈ 1:36
                    srf[fp] = celInv * first(GVM.momSer(GVM.parFac(D,
                        GVM.facPar[1, fp], GVM.facPar[2, fp], sB)...,
                        Complex{BigFloat}(frq), mM))
                end
                GVM.srfSum!(ref, srf)
                got = view(toe, :, :, CartesianIndex((D .+ 1)...))
                wst = max(wst, Float64(maximum(abs, got .- ref) /
                    maximum(abs, ref)))
            end
        end
        @test wst < 1e-13
    end
end
