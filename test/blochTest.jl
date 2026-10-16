# Bloch-periodic operators: against the dense Bloch matrix of blcFil's Toeplitz block, the accuracy records of
# ref/bloch.txt and the 256-bit composite references of ref/blochCmp.txt
using JLD2
import GilaElectromagnetics.Bloch: blcSet, blcFil, blcTns, blcCmp, QI

# dense Bloch matrix in Gila's vector layout (cells..., 3) of the Toeplitz block H + iA over the offsets ms
function blcMat(H, A, ms, ax, N)
    cs = vec(collect(CartesianIndices(N))); n = length(cs); D = zeros(ComplexF64, 3n, 3n)
    for (p, ci) in enumerate(cs), (q, cj) in enumerate(cs)
        m = Tuple(ci) .- Tuple(cj)
        J = ntuple(i -> i in ax ? findfirst(==(m[i]), ms[i]) : findfirst(==(abs(m[i])), ms[i]), 3)
        for a in 1:3, b in 1:3
            # open axes: G_ab(-m) = σ_a σ_b G_ab(m)
            sg = prod(i -> (!(i in ax) && m[i] < 0 && ((a == i) + (b == i)) == 1) ? -1 : 1, 1:3)
            D[p + (a - 1) * n, q + (b - 1) * n] = sg * (H[a, b, J...] + im * A[a, b, J...])
        end
    end
    return D
end

prat(s) = (p = split(s, "//"); QI(parse(BigInt, p[1]), parse(BigInt, p[2])))
ptri(s) = Tuple(prat.(split(s, ",")))

@testset "Bloch" begin
    rng = MersenneTwister(5)
    @testset "Mat" begin
        for (s, f, ax, N, P, θGen) in (((1//8, 1//8, 1//8), 1.0 + 0.1im, (1,), (4, 2, 2), (6,), (0.7,)),
                                       ((1//32, 1//32, 1//32), 1.0 + 0.0im, (1, 2), (4, 4, 2), (4, 4), (0.7, -1.9)),
                                       ((1//4, 1//4, 1//4), 1.0 + 0.1im, (1, 2, 3), (2, 2, 2), (3, 3, 2), (0.7, -1.9, 2.5)))
            vol = GlaVol(N, s); st = blcSet(s, f, ax, N, P)
            for θ in (θGen, map(_ -> 0//1, ax), map(_ -> 1//2, ax))
                H, A, inf = blcFil(st, θ); D = blcMat(H, A, inf[:ms], ax, N)
                blc = (; ax, P, θ)
                opr = GlaOprVac{Float64}(vol; frqPhz = f, blc)
                x = randn(rng, ComplexF64, size(D, 1)); y = randn(rng, ComplexF64, size(D, 1))
                @test norm(opr * x - D * x) < 4eps() * norm(D * x)
                θ === θGen || continue
                @test abs(dot(opr * x, y) - dot(x, adjoint(opr) * y)) < 4eps() * norm(opr * x) * norm(y)
                @test norm(glaSym(opr) * x - (D + D') / 2 * x) < 4eps() * norm(D * x)
                @test norm(asym(opr) * x - (D - D') / 2im * x) < 4eps() * norm(D * x)
                θNew = map(t -> -t / 3, θ)
                @test setBlc!(opr, θNew) * x ≈ GlaOprVac{Float64}(vol; frqPhz = f, blc = (; ax, P, θ = θNew)) * x rtol = 4eps()
            end
        end
    end

    @testset "Rec" begin
        lns = filter(l -> !startswith(l, "#"), readlines(joinpath(@__DIR__, "ref", "bloch.txt")))
        setprecision(BigFloat, 384) do
            for ln in lns
                c = split(ln)
                ax = Tuple(parse.(Int, split(c[2], ","))); L = Tuple(prat.(split(c[3], ",")))
                θ = Tuple(t == "pi" ? 1//2 : parse(Float64, t) for t in split(c[4], ","))
                f = complex(parse(Float64, c[8]), parse(Float64, c[9])); κe = parse(Float64, c[11])
                H, A, _ = blcTns(ptri(c[7]), ptri(c[5]), ptri(c[6]), f, ax, L, θ)
                X = H .+ im .* A; Y = complex.(parse.(BigFloat, c[13:21]), parse.(BigFloat, c[22:30]))
                m = maximum(abs, Y)
                ep = maximum(i -> abs(Y[i]) > 1e-8 * m ? Float64(abs(X[i] - Y[i]) / abs(Y[i])) / eps() : 0.0, 1:9)
                @test ep <= 3 * (1 + κe)
            end
        end
    end

    # one box of 4 × 2 × 2 cells, and the same box as two regions of 2 × 2 × 2 cells
    s = (1//32, 1//32, 1//32); ax = (1,); P = (6,); θ = (0.7,)
    cvol = GlaCmpVol([GlaVol((2, 2, 2), s, (1//32, 1//32, 1//32)), GlaVol((2, 2, 2), s, (3//32, 1//32, 1//32))])
    # composite degree of freedom → box degree of freedom
    lin = LinearIndices((4, 2, 2))
    prm = [lin[c[1] + 2 * (r - 1), c[2], c[3]] + 16 * (a - 1) for r in 1:2 for a in 1:3 for c in CartesianIndices((2, 2, 2))]
    function boxMat(f, θ)
        H, A, inf = blcFil(s, f, ax, (4, 2, 2), P, θ)
        return blcMat(H, A, inf[:ms], ax, (4, 2, 2))[prm, prm]
    end

    @testset "Cmp" begin
        f = 1.0 + 0.1im; blc = (; ax, P, θ)
        D = boxMat(f, θ); x = randn(rng, ComplexF64, 48); y = randn(rng, ComplexF64, 48)
        opr = GlaCmpOprVac{Float64}(cvol; frqPhz = f, blc)
        @test norm(opr * x - D * x) < 4eps() * norm(D * x)
        @test abs(dot(opr * x, y) - dot(x, adjoint(opr) * y)) < 4eps() * norm(opr * x) * norm(y)
        @test norm(glaSym(opr) * x - (D + D') / 2 * x) < 4eps() * norm(D * x)
        @test norm(asym(opr) * x - (D - D') / 2im * x) < 4eps() * norm(D * x)
        θNew = (-1.9,)
        @test setBlc!(opr, θNew) * x ≈ GlaCmpOprVac{Float64}(cvol; frqPhz = f, blc = (; ax, P, θ = θNew)) * x rtol = 4eps()
        Dn = boxMat(f, θNew)
        @test norm(opr * x - Dn * x) < 4eps() * norm(Dn * x)
        # the part is rebuilt at the phases setBlc! left
        @test norm(asym(opr) * x - (Dn - Dn') / 2im * x) < 4eps() * norm(Dn * x)
    end

    # coarse (2, 2, 2) at (1/16, 1/32, 1/32) next to fine (4, 2, 2) along the periodic axis: the 8 × 2 × 2 box at 1/32 with pairs of cells merged
    @testset "CrsCmp" begin
        crs = GlaCmpVol([GlaVol((2, 2, 2), (1//16, 1//32, 1//32), (1//16, 1//32, 1//32)), GlaVol((4, 2, 2), s, (3//16, 1//32, 1//32))])
        # composite degree of freedom → normalized box degrees of freedom
        lnB = LinearIndices((8, 2, 2)); Q = zeros(96, 72); k = 0
        for (r, reg) in enumerate(regions(crs)), a in 1:3, c in CartesianIndices(reg.cel)
            k += 1
            for x in (r == 1 ? (2c[1] - 1, 2c[1]) : (c[1] + 4,))
                Q[lnB[x, c[2], c[3]] + 32 * (a - 1), k] = r == 1 ? 1 / sqrt(2) : 1
            end
        end
        f = 1.0 + 0.1im; Pc = (10,)
        crsMat(θ) = ((H, A, inf) = blcFil(s, f, ax, (8, 2, 2), Pc, θ); Q' * blcMat(H, A, inf[:ms], ax, (8, 2, 2)) * Q)
        D = crsMat(θ); x = randn(rng, ComplexF64, 72)
        opr = GlaCmpOprVac{Float64}(crs; frqPhz = f, blc = (; ax, P = Pc, θ))
        @test norm(opr * x - D * x) < 4eps() * norm(D * x)
        Dn = crsMat((-1.9,))
        @test norm(setBlc!(opr, (-1.9,)) * x - Dn * x) < 4eps() * norm(Dn * x)
    end

    # the lattice survives Serialization and JLD2: same product, and setBlc! still works on the reloaded operator
    @testset "Ser" begin
        serRt(o) = mktemp() do fil, io; serialize(io, o); close(io); open(deserialize, fil); end
        jldRt(o) = mktempdir() do dir; jldsave(joinpath(dir, "o.jld2"); o); load(joinpath(dir, "o.jld2"), "o"); end
        f = 1.0 + 0.1im; x = randn(rng, ComplexF64, 48); θNew = (-1.9,)
        sOpr = GlaOprVac{Float64}(GlaVol((4, 2, 2), s, (1//32, 1//32, 1//32)); frqPhz = f, blc = (; ax, P, θ = (0.7,)))
        for opr in (sOpr, GlaCmpOprVac{Float64}(cvol; frqPhz = f, blc = (; ax, P, θ)))
            y = opr * x[1:size(opr, 2)]
            for rt in (serRt, jldRt)
                des = rt(opr); z = x[1:size(opr, 2)]
                @test des * z == y
                @test setBlc!(des, θNew) * z == setBlc!(deepcopy(opr), θNew) * z
            end
        end
    end

    # the anti-Hermitian part is positive semidefinite at real f and definite at Im f > 0
    @testset "Psd" begin
        vol = GlaVol((4, 2, 2), (1//8, 1//8, 1//8))
        for f in (1.0 + 0.0im, 1.0 + 0.1im)
            for M in (GlaOprVac{Float64}(vol; frqPhz = f, blc = (ax = (1,), P = (6,), θ), hrmPrt = :asy) * Matrix{ComplexF64}(I, 48, 48),
                      GlaOprVac{Float64}(cvol; frqPhz = f, blc = (; ax, P, θ), hrmPrt = :asy) * Matrix{ComplexF64}(I, 48, 48))
                λ = eigvals(Hermitian((M + M') / 2))
                @test iszero(imag(f)) ? minimum(λ) >= -8eps() * maximum(λ) : minimum(λ) > 0
            end
        end
    end

    # entries of the composite operator against the 256-bit references of its cell pairs
    @testset "CmpRef" begin
        ctr = [ntuple(i -> QI(reg.grd[i][c[i]]), 3) for reg in regions(cvol) for c in CartesianIndices(reg.cel)]
        setprecision(BigFloat, 256) do
            for ln in readlines(joinpath(@__DIR__, "ref", "blochCmp.txt"))
                fld = split(ln, '|'); kv = Dict(Pair(split(x, '='; limit = 2)...) for x in fld[2:9])
                fs = split(kv["f"], ";"); f = complex(parse(Float64, fs[1]), parse(Float64, fs[2])); R = ptri(kv["R"])
                Y = reshape([complex(parse.(BigFloat, split(e, ";"))...) for e in split(strip(fld[12]))], 3, 3)
                M = GlaCmpOprVac{Float64}(cvol; frqPhz = f, blc = (; ax, P, θ)) * Matrix{ComplexF64}(I, 48, 48)
                prs = [(p, q) for p in 1:16, q in 1:16 if ctr[p] .- ctr[q] == R]
                @test !isempty(prs)
                for (p, q) in prs
                    # region r's block holds its 8 cells per component
                    dof(c, a) = (c - 1) ÷ 8 * 24 + (a - 1) * 8 + (c - 1) % 8 + 1
                    X = [M[dof(p, a), dof(q, b)] for a in 1:3, b in 1:3]
                    @test maximum(abs, X .- Y) <= 4eps() * maximum(abs, Y)
                end
            end
        end
    end

    @testset "Rfs" begin
        z3 = (QI(0), QI(0), QI(0)); s32 = (QI(1//32), QI(1//32), QI(1//32))
        # an exact Rayleigh-Wood anomaly (L f = 1 at θ = 0), a period off the cell lattice, a unit cell longer than its period
        for fun in (() -> GlaOprVac{Float64}(GlaVol((2, 2, 2), (1//8, 1//8, 1//8)); blc = (ax = (1,), P = (8,), θ = (0//1,))),
                    () -> blcTns(z3, s32, s32, 1.0 + 0im, (1,), (QI(3//64),), (0.7,)),
                    () -> GlaOprVac{Float64}(GlaVol((4, 2, 2), s); blc = (ax = (1,), P = (3,), θ)))
            rsl = @test_throws ErrorException fun()
            @test !occursin('\n', rsl.value.msg)
        end
    end
end
