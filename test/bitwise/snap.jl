# Every output a bitwise refactor must keep, serialized to ARGS[1] as name => array, or the message of a refusal.
# ARGS[2] is an FFTW wisdom file shared by every run, read when it exists and written otherwise: MEASURE plans differ
# from run to run by about 1.5e-16. Snapshot the base commit from a clean worktree, never the tree being edited:
#   git worktree add ../base HEAD && cp Manifest.toml ../base/
#   julia -t 8 --startup-file=no --project=../base test/bitwise/snap.jl base8.jls wisdom
#   julia -t 8 --startup-file=no --project=. test/bitwise/snap.jl new8.jls wisdom
#   julia --startup-file=no --project=. test/bitwise/cmp.jl base8.jls new8.jls
# and again at -t 1, which also checks that the outputs do not depend on the thread count.
using GilaElectromagnetics, DoubleFloats, Random, Serialization
import GilaElectromagnetics: FFTW, GlaKerOpt, KSRDIR, QI, cntBlk!, facPar, farBlk!, farBlkQss!, farBlkX!,
    farFil!, momBlkQss!, parFac, parMom
import GilaElectromagnetics.Bloch: blcCmp, blcFil, blcSet
# older trees have farTol(T)
const FARTOL = isdefined(GilaElectromagnetics, :FARTOL) ? GilaElectromagnetics.FARTOL : GilaElectromagnetics.farTol(Float64)

wis = get(ARGS, 2, nothing)
wis !== nothing && isfile(wis) && FFTW.import_wisdom(wis)
# a cold route-3 cache; the geometry tables are built in memory (disk = false)
KSRDIR[] = mktempdir()
out = Dict{String,Any}()
# a refusal is part of the output: its message is kept
snap(f, key) = (t = time(); out[key] = try f() catch e; e isa ErrorException || e isa ArgumentError || rethrow(); sprint(showerror, e) end;
    println(rpad(key, 60), round(time() - t, digits = 1), " s"); flush(stdout))

s32 = (1//32, 1//32, 1//32); s16 = (1//16, 1//16, 1//16); s8 = (1//8, 1//8, 1//8); s4 = (1//4, 1//4, 1//4)
sl = (1//32, 1//32, 1//512); z = (0//1, 0//1, 0//1)
# a volume from its cell count, cell edges and lower corner
crn(cel, scl, lo) = GlaVol(cel, scl, lo .+ cel .* scl .// 2)

# Operators, each applied with its adjoint to a seeded vector
f = 1.0 + 0.05im
vol = GlaVol((4, 4, 2), s32, z); far = GlaVol((4, 4, 2), s32, (1//2, 0//1, 0//1))
cvl = refine(GlaCmpVol(GlaVol((4, 4, 4), s16, z)), ((-1//16, 0//1, 0//1), (1//8, 1//4, 1//4)))
bcv = GlaCmpVol([GlaVol((2, 2, 2), s32, (1//32, 1//32, 1//32)), GlaVol((2, 2, 2), s32, (3//32, 1//32, 1//32))])
bcr = GlaCmpVol([GlaVol((2, 2, 2), (1//16, 1//32, 1//32), (1//16, 1//32, 1//32)), GlaVol((4, 2, 2), s32, (3//16, 1//32, 1//32))])
oprs = [
    "self" => T -> GlaOprVac{T}(vol; frqPhz = f),
    "ext" => T -> GlaOprVac{T}(far, vol; frqPhz = f, prxWrn = false),
    "asym" => T -> asym(GlaOprVac{T}(vol; frqPhz = f)),
    "sym" => T -> sym(GlaOprVac{T}(vol; frqPhz = f)),
    "qss" => T -> GlaOprVac{T}(vol; frqPhz = f, qssApx = true),
    "cmp" => T -> GlaOprVac{T}(cvl; frqPhz = f),
    "blc1" => T -> GlaOprVac{T}(GlaVol((4, 2, 2), s32); frqPhz = 1.0 + 0.1im, blc = (ax = (1,), P = (6,), θ = (0.7,))),
    "blc2" => T -> GlaOprVac{T}(GlaVol((4, 4, 2), s32); frqPhz = 1.0 + 0.0im, blc = (ax = (1, 2), P = (4, 6), θ = (0.7, 1//2))),
    "blc3" => T -> GlaOprVac{T}(GlaVol((2, 2, 2), s32); frqPhz = 1.0 + 0.1im, blc = (ax = (1, 2, 3), P = (4, 4, 4), θ = (0.3, -1.9, 1.26e-5))),
    "blcCmp" => T -> GlaOprVac{T}(bcv; frqPhz = 1.0 + 0.1im, blc = (ax = (1,), P = (6,), θ = (0.7,))),
    "blcCrs" => T -> GlaOprVac{T}(bcr; frqPhz = 1.0 + 0.1im, blc = (ax = (1,), P = (10,), θ = (0.7,))),
    "blcSet" => T -> setBlc!(GlaOprVac{T}(bcr; frqPhz = 1.0 + 0.1im, blc = (ax = (1,), P = (10,), θ = (0.7,))), (-1.9,)),
    "blcSetSlf" => T -> setBlc!(GlaOprVac{T}(GlaVol((4, 2, 2), s32); frqPhz = 1.0 + 0.1im, blc = (ax = (1,), P = (6,), θ = (0.7,))), (-1.9,)),
]
for (nam, bld) in oprs, T in (Float64, Float32)
    snap("opr $nam $T") do
        A = bld(T); x = randn(MersenneTwister(7), Complex{T}, size(A, 2))
        (A * x, adjoint(A) * x)
    end
end
snap("opr anomaly") do; GlaOprVac{Float64}(GlaVol((2, 2, 2), s8); blc = (ax = (1,), P = (8,), θ = (0//1,))); end

# Moments of every face pair: contact shapes (cube, slender, needle, pancake) and separated ones, in Double64 and BigFloat-128
cnt8 = vec([Tuple(i) .- 1 for i in CartesianIndices((2, 2, 2))])
sep = [(2, 0, 0), (0, 0, 2), (2, 1, 0), (2, 2, 2), (3, 0, 0), (0, 0, 3), (1, 1, 2), (4, 1, 0), (0, 0, 6)]
momCas = [[("cnt", s, cnt8) for s in (s32, (1//10, 1//10, 1//10), sl, (1//32, 1//16, 1//8), (1//1, 1//1, 1//1000),
    (1//4, 1//4, 1//100), (1//4, 1//4, 1//1000))]; [("sep", s, sep) for s in (s32, sl, (1//4, 1//4, 1//1000))]]
for (knd, s, Ds) in momCas, T in (Double64, BigFloat)
    snap("mom $knd $s $T") do
        prs = [(D, fp) for D in Ds for fp in 1:36]; mom = Vector{Vector{T}}(undef, length(prs))
        Threads.@threads for i in eachindex(prs)
            (D, fp) = prs[i]
            mom[i] = setprecision(() -> parMom(parFac(D, facPar[1, fp], facPar[2, fp], T.(s))..., 8), BigFloat, 128)
        end
        reduce(hcat, mom)
    end
end
# the contact block of a slender cell, and the quasistatic block whose near band the moments fill
snap("cntBlk sl") do
    egoToe = zeros(ComplexF64, 3, 3, 4, 4, 4)
    cntBlk!(egoToe, GlaVol((4, 4, 4), sl, z), GlaKerOpt{Float64}(; frqPhz = f))
    egoToe
end
for s in (sl, (1//32, 1//32, 1//128))
    snap("qssBlk $s") do
        egoToe = zeros(ComplexF64, 3, 3, 6, 6, 6); v = GlaVol((6, 6, 6), s, z)
        momBlkQss!(egoToe, v, cnt8)
        momBlkQss!(egoToe, v, farBlkQss!(egoToe, s; tol = FARTOL, disk = false))
        egoToe
    end
end

# Far field: equal-cell blocks (the slender one reaches route 3 at (0, 0, 3)), a sub-cell shift, and cross-scale pairs
for (s, n, frq) in ((s32, (6, 6, 6), f), (s4, (6, 6, 6), 1.0 + 0.0im), (sl, (5, 5, 7), 1.0 + 0.0im))
    snap("farBlk $s $frq") do; farBlk!(zeros(ComplexF64, 3, 3, n...), s, frq; tol = FARTOL, disk = false); end
end
snap("farFil off") do
    Ds = [(i, j, k) for i in -6:6, j in 0:3, k in 0:2 if max(abs(i), j, k) >= 3] |> vec
    G = zeros(ComplexF64, 3, 3, length(Ds))
    farFil!(q -> view(G, :, :, q), Ds, map(QI, s32), f, (1/128, 0.0, 0.0); tol = FARTOL, disk = false)
    G
end
for sT in (s16, s8)
    snap("farBlkX $sT") do
        n = Int.(sT .// s32)
        Rs = [ntuple(d -> QI(2 * m[d] + n[d] - 1) / 64, 3) for m in Iterators.product(-2:6, -2:4, -3:5)
            if !all(d -> abs(2 * m[d] + n[d] - 1) <= n[d] + 1, 1:3)] |> vec
        G = zeros(ComplexF64, 3, 3, length(Rs))
        farBlkX!(G, Rs, sT, s32, f; tol = FARTOL, disk = false)
        G
    end
end
# an offset off the gcd lattice takes the k-series
snap("farBlkX off") do
    G = zeros(ComplexF64, 3, 3, 1)
    farBlkX!(G, [(QI(-7//128), QI(1//64), QI(1//64))], s16, s32, 0.25 + 0.0im; tol = FARTOL, disk = false)
    G
end

# Bloch fills: d = 1, 2, 3, real and complex f, both routes, the hot twist, the exact twist θ = π, refusals
fil = [
    (s32, 1.0 + 0im, (1,), (4, 3, 3), (4,), (0.7,)),
    (sl, 1.0 + 0.1im, (3,), (3, 3, 4), (8,), (1.26e-5,)),
    (s8, 0.37 + 0im, (1, 2), (4, 4, 3), (5, 4), (1//2, 1//2)),
    ((1//8, 1//16, 1//32), 1.0 + 1e-3im, (1, 3), (6, 3, 4), (12, 8), (π - 1e-4, π - 1e-4)),
    (s32, 1.0 + 0im, (1, 2), (4, 4, 3), (4, 4), (1.26e-5, 1.26e-5)),
    (s32, 1.0 + 0.1im, (1, 2), (4, 4, 3), (4, 4), (1.26e-5, 1.26e-5)),
    (s8, 2.0 + 0.3im, (1,), (6, 4, 4), (18,), (0.0,)),
    (s32, 1.0 + 0im, (1, 2, 3), (4, 4, 4), (4, 4, 4), (1.26e-5, 1.26e-5, 1.26e-5)),
    (s4, 1.0 + 1.0im, (1, 2, 3), (4, 4, 4), (4, 4, 4), (0.7, -1.9, 1.3)),
    (s32, 1.0 + 0im, (1,), (4, 2, 2), (3,), (0.7,)),
    (s32, 1.0 + 0im, (1, 2, 3), (2, 2, 2), (32, 32, 32), (0.0, 0.0, 0.0)),
]
for (s, frq, ax, N, P, θ) in fil
    snap("blcFil $s $frq $ax $N $P $θ") do; blcFil(blcSet(s, frq, ax, N, P), θ)[1:2]; end
end
# a set warmed by two other phases gives the fresh result
snap("blcFil warm") do
    st = blcSet(s8, 0.37 + 0im, (1, 2), (4, 4, 3), (5, 4))
    blcFil(st, (0.4, 0.5)); blcFil(st, (0.7, -1.9))
    blcFil(st, (1//2, 1//2))[1:2]
end
cmp = [
    ("A", [crn((4, 2, 2), s32, z), crn((1, 2, 2), (1//16, 1//32, 1//32), (4//32, 0//1, 0//1))], (1,), (6//32,), 1.0 + 0im, (0.7,)),
    ("B", [crn((4, 4, 1), s32, z), crn((1, 1, 1), (1//16, 1//16, 1//32), (4//32, 0//1, 0//1)), crn((1, 1, 1), (1//8, 1//8, 1//32), (0//1, 0//1, 1//32))],
        (1, 2), (8//32, 4//32), 1.0 + 0.1im, (1.26e-5, 1.26e-5)),
    ("C", [crn((2, 2, 2), s32, z), crn((1, 1, 1), s8, (4//32, 0//1, 0//1)), crn((1, 1, 1), (1//16, 1//8, 1//8), (2//32, 0//1, 0//1))],
        (1, 2, 3), (8//32, 4//32, 4//32), 1.0 + 1.0im, (1//2, 1//2, 1//2)),
    ("P2", [crn((2, 2, 1), sl, z), crn((1, 2, 1), (1//16, 1//32, 1//512), (2//32, 0//1, 0//1))], (1, 3), (4//32, 1//512), 1.0 + 0im, (0.7, -1.9)),
]
for (nam, regs, ax, L, frq, θ) in cmp
    snap("blcCmp $nam") do
        B = blcCmp(regs, frq, ax, L, θ)
        # older trees return (B, info)
        B isa Tuple && (B = first(B))
        [B[k] for k in sort(collect(keys(B)))]
    end
end

serialize(ARGS[1], out)
wis !== nothing && !isfile(wis) && FFTW.export_wisdom(wis)
