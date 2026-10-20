# benchpkg --rev=main,dirty --add=Metal --exeflags=--threads=auto, then benchpkgtable; GILA_BENCH_QUICK=1 drops the 64³ apply
using BenchmarkTools, GilaElectromagnetics, Metal

const SUITE = BenchmarkGroup()
const QCK = get(ENV, "GILA_BENCH_QUICK", "0") == "1"
const SCL = (1//32, 1//32, 1//32)
const DEV = (("cpu", CPU(), Returns(nothing)), ("metal", MetalBackend(), Metal.synchronize))

cub(n) = GlaVol((n, n, n), SCL)
# (target, source) of each created operator; the pairs have a 16³ source at the origin and targets
# at face gaps of 0, 1 and 320 cells, or a coarse 2:1 cube half a wavelength away
src = cub(16)
par = Dict(["self", "16"] => (src, src), ["self", "32"] => (cub(32), cub(32)),
    ["ext", "touching", "16"] => (GlaVol((16, 16, 16), SCL, (1//2, 0//1, 0//1)), src),
    ["ext", "close", "16"] => (GlaVol((16, 16, 16), SCL, (17//32, 0//1, 0//1)), src),
    ["ext", "far", "16"] => (GlaVol((16, 16, 16), SCL, (21//2, 0//1, 0//1)), src),
    ["cmp", "16"] => (GlaVol((8, 8, 8), 2 .* SCL, (1//1, 0//1, 0//1)), src))

SUITE[["create", "cold", "cpu", "self", "8"]] = @benchmarkable GlaOprVac($(cub(8))) setup = GilaElectromagnetics.clrCch!() evals = 1 samples = 3 seconds = 1e5
for (dev, bck, syn) in DEV, (key, (trg, srcVol)) in par
    SUITE[["create", "warm", dev, key...]] = @benchmarkable (GlaOprVac($trg, $srcVol; dev = $bck, prxWrn = false); $syn()) evals = 1
end

# applied operators are built once in Float64 on the CPU, then converted and moved
apl = Dict(["self", "$n"] => GlaOprVac{Float64}(cub(n)) for n in (QCK ? (16, 32) : (16, 32, 64)))
apl[["self", "32x16x8"]] = GlaOprVac{Float64}(GlaVol((32, 16, 8), SCL))
apl[["ext", "16"]] = GlaOprVac{Float64}(par[["ext", "close", "16"]]...; prxWrn = false)
apl[["cmp", "16"]] = GlaOprVac{Float64}(par[["cmp", "16"]]...)
apl[["blc", "32"]] = GlaOprVac{Float64}(cub(32); blc = (ax = (1, 2), P = (32, 32), θ = (0.7, 1//2)))
for (dev, bck, syn) in DEV, T in (Float64, Float32)
    dev == "metal" && T == Float64 && continue
    ops = Dict{Vector{String}, Any}(key => adapt(bck, GlaOprVac{T}(opr)) for (key, opr) in apl)
    ops[["adj", "32"]] = adjoint(ops[["self", "32"]])
    ops[["inv", "32"]] = InvSctOpr(ops[["self", "32"]], 2.0 + 0.1im)
    for (key, opr) in ops
        v = adapt(bck, rand(Complex{T}, size(opr, 2)))
        SUITE[["apply", dev, "f$(8sizeof(T))", key...]] = @benchmarkable ($opr * $v; $syn())
    end
end
