#=
Benchmark suite for GilaElectromagnetics.

Defines `SUITE::BenchmarkGroup` following the PkgBenchmark convention. Include
this file after activating the benchmark environment (see bmkEnv.jl). The
run/save/summary logic can be found in runBmk.jl, and the comparison/plotting
scripts are cmpBmk.jl and pltBmk.jl, respectively.

Configuration is via environment variables (set by runBmk.jl flags):
    GILA_BENCH_QUICK=1 : small sizes and fewer samples, for iterating
    GILA_BENCH_BIG=1   : add application-only self sizes (48, 64)

Suite layout (all keys are strings so results survive the JSON round-trip):
    SUITE["create"][dev]["self"]["16x16x16"]
    SUITE["create"][dev]["ext"]["close"]["16x16x16"]
    SUITE["apply"][dev]["self"]["16x16x16"]
    SUITE["apply"][dev]["ext"]["close"]["16x16x16"]
with dev ∈ ("cpu", "gpu"); the "gpu" groups exist only when CUDA.functional().

Note that building this suite is itself expensive: every application benchmark
needs its operator constructed once (numerical integration) at include time.
=#
using BenchmarkTools
using CUDA
using GilaElectromagnetics
using Random

const quickRun = get(ENV, "GILA_BENCH_QUICK", "0") ∈ ("1", "true")
const bigRun = get(ENV, "GILA_BENCH_BIG", "0") ∈ ("1", "true")
const gpuRun = CUDA.functional()

# every volume uses this cell size
const celScl = (1//32, 1//32, 1//32)

# self operators: cubic sizes plus one non-cubic case to catch anisotropy
# effects in the FFT stage
const slfCelLst = quickRun ?
    [(4, 4, 4), (8, 8, 8)] :
    [[(n, n, n) for n ∈ (4, 8, 16, 24, 32)]; (32, 16, 8)]
# application-only self sizes
const bigCelLst = [(48, 48, 48), (64, 64, 64)]
# external operators: same-size cubic source / target pairs
const extCelLst = quickRun ? [8] : [8, 16, 24]
# labelled external cases; the value is the face-to-face gap in cells along x
const extGapLst = ["touching", "close", "mid", "far"]
extGapCel(lbl::String, n::Integer) =
    lbl == "touching" ? 0 :
    lbl == "close" ? 1 :
    lbl == "mid" ? n :
    lbl == "far" ? 20 * n :
    throw(ArgumentError("unknown external gap label: $lbl"))

# creation samples are expensive: few samples, one eval each, gc between
# samples; the seconds budget is set high so the sample count is the limit
const crtSamples = quickRun ? 2 : 5
const crtSeconds = 1e5
const appSeconds = quickRun ? 2 : 10

szKey(cel::NTuple{3,<:Integer}) = join(cel, "x")
noMsk() = ntuple(_ -> 0:0, 3)

slfVol(cel::NTuple{3,<:Integer}) = GlaVol(cel, celScl, (0//1, 0//1, 0//1))
# source at the origin, same-size target separated along x so that the
# face-to-face gap is gapCel cells (gapCel = 0 means the faces touch)
function extVolPar(n::Integer, gapCel::Integer)
    srcVol = GlaVol((n, n, n), celScl, (0//1, 0//1, 0//1))
    trgVol = GlaVol((n, n, n), celScl, ((n + gapCel) * celScl[1], 0//1, 0//1))
    return trgVol, srcVol
end

#=
Operators for the application benchmarks are built at the memory layer
(GlaVacOprMem) rather than through GlaOprVac(trgVol, srcVol): the user-facing
constructor reroutes volumes in face contact through the union/self path, but
the "touching" case exists precisely to exercise the external contact fill
(genCntVol + egoFunExtCnt!). For every other case the two constructions are
equivalent up to the (negligible) wrapper.
=#
function mkOprCpu(trgVol::GlaVol, srcVol::GlaVol)
    memVac = GlaVacOprMem(CPUKerOpt(), trgVol, srcVol)
    return GlaOprVac(memVac, noMsk(), noMsk())
end
# GPU twin of a CPU operator: reuses the CPU-computed Fourier data (the
# integration is identical) instead of paying for a second full creation
function mkOprGpu(oprCpu::GlaOprVac)
    memVac = GlaVacOprMem(GPUKerOpt(), map(CuArray, oprCpu.mem.egoFur),
        oprCpu.mem.trgVol, oprCpu.mem.srcVol)
    return GlaOprVac(memVac, noMsk(), noMsk())
end

# fixed seed so application inputs are reproducible across runs
const rngBmk = Random.Xoshiro(0x67696c61)
randVec(opr::GlaOprVac) = randn(rngBmk, ComplexF64, size(opr, 2))

# register the application benchmarks for one operator under
# SUITE["apply"][dev][subKey...][key]
function addAppBmk!(appGrp::BenchmarkGroup, oprCpu::GlaOprVac,
    subKey::Vector{String}, key::String)
    vecHst = randVec(oprCpu)
    cpuGrp = foldl(getindex, subKey; init = appGrp["cpu"])
    cpuGrp[key] = @benchmarkable $oprCpu * $vecHst seconds = appSeconds
    if gpuRun
        oprGpu = mkOprGpu(oprCpu)
        vecDev = CuArray(vecHst)
        # always synchronize: benchmarking an async GPU launch is meaningless.
        # The input lives on the device, so the (warning, copying) implicit
        # host -> device path is never measured here.
        gpuGrp = foldl(getindex, subKey; init = appGrp["gpu"])
        gpuGrp[key] =
            @benchmarkable CUDA.@sync($oprGpu * $vecDev) seconds = appSeconds
    end
    return nothing
end

if !gpuRun
    @warn "CUDA is not functional: GPU benchmarks are skipped."
end

SUITE = BenchmarkGroup()
let devLst = gpuRun ? ["cpu", "gpu"] : ["cpu"]
    for opNam ∈ ("create", "apply")
        SUITE[opNam] = BenchmarkGroup()
        for devNam ∈ devLst
            devGrp = SUITE[opNam][devNam] = BenchmarkGroup()
            devGrp["self"] = BenchmarkGroup()
            devGrp["ext"] = BenchmarkGroup()
            for gapLbl ∈ extGapLst
                devGrp["ext"][gapLbl] = BenchmarkGroup()
            end
        end
    end
end

# self operators
for cel ∈ slfCelLst
    key = szKey(cel)
    volSlf = slfVol(cel)
    SUITE["create"]["cpu"]["self"][key] =
        @benchmarkable GlaOprVac($volSlf) samples = crtSamples evals = 1 seconds = crtSeconds gcsample = true
    if gpuRun
        SUITE["create"]["gpu"]["self"][key] =
            @benchmarkable GlaOprVac($volSlf; useGpu = true) samples = crtSamples evals = 1 seconds = crtSeconds gcsample = true
    end
    @info "Building self application operator $key"
    addAppBmk!(SUITE["apply"], mkOprCpu(volSlf, volSlf), ["self"], key)
end
if bigRun
    for cel ∈ bigCelLst
        key = szKey(cel)
        volSlf = slfVol(cel)
        @info "Building big self application operator $key (this is slow)"
        addAppBmk!(SUITE["apply"], mkOprCpu(volSlf, volSlf), ["self"], key)
    end
end

# external operators
for n ∈ extCelLst, gapLbl ∈ extGapLst
    key = szKey((n, n, n))
    trgVol, srcVol = extVolPar(n, extGapCel(gapLbl, n))
    if gapLbl == "touching"
        # GlaOprVac(trgVol, srcVol) reroutes face-contact volumes through the
        # union/self path; the contact fill only runs at the memory layer
        SUITE["create"]["cpu"]["ext"][gapLbl][key] =
            @benchmarkable GlaVacOprMem(CPUKerOpt(), $trgVol, $srcVol) samples = crtSamples evals = 1 seconds = crtSeconds gcsample = true
        if gpuRun
            SUITE["create"]["gpu"]["ext"][gapLbl][key] =
                @benchmarkable GlaVacOprMem(GPUKerOpt(), $trgVol, $srcVol) samples = crtSamples evals = 1 seconds = crtSeconds gcsample = true
        end
    else
        SUITE["create"]["cpu"]["ext"][gapLbl][key] =
            @benchmarkable GlaOprVac($trgVol, $srcVol) samples = crtSamples evals = 1 seconds = crtSeconds gcsample = true
        if gpuRun
            SUITE["create"]["gpu"]["ext"][gapLbl][key] =
                @benchmarkable GlaOprVac($trgVol, $srcVol; useGpu = true) samples = crtSamples evals = 1 seconds = crtSeconds gcsample = true
        end
    end
    @info "Building external application operator $gapLbl $key"
    addAppBmk!(SUITE["apply"], mkOprCpu(trgVol, srcVol), ["ext", gapLbl], key)
end
