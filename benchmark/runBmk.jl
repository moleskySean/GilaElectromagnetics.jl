#=
Benchmark runner. Usage:

    julia --project=. -t auto benchmark/runBmk.jl [--quick|--big]

    --quick : small sizes and fewer samples, for iterating
    --big   : add application-only self sizes (48, 64)

Writes three files to benchmark/results/, named bmk_<commit>[-dirty]_<time>:
    .json      BenchmarkTools results (portable; input for cmpBmk.jl)
    .meta.json machine/context tags (commit, threads, CPU/GPU, ...)
    .jld2      results + metadata together (the existing JLD2 flow)
=#

# flags must be in ENV before benchmarks.jl is included
for arg ∈ ARGS
    if arg == "--quick"
        ENV["GILA_BENCH_QUICK"] = "1"
    elseif arg == "--big"
        ENV["GILA_BENCH_BIG"] = "1"
    else
        error("unknown argument: $arg (expected --quick and/or --big)")
    end
end

include("bmkEnv.jl")

using BenchmarkTools
using CUDA
using Dates
using FFTW
using JLD2
using Printf
using LinearAlgebra: BLAS

# check if all threads are being used
function check_threads()
    available_threads = Sys.CPU_THREADS
    active_threads = Threads.nthreads()
    if active_threads < available_threads
        @warn "Julia is using only $active_threads out of $available_threads available threads.\n" *
              "Consider starting Julia with more threads using the `-t` or `--threads` flag.\n" *
              "Example: julia --project=. -t auto benchmark/runBmk.jl"
    else
        println("All available threads ($active_threads) are being used.")
    end
end
check_threads()

# a busy machine poisons every threaded timing: warn before spending an hour
# measuring other people's jobs
loadBeg = Sys.loadavg()
if first(loadBeg) > 0.1 * Sys.CPU_THREADS
    @warn "1-minute load average is $(round(first(loadBeg); digits = 2)) " *
        "with $(Sys.CPU_THREADS) CPU threads: other processes are using " *
        "this machine and timings will not be comparable across runs."
end

println("Building benchmark suite (constructs the application operators)...")
include("bmk.jl")

# tune evals for the cheap application benchmarks; the creation benchmarks
# have fixed samples/evals and are not tuned
println("Tuning application benchmarks...")
tune!(SUITE["apply"]; verbose = true)

println("Running benchmark suite...")
results = run(SUITE; verbose = true)

# metadata tags
function gitMeta()
    rootDir = dirname(@__DIR__)
    try
        cmtHsh = readchomp(`git -C $rootDir rev-parse --short=12 HEAD`)
        gitDrt = !isempty(readchomp(`git -C $rootDir status --porcelain`))
        return cmtHsh, gitDrt
    catch
        return "unknown", false
    end
end
cmtHsh, gitDrt = gitMeta()

metadata = Dict{String,Any}(
    "commit" => cmtHsh,
    "dirty" => gitDrt,
    "timestamp" => Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"),
    "julia" => string(VERSION),
    "nthreads" => Threads.nthreads(),
    "fftw_threads" => FFTW.get_num_threads(),
    "blas_threads" => BLAS.get_num_threads(),
    "cpu" => Sys.cpu_info()[1].model,
    "cpu_threads" => Sys.CPU_THREADS,
    # 1/5/15 minute load averages at run start and at save time: a high load
    # from *other* processes is invisible in nthreads but poisons timings.
    # Note the end-of-run value includes this benchmark's own threads
    "loadavg_start" => join(round.(loadBeg; digits = 2), " "),
    "loadavg" => join(round.(Sys.loadavg(); digits = 2), " "),
    "gpu" => CUDA.functional() ? CUDA.name(CUDA.device()) : "none",
    "quick" => quickRun,
    "big" => bigRun,
)

# save results: JSON (BenchmarkTools) + metadata sidecar + JLD2
jsnEsc(str::AbstractString) = replace(str, "\\" => "\\\\", "\"" => "\\\"")
jsnVal(val::AbstractString) = "\"" * jsnEsc(val) * "\""
jsnVal(val::Bool) = string(val)
jsnVal(val::Integer) = string(val)
function writeJsn(pth::AbstractString, dct::AbstractDict)
    open(pth, "w") do io
        println(io, "{")
        entLst = sort!(collect(dct); by = first)
        for (itr, (key, val)) ∈ enumerate(entLst)
            println(io, "    ", jsnVal(key), ": ", jsnVal(val),
                itr < length(entLst) ? "," : "")
        end
        println(io, "}")
    end
end

resDir = joinpath(@__DIR__, "results")
mkpath(resDir)
baseNam = "bmk_" * cmtHsh * (gitDrt ? "-dirty" : "") * "_" *
    Dates.format(now(), "yyyymmdd-HHMMSS")
jsonPth = joinpath(resDir, baseNam * ".json")
metaPth = joinpath(resDir, baseNam * ".meta.json")
jldPth = joinpath(resDir, baseNam * ".jld2")
BenchmarkTools.save(jsonPth, results)
writeJsn(metaPth, metadata)
jldsave(jldPth; results = results, metadata = metadata)

# summary
println()
@printf("%-52s %14s %14s %10s\n", "benchmark", "median time", "memory", "allocs")
for (keyPth, tri) ∈ sort(BenchmarkTools.leaves(results); by = pr -> join(first(pr), "/"))
    @printf("%-52s %14s %14s %10d\n", join(keyPth, "/"),
        BenchmarkTools.prettytime(time(median(tri))),
        BenchmarkTools.prettymemory(memory(tri)), allocs(tri))
end

println("\nResults written to:")
println("    ", jsonPth)
println("    ", metaPth)
println("    ", jldPth)
println("\nCompare against a previous run with:")
println("    julia --project=. benchmark/cmpBmk.jl <old>.json ", jsonPth)
