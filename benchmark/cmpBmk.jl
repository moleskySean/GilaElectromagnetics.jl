#=
Compare two benchmark result files. Usage:

    julia --project=. benchmark/cmpBmk.jl <old>.(json|jld2) <new>.(json|jld2) [--tolerance=0.05]

For every benchmark present in both files, prints the BenchmarkTools.judge of
the median estimates (new relative to old) and flags time regressions. Exits
with a non-zero status when any time regression is found.
=#
include("bmkEnv.jl")

using BenchmarkTools
using JLD2
using Printf

tolArg = 0.05
pthLst = String[]
for arg ∈ ARGS
    if startswith(arg, "--tolerance=")
        global tolArg = parse(Float64, last(split(arg, "=")))
    else
        push!(pthLst, arg)
    end
end
if length(pthLst) != 2
    error("usage: julia --project=. benchmark/cmpBmk.jl <old>.json <new>.json [--tolerance=0.05]")
end
oldPth, newPth = pthLst

function loadGrp(pth::AbstractString)
    if endswith(pth, ".jld2")
        return load(pth, "results")
    end
    return only(BenchmarkTools.load(pth))
end

# run metadata: from the jld2 directly, or the .meta.json sidecar of a .json
# result file (flat string/bool/int values as written by runBmk.jl)
function loadMta(pth::AbstractString)
    if endswith(pth, ".jld2")
        try
            return load(pth, "metadata")
        catch
            return nothing
        end
    end
    mtaPth = first(splitext(pth)) * ".meta.json"
    isfile(mtaPth) || return nothing
    mta = Dict{String,Any}()
    for lin ∈ eachline(mtaPth)
        mtc = match(r"^\s*\"([^\"]+)\":\s*(.*?),?\s*$", lin)
        isnothing(mtc) && continue
        mta[first(mtc.captures)] = strip(last(mtc.captures), '"')
    end
    return mta
end

oldGrp = loadGrp(oldPth)
newGrp = loadGrp(newPth)

# environment keys that change what a fair time comparison means: a mismatch
# here (fewer threads, different Julia, different machine) shifts every
# CPU-parallel benchmark and masquerades as a code regression / improvement
mtaChkLst = ("julia", "nthreads", "fftw_threads", "blas_threads", "cpu",
    "cpu_threads", "gpu", "quick", "big")
oldMta = loadMta(oldPth)
newMta = loadMta(newPth)
if isnothing(oldMta) || isnothing(newMta)
    @warn "run metadata missing for one or both result files; cannot check " *
        "that the runs are comparable"
else
    difLst = [key for key ∈ mtaChkLst if
        string(get(oldMta, key, missing)) != string(get(newMta, key, missing))]
    if !isempty(difLst)
        println("WARNING: these runs may NOT be comparable; metadata differs:")
        for key ∈ difLst
            println("    ", key, ": old = ", get(oldMta, key, "?"),
                ", new = ", get(newMta, key, "?"))
        end
        println()
    end
end

oldLvs = Dict(join(keyPth, "/") => tri for (keyPth, tri) ∈ BenchmarkTools.leaves(oldGrp))
newLvs = Dict(join(keyPth, "/") => tri for (keyPth, tri) ∈ BenchmarkTools.leaves(newGrp))
shrKey = sort!(collect(intersect(keys(oldLvs), keys(newLvs))))
if isempty(shrKey)
    error("the two result files have no benchmarks in common")
end

println("old: ", oldPth)
println("new: ", newPth)
println("tolerance: ", tolArg, "\n")
@printf("%-52s %12s %12s %8s %-12s %8s %-12s\n", "benchmark", "old", "new",
    "t-ratio", "time", "m-ratio", "memory")
regLst = String[]
for key ∈ shrKey
    oldMed = median(oldLvs[key])
    newMed = median(newLvs[key])
    jdg = judge(newMed, oldMed; time_tolerance = tolArg,
        memory_tolerance = tolArg)
    @printf("%-52s %12s %12s %8.3f %-12s %8.3f %-12s\n", key,
        BenchmarkTools.prettytime(time(oldMed)),
        BenchmarkTools.prettytime(time(newMed)),
        ratio(jdg).time, string(time(jdg)),
        ratio(jdg).memory, string(memory(jdg)))
    if time(jdg) == :regression
        push!(regLst, key)
    end
end

for (nam, mssLst) ∈ (("old", setdiff(keys(newLvs), keys(oldLvs))),
    ("new", setdiff(keys(oldLvs), keys(newLvs))))
    if !isempty(mssLst)
        println("\nBenchmarks missing from the $nam file:")
        foreach(key -> println("    ", key), sort!(collect(mssLst)))
    end
end

if isempty(regLst)
    println("\nNo time regressions (tolerance = $tolArg).")
else
    println("\nTime regressions (tolerance = $tolArg):")
    foreach(key -> println("    ", key), regLst)
    exit(1)
end
