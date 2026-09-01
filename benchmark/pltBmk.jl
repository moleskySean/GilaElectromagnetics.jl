#=
Plot creation / application times from runBmk.jl. Usage:

    julia --project=. benchmark/pltBmk.jl [old.(json|jld2)] [new.(json|jld2)]

With no arguments, plots the newest .json in benchmark/results/. With one
result file, plots that run: time versus size for the cubic self operators,
CPU and GPU overlaid when both are present, saved to benchmark/bmk.png. With
two result files, overlays both runs in the spirit of cmpBmk.jl --- the first
(old) dashed and faded, the second (new) solid --- saved to
benchmark/bmkCmp.png.
=#
include("bmkEnv.jl")

using BenchmarkTools
using JLD2
using Plots
using Statistics

function loadGrp(pth::AbstractString)
    if endswith(pth, ".jld2")
        return load(pth, "results")
    end
    return only(BenchmarkTools.load(pth))
end

# legend label for a result file: the commit part of the runBmk.jl naming
# convention when present, the bare file name otherwise
function runLbl(pth::AbstractString)
    mtc = match(r"^bmk_([0-9a-f]+(?:-dirty)?)_", basename(pth))
    return isnothing(mtc) ? first(splitext(basename(pth))) :
        String(first(mtc.captures))
end

length(ARGS) <= 2 || error("usage: julia --project=. benchmark/pltBmk.jl " *
    "[old.(json|jld2)] [new.(json|jld2)]")
resPthLst = if isempty(ARGS)
    resDir = joinpath(@__DIR__, "results")
    candLst = isdir(resDir) ?
        filter(pth -> endswith(pth, ".json") && !endswith(pth, ".meta.json"),
            readdir(resDir; join = true)) : String[]
    if isempty(candLst)
        error("no result files in $resDir; run benchmark/runBmk.jl " *
            "first or pass a result file explicitly")
    end
    [last(sort!(candLst; by = mtime))]
else
    collect(String, ARGS)
end
cmpRun = length(resPthLst) == 2
foreach(pth -> println("Plotting ", pth), resPthLst)
runLst = [(runLbl(pth), loadGrp(pth)) for pth ∈ resPthLst]

# (n, median seconds, 25% quantile, 75% quantile) for the cubic self
# benchmarks of one device. Medians match what cmpBmk.jl judges, and with the
# interquartile band stay readable for the heavy-tailed (GC / GPU-sync
# outlier) timing distributions that make mean ± std ribbons explode
function slfSrs(resGrp::BenchmarkGroup, opNam::String, devNam::String)
    (haskey(resGrp, opNam) && haskey(resGrp[opNam], devNam) &&
        haskey(resGrp[opNam][devNam], "self")) || return nothing
    ptsLst = Tuple{Int,Float64,Float64,Float64}[]
    for (key, tri) ∈ resGrp[opNam][devNam]["self"]
        dimLst = parse.(Int, split(key, "x"))
        allequal(dimLst) || continue # line plot is cubic sizes only
        tmQnt = quantile(tri.times, (0.25, 0.5, 0.75)) ./ 1e9
        push!(ptsLst, (first(dimLst), tmQnt[2], tmQnt[1], tmQnt[3]))
    end
    isempty(ptsLst) && return nothing
    return sort!(ptsLst; by = first)
end

function pltOpr(runLst::AbstractVector, opNam::String, titNam::String)
    plt = plot(xscale = :log2, yscale = :log10, ylabel = "Time [s]",
        title = titNam * " Benchmarks", legend = :topleft)
    nTck = Int[]
    for (runItr, (lblNam, resGrp)) ∈ enumerate(runLst)
        # in a comparison the first (old) run is dashed and faded; devices
        # keep a fixed color across runs so old/new pairs read together
        oldSty = length(runLst) > 1 && runItr == 1
        for (devItr, (devNam, mrk)) ∈
            enumerate((("cpu", :circle), ("gpu", :utriangle)))
            srs = slfSrs(resGrp, opNam, devNam)
            isnothing(srs) && continue
            union!(nTck, first.(srs))
            medLst = getindex.(srs, 2)
            # asymmetric interquartile ribbon about the median; quantiles of
            # positive times keep the band positive on the log axis
            ribLo = medLst .- getindex.(srs, 3)
            ribHi = getindex.(srs, 4) .- medLst
            plot!(plt, first.(srs), medLst;
                  ribbon = (ribLo, ribHi), m = mrk, color = devItr + (runItr-1)*2,
                linestyle = oldSty ? :dash : :solid,
                label = uppercase(devNam) * " " * titNam *
                    (length(runLst) > 1 ? " ($lblNam)" : ""))
        end
    end
    sort!(nTck)
    isempty(nTck) || plot!(plt; xticks = (nTck, string.(nTck)))
    return plt
end

crtPlt = pltOpr(runLst, "create", "Creation")
appPlt = pltOpr(runLst, "apply", "Application")
xlabel!(appPlt, "Self operator size: (n, n, n)")
plt = plot(crtPlt, appPlt; layout = (2, 1), size = (800, 600))

pngPth = joinpath(@__DIR__, cmpRun ? "bmkCmp.png" : "bmk.png")
plot!(plt, dpi=300)
savefig(plt, pngPth)
println("Plot saved as ", pngPth)
if isinteractive()
    display(plt)
end
