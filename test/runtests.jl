using Test, Random, GilaElectromagnetics, LinearAlgebra
using LinearMaps, LinearOperators, SciMLOperators, Serialization
import GilaElectromagnetics.GilaVacuum: egoOpr!
import GilaElectromagnetics.GilaTypes: dflPrc

# Pkg.test() runs core, Pkg.test(test_args = ["gpu", ...]) the named groups and "all" every group
const tstGrp = [
    "core" => ["volTest.jl", "cmpVolTest.jl", "fldTest.jl", "cmpOprTest.jl", "cmpSctTest.jl",
        "susTest.jl", "vacTest.jl", "slvTest.jl", "prcnTest.jl", "oprTest.jl", "prcTest.jl",
        "serTest.jl", "mulTest.jl", "linAlgTest.jl", "extOpsTest.jl",
        "crsSclTest.jl", "cntTest.jl", "prxTest.jl", "physTest.jl"],
    "gpu" => ["gpuTest.jl"],
    "integrals" => ["vacuum/momTest.jl", "vacuum/farTest.jl", "vacuum/qssTest.jl", "vacuum/posDefTest.jl"],
    "bloch" => ["blochTest.jl"],
]
issubset(ARGS, [first.(tstGrp); "all"]) || error("The test groups are $(join(first.(tstGrp), ", ")) and all, not $(join(ARGS, ", ")).")
const runGrp = "all" in ARGS ? first.(tstGrp) : isempty(ARGS) ? ["core"] : ARGS

Random.seed!(0x67696c61)
include("tstHlp.jl")

@testset "GilaElectromagnetics" begin
    @testset "$grp" for (grp, fils) in filter(g -> first(g) in runGrp, tstGrp)
        foreach(include, fils)
    end
end
