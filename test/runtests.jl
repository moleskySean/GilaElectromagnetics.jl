using Test, Random, GilaElectromagnetics, LinearAlgebra
using LinearMaps, LinearOperators, SciMLOperators, Serialization
# pocl_jll before OpenCL activates its platform. Metal compiles in the world of its __init__,
# so loading it last lets it see the GPUCompiler methods OpenCL adds
using CUDA, pocl_jll, OpenCL, Metal
import GilaElectromagnetics.GilaVolumes: uniVol
import GilaElectromagnetics.GilaVacuum: egoOpr!
import GilaElectromagnetics.GilaTypes: dflPrc

Random.seed!(0x67696c61)
include("tstHlp.jl")

@testset "GilaElectromagnetics" begin
    include("volTest.jl")
    include("cmpVolTest.jl")
    include("fldTest.jl")
    include("cmpOprTest.jl")
    include("cmpSctTest.jl")
    include("susTest.jl")
    include("vacTest.jl")
    include("slvTest.jl")
    include("prcnTest.jl")
    include("oprTest.jl")
    include("prcTest.jl")
    include("serTest.jl")
    include("jld2Test.jl")
    include("mulTest.jl")
    include("linAlgTest.jl")
    include("extOpsTest.jl")
    include("crsSclTest.jl")
    include("cntTest.jl")
    include("prxTest.jl")
    include("physTest.jl")
    include("blochTest.jl")
    include("vacuum/runtests.jl")
end
