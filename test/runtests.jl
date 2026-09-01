using Test, Random, GilaElectromagnetics, LinearAlgebra
using LinearMaps, LinearOperators, SciMLOperators, Serialization, CUDA
import GilaElectromagnetics.GilaVolumes: uniVol

Random.seed!(0x67696c61)
include("tstHlp.jl")

@testset "GilaElectromagnetics" begin
    include("volTest.jl")
    include("cmpVolTest.jl")
    include("fldTest.jl")
    include("cmpOprTest.jl")
    include("cmpSctTest.jl")
    include("vacTest.jl")
    include("slvTest.jl")
    include("oprTest.jl")
    include("serTest.jl")
    include("linAlgTest.jl")
    include("mulTest.jl")
    include("extOpsTest.jl")
    include("crsSclTest.jl")
    include("cntTest.jl")
    include("prxTest.jl")
    include("physTest.jl")
end
