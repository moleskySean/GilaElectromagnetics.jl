using Test

# println("Running vacuum test suite...")

@testset "Vacuum Tests" begin
    include("intConTest.jl")
    include("anaTest.jl")
    include("extSlfTest.jl")
    include("posDefTest.jl")
    include("cpuGpuTest.jl")
    include("serializationTest.jl")
end

# println("Vacuum test suite completed.")
