using GilaElectromagnetics

num_cells = (8, 8, 8)
cell_size = (1//32, 1//32, 1//32)
dev = CPU() # swap for CUDABackend(), MetalBackend() or OpenCLBackend()
const T = Float32 # Float64 for double precision

vol = GlaVol(num_cells, cell_size, (0//1, 0//1, 0//1))
G = GlaOprVac{T}(vol; dev=dev)
source_vec = rand(eltype(G), size(G, 2))
field_vec = G * source_vec # Apply the vacuum Green function to the source vector
println("It works!")
