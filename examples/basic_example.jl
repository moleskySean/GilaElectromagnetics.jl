using GilaElectromagnetics

num_cells = (8, 8, 8)
cell_size = (1//32, 1//32, 1//32)
has_gpu = false # Set to true if you have a CUDA enabled GPU
const T = ComplexF32 # Set to ComplexF64 for double precision

G = GlaOpr(num_cells, cell_size; useGpu=has_gpu, setTyp=T)
source_vec = rand(eltype(G), size(G, 2))
field_vec = G * source_vec # Apply the Greens operator to the source vector
println("It works!")
