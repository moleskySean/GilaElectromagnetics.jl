# Importing the JLL is what activates VkFFT's CUDA backend.
module GilaCUDAExt

using CUDA
using VkFFT_CUDA_jll
import GilaElectromagnetics: devMem

devMem(::CUDABackend) = CUDA.free_memory()

end
