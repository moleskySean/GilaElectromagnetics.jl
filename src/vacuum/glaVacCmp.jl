using KernelAbstractions
using CUDA
using Serialization

"""
    GlaKerOpt

Abstract type for computational information that determines how the Green function
operator is computed. Concrete implementations include `CPUKerOpt` for CPU computation
and `GPUKerOpt` for GPU computation.
"""
abstract type GlaKerOpt end

"""
    CPUKerOpt <: GlaKerOpt

Options for CPU computation of the Green function operator.

`CPUKerOpt` determines the accuracy of the Green function computation for cells in contact through its integration order. Higher values give better accuracy but require more computation. The phase factor allows for complex frequencies, which is useful for modeling dispersive media or for numerical stability.

# Fields
- `frqPhz::Number`: Multiplicative scaling factor allowing for complex frequencies
- `intOrd::Integer`: Gauss-Legendre integration order for cells in contact
- `adjMod::Bool`: Adjoint mode flag (true if adjoint mode is enabled)
- `bckEnd::CPU`: Backend for CPU computation
"""
mutable struct CPUKerOpt <: GlaKerOpt
    frqPhz::Number
    intOrd::Integer
    adjMod::Bool
    bckEnd::CPU
end

"""
    CPUKerOpt()

Construct a CPUKerOpt with default values.

The default integration order of 32 provides a good balance between accuracy and computational cost for most applications.

# Returns
- `CPUKerOpt`: A new CPU kernel options object with:
  - Phase factor of 1.0 + 0.0im
  - Integration order of 32
  - Adjoint mode disabled
  - Default CPU backend
"""
CPUKerOpt() = CPUKerOpt(1.0+0.0im, 32, false, CPU())

"""
    frqPhz(opt::CPUKerOpt) -> Number

Get the multiplicative scaling factor allowing for complex frequencies.

# Arguments
- `opt::CPUKerOpt`: The CPU kernel options object

# Returns
- `Number`: The phase factor
"""
frqPhz(opt::CPUKerOpt) = opt.frqPhz

"""
    intOrd(opt::CPUKerOpt)

Get the integration order from a CPU kernel options object.

# Arguments
- `opt::CPUKerOpt`: The CPU kernel options object

# Returns
- `Integer`: The integration order
"""
intOrd(opt::CPUKerOpt) = opt.intOrd

"""
    adjMod(opt::CPUKerOpt)

Get the adjoint mode flag from a CPU kernel options object.

# Arguments
- `opt::CPUKerOpt`: The CPU kernel options object

# Returns
- `Bool`: True if adjoint mode is enabled, false otherwise
"""
adjMod(opt::CPUKerOpt) = opt.adjMod

"""
    bckEnd(opt::CPUKerOpt)

Get the CPU backend from a CPU kernel options object.

# Arguments
- `opt::CPUKerOpt`: The CPU kernel options object

# Returns
- `CPU`: The CPU backend
"""
bckEnd(opt::CPUKerOpt) = opt.bckEnd

"""
    arrTyp(opt::CPUKerOpt)

Get the array type from a CPU kernel options object.

# Arguments
- `opt::CPUKerOpt`: The CPU kernel options object

# Returns
- `Type`: Array{ComplexF64}
"""
arrTyp(::CPUKerOpt) = Array{ComplexF64}

"""
    useCpu(opt::CPUKerOpt)

Does nothing. This function is a placeholder for consistency with the GPU version.

# Arguments
- `opt::CPUKerOpt`: The CPU kernel options object

# Returns
- `CPUKerOpt`: The same CPU kernel options object
"""
useCpu(opt::CPUKerOpt) = opt

"""
    useGpu(opt::CPUKerOpt)

Switch to GPU computation for the given CPU kernel options object.

Creates a new `GPUKerOpt` object with the same phase factor and integration order as the original `CPUKerOpt`, but with default thread and block counts for GPU computation. The adjoint mode flag is also preserved.

# Arguments
- `opt::CPUKerOpt`: The CPU kernel options object

# Returns
- `GPUKerOpt`: A new GPU kernel options object with:
  - Phase factor of `opt.frqPhz`
  - Integration order of `opt.intOrd`
  - Default thread and block counts for GPU computation
  - Adjoint mode flag from `opt`
  - Default CUDA backend
"""
useGpu(opt::CPUKerOpt) = GPUKerOpt(opt.frqPhz, opt.intOrd, (128, 2, 1), (1, 128, 256), opt.adjMod, CUDABackend())

"""
    GPUKerOpt <: GlaKerOpt

Options for GPU computation of the Green function operator.

`GPUKerOpt` determines the parallelization strategy for GPU computation through its thread and block counts. The integration order determines the accuracy of the Green function computation for cells in contact. Higher values give better accuracy but require more computation. The phase factor allows for complex frequencies, which is useful for modeling dispersive media or for numerical stability.

# Fields
- `frqPhz::Number`: Multiplicative scaling factor allowing for complex frequencies
- `intOrd::Integer`: Gauss-Legendre integration order for cells in contact
- `numTrd::NTuple{3, Integer}`: Number of threads to use when running GPU kernels
- `numBlk::NTuple{3, Integer}`: Number of thread blocks to use when running GPU kernels
- `adjMod::Bool`: Adjoint mode flag (true if adjoint mode is enabled)
- `bckEnd::GPU`: Backend for GPU computation
"""
mutable struct GPUKerOpt <: GlaKerOpt
    frqPhz::Number
    intOrd::Integer
    numTrd::NTuple{3, Integer}
    numBlk::NTuple{3, Integer}
    adjMod::Bool
    bckEnd::GPU
end

"""
    GPUKerOpt()

Construct a GPUKerOpt with default values.

The default thread and block counts are chosen to provide good performance on most NVIDIA GPUs. The default integration order of 32 provides a good balance between accuracy and computational cost for most applications.

# Returns
- `GPUKerOpt`: A new GPU kernel options object with:
  - Phase factor of 1.0 + 0.0im
  - Integration order of 32
  - 128 threads per block
  - 256 blocks
  - Adjoint mode disabled
  - Default CUDA backend
"""
GPUKerOpt() = GPUKerOpt(1.0+0.0im, 32, (128, 2, 1), (1, 128, 256), false, CUDABackend())

"""
    frqPhz(opt::GPUKerOpt)

Get the phase factor from a GPU kernel options object.

# Arguments
- `opt::GPUKerOpt`: The GPU kernel options object

# Returns
- `Number`: The phase factor
"""
frqPhz(opt::GPUKerOpt) = opt.frqPhz

"""
    intOrd(opt::GPUKerOpt)

Get the integration order from a GPU kernel options object.

# Arguments
- `opt::GPUKerOpt`: The GPU kernel options object

# Returns
- `Integer`: The integration order
"""
intOrd(opt::GPUKerOpt) = opt.intOrd

"""
    adjMod(opt::GPUKerOpt)

Get the number of threads to use when running GPU kernels.

The thread count determines the parallelization strategy for GPU computation. Higher values may improve performance but require more GPU resources.
"""
numTrd(opt::GPUKerOpt) = opt.numTrd

"""
    numBlk(opt::GPUKerOpt) -> NTuple{3, Integer}

Get the number of thread blocks to use when running GPU kernels.

The block count determines the parallelization strategy for GPU computation. Higher values may improve performance but require more GPU resources.
"""
numBlk(opt::GPUKerOpt) = opt.numBlk

"""
    adjMod(opt::GPUKerOpt) -> Bool

# Arguments
- `opt::GPUKerOpt`: The GPU kernel options object

# Returns
- `Bool`: True if adjoint mode is enabled, false otherwise
"""
adjMod(opt::GPUKerOpt) = opt.adjMod

"""
    bckEnd(opt::GPUKerOpt)

Get the GPU backend from a GPU kernel options object.

# Arguments
- `opt::GPUKerOpt`: The GPU kernel options object

# Returns
- `GPU`: The GPU backend
"""
bckEnd(opt::GPUKerOpt) = opt.bckEnd

"""
    arrTyp(opt::GPUKerOpt)

Get the array type from a GPU kernel options object.

# Arguments
- `opt::GPUKerOpt`: The GPU kernel options object

# Returns
- `Type`: CuArray{ComplexF64}
"""
arrTyp(::GPUKerOpt) = CuArray{ComplexF64}


"""
    useCpu(opt::GPUKerOpt)

Switch to CPU computation for the given GPU kernel options object.

Creates a new `CPUKerOpt` object with the same phase factor and integration order as the original `GPUKerOpt`, but with default values for CPU computation. The thread and block counts are ignored in this case. The adjoint mode flag is also preserved.

# Arguments
- `opt::GPUKerOpt`: The GPU kernel options object

# Returns
- `CPUKerOpt`: A new CPU kernel options object with:
  - Phase factor of `opt.frqPhz`
  - Integration order of `opt.intOrd`
  - Adjoint mode flag from `opt`
  - Default CPU backend
"""
useCpu(opt::GPUKerOpt) = CPUKerOpt(opt.frqPhz, opt.intOrd, opt.adjMod, CPU())

"""
    useGpu(opt::GPUKerOpt)

Does nothing. This function is a placeholder for consistency with the CPU version.

# Arguments
- `opt::GPUKerOpt`: The GPU kernel options object

# Returns
- `GPUKerOpt`: The same GPU kernel options object
"""
useGpu(opt::GPUKerOpt) = opt

# Add serialization support for GlaKerOpt
function Serialization.serialize(io::IO, opt::CPUKerOpt)
    serialize(io, opt.frqPhz)
    serialize(io, opt.intOrd)
    serialize(io, opt.adjMod)
end

function Serialization.deserialize(io::IO, ::Type{CPUKerOpt})
    frqPhz = deserialize(io)
    intOrd = deserialize(io)
    adjMod = deserialize(io)
    return CPUKerOpt(frqPhz, intOrd, adjMod, CPU())
end

function Serialization.serialize(io::IO, opt::GPUKerOpt)
    serialize(io, opt.frqPhz)
    serialize(io, opt.intOrd)
    serialize(io, opt.numTrd)
    serialize(io, opt.numBlk)
    serialize(io, opt.adjMod)
end

function Serialization.deserialize(io::IO, ::Type{GPUKerOpt})
    frqPhz = deserialize(io)
    intOrd = deserialize(io)
    numTrd = deserialize(io)
    numBlk = deserialize(io)
    adjMod = deserialize(io)
    return GPUKerOpt(frqPhz, intOrd, numTrd, numBlk, adjMod, CUDABackend())
end
