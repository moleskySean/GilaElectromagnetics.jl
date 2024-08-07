# Usage

The following presents typical uses of GilaElectromagnetics.

## Vacuum Green's function

Gila solves Maxwell's equations in a vacuum by employing finite element methods
on a given volume. What Gila computes is the action of the vacuum Green's
function ``\textbf{G}_0`` on a vector. The operator ``\textbf{G}_0`` is
represented in memory by the structure
[`GlaOprMem`](library.md#GilaElectromagnetics.GlaOprMem). A simpler object to
use which acts exactly as you would want an operator to behave is the
[`GlaOpr`](library.md#GilaElectromagnetics.GlaOpr) structure. To initialize
either a [`GlaOprMem`](library.md#GilaElectromagnetics.GlaOprMem) object or a
[`GlaOpr`](library.md#GilaElectromagnetics.GlaOpr) object, information about the
space where the equations are solved must be defined.

In Gila, volumes must be rectangular prisms, which are then subdivided into
smaller volumes called *cells*. The size of these cells in a specific direction
is given in fractions of a wavelength, because Gila solves a system at one given
wavelength.

### Self and external operators

The vacuum Green's function for a given volume is obtained using a
[`GlaOpr`](library.md#GilaElectromagnetics.GlaOpr) constructor. There are two
cases to consider. The first one is the simplest, where the source volume is
the same as the target volume. The resulting operator is called the *self
Green's operator*. The following example shows how to build one:

```julia
# Volume definition
cells = (8, 8, 8) # Defines a volume with 8 cells in each direction
scale = (1//32, 1//32, 1//32) # Each cell is 1/32 of a wavelength in each direction
origin = (0//1, 0//1, 0//1) # OPTIONAL: The volume is located at the origin

# Self Greens operator
G_0 = GlaOpr(cells, scale, origin)
```

!!! note "Origin"
    The origin in this context refers to the cell `(1, 1, 1)`, located in the corner of the volume. 

Two more parameters can be passed: `useGpu` if an *Nvidia* GPU is available, and
`setTyp` to either use 32 or 64 bit complex numbers (single or double
precision). The syntax to use these parameters is shown here:

```julia
G_0 = GlaOpr(cells, scale, origin; useGpu=true, setTyp=ComplexF64)
```

The second case to consider is where there are two separate volumes, or when the defined volume (source) is different from the one where a solution is desired (target). This can be useful if a medium and current sources are defined in a region, but the space to be simulated is either partially or fully contained by the source volume, or if both are completely separated. The Green's function is then called the *external Green's operator*, and it can be constructed just like the self operator, only with two volumes required :

```julia
# Source volume definition
src_cells = (src_nx, src_ny, src_nz) # tuple of Integers
src_scale = (src_sclx, src_scly, src_sclz) # tuple of Rationals
src_origin = (src_orgx, src_orgy, src_orgz) # tuple of Rationals, REQUIRED

# Target volume definition
trg_cells = (trg_nx, trg_ny, trg_nz) # tuple of Integers
trg_scale = (trg_sclx, trg_scly, trg_sclz) # tuple of Rationals
trg_origin = (trg_orgx, trg_orgy, trg_orgz) # tuple of Rationals, REQUIRED

# External Green's operator
G_0 = GlaOpr(src_cells, src_scale, src_origin, trg_cells, trg_scale, trg_origin)
```
The same optionnal parameters for CUDA and the complex type could be given.

!!! note "Origin"
    The origin in this context refers to the cell `(1, 1, 1)`, located in the corner of the volume. 

This operator uses the `GlaOpr` type, which in itself is an abstraction wrapper
for the [`GlaOprMem`](library.md#GilaElectromagnetics.GlaOprMem). It is *not* a
matrix, but it can be used as a linear operator, i.e., you can multiply a vector
by it.

## [Scattering problem](@id scattering)

Gila is designed primarily to tackle the *scattering problem*, which asks to
find the total field ``\textbf{f}_t`` produced, given an incident field
``\textbf{f}_i`` and a dielectric profile. This allows us to solve Maxwell's
equations in matter.

### Theoretical overview

We can always decompose the total field into an incident and a scattered part:
```math
\textbf{f}_t = \textbf{f}_i + \textbf{f}_s
```

Furthermore, we define the matrix ``\textbf{X}`` to represent permittivity and
permeability as such, while still using natural units :

```math
\textbf{X} =
\begin{pmatrix}
\textbf{X}_{je} & \textbf{0} \\
\textbf{0} & \textbf{X}_{mh}
\end{pmatrix}
```

Here, ``\textbf{X}_{je}`` and ``\textbf{X}_{mh}`` are diagonal matrices
described by the electric and magnetic susceptibility ``\chi_e`` and ``\chi_m``
respectively:

```math
\textbf{X}_{je} = \frac{i}{k_0}\chi_e \textbf{I}_{3 \times 3}
```
```math
\textbf{X}_{mh} = -\frac{i}{k_0}\chi_m \textbf{I}_{3 \times 3}
```
where:

```math
\textbf{P} = \epsilon_0 \chi_e \textbf{E}
```
```math
\textbf{M} = \chi_m \textbf{H}
```

with ``\textbf{P}``, the polarization density, and ``\textbf{M}``, the
magnetization field. 

In the same way that the fields can be decomposed into incident, scattered and
total parts, so can the polarization density vector: ``\textbf{p}_t =
\textbf{p}_i + \textbf{p}_s``. After such a decomposition, the constitutive
relations become:

```math
\frac{i}{k_0}\textbf{p}_s = \textbf{X}\textbf{f}_t
```

Recall the [relevant](./concepts.md#maxwell) Maxwell's equations in vacuum. By
their linearity, their scattered part can be written simply as:

```math
\textbf{M}_0 \textbf{f}_s = \frac{i}{k_0}\textbf{p}_s
```
Combining with the previous equation:

```math
\textbf{M}_0 \textbf{f}_s = \textbf{X}\textbf{f}_t
```

With the vacuum Green's function being the inverse of ``\textbf{M}_0``:

```math
\textbf{f}_s = \textbf{G}_0 \textbf{X}\textbf{f}_t
```

With the decomposition of the scattered field into its total and initial parts,
along with substitutions with previous equations, the following relation can be
found:

```math
\textbf{f}_t - \frac{i}{k_0}\textbf{G}_0 \textbf{p}_s =\frac{i}{k_0}\textbf{G}_0 \textbf{p}_i
```

Finally, multiplying both sides by ``\textbf{X}``, using further substitutions
and decomposing the density vector the same way the fields vector was, a crucial
equation is reached:

```math
(\textbf{I}_{6 \times 6} - \textbf{X}\textbf{G}_0)\textbf{p}_t = \textbf{p}_i
```

This is known as the [Lippmann-Schwinger
equation](https://en.wikipedia.org/wiki/Lippmann%E2%80%93Schwinger_equation). In
order to obtain the *total polarization current density* from the material's
properties and from an initial polarization current density, the following needs
to be solved:

```math
\textbf{p}_t = (\textbf{I}_{6 \times 6} - \textbf{X}\textbf{G}_0)^{-1}\textbf{p}_i
```

where ``\textbf{W} = (\textbf{I}_{6 \times 6} - \textbf{X}\textbf{G}_0)^{-1}``
is denoted as the *scattering operator*. With a way to define this operator,
Gila effectively allows the Maxwell's equations to be solved in matter.

!!! danger "Accessing specific matrices"
    It is important to note that the mathematical solutions above would apply to
    a single cell of a system. For a whole system, they are still correct, but
    the matrices and vectors are expanded, and each cell adds six elements (on
    the diagonal for the matrix ``\textbf{X}``). This is explained further in
    the implementation examples. 
    
    It is important to to keep in mind is that these matrices become enormous
    very quickly as the dimensions of a volume increases. Gila's trick is to
    actually *not compute* the Green's function, but to only compute it's
    application on a vector ``\textbf{v}``. The same would go for
    ``\textbf{W}``, since it's composed of the Green's operator. Thus, Gila
    outputs ``\textbf{G}_0 \textbf{v}``, ``\textbf{W}\textbf{p}_i`` (provided
    Lippmann-Schwinger is implemented) or anything similar.

### Implementation

This section will showcase an implementation of the scattering operator, to
solve the Lippmann-Schwinger equation in this context. These can be implemented
as a module in a file of a project. To begin, the following packages must be
imported in order for the following code to function:

```julia
# At the beginning of an exported module
using LinearAlgebra
using JLD2
using CUDA
using GilaElectromagnetics
```

In a nutshell, the part that takes the longest to compute for Gila are *fast
Fourier transforms (FFTs)*. However, the implementation of
[JLD2](https://juliaio.github.io/JLD2.jl/dev/) gives the possibility of storing
(serializing) these FFTs, drastically speeding things up for subsequent uses of
Gila for systems of identical dimensions. The following function simply verifies
the existence of a folder named `preload` where the Fourier transforms will be
stored:

```julia
function get_preload_dir()
    found_dir = false
    dir = "preload/"
    for i in 1:10
        if !isdir(dir)
            dir = "../"^i * "preload/"
        else
            found_dir = true
            break
        end
    end
    if !found_dir
        error("Could not find preload directory. Please create a directory named 'preload' in the current directory or parent directories.")
    end
    return dir
end
```

The following function implements the
[`GlaOpr`](library.md#GilaElectromagnetics.GlaOpr) with the possibility of
obtaining it faster if it was serialized before:

```julia
function load_greens_operator(cells::NTuple{3, Int}, scale::NTuple{3, Rational{Int}};
                              set_type=ComplexF64, use_gpu::Bool=false)

    # Define the name of the FFT file
	preload_dir = get_preload_dir()
	type_str = set_type == ComplexF64 ? "c64" : (set_type == ComplexF32 ? "c32" : "c16")
	fname = "$(type_str)_$(cells[1])x$(cells[2])x$(cells[3])_$(scale[1].num)ss$(scale[1].den)x$(scale[2].num)ss$(scale[2].den)x$(scale[3].num)ss$(scale[3].den).jld2"
	fpath = joinpath(preload_dir, fname)

    # If file exists, unserialise and return GlaOpr
	if isfile(fpath)
		file = jldopen(fpath)
		fourier = file["fourier"]
		if use_gpu
			fourier = CuArray.(fourier)
		end
		options = GlaKerOpt(use_gpu)
		volume = GlaVol(cells, scale, (0//1, 0//1, 0//1))
		mem = GlaOprMem(options, volume; egoFur=fourier, setTyp=set_type)
		return GlaOpr(mem)
	end

    # If file doesn't exist, generate GlaOpr, save it and return it
	operator = GlaOpr(cells, scale; setTyp=set_type, useGpu=use_gpu)
	fourier = operator.mem.egoFur
	if use_gpu
		fourier = Array.(fourier)
	end
	jldsave(fpath; fourier=fourier)
	return operator
end
```

With this function prepared, the memory structure of the Lippmann-Schwinger and
its constructors can be defined. The following simply creates the `struct` for
it with some error checking and preparation for the use of CUDA if required:

```julia
struct LippmannSchwinger
	greens_op::GlaOpr
	medium::AbstractArray{<:Complex, 4}

    # Simple constructor
	function LippmannSchwinger(greens_op::GlaOpr, medium::AbstractArray{<:Complex})

        # Verify if the dimensions and type of GlaOpr and the medium match.
		if glaSze(greens_op, 1)[1:3] != size(medium)
			println(glaSze(greens_op, 1)[1:3])
			println("!=")
			println(size(medium))
			throw(DimensionMismatch("Green's operator and medium must have the same size."))
		end
		if eltype(greens_op) != eltype(medium)
			throw(ArgumentError("Medium must have the same element type as the Green's operator."))
		end

        # Reshape to match the mathematical definitions of the medium
		medium = reshape(medium, glaSze(greens_op, 1)[1:3]..., 1)

        # Make the medium array compatible with CUDA if it's set up
		if greens_op.mem.cmpInf.devMod
			medium = CuArray(medium)
		end

		new(greens_op, medium)
	end
end
```

The definition of the medium as an rank 4 tensor is more intuitive for the user.
The first three dimensions simply describe the indices of a cell, and the
element in the fourth dimension is the complex ``\chi_e`` value at the chosen
cell. This tensor is then reshaped to correspond to how ``\textbf{X}`` was
defined.

!!! danger "Materials treated by Gila"
    GilaElectromagnetics can only treat non-magnetic materials, as they are the
    most common in the field of nano-optics. This is why the medium only defines the
    electric susceptibility. For now, only values of susceptibility with ``\Re(\chi_e) > 0`` 
    converge for the iterative methods used in the following sections.
    However, there is current development on a preconditioner which will allow
    negative real parts of the electric susceptibility to be used without convergence
    problems. This will make simulations of metals possible, and simplify the
    [treatement of empty space](examples.md#empty).

It can also be useful to have a constructor of `LippmannSchwinger` that directly
takes the definition of the cells, the scale, the medium and other parameters:

```julia
function LippmannSchwinger(cells::NTuple{3, Int}, scale::NTuple{3, Rational{Int}},
                           medium::AbstractArray{<:Complex};
                           set_type=ComplexF64, use_gpu::Bool=false)

	greens_op = load_greens_operator(cells, scale; set_type=set_type, use_gpu=use_gpu)
	return LippmannSchwinger(greens_op, medium)
end
```

With this operator not being a matrix but its own memory type, some
mathematical and typical Julia functions ought to be defined:

```julia
# Informations on Lippmann-Schwinger
Base.size(op::LippmannSchwinger) = size(op.greens_op)
Base.size(op::LippmannSchwinger, dim::Int) = size(op.greens_op, dim)
glaSze(op::LippmannSchwinger) = glaSze(op.greens_op)
glaSze(op::LippmannSchwinger, dim::Int) = glaSze(op.greens_op, dim)
Base.eltype(op::LippmannSchwinger) = eltype(op.greens_op)

# Redefinition of the multiplication
function Base.:*(op::LippmannSchwinger, x::AbstractArray)
	if op.greens_op.mem.cmpInf.devMod
		x = CuArray(x)
	end
	gx = reshape(op.greens_op * x, glaSze(op, 1))
	return x - reshape(op.medium .* gx, size(x))
end
LinearAlgebra.mul!(y::AbstractArray, op::LippmannSchwinger, x::AbstractArray) = y .= op * x
```

Similar techniques are implemented with Gila so that multiplying a Green's
operator or attempting to get information on it is more seamless. The final step
consists in using a solver to solve ``\textbf{p}_t = (\textbf{I}_{6 \times 6} -
\textbf{X}\textbf{G}_0)^{-1}\textbf{p}_i``. A simple implementation of one would
go like this:

```julia
using IterativeSolvers

function solve(ls::LippmannSchwinger, i::AbstractArray{<:Complex, 4};
               solver=bicgstabl)

    # Inversion of Lippmann-Schwinger
    out = solver(ls, reshape(deepcopy(i), prod(size(i))))
    return reshape(out, size(i))
end
```

Two main solvers from
[IterativeSolvers.jl](https://iterativesolvers.julialinearalgebra.org/stable/)
were tested and verified to work : BiCGStab and GMRES.

!!! danger "Usage of GPU with solvers"
    Currently, activating the use of CUDA and using a solver from
    `IterativeSolvers` results in an error. This is due to these solvers not working
    with the `CuArray` type of CUDA. Implementing a fix to this problem is feasible
    for a user, as very few changes to these packages are required. A working
    BiCGStab version for this use case is currently in development.

As presented here, the solving function returns a rank 4 tensor, or an array of
size 4, where the first three indices choose a cell, and the fourth contains the
``\textbf{p}_t`` vector at that cell. Maxwell's equations in the medium are thus
solved with this function.

!!! note "Redefinition of the multiplication"
    The redefinition of the multiplication might seem odd, but it is essential
    to achieve the form ``\textbf{I}_{6 \times 6} - \textbf{X}\textbf{G}_0`` in the
    iterative solvers. It is what allows BiCGStab or other algorithms to output
    ``\textbf{W}\textbf{p}_i``, the different multiplications come in handy in their
    underlying workings. 

## Fields

The demonstrated solver for the scattering problem gives the total polarization
current density. If an electric field is desired, the following equation can be
used:

```math
\textbf{e}_t = \textbf{G}_0 \textbf{p}_t
```

The only thing required is to define Green's operator for the volume. With the
definition of the self Green's operator showed above and the solver, finding the
total electric field for the scattering problem can be done as such:

```julia
# Define the volume
cells = (n_x, n_y, n_z)
scale = (scl_x, scl_y, scl_z)

# Define the medium, this is just a simple example
medium = fill(1.0 + 0.0im, n_x, n_y, n_z)

# Define the operator
G_0 = GlaOpr(cells, scale)
LS = LippmanSchwinger(cells, scale, medium)

# Define p_i. For this example, only one  at (i, j, k)
p_i = zeros(eltype(LS), n_x, n_y, n_z, 3)
p_i[i, j, k, :] = [1, 0, 0]

# Solve for p_t
p_t = solve(LS, p_i)

# Obtain the electric field
e_t = G_0 * p_t
```

As mentioned previously, multiplication of a Green's operator with a vector is
already well-defined by Gila.

!!! note "Meaning of the polarization current density"
    In the context of defining an array for ``\textbf{p}_i``, a single vector of
    polarization current density can be seen as an electric dipole at the point
    where the vector is located.
    
    For a linear, non-dispersive and isotropic dielectric, the following
    relationship can relate this polarization density to the electric field:
    
    ```math
    \textbf{p}_i = \chi \textbf{e}_i
    ```
    This would apply for each cell of the defined volume.

## Technical details

### API

The presentation above didn't present every single way to define the operators.
For example, the simplest constructor of `GlaOpr` only takes in a `GlaOprMem`
memory structure. It would be possible to directly define it using it's
[structure definition](library.md#GilaElectromagnetics.GlaOprMem)

Multiple other functions and memory structures are made available in the API.
Some allow to obtain information on the operators, such as their size or their
nature (self or external operator). See the [API reference](library.md) for more
details.

### Boundaries of a volume

The behaviour of space outside the defined volume is designed to be like empty
space. This is referred to as *open boundary conditions*.

### Memory

The operators and the matrices for bigger volumes can take a lot of memory. The
following is a good rule of thumb to make sure the host doesn't run out of
memory.

The size of a `ComplexF64` number is 128 bits. The vector ``\textbf{p}_i``
contains 3 complex numbers per component, and ``n_x \times n_y \times n_z``
vectors. Thus, the size of ``\textbf{p}_i`` is ``384 \times n_x \times n_y
\times n_z`` bits. 

It is strongly advised to have at least 8 times that amount of storage in RAM or
VRAM available. This amount has some buffer in it, but to use all the operations
shown above and other scripts, it is the amount of memory with which no errors
caused by lack of memory should arise.

### Multi-threading

When using Gila on a CPU, many functions can take advantage of multiple compute
threads, which can bring massive speed gains. It is highly advised to use Gila
within a Julia REPL that has access to as many threads as possible. For example,
to launch a REPL with 8 threads, the following line needs to be entered in a
terminal:

```
julia -t 8
```

Then, to make the most use of those threads, two packages that are used by Gila
need to be set to use those threads. This can be done by setting the following
before using any defined operators:

```julia
using Base.Threads # to access the number of threads given to the REPL

using FFTW
using LinearAlgebra.BLAS

num_threads = nthreads()
BLAS.set_num_threads(num_threads)
FFTW.set_num_threads(num_threads)
```
