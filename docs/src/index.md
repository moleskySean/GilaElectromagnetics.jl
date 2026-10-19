# GilaElectromagnetics.jl

[GilaElectromagnetics.jl](https://github.com/moleskySean/GilaElectromagnetics.jl)
is a Julia package that provides a very efficient implementation of the discrete
three-dimensional electromagnetic Green function. For a technical description of
the implementation, see the associated
[paper](https://github.com/moleskySean/GilaElectromagnetics.jl/blob/main/docs/gilaDoc.pdf).
The [Theory](@ref sec-1) section explains what Gila computes and how. The
public [API reference](./library.md) lists every exported function and type.

## Use cases

GilaElectromagnetics, or Gila for short, enables fast and precise sub-wavelength
electromagnetic simulations. Below are some features Gila provides:

- Solving Maxwell's equations numerically in vacuum and in matter.
- Application of the Green function of the vacuum Maxwell vacuum operator.
- Solving the scattering problem in non-magnetic materials.
- GPU accelerated computations with CUDA, Metal or OpenCL.

## Installation

Installation can be done with Julia's package manager:

```julia-repl
julia> using Pkg
julia> Pkg.add("GilaElectromagnetics")
```

Alternatively, in Julia's REPL, a typing `]` puts you in ["Pkg
mode"](https://docs.julialang.org/en/v1/stdlib/Pkg/). In this command line
package manager, installing GilaElectromagnetics can be done as follows:

```
(@v1.10) pkg> add GilaElectromagnetics
```