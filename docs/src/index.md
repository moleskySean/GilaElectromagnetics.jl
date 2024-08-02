# General

Documentation for GilaElectromagnetics.jl, a julia package implementing the three dimensional electromagnetic Green function.

## Use cases

GilaElectromagnetics, or Gila for short, allows to perform fast subwavelength electromagnetic simulations with a high degree of precision. It allows to :

- Solve Maxwell's equations numerically in vacuum and in matter.
- Obtain the Green's function of Maxwell's vacuum operator.
- Solve the scattering problem in non-magnetic materials.
- Use CUDA to accelerate computations.


## Installation

The installation can be done with julia's package manager :

```julia-repl
julia> using Pkg
julia> Pkg.add("GilaElectromagnetics")
```

Alternatively, in julia's REPL, a press of ']' enters "Pkg mode", where simply entering the following installs the package :

```
(@v1.10) pkg> add GilaElectromagnetics
```
