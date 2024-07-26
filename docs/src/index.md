# General

Documentation for GilaElectromagnetics.jl, a julia package implementing the three dimensional electromagnetic Green function.

## Use cases

general : solving maxwell eq

obtaining vacuum green's function

scattering problem

materials non mag ...

subwavelength simulations

(soon) use gpu to solve

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
