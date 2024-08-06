# GilaElectromagnetics.jl

*Greens in linear algebra*

[![Build status (Github Actions)](https://github.com/moleskySean/GilaElectromagnetics.jl/workflows/CI/badge.svg)](https://github.com/moleskySean/GilaElectromagnetics.jl/actions)
[![codecov.io](http://codecov.io/github/moleskySean/GilaElectromagnetics.jl/coverage.svg?branch=main)](http://codecov.io/github/moleskySean/GilaElectromagnetics.jl?branch=main)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://moleskySean.github.io/GilaElectromagnetics.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://moleskySean.github.io/GilaElectromagnetics.jl/dev)

[GilaElectromagnetics.jl](https://github.com/moleskySean/GilaElectromagnetics.jl)
is a Julia package that provides a very efficient implementation of the discrete
three-dimensional electromagnetic Green function. Documentation for the package
can be found
[here](https://moleskysean.github.io/GilaElectromagnetics.jl/stable).

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
