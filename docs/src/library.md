# Library

The following is the exhaustive list of the API available to users, accompanied by explanations.

## Module Index

```@index
Modules = [GilaElectromagnetics]
Order   = [:constant, :type, :function, :macro]
```
## Detailed API

!!! note "Storage precision"
    Operators and fields carry a real type parameter `T<:AbstractFloat` (`Float32`
    or `Float64`); the data itself is `Complex{T}`. Unparameterized constructors
    default to `T = dflPrc` (`Float32`); request another precision with the type
    parameter, e.g. `GlaOprVac{Float64}(vol)`.

```@autodocs
Modules = [GilaElectromagnetics]
Order   = [:constant, :type, :function, :macro]
Private = false
```

`GlaCmpOprVac` is exported as `CompositeVacuumGreenOperator`:

```@docs
GilaElectromagnetics.GlaCmpOprVac
```
