# Library

## Module Index

```@index
Modules = [GilaElectromagnetics]
Order   = [:constant, :type, :function, :macro]
```
## Detailed API

```@autodocs
Modules = [GilaElectromagnetics]
Order   = [:constant, :type, :function, :macro]
```

TEST

```@docs 
GilaElectromagnetics.GlaKerOpt
```
questions pour Paul :

pourquoi est-ce que des fonctions qui sont meme pas exportees par Gila, tel que `egoBrnDev!`, `genEgoExt!` et autres sont generes?

pourquoi `egoOpr!`, qui est clairement exporte, pas inclu par autodocs?

Est-ce que a cause de ca je devrais me contenter d'appeler les fonctions individuellement?
