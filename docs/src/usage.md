# Usage

The following subsections present typical uses of the API given by GilaElectromagnetics.

## Vacuum Green's function

Gila can solve Maxwell's equations in a vacuum by employing finite element methods on a given volume. What is actually returned is the vacuum Green's function ``\textbf{G}_0``. This operator is represented by the memory structure [`GlaOprMem`](library.md#GilaElectromagnetics.GlaOprMem), which can be constructed in different ways. To make it, information about the space where the equations are solved must be defined.

With this solver, volumes must be rectangular prisms, which are then subdivided into smaller volumes called *cells*. The scaling factor of those cells in a specific direction is given in fractions of a wavelength, because Gila solves a system at one given wavelength.

### Self and external operators

The vacuum Green's function for a given volume is obtained using a [`GlaOpr`](library.md#GilaElectromagnetics.GlaOpr) constructor. There are two cases to consider. The first one is the simplest, where the defined volume is the same as the solved volume. The resulting operator is called the *self Green's operator*. The following general example shows how to build it :

```julia
# Volume definition
cells = (n_x, n_y, n_z) # n is of type Integer
scale = (scl_x, scl_y, scl_z) # scl is of type Rational
origin = (org_x, org_y, org_z) # NOT REQUIRED, org is of type Rational

# Self Greens operator
G_0 = GlaOpr(cells, scale, origin)
```

Two more parameters can be given : `useGpu` if an *Nvidia* GPU using CUDA cores can be used, and `setTyp` to either use 32 or 64 bit complex numbers. The syntax to tweak these parameters is shown here :

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

This operator uses the `GlaOpr` type, which in itself is an abstraction wrapper for the [`GlaOprMem`](library.md#GilaElectromagnetics.GlaOprMem).

## Scattering problem

One of the most interesting problems that can be attacked by Gila is the *scattering problem*, which asks to find the total field ``\textbf{f}_t`` produced, given an incident field ``\textbf{f}_i`` and a dielectric profile. 

### Theoretical overview

Because of the linearity of the considered Maxwell's equations, it is possible to decompose the total field with the incident field and the scattered field ``\textbf{f}_s`` in the following way :

```math
\textbf{f}_t = \textbf{f}_i + \textbf{f}_s
```

Furthermore, we define the matrix ``\textbf{X}`` to represent permittivity and permeability in the following way :



W : scattering operator

### Implementation

warning, prévoire 8*v (taille des vecteurs sources) comme rule of thumb

## Fields


## Technical details


