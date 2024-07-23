# Usage

The following subsections present typical uses of the API given by GilaElectromagnetics.

## Vacuum Green's function

Gila can solve Maxwell's equations in a vacuum by employing finite element methods on a given volume. What is actually returned is the vacuum Green's function ``\textbf{G}_0``. This operator is represented by the memory structure [`GlaOprMem`](library.md#GilaElectromagnetics.GlaOprMem), which can be constructed in different ways. To make it, information about the space where the equations are solved must be defined.

With this solver, volumes must be rectangular prisms, which are then subdivided into smaller volumes called *cells*. The scaling factor of those cells in a specific direction is given in fractions of a wavelength, because Gila solves a system at one given wavelength.

### Self and external operators

The vacuum Green's function for a given volume is obtained using a [`GlaOpr`](library.md#GilaElectromagnetics.GlaOpr) constructor. There are two cases to consider. The first one is the simplest, where the defined volume is the same as the solved volume. The following general example shows how to build it :

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

The second case to consider is where there are 

## Scattering problem

### Theoretical overview

W : scattering operator

### Implementation

warning, prévoire 8*v (taille des vecteurs sources) comme rule of thumb

## Fields


## Technical details


