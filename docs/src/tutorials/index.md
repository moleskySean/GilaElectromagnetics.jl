# Tutorials

The tutorials build up Gila one piece at a time. Each is a script in the
`examples` folder of the repository, so you can also run it yourself:

```sh
julia --project=examples examples/01_volumes_and_fields.jl
```

1. [Volumes and Fields](01_volumes_and_fields.md) sets up the grid and the field
   type, and explains why norms carry the cell volume.
2. [The Vacuum Green Operator](02_vacuum_operator.md) computes the field of a
   single radiating cell and checks it against the dipole field.
3. [A First Scattering Calculation](03_first_scattering.md) puts a small
   dielectric sphere in a plane wave and compares its dipole moment with the
   Clausius–Mossotti polarizability.
4. [Geometry with VoxelShapes](04_geometry_with_voxelshapes.md) turns shapes and
   their boolean combinations into susceptibilities, with anti-aliased surfaces.
5. [Refined Meshes](05_refined_meshes.md) uses fine cells only where the
   geometry needs them.
6. [Physical Units](06_physical_units.md) attaches SI units to currents, fields
   and power, and checks the radiated power of a small dipole.
7. [Reflection off a Thin Film](07_thin_film.md) puts a film, made periodic
   with Bloch boundary conditions, in a plane wave and compares its reflection
   with the Airy formula.
8. [Band Structure of a Photonic Crystal](08_band_structure.md) finds the TM
   bands of a rod crystal with a Bloch sweep and a matrix-free eigensolver.
