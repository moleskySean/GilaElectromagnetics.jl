# Physical Units

Gila works in wavelengths and stores everything divided by ``Z_\circ``, so its numbers carry no units. This page shows how to get real units in and out with [Unitful.jl](https://github.com/PainterQubits/Unitful.jl). Gila does not depend on Unitful, so the conversions are done by hand at the boundary.

## What Gila stores

<!-- Short table: lengths in λ, currents in A s m⁻¹, fields as f/Z∘ (so also A s m⁻¹), G∘/Z∘ and χ dimensionless. Point to Table 2 and convention 4.3. Consequence: every operator is dimensionless, a field keeps its unit through `*` and `\`. -->

## Geometry: choosing λ

<!-- The one physical number Gila never sees. Cell sizes are exact Rationals in λ; pick λ (or λ = c0/ν) and express cells in it. Warn against rationalize(cell/λ) from metres. -->

## Currents in

<!-- `GlaFld(ComplexF64.(ustrip.(u"A*s/m", p)), cvol)`. ustrip checks the dimension. Densities, not √ΔV-weighted (convention 3.1). -->

## Fields out: multiply by Z∘

<!-- `Unitful.Z0 .* fld.dat .* u"A*s/m"` gives V s m⁻¹. The one step people forget. Same for G * p and for the scattered field from a solve. -->

## Materials from ε

<!-- `χ = ε/ε0 - 1`, `ustrip(NoUnits, χ)`, passed to `SusOpr`. -->

## Inner products and norms

<!-- `dot`/`norm` weight by ΔV in λ³. Physical value: multiply by λ^3 and the units of both sides. norm(f, p) gets λ^(3/p); norm(f, Inf) none. -->

## Complex frequency

<!-- Convention 4.1/4.3: matrix elements are also scaled by 1/ζ². State how to undo it in the field read-out when ζ ≠ 1. Link to howto/frequency.md. -->

## Worked example

<!-- One end to end script: λ = 1 µm cube, current in A s m⁻¹, scattered field in V/m, absorbed power from a dot in W (check the time-harmonic factor convention). -->
