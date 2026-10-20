# README figure: the field of a small radiating block on a sheet one wavelength away.
# Copied from tutorial 02; run with julia --project=examples examples/readme/radiating_cell.jl
using GilaElectromagnetics, CairoMakie
include(joinpath(@__DIR__, "..", "..", "docs", "tutorial_figures.jl"))  # darken!

source = GlaVol((2, 2, 2), (1//16, 1//16, 1//16))
target = GlaVol((64, 2, 64), (1//16, 1//16, 1//16), (0//1, 1//1, 0//1))
G₀ = VacuumGreenOperator(target, source)
p = discretize!(zeros(source), r -> (0, 0, 1))
e = G₀ * p

x = Float64.(target.grd[1])
z = Float64.(target.grd[3])
e_z = real(regionview(e, 1)[:, 1, :, 3])
source_corner = Float64.(first.(source.grd) .- source.scl .// 2)
fig = Figure()
ax = Axis3(fig[1, 1]; aspect = :data, xlabel = "x (λ)", ylabel = "y (λ)", zlabel = "z (λ)", yticks = [0, 1])
hm = heatmap!(ax, x, z, e_z; colormap = :balance, colorrange = (-1, 1) .* maximum(abs, e_z),
              transformation = (:xz, 1.0))  # the sheet sits at y = 1
mesh!(ax, Rect3f(source_corner, Float64.(source.cel .* source.scl)); color = :darkorange)
limits!(ax, -2, 2, -0.25, 1.25, -2, 2)
Colorbar(fig[1, 2], hm; label = rich("Re e", subscript("z")))
save(joinpath(@__DIR__, "radiating_cell-light.png"), fig)
save(joinpath(@__DIR__, "radiating_cell-dark.png"), TutorialFigures.darken!(fig))
