# README figure: a silicon sphere on a glass pad in a plane wave, solved on a refined mesh.
# Copied from tutorial 05; run with julia --project=examples examples/readme/sphere_on_pad.jl
using GilaElectromagnetics, CairoMakie
include(joinpath(@__DIR__, "..", "..", "docs", "tutorial_figures.jl"))  # darken!
using VoxelShapes: VoxelShapes, Region, Geometry, FillableSphere, FillableCuboid, rasterize, SuperResolutionAntiAliasing

χ_silicon = 11.0
χ_glass = 1.25
sphere = FillableSphere((0.0, 0.0, 0.075), 0.15, χ_silicon)
pad = FillableCuboid((0.0, 0.0, -0.2), (1.25, 1.25, 0.25), χ_glass)
geometry = Geometry([sphere, pad], 0.0, SuperResolutionAntiAliasing(4))
coarse_region = Region((24, 24, 12), (1//16, 1//16, 1//16), (0//1, 0//1, 0//1))
pieces = VoxelShapes.refine(coarse_region, sphere; padding = 1//16)
volume = CompositeVolume(pieces)

G₀ = VacuumGreenOperator(volume)
W = ScatteringOperator(G₀, geometry)
f_inc = discretize!(zeros(volume), r -> (1, 0, 0) .* cis(-2π * r[3]))  # down from above, along x
p_s = W * (sus(W) * f_inc)
f_t = regrid(f_inc + G₀ * p_s, (1//32, 1//32, 1//32))

# The structure, every material cell of the refined mesh as a cube
filling = Geometry([FillableSphere((0.0, 0.0, 0.075), 0.15, 1.0), FillableCuboid((0.0, 0.0, -0.2), (1.25, 1.25, 0.25), 1.0)],
                   0.0, SuperResolutionAntiAliasing(4))
centers, sizes, colors = Point3f[], Float64[], RGBAf[]
for piece in VoxelShapes.regions(pieces)
    grid, Δ = GlaVol(piece).grd, Float64(GlaVol(piece).scl[1])
    χ_piece, filled = rasterize(geometry, piece), rasterize(filling, piece)
    for I in findall(>(0), χ_piece)
        push!(centers, Point3f(grid[1][I[1]], grid[2][I[2]], grid[3][I[3]]))
        push!(sizes, 0.9Δ)
        color = cgrad(:ice)[0.3 + 0.6sqrt(χ_piece[I] / χ_silicon)]
        push!(colors, RGBAf(color.r, color.g, color.b, 0.1 + 0.9filled[I]))
    end
end

fine = GlaVol(Region((48, 48, 24), (1//32, 1//32, 1//32), (0//1, 0//1, 0//1)))
x = Float64.(fine.grd[1])
z = Float64.(fine.grd[3])
fig = Figure(size = (800, 880))
ax = Axis3(fig[1, 1]; aspect = :data, xlabel = "x (λ)", ylabel = "y (λ)", zlabel = "z (λ)",
           zlabeloffset = 50, protrusions = (70, 30, 30, 30))
meshscatter!(ax, centers; marker = Rect3f(Vec3f(-0.5), Vec3f(1)), markersize = sizes, color = colors)
ax = Axis(fig[2, 1]; xlabel = "x (λ)", ylabel = "z (λ)", aspect = DataAspect())
hm = heatmap!(ax, x, z, abs.(f_t[:, 24, :, 1]))  # middle y slice, x component
Colorbar(fig[2, 2], hm; label = "|fₓ|")
rowsize!(fig.layout, 2, Relative(0.42))
save(joinpath(@__DIR__, "sphere_on_pad-light.png"), fig)
save(joinpath(@__DIR__, "sphere_on_pad-dark.png"), TutorialFigures.darken!(fig))
