include("GilaOperators.jl")

using Base.Threads
using FFTW
using IterativeSolvers
using LinearAlgebra
using LinearAlgebra.BLAS
using Random
using ..GilaOperators
using CUDA
using GLMakie
using GeometryBasics
using Colors
using Printf

Random.seed!(0)

num_threads = nthreads()
BLAS.set_num_threads(num_threads)
FFTW.set_num_threads(num_threads)


#######################################################################
# description of system

# dimensions
num_cells_side = 32 # must be even
num_cells_side_vac = 128 # must be even
cells_per_wavelength = 32

# medium
χ_fill = ComplexF32(1.5 + 0im)

# source
pos_x = 16
pos_y = 16
pos_z = 16
dip_x = 0
dip_y = 0
dip_z = 30

# visualisation
dst_slice = 70
slice_id = "xz"
scale_real = true

fig1_name = "cube_1.png"
fig2_name = "cube_2.png"
#######################################################################
# solving

"""
Solves t = (1 - XG)⁻¹i
"""
function solve(ls::LippmannSchwinger, i::AbstractArray{<:Complex, 4}; solver=bicgstabl)
  out = solver(ls, reshape(deepcopy(i), prod(size(i)))) # inversion de LS
  return reshape(out, size(i))
end

println("Making geometry and medium of cube.")
cells = (num_cells_side, num_cells_side, num_cells_side)
scale = (1//cells_per_wavelength, 1//cells_per_wavelength, 1//cells_per_wavelength)
coord = (0//1, 0//1, 0//1);

χ = fill(χ_fill, num_cells_side, num_cells_side, num_cells_side)

println("Making LippmannSchwinger.")
@time ls = LippmannSchwinger(cells, scale, χ; set_type=ComplexF32)
println("LippmannSchwinger created. Creating source.")

p_i = zeros(eltype(ls), num_cells_side, num_cells_side, num_cells_side, 3)
p_i[pos_x, pos_y, pos_z, :] = [dip_x, dip_y, dip_z]

println("Sources created. Solving. ")
@time p_t = solve(ls, p_i)

println("Solved. Embedding cube in empty space...")
cells_vac = (num_cells_side_vac, num_cells_side_vac, num_cells_side_vac)
p_t_vac = zeros(ComplexF32, num_cells_side_vac, num_cells_side_vac, num_cells_side_vac, 3)

pos_cube_begin = Int((num_cells_side_vac / 2) - (num_cells_side / 2))
pos_cube_end = Int((num_cells_side_vac / 2) + (num_cells_side / 2) - 1)

p_t_vac[pos_cube_begin:pos_cube_end, pos_cube_begin:pos_cube_end, pos_cube_begin:pos_cube_end, :] .= p_t

println("Loading Green's Operator, empty space included")
G_0_vac = load_greens_operator(tuple(cells_vac...), scale; set_type=ComplexF32)

println("Loading done.\nMaking field from Green's operator EMPTY SPACE INCLUDED.")
@time e_t_vac = G_0_vac * p_t_vac

#######################################################################
# visualising preparation

struct param_heat
  dist_slice::Union{Int, Nothing}
  plane::Union{String, Nothing}
  cells::Tuple{Int64, Int64, Int64}
  scale::Tuple{Rational{Int64}, Rational{Int64}, Rational{Int64}}
  coord::Tuple{Rational{Int64}, Rational{Int64}, Rational{Int64}}
end

function viz_heat(param::param_heat, pol_field::Array{<:Complex, 4})
    # Matrix for heatmap
    if param.plane == "yz"
        slice = pol_field[param.dist_slice, :, :, :]
        matrix_x = zeros(param.cells[2], param.cells[3])
        matrix_y = zeros(param.cells[2], param.cells[3])
        matrix_z = zeros(param.cells[2], param.cells[3])
        matrix_amp = zeros(param.cells[2], param.cells[3])
        for i in 1:param.cells[2]
            for j in 1:param.cells[3]
                matrix_x[i, j] = real(slice[i, j, 1])
                matrix_y[i, j] = real(slice[i, j, 2])
                matrix_z[i, j] = real(slice[i, j, 3])
                matrix_amp[i, j] = real((norm(slice[i, j, :]))^2)
            end
        end
    elseif param.plane == "xz"
        slice = pol_field[:, param.dist_slice, :, :]
        matrix_x = zeros(param.cells[1], param.cells[3])
        matrix_y = zeros(param.cells[1], param.cells[3])
        matrix_z = zeros(param.cells[1], param.cells[3])
        matrix_amp = zeros(param.cells[1], param.cells[3])
        for i in 1:param.cells[1]
            for j in 1:param.cells[3]
                matrix_x[i, j] = real(slice[i, j, 1])
                matrix_y[i, j] = real(slice[i, j, 2])
                matrix_z[i, j] = real(slice[i, j, 3]) 
                matrix_amp[i, j] = real((norm(slice[i, j, :]))^2)
            end
        end
    elseif param.plane == "xy"
        slice = pol_field[:, :, param.dist_slice, :]
        matrix_x = zeros(param.cells[1], param.cells[2])
        matrix_y = zeros(param.cells[1], param.cells[2])
        matrix_z = zeros(param.cells[1], param.cells[2])
        matrix_amp = zeros(param.cells[1], param.cells[2])
        for i in 1:param.cells[1]
            for j in 1:param.cells[2]
                matrix_x[i, j] = real(slice[i, j, 1])
                matrix_y[i, j] = real(slice[i, j, 2])
                matrix_z[i, j] = real(slice[i, j, 3]) 
                matrix_amp[i, j] = real((norm(slice[i, j, :]))^2)
            end
        end
    end
    return [matrix_x, matrix_y, matrix_z, matrix_amp]
end

include("colors.jl")

plot_param = param_heat(dst_slice, slice_id, cells_vac, scale, coord)
data = viz_heat(plot_param, e_t_vac)

x_ticks = 0:cells_per_wavelength:num_cells_side_vac
y_ticks = 0:cells_per_wavelength:num_cells_side_vac
z_ticks = 0:cells_per_wavelength:num_cells_side_vac

x_labels = [string(i ÷ cells_per_wavelength) for i in x_ticks]
y_labels = [string(i ÷ cells_per_wavelength) for i in y_ticks]
z_labels = [string(i ÷ cells_per_wavelength) for i in z_ticks]

if plot_param.plane == "xz"
  abs = "x"
  ord = "z"
  dst_from = "y"
elseif plot_param.plane =="yz"
  abs = "y"
  ord = "z"
  dst_from = "x"
else
  abs = "x"
  ord = "y"
  dst_from = "z"
end

#######################################################################
# fig1 : map, info and intensity

fig1 = Figure(size = (900,800))

grid1 = fig1[1, 1] = GridLayout(tell = :bbox)

χ_f = @sprintf("%.2f", χ_fill)

ax5 = Axis3(grid1[1:2, 1],
      title = "Position of the heatmap",
      aspect = :data,
      xlabel = "# of wavelengths in X",
      ylabel = "# of wavelengths in Y",
      zlabel = "# of wavelengths in Z", 
      xticks = (x_ticks, x_labels),
      yticks = (y_ticks, y_labels),
      zticks = (z_ticks, z_labels)
)

text_label = Label(grid1[1:2, 2], "Heatmap @ $dst_from = $dst_slice cells
  Number of cells per wavelength : $cells_per_wavelength
  In the box of $num_cells_side x $num_cells_side x $num_cells_side cells :
  Dipole [$dip_x, $dip_y, $dip_z] @ ($pos_x, $pos_y, $pos_z) 
  Medium filled with χ = $χ_f",
  tellwidth=false, padding=(20, 20, 20, 20), halign=:center, fontsize=14)

mesh!(ax5, FRect3D(Point3f0(0, 0, 0), Point3f0(num_cells_side_vac, num_cells_side_vac, num_cells_side_vac)),
      color=:lightblue, transparency=true, alpha=0.3)

mesh!(ax5, FRect3D(Point3f0(pos_cube_begin, pos_cube_begin, pos_cube_begin), Point3f0(num_cells_side, num_cells_side, num_cells_side)),
      color=:gray, transparency=true, alpha=0.2)

if plot_param.plane == "xz"
    lines!(ax5, [0, num_cells_side_vac], [dst_slice, dst_slice], [0, 0], color=:red, linewidth=3, fxaa=true)
    lines!(ax5, [0, num_cells_side_vac], [dst_slice, dst_slice], [num_cells_side_vac, num_cells_side_vac], color=:red, linewidth=3, fxaa=true)
    lines!(ax5, [0, 0], [dst_slice, dst_slice], [0, num_cells_side_vac], color=:red, linewidth=3, fxaa=true)
    lines!(ax5, [num_cells_side_vac, num_cells_side_vac], [dst_slice, dst_slice], [0, num_cells_side_vac], color=:red, linewidth=3, fxaa=true)
elseif plot_param.plane == "yz"
    lines!(ax5, [dst_slice, dst_slice], [0, num_cells_side_vac], [0, 0], color=:red, linewidth=3, fxaa=true)
    lines!(ax5, [dst_slice, dst_slice], [0, num_cells_side_vac], [num_cells_side_vac, num_cells_side_vac], color=:red, linewidth=3, fxaa=true)
    lines!(ax5, [dst_slice, dst_slice], [0, 0], [0, num_cells_side_vac], color=:red, linewidth=3, fxaa=true)
    lines!(ax5, [dst_slice, dst_slice], [num_cells_side_vac, num_cells_side_vac], [0, num_cells_side_vac], color=:red, linewidth=3, fxaa=true)
elseif plot_param.plane == "xy"
    lines!(ax5, [0, num_cells_side_vac], [0, 0], [dst_slice, dst_slice], color=:red, linewidth=3, fxaa=true)
    lines!(ax5, [0, num_cells_side_vac], [num_cells_side_vac, num_cells_side_vac], [dst_slice, dst_slice], color=:red, linewidth=3, fxaa=true)
    lines!(ax5, [0, 0], [0, num_cells_side_vac], [dst_slice, dst_slice], color=:red, linewidth=3, fxaa=true)
    lines!(ax5, [num_cells_side_vac, num_cells_side_vac], [0, num_cells_side_vac], [dst_slice, dst_slice], color=:red, linewidth=3, fxaa=true)
end

# Show the dipole
magnitude = sqrt(dip_x^2 + dip_y^2 + dip_z^2)

norm_dip_x = dip_x / magnitude
norm_dip_y = dip_y / magnitude
norm_dip_z = dip_z / magnitude

scale_factor = 3
scaled_dip_x = norm_dip_x * scale_factor
scaled_dip_y = norm_dip_y * scale_factor
scaled_dip_z = norm_dip_z * scale_factor

arrows!(
    ax5, 
    [Point3f0(pos_x + pos_cube_begin, pos_y + pos_cube_begin, pos_z + pos_cube_begin)], 
    [Vec3f0(scaled_dip_x, scaled_dip_y, scaled_dip_z)],
    arrowsize=7, linewidth=2, color=:red, label="Source Vector"
)

# Heatmap
ax4 = Axis(grid1[3:4, 1:2],
    title = L"Heatmap of the intensity of the electric field $E^2$",
    xlabel = "# of wavelengths in $abs",
    ylabel = "# of wavelengths in $ord",
    xticks = (x_ticks, x_labels),
    yticks = (y_ticks, y_labels)    
)

hm4 = heatmap!(ax4, log10.(data[4]), colormap=colormap_icefire)
cb4 = Colorbar(grid1[3:4, 3], hm4, label="Intensity (powers of 10)")

# Save figure
save("fig_docs/"*fig1_name, fig1)

# Show the figure
# display(fig1)


#######################################################################
# fig2 : real parts

fig2 = Figure(size = (900, 1000))
grid2 = fig2[1, 1] = GridLayout(tell = :bbox)

ax1 = Axis(grid2[1, 1],
    title = L"Heatmap of the real part of the electric field $E_x$",
    xlabel = "# of wavelengths in $abs",
    ylabel = "# of wavelengths in $ord",
    xticks = (x_ticks, x_labels),
    yticks = (y_ticks, y_labels)    
)
ax2 = Axis(grid2[2, 1],
    title = L"Heatmap of the real part of the electric field $E_y$",
    xlabel = "# of wavelengths in $abs",
    ylabel = "# of wavelengths in kk$ord",
    xticks = (x_ticks, x_labels),
    yticks = (y_ticks, y_labels)    
)
ax3 = Axis(grid2[3, 1],
    title = L"Heatmap of the real part of the electric field $E_z$",
    xlabel = "# of wavelengths in $abs",
    ylabel = "# of wavelengths in $ord",
    xticks = (x_ticks, x_labels),
    yticks = (y_ticks, y_labels)    
)

if scale_real
  global_min = minimum([minimum(data[1]), minimum(data[2]), minimum(data[3])])
  global_max = maximum([maximum(data[1]), maximum(data[2]), maximum(data[3])])
 
  hm1 = heatmap!(ax1, data[1], colormap = colormap_icefire, colorrange = (global_min, global_max))
  hm2 = heatmap!(ax2, data[2], colormap = colormap_icefire, colorrange = (global_min, global_max))
  hm3 = heatmap!(ax3, data[3], colormap = colormap_icefire, colorrange = (global_min, global_max))

  cb1 = Colorbar(grid2[1, 2], hm1, label = "Real part of the electric field")
  cb2 = Colorbar(grid2[2, 2], hm1, label = "Real part of the electric field")
  cb3 = Colorbar(grid2[3, 2], hm1, label = "Real part of the electric field")
else
  hm1 = heatmap!(ax1, data[1], colormap=colormap_icefire)
  cb1 = Colorbar(grid2[1, 2], hm1, label="Real part of the electric field")

  hm2 = heatmap!(ax2, data[2], colormap=colormap_icefire)
  cb2 = Colorbar(grid2[2, 2], hm2, label="Real part of the electric field")

  hm3 = heatmap!(ax3, data[3], colormap=colormap_icefire)
  cb3 = Colorbar(grid2[3, 2], hm3, label="Real part of the electric field")
end

# Save figure
save("fig_docs/"*fig2_name, fig2)

# Show the figure
#display(fig2)
