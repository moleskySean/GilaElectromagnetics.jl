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
using ColorSchemes

Random.seed!(0);

num_threads = nthreads()
BLAS.set_num_threads(num_threads)
FFTW.set_num_threads(num_threads)


#######################################################################
# description of system

# wave vector (is normalized later)
k_i = 1.0
k_j = 0.0
k_k = -2.0 # negative value looks better, not mandatory
amp = 3.0

# dimensions
num_cellsx = 256
num_cellsy = 256
num_cellsz = 24
num_cellsz_vac = 128
cells_per_wavelength = 32

# medium
position_film = 52
χ_fill = ComplexF32(1.5 + 0im)
decay_length = 1

# visualisation
dst_slice = 128
slice_id = "xz"
scale_real = true

fig1_name = "film_7.png"
fig2_name = "film_8.png"
#######################################################################
# solving

"""
Solves t = (1 - XG)⁻¹i
"""
function solve(ls::LippmannSchwinger, i::AbstractArray{<:Complex, 4}; solver=bicgstabl)
  out = solver(ls, reshape(deepcopy(i), prod(size(i)))) # inversion de LS
  return reshape(out, size(i))
end

function medium_decay_tanh!(cells, χ, χ_fill, d)
  for i in 1:d
    χ[d-i+1, :, :, :] .= (χ_fill / 2) * ( tanh((6/d) * (-i + (d/2))) + 1 ) + ( (0.1/(d^2))*((i-d-1)^2) )im
    χ[cells[1] - d + i, :, :, :] .= (χ_fill / 2) * ( tanh((6/d) * (-i + (d/2))) + 1 ) + ( (0.1/(d^2))*((i-d-1)^2) )im
  end
  for i in 1:d
    for x in 1:cells[1]
      if real(χ[x, d-i+1, 1, 1]) > real((χ_fill / 2) * ( tanh((6/d) * (-i + (d/2))) + 1 ))
        χ[x, d-i+1, :, :] .= (χ_fill / 2) * ( tanh((6/d) * (-i + (d/2))) + 1 ) + ( (0.1/(d^2))*((i-d-1)^2) )im
        χ[x, cells[2] - d + i, :, :] .= (χ_fill / 2) * ( tanh((6/d) * (-i + (d/2))) + 1 ) + ( (0.1/(d^2))*((i-d-1)^2) )im
      end
    end
  end
end

# Electric wave. By default, parallel to y axis (s polarization)
function electric_field(x, y, z, dir_i, dir_j, dir_k, amp, ω=0, t=0)
  k = normalize([dir_i, dir_j, dir_k])
  
  if abs(k[3]) != 1
    v = [0, 0, 1]  # use z-axis if k is not aligned with z
  else
    v = [0, 1, 0]  # use y-axis if k is aligned with z (would not be s pol anymore)
  end

  # E perpendiculaire à k et v : choix arbitraire de direction de E
  E0 = amp * normalize(cross(k, v))

  return E0 * exp(1im * (dot(2π*k, [x, y, z]) - ω * t))
end

println("Making geometry and medium.")
cells = (num_cellsx, num_cellsy, num_cellsz)
scale = (1//cells_per_wavelength, 1//cells_per_wavelength, 1//cells_per_wavelength)
coord = (0//1, 0//1, 0//1);

χ = fill(χ_fill, num_cellsx, num_cellsy, num_cellsz)

println("Creating decay.")
@time medium_decay_tanh!(cells, χ, χ_fill, decay_length * cells_per_wavelength);

println("Decay created. Making LippmannSchwinger.")
@time ls = LippmannSchwinger(cells, scale, χ; set_type=ComplexF32)
print("LippmannSchwinger created. ")

println("Creating sources.")
p_i = zeros(eltype(ls), num_cellsx, num_cellsy, num_cellsz, 3)

# χ*e_i = p_i everywhere
@time for x in 1:num_cellsx
    for y in 1:num_cellsy
        for z in 1:num_cellsz
            p_i[x, y, z, :] = real(electric_field((x-1) + coord[1], (y-1) + coord[2], (z-1) + coord[3], k_i, k_j, k_k, amp)) * χ[x, y, z, 1]
        end
    end
end

println("Sources created. Solving. ")
@time p_t = solve(ls, p_i)
println(size(p_t))

println("Solved. Embedding film in empty space...")
@show cells_vac = (num_cellsx, num_cellsy, num_cellsz_vac)

end_film = position_film + num_cellsz - 1
p_t_vac = zeros(ComplexF32, num_cellsx, num_cellsy, num_cellsz_vac, 3)
p_t_vac[:, :, position_film:end_film, :] .= p_t

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

plot_param = param_heat( dst_slice, "xz", cells_vac, scale, coord)
data = viz_heat(plot_param, e_t_vac)

# Adjusting tick labels to match tick positions
x_ticks = 0:cells_per_wavelength:num_cellsx
y_ticks = 0:cells_per_wavelength:num_cellsy
z_ticks = 0:cells_per_wavelength:max(num_cellsz_vac, num_cellsz)

# Generating labels
x_labels = [string(i ÷ cells_per_wavelength) for i in x_ticks]
y_labels = [string(i ÷ cells_per_wavelength) for i in y_ticks]
z_labels = [string(i ÷ cells_per_wavelength) for i in z_ticks]

#######################################################################
# fig1 : map, info and intensity

fig1 = Figure(size = (1150, 800))  # Adjusted the figure size for a better layout

# Set up a grid layout
grid1 = fig1[1, 1] = GridLayout(tell = :bbox)

# 3D Axis on the third row
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

k_norm = sqrt(k_i^2 + k_j^2 + k_k^2)
k_i_norm = @sprintf("%.2f", k_i/k_norm)
k_j_norm = @sprintf("%.2f", k_j/k_norm)
k_k_norm = @sprintf("%.2f", k_k/k_norm)
χ_f = @sprintf("%.2f", χ_fill)
thicc = @sprintf("%.2f", num_cellsz/cells_per_wavelength)

text_label = Label(grid1[1:2, 2], "Number of cells per wavelength : $cells_per_wavelength 
  Decay on $decay_length wavelength(s)
  Medium filled with χ = $χ_f
  Film thickness of $thicc wavelength(s)
  Wavevector (normalized) : [ $k_i_norm , $k_j_norm , $k_k_norm ]" , 
  tellwidth=false, padding=(20, 20, 20, 20), halign=:center, fontsize=14)

mesh!(ax5, FRect3D(Point3f0(0, 0, 0), Point3f0(num_cellsx, num_cellsy, num_cellsz_vac)),
      color=:lightblue, transparency=true, alpha=0.5)

mesh!(ax5, FRect3D(Point3f0(0, 0, position_film), Point3f0(num_cellsx, num_cellsy, num_cellsz)),
      color=:purple, transparency=true, alpha=0.7)

lines!(ax5, [0, num_cellsx], [dst_slice, dst_slice], [0, 0], color=:red, linewidth=3, fxaa=true)
lines!(ax5, [0, num_cellsx], [dst_slice, dst_slice], [num_cellsz_vac, num_cellsz_vac], color=:red, linewidth=3, fxaa=true)
lines!(ax5, [0, 0], [dst_slice, dst_slice], [0, num_cellsz_vac], color=:red, linewidth=3, fxaa=true)
lines!(ax5, [num_cellsx, num_cellsx], [dst_slice, dst_slice], [0, num_cellsz_vac], color=:red, linewidth=3, fxaa=true)

arrow_scale_factor = 40
direction_vector = normalize([k_i, k_j, k_k]) * arrow_scale_factor
x_start = (num_cellsx / 2) - direction_vector[1]
z_start = (num_cellsz_vac + 5) - direction_vector[3]
start_point = Point3f0(x_start, dst_slice, z_start)
end_point = Point3f0(num_cellsx / 2, dst_slice, num_cellsz_vac + 5)

lines!(ax5, [start_point[1], end_point[1]], [start_point[2], end_point[2]], [start_point[3], end_point[3]], linewidth=2, color=:black, fxaa=true)

arrowhead_length = 5
arrowhead_width = 5
direction = normalize(end_point - start_point)
perp_direction = normalize(cross(direction, Point3f0(0, 1, 0)))

arrow_tip = end_point
arrow_base = end_point - arrowhead_length * direction
corner1 = arrow_base + arrowhead_width * perp_direction
corner2 = arrow_base - arrowhead_width * perp_direction

vertices = [arrow_tip, corner1, corner2]
faces = [1, 2, 3]

mesh!(ax5, GeometryBasics.Mesh(vertices, faces), color=:black)

ax4 = Axis(grid1[3:4, 1:2],
    title = L"Heatmap of the intensity of the electric field $E^2$",
    xlabel = "# of wavelengths in x",
    ylabel = "# of wavelengths in z",
    xticks = (x_ticks, x_labels),
    yticks = (y_ticks, y_labels)    
)

hm4 = heatmap!(ax4, data[4], colormap = :lajolla)
cb4 = Colorbar(grid1[3:4, 3], hm4, label="Intensity")


save("fig_docs/"*fig1_name, fig1)
# Show the figure
#display(fig1)

#######################################################################
# fig2 : real parts

fig2 = Figure(size = (900, 1000))
grid2 = fig2[1, 1] = GridLayout(tell = :bbox)

ax1 = Axis(grid2[1, 1],
    title = L"Heatmap of the real part of the electric field $E_x$",
    xlabel = "# of wavelengths in x",
    ylabel = "# of wavelengths in z",
    xticks = (x_ticks, x_labels),
    yticks = (y_ticks, y_labels)    
)

ax2 = Axis(grid2[2, 1],
    title = L"Heatmap of the real part of the electric field $E_y$",
    xlabel = "# of wavelengths in x",
    ylabel = "# of wavelengths in z",
    xticks = (x_ticks, x_labels),
    yticks = (y_ticks, y_labels)    
)

ax3 = Axis(grid2[3, 1],
    title = L"Heatmap of the real part of the electric field $E_z$",
    xlabel = "# of wavelengths in x",
    ylabel = "# of wavelengths in z",
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
# display(fig2)
