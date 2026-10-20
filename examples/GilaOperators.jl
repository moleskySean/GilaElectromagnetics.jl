module GilaOperators

using LinearAlgebra
using JLD2
using GilaElectromagnetics
import GilaElectromagnetics: GlaKerOpt, GlaVacOprMem, glaSze

# Load a GPU package (CUDA, Metal or OpenCL) before using this module with
# dev != CPU(); Gila's extension for it then loads automatically.

export load_green_operator, LippmannSchwinger

function get_preload_dir()
	found_dir = false
	dir = "preload/"
	for i in 1:10
		if !isdir(dir)
			dir = "../"^i * "preload/"
		else
			found_dir = true
			break
		end
	end
	if !found_dir
		error("Could not find preload directory. Please create a directory named 'preload' in the current directory or parent directories.")
	end
	return dir
end

function load_green_operator(cells::NTuple{3, Int}, scale::NTuple{3, Rational{Int}}; set_type=ComplexF64, dev=CPU())
	preload_dir = get_preload_dir()
	type_str = set_type == ComplexF64 ? "c64" : (set_type == ComplexF32 ? "c32" : "c16")
	fname = "$(type_str)_$(cells[1])x$(cells[2])x$(cells[3])_$(scale[1].num)ss$(scale[1].den)x$(scale[2].num)ss$(scale[2].den)x$(scale[3].num)ss$(scale[3].den).jld2"
	fpath = joinpath(preload_dir, fname)
	prc = set_type == ComplexF64 ? Float64 : Float32
	if isfile(fpath)
		print("Deserialising : cells = $cells , scale = $scale, type = $set_type ... ")
		file = jldopen(fpath)
		fourier = adapt.(Ref(dev), file["fourier"])
		options = GlaKerOpt{prc}(dev)
		volume = GlaVol(cells, scale, (0//1, 0//1, 0//1))
		mem = GlaVacOprMem(options, fourier, volume)
		println("Done")
		return GlaOprVac(mem)
	end
	print("Serialising : cells = $cells , scale = $scale, type = $set_type ...")
	operator = GlaOprVac{prc}(GlaVol(cells, scale, (0//1, 0//1, 0//1)); dev=dev)
	fourier = Array.(operator.mem.egoFur)
	jldsave(fpath; fourier=fourier)
	println("Done")
	return operator
end

struct LippmannSchwinger
	green_op::AbstractGlaVacOpr
	medium::AbstractArray{<:Complex, 4}

	function LippmannSchwinger(green_op::AbstractGlaVacOpr, medium::AbstractArray{<:Complex})
		if glaSze(green_op, 1)[1:3] != size(medium)
			println(glaSze(green_op, 1)[1:3])
			println("!=")
			println(size(medium))
			throw(DimensionMismatch("Green operator and medium must have the same size."))
		end
		if eltype(green_op) != eltype(medium)
			throw(ArgumentError("Medium must have the same element type as the Green operator."))
		end
		medium = reshape(medium, glaSze(green_op, 1)[1:3]..., 1)
		medium = adapt(get_backend(green_op), medium)
		new(green_op, medium)
	end
end

function LippmannSchwinger(cells::NTuple{3, Int}, scale::NTuple{3, Rational{Int}}, medium::AbstractArray{<:Complex}; set_type=ComplexF64, dev=CPU())
	green_op = load_green_operator(cells, scale; set_type=set_type, dev=dev)
	return LippmannSchwinger(green_op, medium)
end

Base.size(op::LippmannSchwinger) = size(op.green_op)
Base.size(op::LippmannSchwinger, dim::Int) = size(op.green_op, dim)
glaSze(op::LippmannSchwinger) = glaSze(op.green_op)
glaSze(op::LippmannSchwinger, dim::Int) = glaSze(op.green_op, dim)
Base.eltype(op::LippmannSchwinger) = eltype(op.green_op)
function Base.:*(op::LippmannSchwinger, x::AbstractArray)
	x = adapt(get_backend(op.green_op), x)
	gx = reshape(op.green_op * x, glaSze(op, 1))
	return x - reshape(op.medium .* gx, size(x))
end
LinearAlgebra.mul!(y::AbstractArray, op::LippmannSchwinger, x::AbstractArray) = y .= op * x

end
