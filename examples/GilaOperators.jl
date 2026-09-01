module GilaOperators

using LinearAlgebra
using JLD2
using CUDA
using GilaElectromagnetics
import GilaElectromagnetics: glaSze

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

function load_green_operator(cells::NTuple{3, Int}, scale::NTuple{3, Rational{Int}}; set_type=ComplexF64, use_gpu::Bool=false)
	preload_dir = get_preload_dir()
	type_str = set_type == ComplexF64 ? "c64" : (set_type == ComplexF32 ? "c32" : "c16")
	fname = "$(type_str)_$(cells[1])x$(cells[2])x$(cells[3])_$(scale[1].num)ss$(scale[1].den)x$(scale[2].num)ss$(scale[2].den)x$(scale[3].num)ss$(scale[3].den).jld2"
	fpath = joinpath(preload_dir, fname)
	if isfile(fpath)
		print("Deserialising : cells = $cells , scale = $scale, type = $set_type ... ")
		file = jldopen(fpath)
		fourier = file["fourier"]
		if use_gpu
			fourier = CuArray.(fourier)
		end
		options = GlaKerOpt(use_gpu)
		volume = GlaVol(cells, scale, (0//1, 0//1, 0//1))
		mem = GlaOprMem(options, volume; egoFur=fourier, setTyp=set_type)
		println("Done")
		return GlaOpr(mem)
	end
	print("Serialising : cells = $cells , scale = $scale, type = $set_type ...")
	operator = GlaOpr(cells, scale; setTyp=set_type, useGpu=use_gpu)
	fourier = operator.mem.egoFur
	if use_gpu
		fourier = Array.(fourier)
	end
	jldsave(fpath; fourier=fourier)
	println("Done")
	return operator
end

struct LippmannSchwinger
	green_op::GlaOpr
	medium::AbstractArray{<:Complex, 4}

	function LippmannSchwinger(green_op::GlaOpr, medium::AbstractArray{<:Complex})
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
		if green_op.mem.cmpInf.devMod
			medium = CuArray(medium)
		end
		new(green_op, medium)
	end
end

function LippmannSchwinger(cells::NTuple{3, Int}, scale::NTuple{3, Rational{Int}}, medium::AbstractArray{<:Complex}; set_type=ComplexF64, use_gpu::Bool=false)
	green_op = load_green_operator(cells, scale; set_type=set_type, use_gpu=use_gpu)
	return LippmannSchwinger(green_op, medium)
end

Base.size(op::LippmannSchwinger) = size(op.green_op)
Base.size(op::LippmannSchwinger, dim::Int) = size(op.green_op, dim)
glaSze(op::LippmannSchwinger) = glaSze(op.green_op)
glaSze(op::LippmannSchwinger, dim::Int) = glaSze(op.green_op, dim)
Base.eltype(op::LippmannSchwinger) = eltype(op.green_op)
function Base.:*(op::LippmannSchwinger, x::AbstractArray)
	if op.green_op.mem.cmpInf.devMod
		x = CuArray(x)
	end
	gx = reshape(op.green_op * x, glaSze(op, 1))
	return x - reshape(op.medium .* gx, size(x))
end
LinearAlgebra.mul!(y::AbstractArray, op::LippmannSchwinger, x::AbstractArray) = y .= op * x

end
