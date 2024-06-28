#=
glaIntsup evaluates the integrands called by the weakS, weakE, and weakV head 
functions using a series of variable transformations and analytic integral 
evaluations---reducing the four dimensional surface integrals performed for 
``standard'' cells to one dimensional integrals. Minimal comments are included 
in this code, which is mostly a julia translation of DIRECTFN_E by 
Athanasios Polimeridis. For a complete description of the steps 
being performed see the article cited above and references included therein. 
=#
#=
Weak integral evaluation for a panel interacting with itself. 
=#
function wekSInt(rPts::AbstractArray{<:AbstractFloat,2}, 
	glQud::AbstractArray{<:AbstractFloat,2}, cmpInf::GlaKerOpt)::ComplexF64

	glqOrd = size(glQud)[1]
	# integral as reduction
	return eqvJacS(rPts) * ThreadsX.mapreduce(x ->  wekSIntKer(x, glqOrd, 
		rPts, glQud, cmpInf), +, CartesianIndices((1:3, 1:8, 1:glqOrd)); 
		init = 0.0 + im * 0.0)
end
#=
Kernel function for self panel integrals.
=#
function wekSIntKer(slvInd::CartesianIndex{3}, glqOrd::Integer, 
	rPts::AbstractArray{<:AbstractFloat,2}, 
	glQud::AbstractArray{<:AbstractFloat,2}, cmpInf::GlaKerOpt)::ComplexF64

	(ψA, ψB) = ψlimS(slvInd[2])
	θ = θf(ψA, ψB, glQud[slvInd[3],1])
	(ηA, ηB) = ηlimS(slvInd[2], θ)
	intVal =  0.0 + im * 0.0
	# component contribution
	@inbounds for itr ∈ 1:glqOrd
		intVal += glQud[itr,2] * nS(slvInd[1], slvInd[2], θ, 
			θf(ηA, ηB, glQud[itr,1]), rPts, glQud, cmpInf)
	end
	return 0.25 * intVal * glQud[slvInd[3],2] * (ψB - ψA) * (ηB - ηA) * sin(θ)
end	
#=
Weak integral evaluation for two panels sharing an edge. 
=#
function wekEInt(rPts::AbstractArray{<:AbstractFloat,2}, 
	glQud::AbstractArray{<:AbstractFloat,2}, cmpInf::GlaKerOpt)::ComplexF64

	glqOrd = size(glQud)[1]
	# integral implemented as reduction
	return eqvJacEV(rPts) * ThreadsX.mapreduce(x ->  wekEIntKer(x, glqOrd, 
		rPts, glQud, cmpInf), +, CartesianIndices((1:6, 1:glqOrd)); 
		init = 0.0 + im * 0.0)
end
#=
Kernel for edge panel reduction.
=#
function wekEIntKer(slvInd::CartesianIndex{2}, glqOrd::Integer, 
	rPts::AbstractArray{<:AbstractFloat,2}, 
	glQud::AbstractArray{<:AbstractFloat,2}, cmpInf::GlaKerOpt)::ComplexF64

	(ψA, ψB) = ψlimE(slvInd[1])
	θB = θf(ψA, ψB, glQud[slvInd[2], 1])
	(ηA, ηB) = ηlimE(slvInd[1], θB)
	θA = 0.0 + im * 0.0
	intVal = 0.0 + im * 0.0
	# component contribution
	@inbounds for itr ∈ 1:glqOrd
		θA = θf(ηA, ηB, glQud[itr, 1])
		intVal += glQud[itr, 2] * cos(θA) * 
		(nE(slvInd[1], 1, θA, θB, rPts, glQud, cmpInf) + 
		nE(slvInd[1], -1, θA, θB, rPts, glQud, cmpInf)) 
	end
	return 0.25 * intVal * glQud[slvInd[2], 2] * (ψB - ψA) * (ηB - ηA) 
end
#=
Weak integral for panels sharing a vertex.
=#
function wekVInt(sngMod::Bool, rPts::AbstractArray{<:AbstractFloat,2}, 
	glQud::AbstractArray{<:AbstractFloat,2}, cmpInf::GlaKerOpt)::ComplexF64

	glqOrd = size(glQud)[1]
	# integral as mapreduction
	return eqvJacEV(rPts) * ^(π,2) * ThreadsX.mapreduce(x -> wekVIntKer(x, 
		glqOrd, sngMod, rPts, glQud, cmpInf), +, CartesianIndices((1:glqOrd, 
		1:glqOrd, 1:glqOrd,)); init = 0.0 + im * 0.0) / 144.0
end
#=
Kernel for weak vertex integral.
=#
function wekVIntKer(slvInd::CartesianIndex{3}, glqOrd::Integer, sngMod::Bool, 
	rPts::AbstractArray{<:AbstractFloat,2}, 
	glQud::AbstractArray{<:AbstractFloat,2}, cmpInf::GlaKerOpt)
	
	xPts = Array{Float64,2}(undef, 3, 2)
	θA = θf(0.0, π / 3.0, glQud[slvInd[1],1])
	sθA, cθA = sincos(θA)
	LA = 2.0 * sqrt(3.0) / (sθA + sqrt(3.0) * cθA)
	θB = θf(0.0, π / 3.0, glQud[slvInd[2],1])
	sθB, cθB = sincos(θB)
	LB = 2.0 * sqrt(3.0) / (sθB + sqrt(3.0) * cθB) 
	θC = θf(0.0, atan(LB / LA), glQud[slvInd[3],1])
	sθC, cθC = sincos(θC)
	θD = θf(atan(LB / LA), π / 2.0, glQud[slvInd[3],1])
	sθD, cθD = sincos(θD)
	intValC = 0.0 + im * 0.0
	intValD = 0.0 + im * 0.0
	θX = 0.0
	# select kernel mode
	if sngMod == true
		# loop D
		@inbounds for itrC ∈ 1:glqOrd
			θX = θf(0.0, LA / cθC, glQud[itrC,1])
			spxV!(xPts, θX, θC, θB, θA)
			intValC += glQud[itrC,2] * (θX^3) * 
			kerEVN(rPts, xPts, cmpInf.frqPhz)
		end
		# loop E
		@inbounds for itrD ∈ 1:glqOrd
			θX = θf(0.0, LB / sθD, glQud[itrD,1])
			spxV!(xPts, θX, θD, θB, θA)
			intValD += glQud[itrD,2] * (θX^3) * 
			kerEVN(rPts, xPts, cmpInf.frqPhz)
		end
	else
		# loop D
		@inbounds for itrC ∈ 1:glqOrd
			θX = θf(0.0, LA / cθC, glQud[itrC,1])
			spxV!(xPts, θX, θC, θB, θA)
			intValC += glQud[itrC,2] * (θX^3) * 
			kerEV(rPts, xPts, cmpInf.frqPhz)
		end
		# loop E
		@inbounds for itrD ∈ 1:glqOrd
			θX = θf(0.0, LB / sθD, glQud[itrD,1])
			spxV!(xPts, θX, θD, θB, θA)
			intValD += glQud[itrD,2] * (θX^3) * 
			kerEV(rPts, xPts, cmpInf.frqPhz)
		end
	end
	return glQud[slvInd[1],2] * glQud[slvInd[2],2] * 
	glQud[slvInd[3],2] * (atan(LB / LA) * (LA * sθC * intValC - LB * 
	cθD * intValD) + 0.5 * π * LB * cθD * intValD)
end

@inline function ψlimS(idf::Integer)::Tuple{Float64,Float64}
	
	if idf == 1 || idf == 5 || idf == 6 		return (0.0, π / 3.0)
	elseif idf == 2 || idf == 7 				return (π / 3.0, 2.0 * π / 3.0)
	elseif idf == 3 || idf == 4 || idf == 8 	return (2.0 * π / 3.0, π)
	else 									error("Unrecognized identifier.")
	end
end

@inline function ψlimE(idf::Integer)::Tuple{Float64,Float64}
	
	if idf == 1 						return (0.0, π / 3.0)
	elseif idf == 2 || idf == 3 		return (π / 3.0, π / 2.0)
	elseif idf == 4 || idf == 6 		return (π / 2.0, π)
	elseif idf == 5 					return (0.0, π / 2.0)
	else 								error("Unrecognized identifier.")
	end
end

@inline function ηlimS(idf::Integer, θ::AbstractFloat)::Tuple{Float64,Float64}
	
	if idf == 1 || idf == 2
		return (0.0, 1.0)
	elseif idf == 3
		return ((sqrt(3.0) - tan(π - θ)) / (sqrt(3.0) + tan(π - θ)), 
			1.0)
	elseif idf == 4
		return (0.0, (sqrt(3.0) - tan(π - θ)) / (sqrt(3.0) + tan(π - θ)))
	elseif idf == 5
		return ((tan(θ) - sqrt(3.0)) / (sqrt(3.0)  + tan(θ)), 0.0)
	elseif idf == 6
		return (-1.0, (tan(θ) - sqrt(3.0)) / (sqrt(3.0) + tan(θ)))
	elseif idf == 7 || idf == 8
		return (-1.0, 0.0)
	else
		error("Unrecognized identifier.")
	end
end

@inline function ηlimE(idf::Integer, θ::AbstractFloat)::Tuple{Float64,Float64}
	
	sθ, cθ = sincos(θ)
	if idf == 1
		return (0.0, atan(sθ + sqrt(3.0) * cθ))
	elseif idf == 2
		return (atan(sθ - sqrt(3.0) * cθ), 
			atan(sθ + sqrt(3.0) * cθ))
	elseif idf == 3 || idf == 4
		return (0.0, atan(sθ - sqrt(3.0) * cθ))
	elseif idf == 5
		return (atan(sθ + sqrt(3.0) * cθ), 0.5 * π)
	elseif idf == 6
		return (atan(sθ - sqrt(3.0) * cθ), 0.5 * π)
	else
		error("Unrecognized identifier.")
	end
end	

function nS(dir::Integer, idf::Integer, θ1::T, θB::T, rPts::Array{T,2}, 
	glQud::Array{T,2}, cmpInf::GlaKerOpt)::ComplexF64 where T<:AbstractFloat

	int = 0.0 + 0.0im
	glqOrd = size(glQud)[1]
	if idf == 1 || idf == 5
		for n ∈ 1:glqOrd
			int += glQud[n,2] * aS(rPts, θ1, θB, 
				θf(0.0, (1.0 - θB) / cos(θ1), glQud[n,1]), dir, glQud, 
				cmpInf)
		end
		return (1.0 - θB) / (2.0 * cos(θ1)) * int
	elseif idf == 2 || idf == 3
		for n ∈ 1:glqOrd
			int += glQud[n,2] * aS(rPts, θ1, θB, 
				θf(0.0, sqrt(3.0) * (1.0 - θB) / sin(θ1), glQud[n,1]), dir, 
				glQud, cmpInf)
		end
		return sqrt(3.0) * (1.0 - θB) / (2.0 * sin(θ1)) * int
	elseif idf == 6 || idf == 7
		for n ∈ 1:glqOrd
			int += glQud[n,2] * aS(rPts, θ1, θB, 
				θf(0.0, sqrt(3.0) * (1.0 + θB) / sin(θ1), glQud[n,1]), dir, 
				glQud, cmpInf)
		end 
		return sqrt(3.0) * (1.0 + θB) / (2.0 * sin(θ1)) * int
	elseif idf == 4 || idf == 8
		for n ∈ 1:glqOrd
			int += glQud[n,2] * aS(rPts, θ1, θB, 
				θf(0.0, -(1.0 + θB) / cos(θ1), glQud[n,1]), dir, glQud, 
				cmpInf)
		end
		return -(1.0 + θB) / (2.0 * cos(θ1)) * int
	else
		error("Unrecognized identifier.")
	end
end

function nE(idf1::Integer, idf2::Integer, θB::T, θ1::T, rPts::Array{T,2}, 
	glQud::Array{T,2}, cmpInf::GlaKerOpt)::ComplexF64 where T<:AbstractFloat

	γ = 0.0 
	intVal1 = 0.0 + 0.0im 
	intVal2 = 0.0 + 0.0im
	glqOrd = size(glQud)[1]
	if idf1 == 1 || idf1 == 2 
		sθ1, cθ1 = sincos(θ1)
		γ = (sθ1 + sqrt(3.0) * cθ1 - tan(θB)) / 
		(sθ1 + sqrt(3.0) * cθ1 + tan(θB))
		for n ∈ 1:glqOrd
			intVal1 += glQud[n, 2] * intNE(n, 1, γ, θB, θ1, rPts, glQud, 
				idf2, cmpInf)
			intVal2 += glQud[n, 2] * intNE(n, 2, γ, θB, θ1, rPts, glQud, 
				idf2, cmpInf)
		end
		return 0.5 * intVal2 + γ * 0.5 * (intVal1-intVal2)
	elseif idf1 == 3
		γ = sqrt(3.0) / tan(θ1)
		for n ∈ 1:glqOrd
			intVal1 += glQud[n, 2] * intNE(n, 1, γ, θB, θ1, rPts, glQud, 
				idf2, cmpInf)
			intVal2 += glQud[n, 2] * intNE(n, 3, γ, θB, θ1, rPts, glQud, 
				idf2, cmpInf)
		end
		return 0.5 * intVal2 + 0.5 * γ * (intVal1 - intVal2) 
	elseif idf1 == 4
		for n ∈ 1:glqOrd
			intVal1 += glQud[n, 2] * intNE(n, 4, 1.0, θB, θ1, rPts, 
				glQud, idf2, cmpInf)
		end
		return 0.5 * intVal1
	elseif idf1 == 5 || idf1 == 6
		for n ∈ 1:glqOrd
			intVal1 += glQud[n, 2] * intNE(n, 5, 1.0, θB, θ1, rPts, 
				glQud, idf2, cmpInf)
		end
		return 0.5 * intVal1 
	else
		error("Unrecognized identifier.")
	end
end

@inline function intNE(n::Integer, idf1::Integer, γ::T, θB::T, θ1::T, 
	rPts::Array{T,2}, glQud::Array{T,2}, idf2::Integer, 
	cmpInf::GlaKerOpt)::ComplexF64 where T <: AbstractFloat
	
	if idf1 == 1
		η = θf(0.0, γ, glQud[n,1])
		sθ1, cθ1 = sincos(θ1)
		λ = sqrt(3.0) * (1 + η)  /  (cos(θB) * (sθ1 + sqrt(3.0) * cθ1))
	elseif idf1 == 2
		η = θf(γ, 1.0, glQud[n,1])
		λ = sqrt(3.0) * (1.0 - abs(η)) / sin(θB)
	elseif idf1 == 3
		η = θf(γ, 1.0, glQud[n,1])
		sθ1, cθ1 = sincos(θ1)
		λ = sqrt(3.0) * (1.0 - abs(η)) / 
		(cos(θB) * (sθ1 - sqrt(3.0) * cθ1))
	elseif idf1 == 4
		η = θf(0.0, 1.0, glQud[n,1])
		sθ1, cθ1 = sincos(θ1)
		λ = sqrt(3.0) * (1.0 - abs(η)) / 
		(cos(θB) * (sθ1 - sqrt(3.0) * cθ1))
	elseif idf1 == 5
		η = θf(0.0, 1.0, glQud[n,1])
		λ = sqrt(3.0) * (1.0 - η) / sin(θB)
	else
		error("Unrecognized identifier.")
	end
	return aE(rPts, λ, η, θB, θ1, glQud, idf2, cmpInf)
end

function aS(rPts::Array{T,2}, θ1::T, θB::T, θ::T, dir::Integer, 
	glQud::Array{T,2}, cmpInf::GlaKerOpt)::ComplexF64 where T<:AbstractFloat

	xPts = Array{Float64,2}(undef, 3, 2)
	glqOrd = size(glQud)[1]
	aInt = 0.0 + 0.0im
	η1 = 0.0 
	η2 = 0.0 
	ξ1 = 0.0
	sθ1, cθ1 = sincos(θ1)
	@inbounds for n ∈ 1:glqOrd
		(η1, ξ1) = subTri(θB, θ * sθ1, dir)
		(η2, ξ2) = subTri(θf(0.0, θ, glQud[n,1]) * cθ1 + θB, 
			(θ - θf(0.0, θ, glQud[n,1])) * sθ1, dir)
		spx!(xPts, η1, η2, ξ1, ξ2)
		aInt += glQud[n,2] * θf(0.0, θ, glQud[n,1]) * 
		kerSN(rPts, xPts, cmpInf.frqPhz)
	end
	return 0.5 * θ * aInt 
end

function aE(rPts::Array{T,2}, λ::T, η::T, θB::T, θ1::T, glQud::Array{T,2}, 
	idf::Integer, cmpInf::GlaKerOpt)::ComplexF64 where T<:AbstractFloat

	xPts = Array{Float64,2}(undef, 3, 2)
	glqOrd = size(glQud)[1]
	intVal = 0.0 + 0.0im
	ζ = 0.0
	@inbounds for n ∈ 1:glqOrd
		ζ = θf(0.0, λ, glQud[n,1])
		spxE!(xPts, ζ, η, θB, θ1, idf)
		intVal += glQud[n,2] * ζ * ζ * kerEVN(rPts, xPts, cmpInf.frqPhz)
	end
	return 0.5 * λ * intVal 
end

@inline function subTri(λ1::AbstractFloat, λ2::AbstractFloat, 
	dir::Integer)::Tuple{Float64,Float64}

	if dir == 1 		
		return (λ1, λ2)
	elseif dir == 2
		return (0.5 * (1.0 - λ1 - λ2 * sqrt(3)), 
			0.5 * (sqrt(3.0) + λ1 * sqrt(3.0) - λ2))
	elseif dir == 3
		return (0.5 * (- 1.0 - λ1 + λ2 * sqrt(3)), 
			0.5 * (sqrt(3.0) - λ1 * sqrt(3.0) - λ2))
	else
		error("Unrecognized identifier.")
	end
end

@inline function eqvJacEV(rPts::Array{T,2})::Float64 where T<:AbstractFloat

	return sqrt(dot(cross(rPts[:,2] - rPts[:,1], 
		rPts[:,3] - rPts[:,1]), cross(rPts[:,2] - rPts[:,1], 
		rPts[:,3] - rPts[:,1]))) * sqrt(dot(cross(rPts[:,5] - 
		rPts[:,4], rPts[:,6] - rPts[:,4]), cross(rPts[:,5] - 
		rPts[:,4], rPts[:,6] - rPts[:,4]))) / 12.0
end

@inline function eqvJacS(rPts::Array{T,2})::Float64 where T<:AbstractFloat

	return dot(cross(rPts[:,1] - rPts[:,2], rPts[:,3] - rPts[:,1]), 
		cross(rPts[:,1] - rPts[:,2], rPts[:,3] - rPts[:,1])) / 12.0
end

@inline function θf(θa::T, θb::T, pos::T)::Float64	where T<:AbstractFloat
	
	return 0.5 * ((θb - θa) * pos + θa + θb)  
end

function spxV!(xPts::Array{T,2}, θ4::T, θ3::T, θB::T, θ1::T)::Nothing where 
	T <: AbstractFloat

	sθB, cθB = sincos(θB)
	sθ1, cθ1 = sincos(θ1)
	sθ3, cθ3 = sincos(θ3)
	spx!(xPts, θ4 * cθ3 * cθ1 - 1.0, θ4 * sθ3 * cθB - 1.0, θ4 * cθ3 * sθ1, θ4 *
		 sθ3 * sθB)
	return nothing
end

function spxE!(xPts::Array{T,2}, λ::T, η::T, θB::T, θ1::T, 
	idf::Integer)::Nothing where T<:AbstractFloat

	sθB, cθB = sincos(θB)
	sθ1, cθ1 = sincos(θ1)
	if idf == 1
		spx!(xPts, η, λ * cθB * cθ1 - η , λ * sθB, 
			λ * cθB * sθ1)
	elseif idf ==  - 1
		spx!(xPts,  -η,  -(λ * cθB * cθ1 - η) , λ * sθB, 
			λ * cθB * sθ1)
	else
		error("Unrecognized identifier.")
	end
	return nothing
end
#=
Two versions of the simplex function.
=#
@inline function spx!(xPts::Array{T,2}, η1::T, η2::T, ξ1::T, 
	ξ2::T)::Nothing where T <: AbstractFloat

	xPts[1,1] = (sqrt(3.0) * (1.0 - η1) - ξ1) / (2.0 * sqrt(3.0))
	xPts[2,1] = (sqrt(3.0) * (1.0 + η1) - ξ1) / (2.0 * sqrt(3.0))
	xPts[3,1] = ξ1 / sqrt(3.0)
	xPts[1,2] = (sqrt(3.0) * (1.0 - η2) - ξ2) / (2.0 * sqrt(3.0))
	xPts[2,2] = (sqrt(3.0) * (1.0 + η2) - ξ2) / (2.0 * sqrt(3.0))
	xPts[3,2] = ξ2 / sqrt(3.0)
	return nothing
end

@inline function kerEV(rPts::Array{T,2}, xPts::AbstractArray{T,2}, 
	frqPhz::Union{ComplexF64,ComplexF32})::ComplexF64 where T<:AbstractFloat

	return sclEgo(dstMag(xPts[1,1] * rPts[1,1] + xPts[2,1] * 
		rPts[1,2] + xPts[3,1] * rPts[1,3] - (xPts[1,2] * 
			rPts[1,4] + xPts[2,2] * rPts[1,5] + xPts[3,2] * 
			rPts[1,6]), xPts[1,1] * rPts[2,1] + xPts[2,1] * 
		rPts[2,2] + xPts[3,1] * rPts[2,3] - (xPts[1,2] * 
			rPts[2,4] + xPts[2,2] * rPts[2,5] + xPts[3,2] * 
			rPts[2,6]), xPts[1,1] * rPts[3,1] + xPts[2,1] * 
		rPts[3,2] + xPts[3,1] * rPts[3,3] - (xPts[1,2] * 
			rPts[3,4] + xPts[2,2] * rPts[3,5] + xPts[3,2] * 
			rPts[3,6])), frqPhz)
end

@inline function kerEVN(rPts::Array{T,2}, xPts::AbstractArray{T,2}, 
	frqPhz::Union{ComplexF64,ComplexF32})::ComplexF64 where T<:AbstractFloat
	
	return sclEgoN(dstMag(xPts[1,1] * rPts[1,1] + xPts[2,1] * 
		rPts[1,2] + xPts[3,1] * rPts[1,3] - (xPts[1,2] * 
			rPts[1,4] + xPts[2,2] * rPts[1,5] + xPts[3,2] * 
			rPts[1,6]), xPts[1,1] * rPts[2,1] + xPts[2,1] * 
		rPts[2,2] + xPts[3,1] * rPts[2,3] - (xPts[1,2] * 
			rPts[2,4] + xPts[2,2] * rPts[2,5] + xPts[3,2] * 
			rPts[2,6]), xPts[1,1] * rPts[3,1] + xPts[2,1] * 
		rPts[3,2] + xPts[3,1] * rPts[3,3] - (xPts[1,2] * 
			rPts[3,4] + xPts[2,2] * rPts[3,5] + xPts[3,2] * 
			rPts[3,6])), frqPhz)
end

@inline function kerSN(rPts::Array{T,2}, xPts::Array{T,2}, 
	frqPhz::Union{ComplexF64,ComplexF32})::ComplexF64 where T<:AbstractFloat
	
	return	sclEgoN(dstMag(xPts[1,1] * rPts[1,1] + xPts[2,1] * 
		rPts[1,2] + xPts[3,1] * rPts[1,3] - (xPts[1,2] * 
			rPts[1,1] + xPts[2,2] * rPts[1,2] + xPts[3,2] * 
			rPts[1,3]), xPts[1,1] * rPts[2,1] + xPts[2,1] * 
		rPts[2,2] + xPts[3,1] * rPts[2,3] - (xPts[1,2] * 
			rPts[2,1] + xPts[2,2] * rPts[2,2] + xPts[3,2] * 
			rPts[2,3]), xPts[1,1] * rPts[3,1] + xPts[2,1] * 
		rPts[3,2] + xPts[3,1] * rPts[3,3] - (xPts[1,2] * 
			rPts[3,1] + xPts[2,2] * rPts[3,2] + xPts[3,2] * 
			rPts[3,3])), frqPhz)
end