#=
Compare self Green function against analytic form
=#
const π = 3.1415926535897932384626433832795028841971693993751058209749445923
const sepTol = 1.0e-9
const lowTol = 1.0e-12
#=
egoAna is the analytic Green function.
=#
function egoAna!(anaOut::AbstractArray{T}, slfVol::GlaVol, 
	trgRng::Vector{<:StepRange}, dipPos::Vector{<:Rational}, 
	dipVec::Vector{T})::Nothing where T<:Union{ComplexF64,ComplexF64}
	# memory allocation
	linItr = zeros(Int,3)
	egoCel = Array{eltype(anaOut)}(undef, slfVol.cel..., 3)
	# separation magnitude and unit vector
	sep = 0.0
	sH = zeros(eltype(anaOut),3)
	# operators components
	sHs  = zeros(eltype(anaOut), 3, 3)
	id = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
	egoPar = zeros(eltype(anaOut), 3, 3)
	# Green function computation 
	for crtItr ∈ CartesianIndices(tuple(length.(trgRng)...))
		# index positions
		for dirItr ∈ 1:3 
			linItr[dirItr] = LinearIndices(egoCel)[crtItr, dirItr]
		end
		# separation vector
		sep = 2.0 * π * sqrt((trgRng[1][crtItr[1]] - dipPos[1])^2 + 
			(trgRng[2][crtItr[2]] - dipPos[2])^2 + 
			(trgRng[3][crtItr[3]] - dipPos[3])^2)
		# separation too small, use self-point approximation
		if sep < sepTol
			mul!(view(anaOut, linItr), (2.0 * ^(π, 2.0) * 2.0 * im / 3.0) .* id, dipVec)
		else
			sH = (2 * π) .* (trgRng[1][crtItr[1]] - dipPos[1], 
				trgRng[2][crtItr[2]] - dipPos[2], 
				trgRng[3][crtItr[3]] - dipPos[3]) ./ sep 
			sHs = [(sH[1] * sH[1]) (sH[1] * sH[2]) (sH[1] * sH[3]); 
				(sH[2] * sH[1]) (sH[2] * sH[2]) (sH[2] * sH[3]);  
		 		(sH[3] * sH[1]) (sH[3] * sH[2]) (sH[3] * sH[3])]
			egoPar = (2.0 * ^(π, 2.0) * exp(im * sep) / sep) .* 
					(((1.0 + (im * sep - 1.0) / ^(sep, 2.0)) .* id) .- 
					((1.0 + 3.0 * (im * sep - 1.0) / ^(sep, 2.0)) .* sHs))
			LinearAlgebra.mul!(view(anaOut, linItr), egoPar, dipVec)
		end
	end
	return nothing 
end
# compare against analytic discrete dipole solution
dipVec = zeros(ComplexF64, 3)
relErr = zeros(Float64, 3)
anaOut = Array{ComplexF64}(undef, 3 * prod(oprSlfHst.srcVol.cel))
numOut = Array{ComplexF64}(undef, oprSlfHst.srcVol.cel..., 3)
difMat = Array{Float64}(undef, oprSlfHst.srcVol.cel..., 3)
innVecHst = Array{eltype(oprSlfHst.egoFur[1])}(undef, oprSlfHst.srcVol.cel..., 3)
# window to remove for field comparisons 
winSze = 4 * minimum(oprSlfHst.srcVol.scl)
winInt = minimum([Int(div(1//2 * winSze, minimum(oprSlfHst.srcVol.scl))), 
	minimum(oprSlfHst.srcVol.cel)])
# check window size
if winInt > minimum(oprSlfHst.srcVol.cel)
	error("Excluded window is too large for slfVolume.")
end
winSze = winInt * minimum(oprSlfHst.srcVol.scl)
# cartesian direction loop
for dipDir ∈ eachindex(1:3)
	dipLoc = Int.([div(oprSlfHst.srcVol.cel[1], 2), 
		div(oprSlfHst.srcVol.cel[1], 2), 
		div(oprSlfHst.srcVol.cel[1], 2)])
	# dipole direction and location 
	dipVec[:] .= 0.0 + 0.0im
	dipVec[dipDir] = 1.0 + 0.0im
	local dipPos = [oprSlfHst.srcVol.grd[1][dipLoc[1]], 
		oprSlfHst.srcVol.grd[2][dipLoc[2]], 
		oprSlfHst.srcVol.grd[3][dipLoc[3]]]
	# output range for discrete dipole computation
	local trgRng = copy(oprSlfHst.srcVol.grd)
	# preform computations
	innVecHst[:,:,:,:] .= 0.0 + 0.0im;
	innVecHst[dipLoc[1], dipLoc[2], dipLoc[2], dipDir] = 
		(1.0 + 0.0im) / prod(oprSlfHst.srcVol.scl)
	outVecHst = egoOpr!(oprSlfHst, innVecHst);
	copyto!(numOut, outVecHst);
	egoAna!(anaOut, oprSlfHst.srcVol, trgRng, dipPos, dipVec);
	global anaOut = reshape(anaOut, oprSlfHst.srcVol.cel..., 3);
	# comparison array
	fldDif = 0.0;
	for crtItr ∈ CartesianIndices((oprSlfHst.srcVol.cel..., 3))
		# field difference
		fldDif = abs(anaOut[crtItr] - numOut[crtItr])
	 	difMat[crtItr] = min(fldDif, fldDif / max(abs(numOut[crtItr]), lowTol))
	end
	# remove points adjacent to dipole for comparison
	difMat[(dipLoc[1] - winInt):(dipLoc[1] + winInt), 
		(dipLoc[2] - winInt):(dipLoc[2] + winInt),
		(dipLoc[3] - winInt):(dipLoc[3] + winInt), :] .= 0.0 + 0.0im;
	# record maximum relative error in remaining vectors
	global relErr[dipDir] = maximum(difMat)
	# reset for further tests
	copyto!(innVecHst, zeros(eltype(innVecHst), 
		oprSlfHst.srcVol.cel..., 3));
	global anaOut = reshape(anaOut, 3 * prod(oprSlfHst.srcVol.cel));
end
global anaOut = reshape(anaOut, oprSlfHst.srcVol.cel..., 3);
println("Maximum relative field difference outside of ", winSze, 
	" exclusion window.")
@show relErr;