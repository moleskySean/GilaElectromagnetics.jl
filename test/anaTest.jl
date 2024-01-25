const π = 3.1415926535897932384626433832795028841971693993751058209749445923
const sepTol = 1.0e-9
const lowTol = 1.0e-12
#=
egoAna is the analytic Green function.
=#
function egoAna!(anaOut::AbstractArray{T}, vol::GlaVol, 
	trgRng::Vector{<:StepRangeLen}, dipPos::Vector{<:AbstractFloat}, 
	dipVec::Vector{T})::Nothing where T<:Union{ComplexF64,ComplexF64}
	# memory allocation
	linItrs = zeros(Int,3)
	egoCell = Array{eltype(anaOut)}(undef, vol.cel[1], 
		vol.cel[2], vol.cel[3], 3)
	# separation magnitude and unit vector
	sep = 0.0
	sh = zeros(eltype(anaOut),3)
	# operators components
	shs = zeros(eltype(anaOut), 3, 3)
	id = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
	egoPar = zeros(eltype(anaOut), 3, 3)
	# Green function computation 
	for crtItr ∈ CartesianIndices((length(trgRng[1]), 
		length(trgRng[2]), length(trgRng[3])))
		# index positions
		for dirItr ∈ 1:3 
			linItrs[dirItr] = LinearIndices(egoCell)[crtItr[1], crtItr[2], 
			crtItr[3], dirItr]
		end
		# separation vector
		sep = 2 * π * sqrt((trgRng[1][crtItr[1]] - dipPos[1])^2 + 
			(trgRng[2][crtItr[2]] - dipPos[2])^2 + 
			(trgRng[3][crtItr[3]] - dipPos[3])^2)
		# separation too small, use self-point approximation
		if sep < sepTol
			mul!(view(anaOut, linItrs), (2 * π^2  * 2 * im / 3) .* id, dipVec)
		else
			sh = (2 * π) .* (trgRng[1][crtItr[1]] - dipPos[1], 
				trgRng[2][crtItr[2]] - dipPos[2], 
				trgRng[3][crtItr[3]] - dipPos[3]) ./ sep 
			sHs = [(sh[1] * sh[1]) (sh[1] * sh[2]) (sh[1] * sh[3]); 
		 		(sh[2] * sh[1]) (sh[2] * sh[2]) (sh[2] * sh[3]);  
		 		(sh[3] * sh[1]) (sh[3] * sh[2]) (sh[3] * sh[3])]
			egoPar = (2 * π^2 * exp(im * sep) / sep) .* 
					(((1 + (im * sep - 1) / sep^2) .* id) .- 
					((1 + 3 * (im * sep - 1) / sep^2) .* sHs))
			mul!(view(anaOut, linItrs), egoPar, dipVec)
		end
	end
	return nothing 
end
# compare against analytic discrete dipole solution
dipVec = zeros(ComplexF64, 3)
relErr = zeros(Float64, 3)
anaOut = Array{ComplexF64}(undef, 3 * prod(gSlfMemHst.srcVol.cel))
numOut = Array{ComplexF64}(undef, gSlfMemHst.srcVol.cel..., 3)
difMat = Array{Float64}(undef, gSlfMemHst.srcVol.cel..., 3)
# window to remove for field comparisons 
winSze = 1//20
winInt = minimum([Int(div(1//2 * winSze, minimum(gSlfMemHst.srcVol.scl))), 
	minimum(gSlfMemHst.srcVol.cel)])
# check window size
if winInt > minimum(gSlfMemHst.srcVol.cel)
	error("Excluded window is too large for volume.")
end
winSze = winInt * minimum(gSlfMemHst.srcVol.scl)
# current density vector

# cartesian direction loop
for dipDir ∈ 1:3
	dipLoc = Int.([div(gSlfMemHst.srcVol.cel[1], 2), 
		div(gSlfMemHst.srcVol.cel[1], 2), 
		div(gSlfMemHst.srcVol.cel[1], 2)])
	# dipole direction and location 
	dipVec[:] .= 0.0 + 0.0im
	dipVec[dipDir] = 1.0 + 0.0im
	local dipPos = [gSlfMemHst.srcVol.grd[1][dipLoc[1]], 
		gSlfMemHst.srcVol.grd[2][dipLoc[2]], 
		gSlfMemHst.srcVol.grd[3][dipLoc[3]]]
	# output range for discrete dipole computation
	local trgRng = copy(gSlfMemHst.srcVol.grd)
	# preform computations
	gSlfMemHst.actVec[:,:,:,:] .= 0.0 + 0.0im;
	gSlfMemHst.actVec[dipLoc[1], dipLoc[2], dipLoc[2], dipDir] = 
		(1.0 + 0.0im) / prod(gSlfMemHst.srcVol.scl)
	egoOpr!(gSlfMemHst);
	copyto!(numOut, gSlfMemHst.actVec);
	egoAna!(anaOut, gSlfMemHst.srcVol, trgRng, dipPos, dipVec);
	global anaOut = reshape(anaOut, gSlfMemHst.srcVol.cel..., 3);
	# comparison array
	fldDif = 0.0;
	for crtItr ∈ CartesianIndices((gSlfMemHst.srcVol.cel[1], 
		gSlfMemHst.srcVol.cel[2], gSlfMemHst.srcVol.cel[3], 3))
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
	copyto!(gSlfMemHst.actVec,zeros(eltype(anaOut), 
		gSlfMemHst.srcVol.cel..., 3));
	global anaOut = reshape(anaOut, 3 * prod(gSlfMemHst.srcVol.cel));
end
global anaOut = reshape(anaOut, gSlfMemHst.srcVol.cel..., 3);
println("Maximum relative field difference outside of ", winSze, 
	" exclusion window.")
@show relErr;