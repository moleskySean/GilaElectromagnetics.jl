using Test, GilaElectromagnetics, LinearAlgebra

const sepTol = 1.0e-9
const lowTol = 1.0e-12
#=
egoAna is the analytic Green function.
=#
function egoAna!(anaOut::AbstractArray{ComplexF64}, slfVol::GlaVol, trgRng::Vector{<:StepRange}, dipPos::Vector{<:Rational}, dipVec::Vector{ComplexF64})
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
			mul!(view(anaOut, linItr), (2.0 * ^(π, 2.0) * 2.0 * im / 3.0) .* 
				id, dipVec)
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


@testset "Analytical Self Green Function Tests" begin
    volSizes = [
        (6, 6, 6),
        (8, 8, 8),
        (6, 10, 8),
        (16, 16, 16)
    ]
    sclArr = (1//32, 1//32, 1//32)

    for volDim in volSizes
        # println("Testing volume size: ", volDim)
        volObj = GlaVol(volDim, sclArr, (0//1, 0//1, 0//1))
        oprMem = GlaVacOprMem(CPUKerOpt(), volObj)

        dipVec = zeros(ComplexF64, 3)
        relErr = zeros(Float64, 3)
        anaOut = Array{ComplexF64}(undef, 3 * prod(oprMem.srcVol.cel))
        numOut = Array{ComplexF64}(undef, oprMem.srcVol.cel..., 3)
        difMat = Array{Float64}(undef, oprMem.srcVol.cel..., 3)
        innVecHst = Array{eltype(oprMem.egoFur[1])}(undef, oprMem.srcVol.cel..., 3)

        # Exclusion window for field comparisons
        winSze = 4 * minimum(oprMem.srcVol.scl)
        winInt = minimum([Int(div(1//2 * winSze, minimum(oprMem.srcVol.scl))),
                          minimum(oprMem.srcVol.cel)])
        if winInt > minimum(oprMem.srcVol.cel)
            error("Excluded window is too large for self volume.")
        end
        winSze = winInt * minimum(oprMem.srcVol.scl)

        # Cartesian direction loop
        for dipDir ∈ 1:3
            dipLoc = Int.([div(oprMem.srcVol.cel[1], 2),
                           div(oprMem.srcVol.cel[2], 2),
                           div(oprMem.srcVol.cel[3], 2)])
            dipVec[:] .= 0.0 + 0.0im
            dipVec[dipDir] = 1.0 + 0.0im
            dipPos = [oprMem.srcVol.grd[1][dipLoc[1]],
                      oprMem.srcVol.grd[2][dipLoc[2]],
                      oprMem.srcVol.grd[3][dipLoc[3]]]
            trgRng = deepcopy(oprMem.srcVol.grd)

            # Perform computations
            fill!(innVecHst, zero(eltype(innVecHst)))
            innVecHst[dipLoc[1], dipLoc[2], dipLoc[3], dipDir] =
                (1.0 + 0.0im) / prod(oprMem.srcVol.scl)
            outVecHst = egoOpr!(oprMem, innVecHst)
            copyto!(numOut, outVecHst)
            egoAna!(anaOut, oprMem.srcVol, trgRng, dipPos, dipVec)
            anaOut = reshape(anaOut, oprMem.srcVol.cel..., 3)

            # Compute field differences
            for crtItr ∈ CartesianIndices((oprMem.srcVol.cel..., 3))
                fldDif = abs(anaOut[crtItr] - numOut[crtItr])
                difMat[crtItr] = min(fldDif, fldDif / max(abs(numOut[crtItr]), lowTol))
            end

            # Exclude points adjacent to dipole for comparison
            difMat[(dipLoc[1] - winInt):(dipLoc[1] + winInt),
                   (dipLoc[2] - winInt):(dipLoc[2] + winInt),
                   (dipLoc[3] - winInt):(dipLoc[3] + winInt), :] .= 0.0

            # Record maximum relative error
            relErr[dipDir] = maximum(difMat)
        end

        # println("Maximum relative field difference outside of exclusion window:")
        # @show relErr
        @test all(relErr .< 0.05)
    end
end
