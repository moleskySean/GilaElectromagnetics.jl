using Test, GilaElectromagnetics

function dnsMat(G::GlaVacOprMem)
	mat = zeros(ComplexF64, prod(G.trgVol.cel)*3, prod(G.srcVol.cel)*3)
	for i in axes(mat, 2)
		v = zeros(ComplexF64, size(mat, 2))
		v[i] = one(ComplexF64)
        v = reshape(v, G.srcVol.cel..., 3)
        mat[:, i] = vec(egoOpr!(G, v))
	end
	return mat
end

asym(mat::AbstractMatrix{<:Complex}) = (mat - adjoint(mat)) / 2im

@testset "Positive Semi-Definiteness Tests" begin
    volSizes = [
        (2, 2, 2),
        (4, 4, 4),
        (6, 6, 6),
        (8, 8, 8),
        (6, 4, 8),
        (8, 2, 10)
    ]
    sclArr = (1//32, 1//32, 1//32)

    for volDim in volSizes
        # println("Testing volume size: ", volDim)
        volObj = GlaVol(volDim, sclArr, (0//1, 0//1, 0//1))
        oprMem = GlaVacOprMem(CPUKerOpt(), volObj)

        mat = dnsMat(oprMem)
        asmMat = asym(mat)
        sVals = eigvals(asmMat)
        # println("Computed minimum eigenvalue of ", minimum(sVals), ".")
        # println("A minimal value greater than -1.0e-6 is considered normal.")
        @test minimum(sVals) > -1.0e-6
    end
end
