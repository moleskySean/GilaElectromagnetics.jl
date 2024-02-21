#= 
Compare external and self Green functions
=#
## host / device agreement
# input vector for merged Green function
innVecMrgHst = zeros(ComplexF32, celU..., 3)
# source current limited to celA half
rand!(view(innVecMrgHst, :, 1:celA[2], :, :))
innVecMrgDev = CUDA.zeros(ComplexF32, celU..., 3)
# transfer information to GPU
copyto!(innVecMrgDev, innVecMrgHst)
# confirm host and device computation give equivalent results
outVecMrgHst = egoOpr!(oprMrgHst, innVecMrgHst)
outVecMrgDev = egoOpr!(oprMrgDev, innVecMrgDev)
println("Host device self compute agreement: ", 
	@test all(outVecMrgHst .≈ Array(outVecMrgDev)))
## self / external agreement
# input vector for merged Green function
innVecMrgHst = zeros(ComplexF32, celU..., 3)
# source current limited to celA half
rand!(view(innVecMrgHst, :, 1:celA[2], :, :))
# input vector for host external Green function
innVecExtHst = zeros(ComplexF32, celA..., 3)
# input vector for device external Green function
innVecExtDev = CUDA.zeros(ComplexF32, celA..., 3)
# transfer information to external Green function 
copyto!(innVecExtHst, view(innVecMrgHst, 1:celA[1], 1:celA[2], 1:celA[3], :))
# transfer information to external device Green function 
copyto!(innVecExtDev, innVecExtHst)
# confirm host and device computation give equivalent results
outVecMrgHst = egoOpr!(oprMrgHst, innVecMrgHst)
outVecExtHst = egoOpr!(oprExtHst, innVecExtHst)
outVecExtDev = egoOpr!(oprExtDev, innVecExtDev)
# confirm host and device computation give equivalent results
println("Host device external compute agreement: ", 
	@test all(outVecExtHst .≈ Array(outVecExtDev)))
# confirm external and self computation give equivalent results
println("Self external compute agreement: ", 
	@test all(view(outVecMrgHst, 1:celB[1], (celB[2] + 1):celU[2], 1:celB[3], 
		:) .≈ outVecExtHst))