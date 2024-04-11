#= 
Compare external and self Green functions
=#
## verify agreement of host and device computation
# input vector for merged Green function
innVecMrgHst = zeros(ComplexF32, celU..., 3)
# source current limited to celA half
rand!(view(innVecMrgHst, :, 1:celA[2], :, :))
# prepare device computation if CUDA is functional
if CUDA.functional()
	innVecMrgDev = CUDA.zeros(ComplexF32, celU..., 3)
	# transfer information to GPU
	copyto!(innVecMrgDev, innVecMrgHst)
	# confirm host and device computation give equivalent results
	outVecMrgHst = egoOpr!(oprMrgHst, innVecMrgHst)
	outVecMrgDev = egoOpr!(oprMrgDev, innVecMrgDev)
	println("Host device self compute agreement: ", 
		@test all(outVecMrgHst .≈ Array(outVecMrgDev)))
end
## verify agreement of self and merged external Green function
# input vector for merged Green function
innVecMrgHst = zeros(ComplexF32, celU..., 3)
# source current limited to celA half
rand!(view(innVecMrgHst, :, 1:celA[2], :, :))
# input vector for host external Green function
innVecExtHst = zeros(ComplexF32, celA..., 3)
# transfer information to external Green function 
copyto!(innVecExtHst, view(innVecMrgHst, 1:celA[1], 1:celA[2], 1:celA[3], :))
if CUDA.functional()
	# input vector for device external Green function
	innVecExtDev = CUDA.zeros(ComplexF32, celA..., 3)
	# transfer information to external device Green function 
	copyto!(innVecExtDev, innVecExtHst)
	# confirm host and device computation give equivalent results
	outVecExtDev = egoOpr!(oprExtDev, innVecExtDev)
end
outVecMrgHst = egoOpr!(oprMrgHst, innVecMrgHst)
outVecExtHst = egoOpr!(oprExtHst, innVecExtHst)
# confirm host and device computation give equivalent results
if CUDA.functional()
	println("Host device external compute agreement: ", 
		@test all(outVecExtHst .≈ Array(outVecExtDev)))
end
# confirm external and self computation give equivalent results
println("Self external compute agreement: ", 
	@test all(view(outVecMrgHst, 1:celB[1], (celB[2] + 1):celU[2], 1:celB[3], 
		:) .≈ outVecExtHst))