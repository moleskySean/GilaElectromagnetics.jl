# memory declaration 
global linItr = 0
innVecHst = zeros(eltype(oprSlfHst.egoFur[1]), oprSlfHst.srcVol.cel..., 3)
egoCel = Array{eltype(innVecHst)}(undef, oprSlfHst.srcVol.cel..., 3)
egoDenMat = Array{eltype(innVecHst)}(undef, 3 * prod(oprSlfHst.srcVol.cel), 
	3 * prod(oprSlfHst.srcVol.cel))
egoDenMatAsm = Array{eltype(innVecHst)}(undef, 3 * prod(oprSlfHst.srcVol.cel), 
	3 * prod(oprSlfHst.srcVol.cel))
# fill Green function matrix
for crtItr ∈ CartesianIndices(tuple(oprSlfHst.srcVol.cel..., 3))
	# linear index
	global linItr = LinearIndices(egoCel)[crtItr]
	# set source
	innVecHst[crtItr] = (1.0 + 0.0im) 
	# calculate resulting field
	outVecHst = egoOpr!(oprSlfHst, innVecHst)
	# save field result
	copyto!(view(egoDenMat, :, linItr), outVecHst)
	# reset input vector
	copyto!(innVecHst, zeros(eltype(innVecHst), oprSlfHst.srcVol.cel..., 3))
end
# compute anti-symmetric component
adjoint!(egoDenMatAsm, egoDenMat);
lmul!((0.0 + 0.5im) * conj(cmpInfHst.frqPhz)^2, egoDenMatAsm);
axpy!((0.0 - 0.5im) * (cmpInfHst.frqPhz)^2, egoDenMat, egoDenMatAsm);
# numerical eigenvalues of anti-symmetric component of the Green function
sVals = eigvals(egoDenMatAsm);
println("Computed minimum eigenvalue of ", minimum(sVals), ".")
println("A minimal value greater than -1.0e-6 is considered normal.")