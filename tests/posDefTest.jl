# memory declaration 
global linItr = 0
egoCell = Array{ComplexF64}(undef, gSlfOprMemHst.srcVol.cells[1], 
	gSlfOprMemHst.srcVol.cells[2], gSlfOprMemHst.srcVol.cells[3], 3)
egoDenMat = Array{ComplexF64}(undef, 3 * prod(gSlfOprMemHst.srcVol.cells), 
	3 * prod(gSlfOprMemHst.srcVol.cells))
egoDenMatAsm = Array{ComplexF64}(undef, 3 * prod(gSlfOprMemHst.srcVol.cells), 
	3 * prod(gSlfOprMemHst.srcVol.cells))
# fill Green function matrix
for crtItr ∈ CartesianIndices((cells[1], cells[2], cells[3], 3))
	# linear index
	global linItr = LinearIndices(egoCell)[crtItr]
	# set source
	gSlfOprMemHst.actVec[crtItr] = (1.0 + 0.0im) 
	# calculate resulting field
	egoOpr!(gSlfOprMemHst)
	# save field result
	copyto!(view(egoDenMat, :, linItr), gSlfOprMemHst.actVec)
	# reset action vector
	copyto!(gSlfOprMemHst.actVec,zeros(eltype(anaOut), 
	gSlfOprMemHst.srcVol.cells..., 3))
end
# compute anti-symmetric component
adjoint!(egoDenMatAsm, egoDenMat);
lmul!((0.0 + 0.5im) * conj(cmpInfHst.frqPhz)^2, egoDenMatAsm);
axpy!((0.0 - 0.5im) * (cmpInfHst.frqPhz)^2, egoDenMat, egoDenMatAsm);
# numerical eigenvalues of anti-symmetric component of the Green function
sVals = eigvals(egoDenMatAsm);
println("Computed minimum eigenvalue of ", minimum(sVals), ".")
println("A minimal value greater than -1.0e-6 is considered normal.")