const cubRelTol = 1e-8;
const cubAbsTol = 1e-12; 
#=
verify that increasing quadrature order does not change integral values.
=#
function wekIntChk(scl::AbstractFloat, glOrd::Integer)::Array{T,1} where 
	T <: Union{ComplexF64,ComplexF32}
	
	sclV = (scl, scl, scl)
	opts = GlaKerOpt(false)
	# weak integral values for internally set quadrature order
	ws1 = weakS(sclV, gaussQuad1(glOrd), opts)
	we1 = weakE(sclV, gaussQuad1(glOrd), opts)
	wv1 = weakV(sclV, gaussQuad1(glOrd), opts)
	# weak integral values for additional quadrature points
	ws2 = weakS(sclV, gaussQuad1(glOrd + 8), opts)
	we2 = weakE(sclV, gaussQuad1(glOrd + 8), opts)
	wv2 = weakV(sclV, gaussQuad1(glOrd + 8), opts)
	# differeneces
	intDifS = maximum(abs.(ws1 .- ws2) ./ abs.(ws2))
	intDifE = maximum(abs.(we1 .- we2) ./ abs.(we2))
	intDifV = maximum(abs.(wv1 .- wv2) ./ abs.(wv2))
	return [intDifS, intDifE, intDifV]
end
# perform test
intDif = wekIntChk(gSlfOprMemHst.srcVol.scl, 48)
if maximum(abs.(intDiff)) < cubRelTol
	println("Integration convergence test passed.")
else
	println("Integration convergence test failed!")	
end