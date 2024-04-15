const cubRelTol = 1e-8;
const cubAbsTol = 1e-12; 
#=
verify that increasing quadrature order does not change integral values.
=#
function wekIntChk(scl::NTuple{3,<:Rational}, glOrd::Integer)::Array{Float64,1} 

	opts = GlaKerOpt(false)
	# weak integral values for internally set quadrature order
	ws1 = GilaElectromagnetics.wekS(scl, GilaElectromagnetics.gauQud(glOrd), 
		opts)
	we1 = GilaElectromagnetics.wekE(scl, GilaElectromagnetics.gauQud(glOrd), 
		opts)
	wv1 = GilaElectromagnetics.wekV(scl, GilaElectromagnetics.gauQud(glOrd), 
		opts)
	# weak integral values for additional quadrature points
	ws2 = GilaElectromagnetics.wekS(scl, GilaElectromagnetics.gauQud(glOrd + 8), 
		opts)
	we2 = GilaElectromagnetics.wekE(scl, GilaElectromagnetics.gauQud(glOrd + 8), 
		opts)
	wv2 = GilaElectromagnetics.wekV(scl, GilaElectromagnetics.gauQud(glOrd + 8), 
		opts)
	# differences
	intDifS = real(maximum(abs.(ws1 .- ws2) ./ abs.(ws2)))
	intDifE = real(maximum(abs.(we1 .- we2) ./ abs.(we2)))
	intDifV = real(maximum(abs.(wv1 .- wv2) ./ abs.(wv2)))
	return [intDifS, intDifE, intDifV]
end
# perform test
intDif = wekIntChk(oprSlfHst.srcVol.scl, oprSlfHst.cmpInf.intOrd)
if maximum(abs.(intDif)) < cubRelTol
	println("Integration convergence test passed for self volume.")
else
	println("Integration convergence test failed. Default GlaKerOpt integration order may be insufficient, consult glaMem.jl.")
end