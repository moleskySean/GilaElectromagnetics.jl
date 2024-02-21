bId = 4 
# determine source dominated directions
srcDomDir = map(<, oprInfHst.mixInf.trgCel, oprInfHst.mixInf.srcCel)
trgDomDir = map(!, srcDomDir) 
# total size of branch
brnSze = (oprInfHst.mixInf.trgCel .+ oprInfHst.mixInf.srcCel) .÷ 2
# zero if branch is even in that direction, one if branch is odd
brnSym = Bool.([mod(div(bId, ^(2, 3 - k)), 2) for k ∈ 1:3])
# declare memory 
if sum(oprInfHst.cmpInf.devMod) == true
	dimInfDev = CuArray{Int32}(undef, 3, 8)
	dirNumDev = CuArray{Int32}(undef, 3, 8)
	dirSymDev = CuArray{Float32}(undef, 3, 8)
else
	dirSymHst = Array{Float32}(undef, 3, 8)
end
# selection ranges for Green function and vector
modRng = Array{StepRange}(undef, 3)
vecRng = Array{StepRange}(undef, 3)
# perform Hadamard---forward and reverse operation in each direction
for divItr ∈ 0:7
	# one for reverse direction, zero for forward
	revSwt = [mod(div(divItr, ^(2, k - 1)), 2) for k ∈ 1:3]
	# negative one for reverse direction, one for forward
	symSgn = [1,1,1] .- 2 .* revSwt
	# set copy ranges
	for dirItr ∈ 1:3
		# switch between even and odd branch cases
		if brnSym[dirItr] == 0
			# switch between source and target portions of multiplication
			if revSwt[dirItr] == 0
				begInd = 1
				endInd = Integer.(ceil.(oprInfHst.mixInf.trgCel[dirItr] / 2)) + 
					trgDomDir[dirItr] * iseven(oprInfHst.mixInf.trgCel[dirItr])
			# source coefficients
			else
				begInd = 2
				endInd = Integer.(floor.(oprInfHst.mixInf.srcCel[dirItr] / 2)) +
					1 - trgDomDir[dirItr] * 
					iseven(oprInfHst.mixInf.trgCel[dirItr])
			end
		# odd coefficient branch
		else
			begInd = 1 + srcDomDir[dirItr]
			# target coefficients
			if revSwt[dirItr] == 0
				endInd = srcDomDir[dirItr] * 
					Integer.(floor.(oprInfHst.mixInf.trgCel[dirItr] / 2)) + 
					trgDomDir[dirItr] * 
					Integer.(ceil.(oprInfHst.mixInf.trgCel[dirItr] / 2)) + 
					srcDomDir[dirItr]
			# source coefficients
			else
				endInd = srcDomDir[dirItr] * 
					Integer.(ceil.(oprInfHst.mixInf.srcCel[dirItr] / 2)) + 
					trgDomDir[dirItr] * 
					Integer.(floor.(oprInfHst.mixInf.srcCel[dirItr] / 2)) + 
					srcDomDir[dirItr]
			end
		end
		# Green function range
		modRng[dirItr] = begInd:1:endInd
		# vector range
		if revSwt[dirItr] == 0
			vecRng[dirItr] = 1:1:(1 + endInd - begInd)
		else
			vecRng[dirItr] = brnSze[dirItr]:-1:(brnSze[dirItr] - endInd  + 
				begInd)
		end
	end
	if !isempty(CartesianIndices(tuple(modRng...)))
		@show modRng
		@show vecRng
	end
end