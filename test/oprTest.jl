using Test

# Make sure the constructors throw if the type is not a subtype of Complex
@test_throws ArgumentError GlaOpr((1, 1, 1), (1//1, 1//1, 1//1), 
	(0//1, 0//1, 0//1), setTyp=Float64)

self_operator_host = GlaOpr(oprSlfHst)
external_operator_host = GlaOpr(oprExtHst)
merge_operator_host = GlaOpr(oprMrgHst)
if CUDA.functional()
	external_operator_device = GlaOpr(oprExtDev)
	merge_operator_device = GlaOpr(oprMrgDev)
end

function generic_tests(opr::GlaOpr)
	# Make sure the eltype is complex
	@test eltype(opr) <: Complex
	# Create some data to feed to Gila
	x = ones(eltype(opr), size(opr, 2))
	x_gilashape = reshape(x, glaSze(opr, 2))
	x_gilashape_copy = deepcopy(x_gilashape)
	if opr.mem.cmpInf.devMod
		x_gilashape_copy = CuArray(x_gilashape_copy)
	end
	reference_acted_x = egoOpr!(opr.mem, x_gilashape_copy)
	# Make sure the tensor multiplication is correct
	acted_x_gilashape = opr * x_gilashape
	@test acted_x_gilashape ≈ reference_acted_x
	# Make sure the input vector is not mutated
	@test x_gilashape == reshape(ones(eltype(opr), size(opr, 2)), glaSze(opr, 2))
	# Make sure the vector multiplication is correct
	acted_x = opr * x
	@test acted_x ≈ vec(reference_acted_x)
	# Make sure the input vector is not mutated
	@test x == ones(eltype(opr), size(opr, 2))
	# Make sure we can't multiply by the wrong complex type
	t = eltype(opr) <: ComplexF32 ? ComplexF64 : ComplexF32
	bad_input = ones(t, size(opr, 2))
	@test_throws AssertionError opr * bad_input
	# Make sure we get an error if the size of the input vector is wrong
	bad_input = ones(eltype(opr), size(opr, 2) + 1)
	@test_throws DimensionMismatch opr * bad_input
	# Make sure the adjoint of the adjoint is the original operator
	adj = adjoint(opr)
	adj_adj = adjoint(adj)
	@test adjoint(adj).mem.egoFur == opr.mem.egoFur
	@test adj_adj * x ≈ opr * x
	@test isadjoint(opr) == false
	@test isadjoint(adj) == true
	# Some proprerties
	@test issymmetric(opr) == false
	@test isposdef(opr) == false
	@test ishermitian(opr) == false
	@test isdiag(opr) == false
end

@testset "Self operator host" begin
	@test size(self_operator_host) == (16*16*16*3, 16*16*16*3)
	@test glaSze(self_operator_host) == ((16, 16, 16, 3), (16, 16, 16, 3))
	@test isselfoperator(self_operator_host) == true
	@test isexternaloperator(self_operator_host) == false
	generic_tests(self_operator_host)
end

@testset "External operator host" begin
	@test size(external_operator_host) == (16*16*16*3, 16*16*16*3)
	@test glaSze(external_operator_host) == ((16, 16, 16, 3), (16, 16, 16, 3))
	@test isselfoperator(external_operator_host) == false
	@test isexternaloperator(external_operator_host) == true
	generic_tests(external_operator_host)
end

@testset "Merge operator host" begin
	@test size(merge_operator_host) == (16*32*16*3, 16*32*16*3)
	@test glaSze(merge_operator_host) == ((16, 32, 16, 3), (16, 32, 16, 3))
	@test isselfoperator(merge_operator_host) == true
	@test isexternaloperator(merge_operator_host) == false
	generic_tests(merge_operator_host)
end

if CUDA.functional()
	@testset "External operator device" begin
		@test size(external_operator_device) == (16*16*16*3, 16*16*16*3)
		@test glaSze(external_operator_device) == ((16, 16, 16, 3), (16, 16, 
			16, 3))
		@test isselfoperator(external_operator_device) == false
		@test isexternaloperator(external_operator_device) == true
		generic_tests(external_operator_device)
	end

	@testset "Merge operator device" begin
		@test size(merge_operator_device) == (16*32*16*3, 16*32*16*3)
		@test glaSze(merge_operator_device) == ((16, 32, 16, 3), (16, 32, 16, 
			3))
		@test isselfoperator(merge_operator_device) == true
		@test isexternaloperator(merge_operator_device) == false
		generic_tests(merge_operator_device)
	end
end
