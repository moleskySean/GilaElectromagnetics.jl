using Test

# Make sure the constructors throw if the type is not a subtype of Complex
@test_throws ArgumentError GlaOpr((1, 1, 1), (1//1, 1//1, 1//1), (0//1, 0//1, 0//1), setType=Float64)

self_operator_host = GlaOpr(oprSlfHst)
external_operator_host = GlaOpr(oprExtHst)
merge_operator_host = GlaOpr(oprMrgHst)
if CUDA.functional()
	external_operator_device = GlaOpr(oprExtDev)
	merge_operator_device = GlaOpr(oprMrgDev)
end

function generic_tests(op::GlaOpr)
	# Make sure the eltype is complex
	@test eltype(op) <: Complex

	# Create some data to feed to Gila
	x = ones(eltype(op), size(op, 2))
	x_gilashape = reshape(x, gilaSize(op, 2))
	x_gilashape_copy = deepcopy(x_gilashape)
	if op.mem.cmpInf.devMod
		x_gilashape_copy = CuArray(x_gilashape_copy)
	end
	reference_acted_x = egoOpr!(op.mem, x_gilashape_copy)

	# Make sure the tensor multiplication is correct
	acted_x_gilashape = op * x_gilashape
	@test acted_x_gilashape ≈ reference_acted_x

	# Make sure the input vector is not mutated
	@test x_gilashape == reshape(ones(eltype(op), size(op, 2)), gilaSize(op, 2))

	# Make sure the vector multiplication is correct
	acted_x = op * x
	@test acted_x ≈ vec(reference_acted_x)

	# Make sure the input vector is not mutated
	@test x == ones(eltype(op), size(op, 2))

	# Make sure we can't multiply by the wrong complex type
	t = eltype(op) <: ComplexF32 ? ComplexF64 : ComplexF32
	bad_input = ones(t, size(op, 2))
	@test_throws AssertionError op * bad_input

	# Make sure we get an error if the size of the input vector is wrong
	bad_input = ones(eltype(op), size(op, 2) + 1)
	@test_throws DimensionMismatch op * bad_input

	# Make sure the adjoint of the adjoint is the original operator
	adj = adjoint(op)
	adj_adj = adjoint(adj)
	@test adjoint(adj).mem.egoFur == op.mem.egoFur
	@test adj_adj * x ≈ op * x
	@test isadjoint(op) == false
	@test isadjoint(adj) == true

	# Some properties
	@test issymmetric(op) == true
	@test isposdef(op) == true
	@test ishermitian(op) == false
	@test isdiag(op) == false
end

@testset "Self operator host" begin
	@test size(self_operator_host) == (16*16*16*3, 16*16*16*3)
	@test gilaSize(self_operator_host) == ((16, 16, 16, 3), (16, 16, 16, 3))
	@test isselfoperator(self_operator_host) == true
	@test isexternaloperator(self_operator_host) == false
	generic_tests(self_operator_host)
end

@testset "External operator host" begin
	@test size(external_operator_host) == (16*16*16*3, 16*16*16*3)
	@test gilaSize(external_operator_host) == ((16, 16, 16, 3), (16, 16, 16, 3))
	@test isselfoperator(external_operator_host) == false
	@test isexternaloperator(external_operator_host) == true
	generic_tests(external_operator_host)
end

@testset "Merge operator host" begin
	@test size(merge_operator_host) == (16*32*16*3, 16*32*16*3)
	@test gilaSize(merge_operator_host) == ((16, 32, 16, 3), (16, 32, 16, 3))
	@test isselfoperator(merge_operator_host) == true
	@test isexternaloperator(merge_operator_host) == false
	generic_tests(merge_operator_host)
end

if CUDA.functional()
	@testset "External operator device" begin
		@test size(external_operator_device) == (16*16*16*3, 16*16*16*3)
		@test gilaSize(external_operator_device) == ((16, 16, 16, 3), (16, 16, 16, 3))
		@test isselfoperator(external_operator_device) == false
		@test isexternaloperator(external_operator_device) == true
		generic_tests(external_operator_device)
	end

	@testset "Merge operator device" begin
		@test size(merge_operator_device) == (16*32*16*3, 16*32*16*3)
		@test gilaSize(merge_operator_device) == ((16, 32, 16, 3), (16, 32, 16, 3))
		@test isselfoperator(merge_operator_device) == true
		@test isexternaloperator(merge_operator_device) == false
		generic_tests(merge_operator_device)
	end
end
