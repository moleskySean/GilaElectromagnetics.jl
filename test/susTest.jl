# Susceptibility operator tests, isotropic and anisotropic
import GilaElectromagnetics: setSus!
# StaticArrays is a dependency of the package, not of the test environment
const SusSMt = GilaElectromagnetics.SMatrix{3, 3, ComplexF64}

const susVol = mkVol((2,2,2))
const susCvl = GlaCmpVol(susVol)
const susArr = reshape(collect(1:8) .* (0.1 + 0.02im) .+ 0.3, 2, 2, 2)
const susDof = repeat(vec(susArr), 3)
# A gyrotropic medium: complex, not symmetric, and coupling the three components
const susGyr = SusSMt(reshape(ComplexF64[0.6 + 0.1im, -0.3im, 0.02,
    0.3im, 0.6 + 0.1im, 0.0, 0.05, 0.0, 0.4 + 0.05im], 3, 3))

susDia(arr) = (ten = zeros(ComplexF64, size(arr)..., 3, 3);
    for dir in 1:3; ten[:, :, :, dir, dir] .= arr; end; ten)
susVec() = randn(ComplexF64, 24)

@testset "Susceptibility shapes" begin
    isoOpr = SusOpr{Float64}(susVol, susArr)
    @test isoOpr isa SusOpr{Float64, Vector{ComplexF64}}
    @test isoOpr.sus == susDof
    @test isselfoperator(isoOpr) && !isexternaloperator(isoOpr)
    @test !isoverlappingoperator(isoOpr)
    @test SusOpr(susVol, 0.5) isa SusOpr{dflPrc}
    # Every accepted input, against the same reference
    @test SusOpr{Float64}(susVol, 0.5 + 0.05im).sus == fill(ComplexF64(0.5, 0.05), 24)
    @test SusOpr{Float64}(susVol, vec(susArr)).sus == susDof
    @test SusOpr{Float64}(susVol, susDof).sus == susDof
    @test SusOpr{Float64}(susCvl, [susArr]).sus == susDof
    @test SusOpr{Float64}(susVol, pos -> 0.5 + 0.05im).sus == fill(ComplexF64(0.5, 0.05), 24)
    # The tensor shapes, told apart by the storage they ask for
    @test SusOpr{Float64}(susVol, susDia(susArr)).sus == reshape(susDia(susArr), 8, 3, 3)
    @test SusOpr{Float64}(susVol, Matrix(susGyr)) isa SusOpr{Float64, Array{ComplexF64, 3}}
    @test SusOpr{Float64}(susVol, susGyr).sus == SusOpr{Float64}(susVol, Matrix(susGyr)).sus
    @test SusOpr{Float64}(susVol, pos -> susGyr).sus == SusOpr{Float64}(susVol, susGyr).sus
    @test SusOpr{Float64}(susCvl, [susDia(susArr)]).sus == reshape(susDia(susArr), 8, 3, 3)
    # Shapes that fit no region of the tiling
    @test_throws ArgumentError SusOpr{Float64}(susVol, rand(ComplexF64, 5))
    @test_throws ArgumentError SusOpr{Float64}(susVol, rand(ComplexF64, 4, 4, 4))
    @test_throws ArgumentError SusOpr{Float64}(susVol, rand(ComplexF64, 2, 2))
    @test_throws ArgumentError SusOpr{Float64}(susVol, rand(ComplexF64, 2, 2, 2, 3))
    @test_throws ArgumentError SusOpr{Float64}(susVol, pos -> rand(ComplexF64, 2, 2))
    # Precision conversion, as every other operator has
    @test SusOpr{Float32}(isoOpr).sus == ComplexF32.(susDof)
    @test SusOpr{Float64}(isoOpr) === isoOpr
end

@testset "Susceptibility apply" begin
    isoOpr = SusOpr{Float64}(susVol, susArr)
    diaOpr = SusOpr{Float64}(susVol, susDia(susArr))
    gyrOpr = SusOpr{Float64}(susVol, susGyr)
    vec1 = susVec()
    @test isoOpr * vec1 == susDof .* vec1
    # A diagonal tensor is the isotropic answer to the last bit
    @test diaOpr * vec1 ≈ isoOpr * vec1
    @test frbErr(diaOpr * vec1, isoOpr * vec1) < 1e-15
    # Both tensor shapes take a (cel..., 3) tensor and a matrix of columns
    @test vec(isoOpr * reshape(vec1, 2, 2, 2, 3)) == isoOpr * vec1
    @test vec(gyrOpr * reshape(vec1, 2, 2, 2, 3)) == gyrOpr * vec1
    @test (gyrOpr * hcat(vec1, vec1))[:, 2] == gyrOpr * vec1
    # The dense form is the block diagonal of the cell tensors
    gyrMat = dnsMat(gyrOpr)
    @test all(gyrMat[(row - 1) * 8 + cel, (col - 1) * 8 + cel] == susGyr[row, col]
        for row in 1:3, col in 1:3, cel in 1:8)
    @test count(!iszero, gyrMat) == 8 * count(!iszero, susGyr)
    # Mixed precision is an error rather than a conversion
    @test_throws ArgumentError isoOpr * randn(ComplexF32, 24)
    @test_throws ArgumentError isoOpr * randn(ComplexF64, 12)
end

@testset "Susceptibility inverse" begin
    isoOpr = SusOpr{Float64}(susVol, susArr)
    gyrOpr = SusOpr{Float64}(susVol, susGyr)
    vec1 = susVec()
    @test isoOpr \ (isoOpr * vec1) ≈ vec1
    @test gyrOpr \ (gyrOpr * vec1) ≈ vec1
    @test frbErr(gyrOpr \ vec1, inv(dnsMat(gyrOpr)) * vec1) < 1e-13
    @test ldiv!(similar(vec1), isoOpr, isoOpr * vec1) ≈ vec1
    # Vacuum cells have no inverse, in either storage shape
    vacArr = copy(susArr); vacArr[1] = 0
    @test_throws ArgumentError SusOpr{Float64}(susVol, vacArr) \ vec1
    @test_throws ArgumentError SusOpr{Float64}(susVol, susDia(vacArr)) \ vec1
    @test_throws ArgumentError SusOpr{Float64}(susVol, zeros(ComplexF64, 3, 3)) \ vec1
end

@testset "Susceptibility fields" begin
    isoOpr = SusOpr{Float64}(susVol, susArr)
    gyrOpr = SusOpr{Float64}(susVol, susGyr)
    fld = discretize!(zerofield(Float64, susVol), tstDns)
    for opr in (isoOpr, gyrOpr)
        out = opr * fld
        @test out isa GlaFld && out.cvol == fld.cvol
        # χ is dimensionless and diagonal, so no normalization rides along
        @test out.dat == opr * collect(fld.dat)
        @test (opr \ out).dat ≈ collect(fld.dat)
    end
    # A field on another tiling does not fit
    @test_throws ArgumentError isoOpr * zerofield(Float64, mkVol((4,4,4)))
    @test_throws ArgumentError isoOpr \ zerofield(Float64, mkVol((4,4,4)))
end

@testset "Anisotropic scattering" begin
    oprVac = _g0s()
    isoInv = InvSctOpr(oprVac, _sus2s)
    #= A diagonal tensor is the isotropic operator entry for entry, which is the
    check that the tensor route agrees with the route it generalizes. =#
    diaInv = InvSctOpr(oprVac, susDia(_sus2s))
    @test sus(diaInv).sus isa Array{ComplexF64, 3}
    @test prbErr(diaInv, isoInv) < 1e-15
    vec1 = susVec()
    for (isoOpr, diaOpr) in ((isoInv, diaInv),
        (SctOpr(oprVac, _sus2s), SctOpr(oprVac, susDia(_sus2s))),
        (GlaOpr(oprVac, _sus2s), GlaOpr(oprVac, susDia(_sus2s))))
        @test frbErr(diaOpr * vec1, isoOpr * vec1) < 1e-7
    end
    # A gyrotropic medium, through the Lippmann-Schwinger residual
    gyrInv = InvSctOpr(oprVac, susGyr)
    gyrSct = SctOpr(oprVac, susGyr)
    fldInc = susVec()
    curInc = sus(gyrInv) * fldInc
    curTot = gyrSct * copy(curInc)
    @test norm(gyrInv * copy(curTot) - curInc) < 1e-6 * norm(curInc)
    # The setter swaps storage shapes as readily as values
    opr = InvSctOpr(oprVac, _sus2s)
    setSus!(opr, susGyr)
    @test sus(opr).sus == sus(gyrInv).sus
    setSus!(opr, _sus2s)
    @test sus(opr).sus == sus(isoInv).sus
end

@testset "Susceptibility show" begin
    @test shwChk(SusOpr{Float64}(susVol, susArr)) && shwChk(SusOpr{Float64}(susVol, susDia(susArr)))
end
