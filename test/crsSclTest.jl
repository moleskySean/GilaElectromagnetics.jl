# Cross-scale vacuum operator tests
# References are same-scale operators on a uniformly refined copy of the coarse
# volume, aggregated with the maps of the pulse basis: a coarse target row is the
# mean of the eight fine target rows it covers, and a coarse source column
# injects unit current density into those same eight cells.
import GilaElectromagnetics.GilaVolumes: uniVol

const _xsSep    = (1//2, 0//1, 0//1)

# Coarse (2,2,2) cube at the origin, the same cube on the fine mesh, and a fine
# (4,4,4) cube half a wavelength away. Per-partition cells sum to (4,4,4).
const _xsVolCrs = GlaVol((2,2,2), scl16, stdOrg)
const _xsVolRef = GlaVol((4,4,4), stdScl, stdOrg)
const _xsVolFin = GlaVol((4,4,4), stdScl, _xsSep)

#= sum-aggregation matrix from the fine cells of a factor-two refinement onto the
coarse cells, in the (cel..., 3) storage order of the operators =#
function _xsAgrMap(celCrs::NTuple{3,Integer})
    celFin = celCrs .* 2
    indCrs = LinearIndices((celCrs..., 3))
    indFin = LinearIndices((celFin..., 3))
    agrMap = zeros(prod(celCrs) * 3, prod(celFin) * 3)
    for dirItr in 1:3, celItr in CartesianIndices(celCrs)
        for offItr in CartesianIndices((0:1, 0:1, 0:1))
            finItr = 2 .* Tuple(celItr) .- 1 .+ Tuple(offItr)
            agrMap[indCrs[Tuple(celItr)..., dirItr], indFin[finItr..., dirItr]] = 1.0
        end
    end
    return agrMap
end

# Each operator costs a Green function build, so the four dense forms are built
# once here rather than inside the testsets that use them
const _xsAgr = _xsAgrMap((2,2,2))
const _xsInj = collect(transpose(_xsAgr))
const _xsOprCrsTrg = GlaOprVac{Float64}(_xsVolCrs, _xsVolFin)
const _xsOprFinTrg = GlaOprVac{Float64}(_xsVolFin, _xsVolCrs)
const _xsMatCrsTrg = dnsMat(_xsOprCrsTrg)
const _xsMatFinTrg = dnsMat(_xsOprFinTrg)
const _xsRefCrsTrg = dnsMat(GlaOprVac{Float64}(_xsVolRef, _xsVolFin))
const _xsRefFinTrg = dnsMat(GlaOprVac{Float64}(_xsVolFin, _xsVolRef))

@testset "Cross-scale separated, coarse target" begin
    # The finer source is partitioned, so this orientation runs genPrt!
    @test isexternaloperator(_xsOprCrsTrg)

    # coarse target row is the mean of the eight fine target rows
    @test frbErr(_xsMatCrsTrg, (_xsAgr ./ 8) * _xsRefCrsTrg) < 1e-12
    # the sum convention is wrong by a factor of eight, so the test has teeth
    @test frbErr(_xsMatCrsTrg, _xsAgr * _xsRefCrsTrg) > 1e-3
end

@testset "Cross-scale separated, fine target" begin
    # The finer target is partitioned, so this orientation runs mrgPrt!
    @test isexternaloperator(_xsOprFinTrg)

    # coarse source column injects unit density into the eight fine cells
    @test frbErr(_xsMatFinTrg, _xsRefFinTrg * _xsInj) < 1e-12
    @test frbErr(_xsMatFinTrg, _xsRefFinTrg * (_xsInj ./ 8)) > 1e-3
end

@testset "Cross-scale reciprocity" begin
    # diag(ΔV_trg) * G is complex-symmetric, so weighting each orientation by its
    # own target cell volume makes the two transposes of each other
    celVolCrs = Float64(prod(scl16))
    celVolFin = Float64(prod(stdScl))
    @test frbErr(celVolCrs .* _xsMatCrsTrg, celVolFin .* transpose(_xsMatFinTrg)) < 1e-12
    # the same-scale references obey the plain transpose relation
    @test frbErr(_xsRefCrsTrg, transpose(_xsRefFinTrg)) < 1e-12
end

@testset "Same-scale touching goes external" begin
    volSrc = GlaVol((4,4,4), stdScl, stdOrg)
    volTrg = GlaVol((4,4,4), stdScl, (4//32, 0//1, 0//1))
    opr = GlaOprVac{Float64}(volTrg, volSrc)
    # face contact is not overlap: the external path has contact corrections
    @test isexternaloperator(opr)
    @test !isoverlappingoperator(opr)
    # the masked union is the route GlaOprVac took for touching volumes before
    # the strict check
    @test uniVol(volTrg, volSrc).cel == (8, 4, 4)
    @test frbErr(dnsMat(opr), uniMskMat(volTrg, volSrc)) < 1e-13
end

@testset "Cross-scale parity trap throws" begin
    # (2,2,2) coarse against (2,2,2) fine gives one-cell source partitions, so
    # the per-partition cells sum to 3 and the branching algorithm would return
    # finite but wrong values
    volBad = GlaVol((2,2,2), stdScl, _xsSep)
    @test_throws ArgumentError GlaOprVac{Float64}(_xsVolCrs, volBad)
    @test_throws ArgumentError GlaVacOprMem(GlaKerOpt{Float64}(), _xsVolCrs, volBad)
end

@testset "Cross-scale touching" begin
    #= Partitioned sub-lattices in contact average the self block of the gcd cell,
    so the coarse volume remeshed at the fine scale is exact, not approximate. =#
    volTch = GlaVol((4,4,4), stdScl, (4//32, 0//1, 0//1))
    opr = GlaOprVac{Float64}(_xsVolCrs, volTch)
    @test isexternaloperator(opr)
    matTch = dnsMat(opr)
    @test all(isfinite, matTch)
    # the coarse volume remeshed at the fine scale gives the exact answer
    @test frbErr(matTch, (_xsAgr ./ 8) * dnsMat(GlaOprVac{Float64}(_xsVolRef, volTch))) < 1e-12
end

@testset "Cross-scale anisotropic reciprocity" begin
    #= A tensor component (a, b) scaled by (sclS[b]/sclT[b]) / (sclS[a]/sclT[a])
    is hidden by every isotropic and every same-scale pair, whose scale ratio is
    direction independent. Volume-weighted reciprocity catches it at order one. =#
    volAni = GlaVol((4,4,4), (1//64, 1//32, 1//32), stdOrg)
    volIso = GlaVol((4,4,4), stdScl, (5//32, 0//1, 0//1))
    matAI = dnsMat(GlaOprVac{Float64}(volAni, volIso))
    matIA = dnsMat(GlaOprVac{Float64}(volIso, volAni))
    volA, volI = prod(volAni.scl), prod(volIso.scl)
    @test frbErr(volA .* matAI, transpose(volI .* matIA)) < 1e-12
end
