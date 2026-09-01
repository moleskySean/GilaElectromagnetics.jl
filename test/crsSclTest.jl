# Cross-scale vacuum operator tests
# References are same-scale operators on a uniformly refined copy of the coarse
# volume, aggregated with the maps of the pulse basis: a coarse target row is the
# mean of the eight fine target rows it covers, and a coarse source column
# injects unit current density into those same eight cells.
import GilaElectromagnetics.GilaOperators: mskRng
import GilaElectromagnetics.GilaVolumes: uniVol

const _xsSclCrs = (1//16, 1//16, 1//16)
const _xsSclFin = (1//32, 1//32, 1//32)
const _xsSep    = (1//2, 0//1, 0//1)

# Coarse (2,2,2) cube at the origin, the same cube on the fine mesh, and a fine
# (4,4,4) cube half a wavelength away. Per-partition cells sum to (4,4,4).
const _xsVolCrs = GlaVol((2,2,2), _xsSclCrs, stdOrg)
const _xsVolRef = GlaVol((4,4,4), _xsSclFin, stdOrg)
const _xsVolFin = GlaVol((4,4,4), _xsSclFin, _xsSep)

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

relFro(a, b) = norm(a - b) / norm(b)

# Each operator costs a numerical integration, so the four dense forms are built
# once here rather than inside the testsets that use them
const _xsAgr = _xsAgrMap((2,2,2))
const _xsInj = collect(transpose(_xsAgr))
const _xsOprCrsTrg = GlaOprVac(_xsVolCrs, _xsVolFin)
const _xsOprFinTrg = GlaOprVac(_xsVolFin, _xsVolCrs)
const _xsMatCrsTrg = dnsMat(_xsOprCrsTrg)
const _xsMatFinTrg = dnsMat(_xsOprFinTrg)
const _xsRefCrsTrg = dnsMat(GlaOprVac(_xsVolRef, _xsVolFin))
const _xsRefFinTrg = dnsMat(GlaOprVac(_xsVolFin, _xsVolRef))

@testset "Cross-scale separated, coarse target" begin
    # srcDiv > 1 partitions the input, so this orientation runs genPrt!
    @test isexternaloperator(_xsOprCrsTrg)
    @test prod(_xsOprCrsTrg.mem.mixInf.srcDiv) == 8
    @test prod(_xsOprCrsTrg.mem.mixInf.trgDiv) == 1
    @test size(_xsOprCrsTrg) == (24, 192)

    # coarse target row is the mean of the eight fine target rows
    @test relFro(_xsMatCrsTrg, (_xsAgr ./ 8) * _xsRefCrsTrg) < 1e-6
    # the sum convention is wrong by a factor of eight, so the test has teeth
    @test relFro(_xsMatCrsTrg, _xsAgr * _xsRefCrsTrg) > 1e-3
end

@testset "Cross-scale separated, fine target" begin
    # trgDiv > 1 partitions the output, so this orientation runs mrgPrt!
    @test isexternaloperator(_xsOprFinTrg)
    @test prod(_xsOprFinTrg.mem.mixInf.trgDiv) == 8
    @test prod(_xsOprFinTrg.mem.mixInf.srcDiv) == 1
    @test size(_xsOprFinTrg) == (192, 24)

    # coarse source column injects unit density into the eight fine cells
    @test relFro(_xsMatFinTrg, _xsRefFinTrg * _xsInj) < 1e-6
    @test relFro(_xsMatFinTrg, _xsRefFinTrg * (_xsInj ./ 8)) > 1e-3
end

@testset "Cross-scale reciprocity" begin
    # diag(ΔV_trg) * G is complex-symmetric, so weighting each orientation by its
    # own target cell volume makes the two transposes of each other
    celVolCrs = Float64(prod(_xsSclCrs))
    celVolFin = Float64(prod(_xsSclFin))
    @test relFro(celVolCrs .* _xsMatCrsTrg, celVolFin .* transpose(_xsMatFinTrg)) < 1e-6
    # the same-scale references obey the plain transpose relation
    @test relFro(_xsRefCrsTrg, transpose(_xsRefFinTrg)) < 1e-6
end

@testset "Cross-scale adjoint" begin
    adjMat = dnsMat(adjoint(_xsOprCrsTrg))
    @test size(adjMat) == reverse(size(_xsMatCrsTrg))
    @test relFro(adjMat, _xsMatCrsTrg') < 1e-10
    # the original operator is untouched by adjoint
    @test !isadjoint(_xsOprCrsTrg)
end

@testset "Same-scale touching goes external" begin
    volSrc = GlaVol((4,4,4), _xsSclFin, stdOrg)
    volTrg = GlaVol((4,4,4), _xsSclFin, (4//32, 0//1, 0//1))
    opr = GlaOprVac(volTrg, volSrc)
    # face contact is not overlap: the external path has contact corrections
    @test isexternaloperator(opr)
    @test !isoverlappingoperator(opr)
    @test all(==(0:0), opr.srcMsk)
    @test all(==(0:0), opr.trgMsk)
    mat = dnsMat(opr)

    # reference: a self operator on the union volume, masked by hand, which is
    # the route GlaOprVac took for touching volumes before the strict check
    volUni = uniVol(volTrg, volSrc)
    @test volUni.cel == (8, 4, 4)
    oprUni = GlaOprVac(volUni)
    innMsk = mskRng(volSrc, volUni)
    outMsk = mskRng(volTrg, volUni)
    matUni = zeros(ComplexF64, 192, 192)
    for colItr in 1:192
        srcVec = zeros(ComplexF64, volSrc.cel..., 3)
        srcVec[colItr] = one(ComplexF64)
        embVec = zeros(ComplexF64, volUni.cel..., 3)
        embVec[innMsk..., :] .= srcVec
        matUni[:, colItr] .= vec((oprUni * embVec)[outMsk..., :])
    end
    @test relFro(mat, matUni) < 1e-10
end

@testset "Cross-scale parity trap throws" begin
    # (2,2,2) coarse against (2,2,2) fine gives one-cell source partitions, so
    # the per-partition cells sum to 3 and the branching algorithm would return
    # finite but wrong values
    volBad = GlaVol((2,2,2), _xsSclFin, _xsSep)
    @test_throws ArgumentError GlaOprVac(_xsVolCrs, volBad)
    @test_throws ArgumentError GlaVacOprMem(CPUKerOpt(), _xsVolCrs, volBad)
    # doubling the fine cell count in every direction fixes the parity
    @test GilaElectromagnetics.GilaVolumes.genEveExtInf(_xsVolCrs, _xsVolFin) isa
        GilaElectromagnetics.GilaVolumes.GlaExtInf
end

@testset "Cross-scale touching" begin
    #= Partitioned sub-lattices in contact go through the contact quadrature, at
    the accuracy of the cross-scale path rather than the machine precision of
    the same-scale one, which is why the composite layer prefers the sandwich. =#
    volTch = GlaVol((4,4,4), _xsSclFin, (4//32, 0//1, 0//1))
    opr = GlaOprVac(_xsVolCrs, volTch)
    @test isexternaloperator(opr)
    matTch = dnsMat(opr)
    @test all(isfinite, matTch)
    # the coarse volume remeshed at the fine scale gives the exact answer
    @test relFro(matTch, (_xsAgr ./ 8) * dnsMat(GlaOprVac(_xsVolRef, volTch))) < 1e-4
end

@testset "Cross-scale anisotropic reciprocity" begin
    #= The srfScl face pair table is consumed keyed as (target face, source
    face). Written transposed it scales tensor component (a, b) by
    (sclS[b]/sclT[b]) / (sclS[a]/sclT[a]), which every isotropic and every
    same-scale pair hides because their scale ratio is direction independent.
    Volume-weighted reciprocity catches it at order one. =#
    volAni = GlaVol((4,4,4), (1//64, 1//32, 1//32), stdOrg)
    volIso = GlaVol((4,4,4), _xsSclFin, (5//32, 0//1, 0//1))
    matAI = dnsMat(GlaOprVac(volAni, volIso; prxWrn=false))
    matIA = dnsMat(GlaOprVac(volIso, volAni; prxWrn=false))
    volA, volI = prod(volAni.scl), prod(volIso.scl)
    @test relFro(volA .* matAI, transpose(volI .* matIA)) < 1e-5
end
