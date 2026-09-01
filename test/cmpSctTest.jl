# Composite scattering tests
# The susceptibility is diagonal, so it commutes with the √ΔV normalization and
# the inverse scattering operator in the normalized basis is x - X .* (G̃ * x),
# with X entry by entry on the flat degree of freedom layout.
using Test, GilaElectromagnetics, LinearAlgebra, CUDA

import GilaElectromagnetics.GilaOperators: setSus!

const sctScl16 = (1//16, 1//16, 1//16)
const sctScl32 = (1//32, 1//32, 1//32)
const sctOrg0 = (0//1, 0//1, 0//1)

sctRelFro(matA, matB) = norm(matA - matB) / norm(matB)
sctChi(pos) = (0.5 + 0.1im) * (1 + 0.4 * sin(2pi * pos[1]) * cos(2pi * pos[2]))
sctCur(pos) = (exp(2im * pi * pos[3]), 0.5 * exp(2im * pi * pos[1]), 0)

# The flat per degree of freedom susceptibility a sampled function should give
function sctSusRef(cvol::GlaCmpVol, f)
    susCel = [ComplexF64(f(pos)) for (pos, _, _) in coordinates(cvol)]
    susDof, celOff = ComplexF64[], 0
    for reg in regions(cvol)
        celNum = prod(reg.cel)
        blk = susCel[(celOff + 1):(celOff + celNum)]
        for _ in 1:3
            append!(susDof, blk)
        end
        celOff += celNum
    end
    return susDof
end

# Mean of every 2×2×2 block of a (cel..., 3) density array
function sctAgr(arr::Array{ComplexF64,4})
    cel = size(arr)[1:3] .÷ 2
    blk = reshape(arr, 2, cel[1], 2, cel[2], 2, cel[3], 3)
    return reshape(sum(blk; dims=(1, 3, 5)), cel..., 3) ./ 8
end

#= The carved half-box of the composite operator tests, shrunk in y and z so the
dense checks stay small: a fine (4,4,4) region face to face with a coarse
(2,2,2) one, 216 degrees of freedom in all. Building a composite operator is the
expensive part of these tests, so one vacuum operator serves them all. =#
const sctVol = GlaVol((4, 2, 2), sctScl16, sctOrg0)
const sctCvl = refine(GlaCmpVol(sctVol), ((-1//16, 0//1, 0//1), (1//8, 1//8, 1//8)))
const sctVac = GlaCmpOprVac(sctCvl)
const sctInv = InvSctOpr(sctVac, sctChi)
const sctMat = dnsMat(sctInv)
const sctOne = GlaCmpVol(GlaVol((2, 2, 2), sctScl16, sctOrg0))

@testset "Composite scattering traits" begin
    @test nregions(sctCvl) == 2
    @test regions(sctCvl)[1].cel == (4, 4, 4)
    @test regions(sctCvl)[1].scl == sctScl32
    @test regions(sctCvl)[2].cel == (2, 2, 2)
    @test size(sctInv) == (216, 216)
    @test size(sctInv, 1) == 216 && size(sctInv, 2) == 216
    @test eltype(sctInv) == ComplexF64
    @test length(sctInv.sus) == 216
    @test sctInv.sus isa Vector{ComplexF64}
    @test isselfoperator(sctInv)
    @test !isexternaloperator(sctInv)
    @test !isadjoint(sctInv)
    @test !isgpu(sctInv)
    @test !isoverlappingoperator(sctInv)
    @test glaSze(sctInv) == glaSze(sctVac)
    @test slv(sctInv) isa GlaSlv
    sct = SctOpr(sctVac, sctChi)
    gla = GlaOpr(sctVac, sctChi)
    @test size(sct) == (216, 216) && size(gla) == (216, 216)
    @test isselfoperator(sct) && isselfoperator(gla)
    @test slv(sct) isa BiCGStabSolver
    @test occursin("composite", sprint(show, sctInv))
    @test occursin("composite", sprint(show, gla))
    # The composite volume takes the place of the volume in every constructor
    @test InvSctOpr(sctOne, 0.5 + 0.1im) isa InvSctOpr
    @test SctOpr(sctOne, 0.5 + 0.1im; slv=GMRESSolver()) isa SctOpr
    @test GreenOperator(sctOne, 0.5 + 0.1im) isa GlaOpr
    @test InverseScatteringOperator(sctVac, 0.5 + 0.1im) isa InvSctOpr
    @test ScatteringOperator(sctVac, 0.5 + 0.1im) isa SctOpr
    # A two body vacuum operator is not a scattering geometry
    extOpr = GlaCmpOprVac(sctOne, GlaCmpVol(GlaVol((2, 2, 2), sctScl16, (1//2, 0//1, 0//1))))
    @test_throws ArgumentError InvSctOpr(extOpr, 0.5 + 0.1im)
end

@testset "Composite scattering dense identity" begin
    vacMat = dnsMat(sctVac)
    ref = Matrix{ComplexF64}(I, 216, 216) - Diagonal(sctInv.sus) * vacMat
    @test sctRelFro(sctMat, ref) < 1e-13
    # The susceptibility is one value per cell, repeated over the components
    susCel = [ComplexF64(sctChi(pos)) for (pos, _, _) in coordinates(sctCvl)]
    @test sctInv.sus[1:64] ≈ susCel[1:64]
    @test sctInv.sus[65:128] ≈ susCel[1:64]
end

@testset "Composite scattering adjoint" begin
    adjInv = adjoint(sctInv)
    @test isadjoint(adjInv)
    @test !isadjoint(sctInv)
    @test sctRelFro(dnsMat(adjInv), sctMat') < 1e-13
    @test dnsMat(sctInv) == sctMat
    # The full operator satisfies the same identity on random vectors
    gla = GlaOpr(sctVac, sctChi)
    adjGla = adjoint(gla)
    vecX = randn(ComplexF64, 216)
    vecY = randn(ComplexF64, 216)
    lhs = dot(vecY, gla * vecX)
    rhs = dot(adjGla * vecY, vecX)
    @test abs(lhs - rhs) < 1e-6 * abs(lhs)
end

@testset "Composite scattering solver" begin
    sct = SctOpr(sctVac, sctChi)
    invSct = sct.invSctOpr
    vecX = randn(ComplexF64, 216)
    @test norm(invSct * (sct * vecX) - vecX) < 1e-7 * norm(vecX)
    # GMRES reaches the same solution
    sctGmr = SctOpr(sctVac, sctChi; slv=GMRESSolver())
    @test norm(invSct * (sctGmr * vecX) - vecX) < 1e-7 * norm(vecX)
    @test norm(sctGmr * vecX - sct * vecX) < 1e-6 * norm(sct * vecX)
    # A field goes in and a field on the same tiling comes out
    fld = discretize!(zerofield(sctCvl), sctCur)
    out = sct * fld
    @test out isa GlaFld
    @test out.cvol === sctCvl
    @test length(out) == 216
    @test norm((invSct * out).dat - fld.dat) < 1e-7 * norm(fld.dat)
    @test norm(out.dat - sct * collect(fld.dat)) < 1e-7 * norm(out.dat)
    invOut = invSct * fld
    @test invOut isa GlaFld && invOut.cvol === sctCvl
    @test invOut.dat ≈ invSct * collect(fld.dat)
    # The full operator on a field is the vacuum operator after the solve
    gla = GlaOpr(sctVac, sctChi)
    glaOut = gla * fld
    @test glaOut isa GlaFld
    @test norm(glaOut.dat - (sctVac * out).dat) < 1e-7 * norm(glaOut.dat)
    # A field on another tiling does not fit
    @test_throws ArgumentError sct * zerofield(sctOne)
    @test_throws ArgumentError invSct * zeros(ComplexF64, 215)
end

@testset "Composite scattering susceptibility forms" begin
    susFun = sctSusRef(sctCvl, sctChi)
    # A number fills every cell
    @test InvSctOpr(sctVac, 0.5 + 0.1im).sus == fill(ComplexF64(0.5, 0.1), 216)
    # A function of position is sampled at the cell centers
    @test InvSctOpr(sctVac, sctChi).sus ≈ susFun
    # One value per cell, and one value per degree of freedom
    susCel = [ComplexF64(sctChi(pos)) for (pos, _, _) in coordinates(sctCvl)]
    @test length(susCel) == 72
    @test InvSctOpr(sctVac, susCel).sus ≈ susFun
    @test InvSctOpr(sctVac, susFun).sus == susFun
    # One tensor per region
    tenSus = [reshape(susCel[1:64], (4, 4, 4)), reshape(susCel[65:72], (2, 2, 2))]
    @test InvSctOpr(sctVac, tenSus).sus ≈ susFun
    # A bare tensor only fits a tiling of one region
    @test InvSctOpr(sctOne, fill(ComplexF64(0.3), 2, 2, 2)).sus == fill(ComplexF64(0.3), 24)
    @test_throws ArgumentError InvSctOpr(sctVac, fill(ComplexF64(0.3), 4, 4, 4))
    # Sizes that do not fit the tiling
    @test_throws ArgumentError InvSctOpr(sctVac, [fill(ComplexF64(0.3), 4, 4, 4), fill(ComplexF64(0.3), 3, 3, 3)])
    @test_throws ArgumentError InvSctOpr(sctVac, [fill(ComplexF64(0.3), 4, 4, 4)])
    @test_throws ArgumentError InvSctOpr(sctVac, randn(ComplexF64, 100))
    @test_throws ArgumentError InvSctOpr(sctVac, randn(ComplexF64, 1536))
end

@testset "Composite scattering setSus!" begin
    opr = InvSctOpr(sctVac, 0.2 + 0.0im)
    @test opr.sus == fill(ComplexF64(0.2), 216)
    susFun = sctSusRef(sctCvl, sctChi)
    setSus!(opr, sctChi)
    @test opr.sus ≈ susFun
    setSus!(opr, 0.2 + 0.0im)
    @test opr.sus == fill(ComplexF64(0.2), 216)
    setSus!(opr, susFun)
    @test opr.sus == susFun
    @test_throws ArgumentError setSus!(opr, randn(ComplexF64, 100))
    # The setter reaches through the wrappers
    sct = SctOpr(sctVac, 0.2 + 0.0im)
    setSus!(sct, sctChi)
    @test sct.invSctOpr.sus ≈ susFun
    gla = GlaOpr(sctVac, 0.2 + 0.0im)
    setSus!(gla, sctChi)
    @test gla.sctOpr.invSctOpr.sus ≈ susFun
end

@testset "Composite scattering on a uniform mesh" begin
    #= A tiling of one region is the plain volume, and the normalization is a
    global scalar that commutes with everything, so the two agree exactly. =#
    vol = GlaVol((4, 4, 4), sctScl16, sctOrg0)
    chi = 0.5 + 0.1im
    susTen = fill(ComplexF64(chi), 4, 4, 4)
    cmpVac = GlaCmpOprVac(GlaCmpVol(vol))
    plnVac = GlaOprVac(vol)
    cmpInv = InvSctOpr(cmpVac, chi)
    plnInv = InvSctOpr(plnVac, susTen)
    @test dnsMat(cmpInv) == dnsMat(plnInv)
    vecX = randn(ComplexF64, 192)
    @test SctOpr(cmpVac, chi) * vecX ≈ SctOpr(plnVac, susTen) * vecX
    @test GlaOpr(cmpVac, chi) * vecX ≈ GlaOpr(plnVac, susTen) * vecX
    # A plain operator also takes a field
    fld = discretize!(zerofield(vol), sctCur)
    @test (plnInv * fld).dat ≈ cmpInv * collect(fld.dat)
end

@testset "Composite scattering mesh consistency" begin
    #= The same physical problem on a uniform fine mesh and on a mesh refined
    over half the domain. The two answers agree to the discretization error of
    the coarse half, which is what the refinement is there to control. =#
    finVol = GlaVol((8, 4, 4), sctScl32, sctOrg0)
    finFld = discretize!(zerofield(finVol), sctCur)
    cmpFld = discretize!(zerofield(sctCvl), sctCur)
    finOut = SctOpr(GlaCmpVol(finVol), sctChi) * finFld
    cmpOut = SctOpr(sctVac, sctChi) * cmpFld
    finArr = regrid(finOut, sctScl32)
    cmpArr = regrid(cmpOut, sctScl32)
    @test size(finArr) == (8, 4, 4, 3) && size(cmpArr) == (8, 4, 4, 3)
    # The refined half sits on the same cells in both meshes
    finErr = sctRelFro(cmpArr[1:4, :, :, :], finArr[1:4, :, :, :])
    # The coarse half is compared through the exact aggregation map
    finAgr, cmpAgr = sctAgr(finArr), sctAgr(cmpArr)
    crsErr = sctRelFro(cmpAgr[3:4, :, :, :], finAgr[3:4, :, :, :])
    @info "Composite scattering mesh consistency" finErr crsErr
    @test finErr < 5e-2
    @test crsErr < 5e-2
    # Both meshes carry the same physical current, so the L² norms agree as well
    @test abs(norm(cmpFld) - norm(finFld)) < 1e-2 * norm(finFld)
end

@testset "Composite scattering GPU" begin
    if CUDA.functional()
        oprGpu = InvSctOpr(sctCvl, sctChi; useGpu=true)
        @test isgpu(oprGpu)
        @test oprGpu.sus isa CuVector{ComplexF64}
        fldGpu = discretize!(zerofield(sctCvl; useGpu=true), sctCur)
        outGpu = oprGpu * fldGpu
        @test outGpu isa GlaFld
        @test parent(outGpu) isa CuVector{ComplexF64}
        fldCpu = discretize!(zerofield(sctCvl), sctCur)
        outCpu = sctInv * fldCpu
        @test norm(Array(outGpu.dat) - outCpu.dat) < 1e-10 * norm(outCpu.dat)
        sctGpu = SctOpr(oprGpu.oprVac, sctChi)
        @test norm((oprGpu * (sctGpu * fldGpu)).dat - fldGpu.dat) < 1e-6 * norm(fldGpu.dat)
    end
end
