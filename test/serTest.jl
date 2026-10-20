# Round trips through Serialization and JLD2
#= Only GlaVacOprMem and SusOpr have hooks; every wrapper is plain data. Each case
goes through both formats, and serChk asks for the same bytes back and a working
matvec, which needs the FFTW plans rebuilt on load. The Bloch lattice is checked
in blochTest.jl and device memory in gpuTest.jl. =#
using JLD2

function jldRnd(obj)
    fil = tempname() * ".jld2"
    try
        jldsave(fil; obj)
        return load(fil, "obj")
    finally
        rm(fil; force=true)
    end
end

# A coarse region face to face with one refined in x, so both cross-scale blocks are sandwiches
const serCvl = refine(GlaCmpVol(GlaVol((4, 2, 2), scl16, stdOrg)),
    ((-1//16, 0//1, 0//1), (1//8, 1//8, 1//8)); factor=(2, 1, 1))
const serCmp = GlaCmpOprVac{Float64}(serCvl)
#= Every stored option at once: Float32 storage, quasistatic, complex frequency and
adjoint, on two volumes sharing interior, so the masks are the only record of them. =#
const serMem = adjoint!(GlaOprVac{Float32}(GlaVol((2, 2, 2), scl16, stdOrg),
    GlaVol((2, 2, 2), scl16, (1//16, 1//16, 1//16)); qssApx=true, frqPhz=1.0+0.1im))
const serTen = SusOpr{Float64}(_vol2s, randn(ComplexF64, 3, 3))

@testset "Round trips, $(nameof(rnd))" for rnd in (serRnd, jldRnd)
    @test isoverlappingoperator(serMem)
    for opr in (serMem, SusOpr{Float64}(_vol2s, _sus2s), serTen, serCmp,
        GlaOpr(SctOpr(InvSctOpr(serCmp, 0.5 + 0.05im), BiCGStabSolver())))
        serChk(opr; rnd)
    end
    @test rnd(serCmp).srcCvl == serCvl
    # Nested in containers, the route a hook can miss
    innVec = rand(ComplexF64, size(_g0s(), 2))
    @test only(rnd((box = [_g0s()],)).box) * innVec == _g0s() * innVec
end

@testset "No FFTW plan pointers on disk" begin
    opr = GlaOpr(serCmp, 0.5 + 0.05im)
    fil = tempname() * ".jld2"
    try
        jldsave(fil; opr)
        @test isnothing(findfirst(codeunits("FFTW"), read(fil)))
    finally
        rm(fil; force=true)
    end
    @test isnothing(findfirst(codeunits("FFTW"), serByt(opr)))
end
