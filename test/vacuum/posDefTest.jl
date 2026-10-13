#= Asym(G0) is positive semi-definite in vacuum, exactly at real frequency and
strictly at complex frequency, so its worst eigenvalue measures the whole build.
Normalized by lamMax, the bar is a multiple of eps rather than an absolute
number that hides behind a small operator. =#
using Test, GilaElectromagnetics, LinearAlgebra

function pdAsyEig(cel, scl, frq)
    opt = GlaKerOpt{Float64}()
    opt.frqPhz = frq
    mat = dnsMat(GlaVacOprMem(opt, GlaVol(cel, scl, stdOrg)))
    return eigvals(Hermitian((mat - adjoint(mat)) / 2im))
end

@testset "Positive Semi-Definiteness Tests" begin
    sclSlv = (1//32, 1//32, 1//512)

    @testset "cubic cells" begin
        # Two shapes run the same routes as any other
        for cel ∈ ((2,2,2), (8,2,10))
            ev = pdAsyEig(cel, stdScl, 1.0 + 0.0im)
            rat = minimum(ev) / maximum(ev)
            @test rat > -1e-12
            @test minimum(pdAsyEig(cel, stdScl, 1.0 + 0.1im)) > 0
        end
    end

    #= The 1:1:16 cell, the shape both halves of the old build were wrong on,
    in opposite directions. =#
    @testset "slender cells" begin
        ev = pdAsyEig((6, 6, 12), sclSlv, 1.0 + 0.0im)
        rat = minimum(ev) / maximum(ev)
        @test rat > -1e-12
        lam = minimum(pdAsyEig((6, 6, 12), sclSlv, 1.0 + 0.1im))
        @test lam > 0
    end

    #= pdAsyEig subtracts the operator from its own adjoint, at a cost of
    eps * ‖Sym‖ / ‖Asym‖, which is why the bars above are loose. The operator
    takes its part before the transform and holds machine precision however
    fine the mesh. =#
    @testset "fine mesh" begin
        for den ∈ (512, 2048, 8192)
            mat = dnsMat(AsyGlaOprVac{Float64}(GlaVol((4, 4, 4), (1//den, 1//den, 1//den), stdOrg)))
            ev = eigvals(Hermitian((mat + mat') / 2))
            rat = minimum(ev) / maximum(ev)
            @test rat > -1e-14
        end
    end
end
