# Physics tests
# Requires: uniVol imported in volTest.jl (same scope), eigvals from LinearAlgebra

# PSD tolerance: assert min eigenvalue > -rtol * opnorm(Asym).
# Starting from 1e-9; raise to next power of ten if empirically needed.
const psdRtol = 1e-9

function checkPsd(mat, label)
    # mat should already be the matrix to check for PSD (callers extract Asym if needed)
    nrm = opnorm(mat)
    if nrm == 0
        @info "$label: zero matrix (trivially PSD)"
        @test true
        return
    end
    λs  = eigvals(Hermitian((mat + mat') / 2))
    wrs = minimum(λs)
    @info "$label: worst normalized eigenvalue = $(wrs / nrm)"
    @test wrs >= -psdRtol * nrm
end

# Analytic free-space dyadic Green function for a point dipole (see anaTest.jl)
function egoAna!(anaOut::AbstractVector{ComplexF64}, slfVol::GlaVol,
                 trgRng::Vector{<:StepRange}, dipPos::Vector{<:Rational},
                 dipVec::Vector{ComplexF64})
    sepTol = 1.0e-9
    linItr = zeros(Int, 3)
    egoCel = Array{ComplexF64}(undef, slfVol.cel..., 3)
    id = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
    for crtItr in CartesianIndices(tuple(length.(trgRng)...))
        for d in 1:3
            linItr[d] = LinearIndices(egoCel)[crtItr, d]
        end
        sep = 2π * sqrt((trgRng[1][crtItr[1]] - dipPos[1])^2 +
                        (trgRng[2][crtItr[2]] - dipPos[2])^2 +
                        (trgRng[3][crtItr[3]] - dipPos[3])^2)
        if sep < sepTol
            mul!(view(anaOut, linItr), (2.0 * π^2 * 2.0im / 3.0) .* id, dipVec)
        else
            sH = (2π) .* ((trgRng[1][crtItr[1]] - dipPos[1]),
                          (trgRng[2][crtItr[2]] - dipPos[2]),
                          (trgRng[3][crtItr[3]] - dipPos[3])) ./ sep
            sHs = [sH[i] * sH[j] for i in 1:3, j in 1:3]
            egoPar = (2π^2 * exp(im * sep) / sep) .*
                     (((1.0 + (im * sep - 1.0) / sep^2) .* id) .-
                      ((1.0 + 3.0 * (im * sep - 1.0) / sep^2) .* sHs))
            mul!(view(anaOut, linItr), egoPar, dipVec)
        end
    end
    return nothing
end

@testset "Asym(G₀) PSD — self GlaOprVac" begin
    # Use precomputed _selfMem4 — no new GlaVacOprMem construction
    checkPsd(asymMat(dnsMat(_g0())), "self GlaOprVac (4,4,4)")
end

@testset "Asym(G₀) PSD — AsyGlaOprVac itself" begin
    # AsyGlaOprVac(_g0()) deepcopies egoFur then takes Im part — cheap (no integration)
    mat = dnsMat(_asy())
    @test mat ≈ mat'
    checkPsd(mat, "AsyGlaOprVac (4,4,4)")
end

@testset "Asym(G₀) PSD — overlapping as union" begin
    # The union GlaOprVac uses volume masks, which break the PSD property of Asym(G₀).
    # The masked operator's asymmetric part has large negative eigenvalues (~0.4 normalized).
    # This is a known limitation: PSD only holds for un-masked self-operators.
    @test_broken begin
        ovrOrg = ntuple(i -> Rational(4) * stdScl[i] // 2, 3)
        volOvr = GlaVol((4,4,4), stdScl, ovrOrg)
        gOvr   = GlaOprVac(volOvr, _vol4)
        mat = asymMat(dnsMat(gOvr))
        nrm = opnorm(mat)
        minimum(eigvals(Hermitian((mat + mat') / 2))) >= -1e-9 * nrm
    end
end

@testset "Asym(G₀) PSD — MulRegGlaOprVac" begin
    op = MulRegGlaOprVac([_vol4, _trgV4], [_vol4, _trgV4])
    checkPsd(asymMat(dnsMat(op)), "MulRegGlaOprVac 2 regions")
end

@testset "Asym undefined for external (negative control)" begin
    # _gExt() reuses precomputed _extMem4 — no new integration
    gExt = _gExt()
    @test (try AsyGlaOprVac(gExt); false catch e; isa(e, ArgumentError) end)
end

@testset "G₀ = Sym + i·Asym" begin
    # All built from precomputed _selfMem4 — no new GlaVacOprMem constructions
    D = dnsMat(_g0())
    S = dnsMat(_sym())
    A = dnsMat(_asy())
    @test S + im .* A ≈ D   rtol=1e-12
end

@testset "Sym/Asym vs LinearMap decomposition" begin
    g = _g0()
    D = Matrix(LinearMap(g))
    @test relErr(dnsMat(_asy()), asymMat(D)) < 1e-14
    @test relErr(dnsMat(_sym()), symMat(D))  < 1e-14
end

@testset "Analytic dyadic Green function" begin
    lowTol = 1.0e-12
    # Use (6,6,6) — enough cells for Green function to converge; (4,4,4) gives winInt=2
    # which would require index 0 (out of bounds for dipLoc=[2,2,2]).
    for volDim in [(6,6,6)]
        vol    = mkVol(volDim)
        oprMem = GlaVacOprMem(CPUKerOpt(), vol)
        dipVec = zeros(ComplexF64, 3)
        relErrDir = zeros(Float64, 3)
        anaOut = Array{ComplexF64}(undef, 3 * prod(vol.cel))
        numOut = Array{ComplexF64}(undef, vol.cel..., 3)
        difMat = Array{Float64}(undef, vol.cel..., 3)
        innVec = Array{ComplexF64}(undef, vol.cel..., 3)

        winInt = min(Int(div(1//2 * 4 * minimum(vol.scl), minimum(vol.scl))),
                     minimum(vol.cel))

        for dipDir in 1:3
            dipLoc = [div(vol.cel[1], 2), div(vol.cel[2], 2), div(vol.cel[3], 2)]
            dipVec .= 0.0im
            dipVec[dipDir] = 1.0 + 0.0im
            dipPos = Rational.([vol.grd[1][dipLoc[1]], vol.grd[2][dipLoc[2]], vol.grd[3][dipLoc[3]]])
            trgRng = deepcopy(vol.grd)

            fill!(innVec, zero(ComplexF64))
            innVec[dipLoc[1], dipLoc[2], dipLoc[3], dipDir] = (1.0 + 0.0im) / prod(vol.scl)
            copyto!(numOut, egoOpr!(oprMem, innVec))
            egoAna!(anaOut, vol, trgRng, dipPos, dipVec)
            anaOut_r = reshape(anaOut, vol.cel..., 3)

            for crtItr in CartesianIndices((vol.cel..., 3))
                fldDif = abs(anaOut_r[crtItr] - numOut[crtItr])
                difMat[crtItr] = min(fldDif, fldDif / max(abs(numOut[crtItr]), lowTol))
            end
            difMat[(dipLoc[1]-winInt):(dipLoc[1]+winInt),
                   (dipLoc[2]-winInt):(dipLoc[2]+winInt),
                   (dipLoc[3]-winInt):(dipLoc[3]+winInt), :] .= 0.0
            relErrDir[dipDir] = maximum(difMat)
        end
        @test all(relErrDir .< 0.05)
    end
end

@testset "Scattering composition" begin
    v      = rand(ComplexF64, prod((4,4,4)) * 3)
    invSct = _invSct()
    sct    = _sct()
    gVac   = _g0()
    gla    = _gla()
    glaVac = _glaVac()

    # InvSctOpr * (SctOpr * v) ≈ v (self and external)
    @test invSct * (sct * v) ≈ v

    # Zero-susceptibility GlaOpr matches GlaOprVac
    @test glaVac * v ≈ gVac * v

    # GlaOpr field: G₀ * (SctOpr * v)
    @test gla * v ≈ gVac * (sct * v)
end
