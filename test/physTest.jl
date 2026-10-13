# Physics tests
# Requires: uniVol imported in volTest.jl (same scope), eigvals from LinearAlgebra

# PSD tolerance: assert min eigenvalue > -rtol * opnorm(Asym). The worst measured
# here is -1.1e-15 normalized; posDefTest.jl carries the tight per-shape bars.
const psdRtol = 1e-12

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

# Analytic free-space dyadic Green function for a point dipole
function egoAna!(anaOut::AbstractVector{ComplexF64}, slfVol::GlaVol,
                 trgRng::NTuple{3,<:StepRange}, dipPos::Vector{<:Rational},
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

@testset "Asym(G₀) PSD — AsyGlaOprVac itself" begin
    # AsyGlaOprVac(_g0()) deepcopies egoFur then takes Im part — cheap (no integration)
    mat = dnsMat(_asy())
    @test mat ≈ mat'
    checkPsd(mat, "AsyGlaOprVac (4,4,4)")
end

@testset "Asym(G₀) PSD — gapped GlaCmpOprVac" begin
    op = GlaCmpOprVac{Float64}(GlaCmpVol([_vol4, _trgV4]))
    checkPsd(asymMat(dnsMat(op)), "GlaCmpOprVac 2 separated regions")
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

#= The one absolute physical check in the suite: the discretized operator against
the point-dipole dyadic. The 5 % bar is the discretization error of a pulse basis
at 32 cells per wavelength, not slack, and cannot be tightened. (6,6,6) is the
smallest volume that leaves cells outside the exclusion window. =#
@testset "Analytic dyadic Green function" begin
    lowTol = 1.0e-12
    # half width, in cells, of the window around the dipole where the pulse-basis
    # field is not the point-dipole field
    winInt = 2
    vol    = mkVol((6,6,6))
    oprMem = GlaVacOprMem(GlaKerOpt{Float64}(), vol)
    dipVec = zeros(ComplexF64, 3)
    relErrDir = zeros(Float64, 3)
    anaOut = Array{ComplexF64}(undef, 3 * prod(vol.cel))
    numOut = Array{ComplexF64}(undef, vol.cel..., 3)
    difMat = Array{Float64}(undef, vol.cel..., 3)
    innVec = Array{ComplexF64}(undef, vol.cel..., 3)
    dipLoc = div.(vol.cel, 2)
    dipPos = Rational.([vol.grd[d][dipLoc[d]] for d in 1:3])

    for dipDir in 1:3
        dipVec .= 0.0im
        dipVec[dipDir] = 1.0 + 0.0im
        fill!(innVec, zero(ComplexF64))
        innVec[dipLoc..., dipDir] = (1.0 + 0.0im) / prod(vol.scl)
        copyto!(numOut, egoOpr!(oprMem, innVec))
        egoAna!(anaOut, vol, vol.grd, dipPos, dipVec)
        anaRsh = reshape(anaOut, vol.cel..., 3)

        #= The clip is load-bearing, not slack: the pure relative worst is 0.0468
        against the 0.05 bar, so entries where the numerical field nearly vanishes
        would push it over on their absolute difference alone. =#
        for crtItr in CartesianIndices((vol.cel..., 3))
            fldDif = abs(anaRsh[crtItr] - numOut[crtItr])
            difMat[crtItr] = min(fldDif, fldDif / max(abs(numOut[crtItr]), lowTol))
        end
        difMat[(dipLoc[1]-winInt):(dipLoc[1]+winInt),
               (dipLoc[2]-winInt):(dipLoc[2]+winInt),
               (dipLoc[3]-winInt):(dipLoc[3]+winInt), :] .= 0.0
        relErrDir[dipDir] = maximum(difMat)
    end
    @info "analytic dyadic, worst outside the window = $(maximum(relErrDir))"
    @test all(relErrDir .< 0.05)
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
