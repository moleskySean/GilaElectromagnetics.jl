#= The far-field expansion against the 220-bit references of test/ref/far.txt and farx.txt, and
the k-series it falls back on where neither expansion converges. =#
using Test, GilaElectromagnetics
const GVF = GilaElectromagnetics
const farQI = Rational{BigInt}

farSql(w) = ntuple(dir -> (p = split(split(w, ',')[dir], "//");
    farQI(parse(BigInt, p[1]), parse(BigInt, p[2]))), 3)

# (cell scale, frequency) => the offsets and reference tensors cached for it
function farRef()
    grp = Dict{Tuple{NTuple{3,farQI},ComplexF64},
        Vector{Tuple{NTuple{3,Int},Matrix{ComplexF64}}}}()
    for lin ∈ eachline(joinpath(@__DIR__, "..", "ref", "far.txt"))
        (isempty(strip(lin)) || startswith(lin, "#")) && continue
        wrd = split(lin)
        key = (farSql(wrd[1]), complex(parse(Float64, wrd[2]), parse(Float64, wrd[3])))
        push!(get!(grp, key, []),
            (ntuple(dir -> parse(Int, wrd[3 + dir]), 3),
             reshape([(z = split(x, ';');
                complex(parse(Float64, z[1]), parse(Float64, z[2]))) for x ∈ wrd[7:15]], 3, 3)))
    end
    return grp
end

# (target cell, source cell, frequency) => the exact offsets and reference tensors cached for it
function farRefX()
    grp = Dict{Tuple{NTuple{3,farQI},NTuple{3,farQI},ComplexF64},
        Vector{Tuple{NTuple{3,farQI},Matrix{ComplexF64}}}}()
    for lin ∈ eachline(joinpath(@__DIR__, "..", "ref", "farx.txt"))
        (isempty(strip(lin)) || startswith(lin, "#")) && continue
        wrd = split(lin)
        key = (farSql(wrd[1]), farSql(wrd[2]),
            complex(parse(Float64, wrd[3]), parse(Float64, wrd[4])))
        push!(get!(grp, key, []), (farSql(wrd[5]),
            reshape([(z = split(x, ';');
                complex(parse(Float64, z[1]), parse(Float64, z[2]))) for x ∈ wrd[6:14]], 3, 3)))
    end
    return grp
end

# the equal-cell tensors at the cell offsets sepLst, and the cross-scale ones at the exact offsets
function farFil(sepLst, sclQ, frq; kw...)
    ego = Array{ComplexF64}(undef, 3, 3, length(sepLst))
    GVF.farFil!(itr -> view(ego, :, :, itr), collect(sepLst), sclQ, frq; kw...)
    return ego
end
function farBlkX(sepLst, sclTrg, sclSrc, frq)
    ego = Array{ComplexF64}(undef, 3, 3, length(sepLst))
    GVF.farBlkX!(ego, collect(sepLst), sclTrg, sclSrc, frq)
    return ego
end

@testset "Far Field Tests" begin
    @testset "reference tensors" begin
        for ((sclQ, frq), cas) ∈ sort(collect(farRef());
                by = ent -> (Float64.(ent[1][1]), reim(ent[1][2])))
            egoNew = farFil(first.(cas), sclQ, frq)
            @test maximum(itr -> maximum(abs, egoNew[:, :, itr] .- cas[itr][2]) /
                maximum(abs, cas[itr][2]), eachindex(cas)) < 1e-13
        end
    end

    #= Two cells apart the octant expansion and the k-series on the 36 face pairs both converge,
    and they share no code below the moment library, so agreement there checks the panel
    bookkeeping and the normalization of both. =#
    @testset "k-series against the expansion" begin
        scl = (1//32, 1//32, 1//32)
        sclQ = ntuple(dir -> farQI(scl[dir]), 3)
        celSep = (2, 1, 1)
        # the set farFil! builds for this one offset
        @test GVF.farRte(GVF.farSetMem(sclQ, 1.0 + 0.0im; nBlk = 3), celSep)[1] != 3
        egoKsr = first(GVF.tnsKsr(sclQ, celSep, 1.0 + 0.0im))
        egoExp = farFil((celSep,), sclQ, 1.0 + 0.0im)[:, :, 1]
        @test maximum(abs, egoKsr .- egoExp) < 1e-13 * maximum(abs, egoExp)
    end

    @testset "reflection and transposition" begin
        for (scl, frq) ∈ (((1//32, 1//32, 1//32), 1.0 + 0.1im),
                ((1//32, 1//32, 1//512), 1.0 + 0.0im))
            sclQ = ntuple(dir -> farQI(scl[dir]), 3)
            sepLst = [(2, 3, 4), (5, 0, 2), (3, 3, 3)]
            egoPls = farFil(sepLst, sclQ, frq)
            egoMns = farFil([.-sep for sep ∈ sepLst], sclQ, frq)
            for itr ∈ eachindex(sepLst)
                @test maximum(abs, egoPls[:, :, itr] .- transpose(egoMns[:, :, itr])) <
                    1e-14 * maximum(abs, egoPls[:, :, itr])
            end
        end
    end

    @testset "block fill and set reuse" begin
        scl = (1//32, 1//32, 1//32)
        sclQ = ntuple(dir -> farQI(scl[dir]), 3)
        frq = 1.0 + 0.1im
        egoToe = Array{ComplexF64}(undef, 3, 3, 5, 4, 3)
        GVF.farBlk!(egoToe, scl, frq)
        egoRpt = Array{ComplexF64}(undef, 3, 3, 5, 4, 3)
        GVF.farBlk!(egoRpt, scl, frq)
        frqSet = GVF.farSetMem(sclQ, frq; nBlk = 5)
        sepLst = [itr for itr ∈ CartesianIndices(axes(egoToe)[3:5]) if maximum(Tuple(itr)) >= 3]
        # nothing accumulates across offsets, so the threaded fill is the per-offset path
        @test all(egoToe[:, :, itr] == egoRpt[:, :, itr] for itr ∈ sepLst)
        @test all(farFil((Tuple(itr) .- 1,), sclQ, frq; fs = frqSet)[:, :, 1] == egoToe[:, :, itr]
            for itr ∈ sepLst)
        # rHi = 4 nBlk r_d, so a set built for a small block certifies a shorter interval and the
        # memo must not hand it to a large one
        @test GVF.farSetMem(sclQ, frq; nBlk = 4).rHi < frqSet.rHi
    end

    @testset "cross-scale reference tensors" begin
        for ((sclTrg, sclSrc, frq), cas) ∈ sort(collect(farRefX());
                by = ent -> (Float64.(ent[1][2]), reim(ent[1][3])))
            egoNew = farBlkX(first.(cas), sclTrg, sclSrc, frq)
            @test maximum(itr -> maximum(abs, egoNew[:, :, itr] .- cas[itr][2]) /
                maximum(abs, cas[itr][2]), eachindex(cas)) < 1e-13
        end
    end

    #= The pair is a target box at R against a source box at 0, so swapping the two cells and
    negating R must return the same interaction up to the target volume the tensor is divided
    by, and reflecting an axis flips every tensor entry carrying that axis once. The offsets come
    from the reference groups because both identities need them on the common gcd lattice. =#
    @testset "cross-scale swap and reflection" begin
        casRef = farRefX()
        for (sclSrc, frq) ∈ (((1//8, 1//8, 1//8), 1.0 + 0.0im),
                ((1//4, 1//32, 1//32), 1.0 + 0.1im), ((1//2, 1//2, 1//32), 1.0 + 0.0im))
            sclTrg = ntuple(dir -> farQI(1, 32), 3)
            sclSrcQ = ntuple(dir -> farQI(sclSrc[dir]), 3)
            sepLst = first.(casRef[(sclTrg, sclSrcQ, frq)])[1:3]
            volRat = Float64(prod(sclSrcQ) / prod(sclTrg))
            ego = farBlkX(sepLst, sclTrg, sclSrcQ, frq)
            egoSwp = farBlkX([.-sep for sep ∈ sepLst], sclSrcQ, sclTrg, frq)
            @test all(maximum(abs, ego[:, :, itr] .- volRat .* egoSwp[:, :, itr]) <
                1e-13 * maximum(abs, ego[:, :, itr]) for itr ∈ eachindex(sepLst))
            for sgn ∈ ((-1, 1, 1), (1, -1, 1), (1, 1, -1), (-1, -1, -1))
                egoFlp = farBlkX([sgn .* sep for sep ∈ sepLst], sclTrg, sclSrcQ, frq)
                @test all(maximum(abs, egoFlp[:, :, itr] .-
                    [sgn[a] * sgn[b] * ego[a, b, itr] for a ∈ 1:3, b ∈ 1:3]) <
                    1e-13 * maximum(abs, ego[:, :, itr]) for itr ∈ eachindex(sepLst))
            end
        end
    end

    @testset "cross-scale cell ratio guard" begin
        sclTrg = (farQI(3, 64), farQI(1, 32), farQI(1, 32))
        sclSrc = ntuple(dir -> farQI(1, 32), 3)
        @test_throws "integer multiples" GVF.farSetX(sclTrg, sclSrc, 1.0 + 0.0im)
    end

    @testset "cross-scale block fill" begin
        sclTrg = ntuple(dir -> farQI(1, 32), 3)
        sclSrc = (farQI(1, 8), farQI(1, 8), farQI(1, 32))
        frq = 1.0 + 0.0im
        sepLst = first.(farRefX()[(sclTrg, sclSrc, frq)])[1:6]
        egoBlk = Array{ComplexF64}(undef, 3, 3, length(sepLst))
        rteLst = GVF.farBlkX!(egoBlk, sepLst, sclTrg, sclSrc, frq)
        # offsets on the gcd lattice never reach the k-series
        @test all(!=(3), rteLst)
    end

    # an offset off the gcd lattice, as between GlaVol((2, 2, 2), 1/16, 0) and GlaVol((4, 4, 4),
    # 1/32, (1/8 + 1/128, 0, 0)), takes the k-series; its only check is the swap of the two cells
    @testset "cross-scale k-series off the lattice" begin
        sclTrg = (1//16, 1//16, 1//16); sclSrc = (1//32, 1//32, 1//32); frq = 0.25 + 0.0im
        sep = (farQI(-7, 128), farQI(1, 64), farQI(1, 64))
        ego = Array{ComplexF64}(undef, 3, 3, 1); egoSwp = similar(ego)
        # a cold k-series cache, or a cached line would be read instead
        ksrDir = GVF.KSRDIR[]; GVF.KSRDIR[] = mktempdir()
        try
            @test GVF.farBlkX!(ego, [sep], sclTrg, sclSrc, frq; disk = false) == [3]
            @test GVF.farBlkX!(egoSwp, [.-sep], sclSrc, sclTrg, frq; disk = false) == [3]
        finally
            GVF.KSRDIR[] = ksrDir
        end
        @test maximum(abs, ego .- prod(sclSrc) / prod(sclTrg) .* egoSwp) < 1e-13 * maximum(abs, ego)
    end
end

@testset "clrCch! empties the caches and the rebuild is bitwise equal" begin
    B = GVF.Bloch
    vol = mkVol((8,8,8))
    mem = GlaVacOprMem(GlaKerOpt{Float64}(), vol)
    @test !isempty(GVF.SHPC) && !isempty(GVF.momMem)
    GVF.clrCch!()
    @test all(isempty, (GVF.SHPC, GVF.FRQC, GVF.MOMCX, GVF.FRQCX, GVF.momMem, B.GLC, B.MUF, B.MOMMC.d...))
    @test GlaVacOprMem(GlaKerOpt{Float64}(), vol).egoFur == mem.egoFur
end
