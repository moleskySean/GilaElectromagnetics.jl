#= The far-field expansion against the 220-bit references of test/ref/far.txt, its own a
posteriori certificate, and the k-series it falls back on where neither expansion converges. =#
using Test, GilaElectromagnetics
const GVF = GilaElectromagnetics.GilaVacuum
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

# a set covering exactly the offsets it is handed, as farBlkX! builds its own
function farSetCas(sclTrg, sclSrc, frq, sepLst)
    gcdScl = ntuple(dir -> min(sclTrg[dir], sclSrc[dir]), 3)
    celTrg = ntuple(dir -> Int(sclTrg[dir] / gcdScl[dir]), 3)
    celSrc = ntuple(dir -> Int(sclSrc[dir] / gcdScl[dir]), 3)
    GVF.farSetX(sclTrg, sclSrc, frq; offs = sepLst,
        nBlk = GVF.finBlk(sepLst, gcdScl, celTrg, celSrc))
end

@testset "Far Field Tests" begin
    @testset "reference tensors" begin
        for ((sclQ, frq), cas) ∈ sort(collect(farRef());
                by = ent -> (Float64.(ent[1][1]), reim(ent[1][2])))
            frqSet = GVF.farSet(sclQ, frq; offs = first.(cas))
            errMax = 0.0
            crtBad = 0
            for (celSep, egoRef) ∈ cas
                egoNew = GVF.farTns(celSep, sclQ, frq; fs = frqSet)
                err = maximum(abs, egoNew .- egoRef)
                errMax = max(errMax, err / maximum(abs, egoRef))
                knd, lvl, lvlOct, _ = GVF.farRte(frqSet, celSep)
                # est is a scale, not a bound: the certificate discharges the hypothesis.  The
                # rounding allowance is the 40 eps per entry this reference set is measured at
                err > GVF.certEq(frqSet, celSep, knd, lvl, lvlOct) +
                    40 * eps() * maximum(abs, egoNew) && (crtBad += 1)
            end
            @test errMax < 1e-13
            @test crtBad == 0
        end
    end

    #= Two cells apart the octant expansion and the k-series on the 36 face pairs both converge,
    and they share no code below the moment library, so agreement there checks the panel
    bookkeeping and the normalization of both. =#
    @testset "k-series against the expansion" begin
        scl = (1//32, 1//32, 1//32)
        sclQ = ntuple(dir -> farQI(scl[dir]), 3)
        celSep = (2, 1, 1)
        frqSet = GVF.farSet(sclQ, 1.0 + 0.0im; offs = (celSep,))
        @test GVF.farRte(frqSet, celSep)[1] != 3
        egoKsr = first(GVF.tnsKsr(sclQ, celSep, 1.0 + 0.0im))
        egoExp = GVF.farTns(celSep, scl, 1.0 + 0.0im; fs = frqSet)
        @test maximum(abs, egoKsr .- egoExp) < 1e-13 * maximum(abs, egoExp)
    end

    @testset "reflection and transposition" begin
        for (scl, frq) ∈ (((1//32, 1//32, 1//32), 1.0 + 0.1im),
                ((1//32, 1//32, 1//512), 1.0 + 0.0im))
            sclQ = ntuple(dir -> farQI(scl[dir]), 3)
            sepLst = ((2, 3, 4), (5, 0, 2), (3, 3, 3))
            frqSet = GVF.farSet(sclQ, frq; offs = sepLst)
            for celSep ∈ sepLst
                egoPls = GVF.farTns(celSep, scl, frq; fs = frqSet)
                egoMns = GVF.farTns(ntuple(dir -> -celSep[dir], 3), scl, frq; fs = frqSet)
                @test maximum(abs, egoPls .- transpose(egoMns)) < 1e-14 * maximum(abs, egoPls)
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
        @test all(GVF.farTns(Tuple(itr) .- 1, scl, frq; fs = frqSet) == egoToe[:, :, itr]
            for itr ∈ sepLst)
        # rHi = 4 nBlk r_d, so a set built for a small block certifies a shorter interval and the
        # memo must not hand it to a large one
        @test GVF.farSetMem(sclQ, frq; nBlk = 4).rHi < frqSet.rHi
    end

    @testset "cross-scale reference tensors" begin
        for ((sclTrg, sclSrc, frq), cas) ∈ sort(collect(farRefX());
                by = ent -> (Float64.(ent[1][2]), reim(ent[1][3])))
            frqSet = farSetCas(sclTrg, sclSrc, frq, first.(cas))
            errMax = 0.0
            crtBad = 0
            for (sepVec, egoRef) ∈ cas
                egoNew, crt = GVF.farTnsX(sepVec, sclTrg, sclSrc, frq; fs = frqSet, cert = true)
                err = maximum(abs, egoNew .- egoRef)
                errMax = max(errMax, err / maximum(abs, egoRef))
                # farTnsX's certificate already carries 20 eps; this set is measured at 40
                err > crt + 20 * eps() * maximum(abs, egoNew) && (crtBad += 1)
            end
            @test errMax < 1e-13
            @test crtBad == 0
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
            frqSet = farSetCas(sclTrg, sclSrcQ, frq, sepLst)
            swpSet = farSetCas(sclSrcQ, sclTrg, frq, [ntuple(dir -> -sep[dir], 3) for sep ∈ sepLst])
            volRat = Float64(prod(sclSrcQ) / prod(sclTrg))
            for sep ∈ sepLst
                ego = GVF.farTnsX(sep, sclTrg, sclSrcQ, frq; fs = frqSet)
                egoSwp = GVF.farTnsX(ntuple(dir -> -sep[dir], 3), sclSrcQ, sclTrg, frq; fs = swpSet)
                @test maximum(abs, ego .- volRat .* egoSwp) < 1e-13 * maximum(abs, ego)
                for sgn ∈ ((-1, 1, 1), (1, -1, 1), (1, 1, -1), (-1, -1, -1))
                    egoFlp = GVF.farTnsX(ntuple(dir -> sgn[dir] * sep[dir], 3), sclTrg, sclSrcQ,
                        frq; fs = frqSet)
                    @test maximum(abs, egoFlp .-
                        [sgn[a] * sgn[b] * ego[a, b] for a ∈ 1:3, b ∈ 1:3]) <
                        1e-13 * maximum(abs, ego)
                end
            end
        end
    end

    # the ratio check has to come before the Int() conversion it protects, which would
    # otherwise throw InexactError on the ratio instead
    @testset "cross-scale cell ratio guard" begin
        sclTrg = (farQI(3, 64), farQI(1, 32), farQI(1, 32))
        sclSrc = ntuple(dir -> farQI(1, 32), 3)
        @test_throws "integer multiples" GVF.farTnsX((farQI(1), farQI(0), farQI(0)),
            sclTrg, sclSrc, 1.0 + 0.0im)
        @test_throws "integer multiples" GVF.farSetX(sclTrg, sclSrc, 1.0 + 0.0im)
    end

    @testset "cross-scale block fill" begin
        sclTrg = ntuple(dir -> farQI(1, 32), 3)
        sclSrc = (farQI(1, 8), farQI(1, 8), farQI(1, 32))
        frq = 1.0 + 0.0im
        sepLst = first.(farRefX()[(sclTrg, sclSrc, frq)])[1:6]
        egoBlk = Array{ComplexF64}(undef, 3, 3, length(sepLst))
        rteLst = GVF.farBlkX!(egoBlk, sepLst, sclTrg, sclSrc, frq)
        # a lattice-aligned pair never reaches the k-series, which is why no generation path does
        @test all(!=(3), rteLst)
        frqSet = farSetCas(sclTrg, sclSrc, frq, sepLst)
        @test all(GVF.farTnsX(sepLst[itr], sclTrg, sclSrc, frq; fs = frqSet) ==
            egoBlk[:, :, itr] for itr ∈ eachindex(sepLst))
    end
end
