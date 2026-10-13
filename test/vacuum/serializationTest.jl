#= A GPU-built mem serializes from device memory and deserializes to a CPU mem
whose written data still applies. The CPU round trip is checked entrywise by
serTest.jl. =#
using Test, Serialization, GilaElectromagnetics

@testset "GlaVacOprMem serialization GPU" begin
    @testset "$(nameof(typeof(dev))) $T" for dev in gpuBackends, T in gpuPrcs(dev)
        vol = mkVol((4, 4, 4))
        mem = GlaVacOprMem(GlaKerOpt{T}(dev), vol, vol)
        tmpFil = tempname()
        try
            open(io -> serialize(io, mem), tmpFil, "w")
            desMem = open(deserialize, tmpFil, "r")
            @test get_backend(desMem) isa CPU
            for fld in (:genPrc, :frqPhz, :adjMod)
                @test getfield(desMem.cmpInf, fld) == getfield(mem.cmpInf, fld)
            end
            for fld in (:minScl, :trgDiv, :srcDiv, :trgCel, :srcCel, :trgPar, :srcPar)
                @test getfield(desMem.mixInf, fld) == getfield(mem.mixInf, fld)
            end
            @test desMem.trgVol == mem.trgVol && desMem.srcVol == mem.srcVol
            for (furDes, fur) in zip(desMem.egoFur, mem.egoFur)
                @test furDes == Array(fur)
            end
            actVec = ones(Complex{T}, (4, 4, 4, 3))
            @test egoOpr!(mem, adapt(dev, deepcopy(actVec))) ≈
                adapt(dev, egoOpr!(desMem, deepcopy(actVec)))
        finally
            isfile(tmpFil) && rm(tmpFil)
        end
    end
end
