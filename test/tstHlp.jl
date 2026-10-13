# Shared helpers — included before all topic files in runtests.jl
# Requires: using GilaElectromagnetics, LinearAlgebra, CUDA, Metal, OpenCL (done in runtests.jl)
import GilaElectromagnetics.GilaOperators: mskRng
import GilaElectromagnetics.GilaVolumes: uniVol
using KernelAbstractions: functional, supports_float64

# OpenCL.jl leaves functional unimplemented (missing); pocl always provides a device
const gpuBackends = filter(dev -> coalesce(functional(dev), true), [CUDABackend(), MetalBackend(), OpenCLBackend()])

# Storage precisions to test on a backend: Float32 always, Float64 only where supported
gpuPrcs(dev) = supports_float64(dev) ? (Float32, Float64) : (Float32,)

# Relative CPU-GPU parity bound; Float64 leaves room for device FFTs less exact than pocl's
gpuTol(T) = T === Float64 ? 1e-12 : 1e-5

# sym, unlike asym, is not exported; the fine mesh block type is internal as well
const glaSym = GilaElectromagnetics.GilaOperators.sym
const GlaSnd = GilaElectromagnetics.GilaOperators.GlaSndOprVac

const volSizes = [(2,2,2), (4,4,4), (6,6,6), (8,8,8), (6,4,8), (8,2,10)]
const stdScl = (1//32, 1//32, 1//32)
const scl16  = (1//16, 1//16, 1//16)
const stdOrg = (0//1, 0//1, 0//1)
const extOrg = (1//1, 1//1, 1//1)

# the density every discretize! test uses: one period across x, linear in y
tstDns = pos -> (exp(2im * pi * pos[1]), pos[2], 0)

mkVol(dim; org=stdOrg, scl=stdScl) = GlaVol(dim, scl, org)
mkSus(::Type{T}, dim; val=0.5+0.05im) where T<:AbstractFloat = fill(Complex{T}(val), dim...)
mkSus(dim; val=0.5+0.05im) = mkSus(Float64, dim; val=val)

# ---------------------------------------------------------------------------
# Shared precomputed objects — expensive GlaVacOprMem built once, reused everywhere
# ---------------------------------------------------------------------------
const _vol4  = mkVol((4,4,4))
const _sus4  = mkSus((4,4,4))
const _trgV4 = mkVol((4,4,4); org=extOrg)

const _selfMem4 = GlaVacOprMem(GlaKerOpt{Float64}(), _vol4)
const _extMem4  = GlaVacOprMem(GlaKerOpt{Float64}(), _trgV4, _vol4)

# Tiny (2,2,2) operator for tests that loop over many mul! calls (linAlg, extOps)
const _vol2s   = mkVol((2,2,2))
const _sus2s   = mkSus((2,2,2))
const _mem2s   = GlaVacOprMem(GlaKerOpt{Float64}(), _vol2s)
_g0s()     = GlaOprVac(_mem2s)
_asys()    = AsyGlaOprVac(_g0s())
_invScts() = InvSctOpr(_g0s(), _sus2s)
_scts()    = SctOpr(_g0s(), _sus2s)
_glas()    = GlaOpr(_g0s(), _sus2s)

# Cheap operator builders (share the precomputed egoFur — no integration cost)
_g0()     = GlaOprVac(_selfMem4)
_gExt()   = GlaOprVac(_extMem4)
_invSct() = InvSctOpr(_g0(), _sus4)
_sct()    = SctOpr(_g0(), _sus4)
_gla()    = GlaOpr(_g0(), _sus4)
_glaVac() = GlaOpr(_g0(), zeros(ComplexF64, 4, 4, 4))
# Cheap AsyGlaOprVac/SymGlaOprVac from precomputed GlaOprVac (deepcopy of egoFur, no integration)
_asy()    = AsyGlaOprVac(_g0())
_sym()    = SymGlaOprVac(_g0())

function dnsMat(opr::AbstractGlaOpr)
    T = eltype(opr)
    n = size(opr, 2)
    mat = zeros(T, size(opr, 1), n)
    for i in 1:n
        v = zeros(T, n)
        v[i] = one(T)
        mat[:, i] .= opr * v
    end
    return mat
end

function dnsMat(mem::GlaVacOprMem)
    T = eltype(first(mem.egoFur))
    n = prod(mem.srcVol.cel) * 3
    m = prod(mem.trgVol.cel) * 3
    mat = zeros(T, m, n)
    for i in 1:n
        v = zeros(T, mem.srcVol.cel..., 3)
        v[i] = one(T)
        mat[:, i] .= vec(egoOpr!(mem, v))
    end
    return mat
end

# The self operator on the union of a same-scale pair, masked down to the target
# and source cells: the reference an external operator has to reproduce
function uniMskMat(trgVol::GlaVol, srcVol::GlaVol)
    uniVolume = uniVol(trgVol, srcVol)
    oprUni = GlaOprVac{Float64}(uniVolume)
    innMsk, outMsk = mskRng(srcVol, uniVolume), mskRng(trgVol, uniVolume)
    colNum = prod(srcVol.cel) * 3
    mat = zeros(ComplexF64, prod(trgVol.cel) * 3, colNum)
    for colItr in 1:colNum
        srcVec = zeros(ComplexF64, srcVol.cel..., 3)
        srcVec[colItr] = one(ComplexF64)
        embVec = zeros(ComplexF64, uniVolume.cel..., 3)
        embVec[innMsk..., :] .= srcVec
        mat[:, colItr] .= vec((oprUni * embVec)[outMsk..., :])
    end
    return mat
end

function serRnd(opr)
    buf = IOBuffer()
    serialize(buf, opr)
    seekstart(buf)
    return deserialize(buf)
end

# round trip an operator and compare it entrywise, not just through one matvec
function serChk(opr; tol = 1e-12)
    desOpr = serRnd(opr)
    @test desOpr isa typeof(opr)
    @test frbErr(dnsMat(desOpr), dnsMat(opr)) < tol
    innVec = rand(ComplexF64, size(opr, 2))
    @test norm(desOpr * innVec - opr * innVec) < tol * norm(opr * innVec)
    return desOpr
end

asymMat(m) = (m - m') / 2im
symMat(m)  = (m + m') / 2

relErr(a, b) = opnorm(a - b) / opnorm(b)
# Frobenius version, for vectors and for arrays opnorm does not take
frbErr(a, b) = norm(a .- b) / norm(b)
