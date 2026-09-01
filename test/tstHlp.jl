# Shared helpers — included before all topic files in runtests.jl
# Requires: using GilaElectromagnetics, LinearAlgebra, CUDA (done in runtests.jl)

const volSizes    = [(2,2,2), (4,4,4), (6,6,6), (8,8,8), (6,4,8), (8,2,10)]
const volSizesAna = [(6,6,6), (8,8,8), (6,10,8), (16,16,16)]
# Smaller subset for tests that call GlaVacOprMem — large volumes are very slow to construct
const vacVolSizes  = [(2,2,2), (4,4,4), (2,4,6)]
const stdScl = (1//32, 1//32, 1//32)
const stdOrg = (0//1, 0//1, 0//1)
const extOrg = (1//1, 1//1, 1//1)

mkVol(dim; org=stdOrg, scl=stdScl) = GlaVol(dim, scl, org)
mkSus(dim; val=0.5+0.05im) = fill(ComplexF64(val), dim...)
mkVac(dim) = zeros(ComplexF64, dim...)

# ---------------------------------------------------------------------------
# Shared precomputed objects — expensive GlaVacOprMem built once, reused everywhere
# ---------------------------------------------------------------------------
const _vol4  = mkVol((4,4,4))
const _sus4  = mkSus((4,4,4))
const _vac4  = mkVac((4,4,4))
const _trgV4 = mkVol((4,4,4); org=extOrg)

const _selfMem4 = GlaVacOprMem(CPUKerOpt(), _vol4)
const _extMem4  = GlaVacOprMem(CPUKerOpt(), _trgV4, _vol4)

# Tiny (2,2,2) operator for tests that loop over many mul! calls (linAlg, extOps)
const _vol2s   = mkVol((2,2,2))
const _sus2s   = mkSus((2,2,2))
const _mem2s   = GlaVacOprMem(CPUKerOpt(), _vol2s)
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
_glaVac() = GlaOpr(_g0(), _vac4)
# Cheap AsyGlaOprVac/SymGlaOprVac from precomputed GlaOprVac (deepcopy of egoFur, no integration)
_asy()    = AsyGlaOprVac(_g0())
_sym()    = SymGlaOprVac(_g0())

function dnsMat(opr::AbstractGlaOpr)
    n, m = size(opr, 2), size(opr, 2)
    rows = size(opr, 1)
    mat = zeros(ComplexF64, rows, n)
    for i in 1:n
        v = zeros(ComplexF64, n)
        v[i] = one(ComplexF64)
        mat[:, i] .= opr * v
    end
    return mat
end

function dnsMat(mem::GlaVacOprMem)
    n = prod(mem.srcVol.cel) * 3
    m = prod(mem.trgVol.cel) * 3
    mat = zeros(ComplexF64, m, n)
    for i in 1:n
        v = zeros(ComplexF64, mem.srcVol.cel..., 3)
        v[i] = one(ComplexF64)
        mat[:, i] .= vec(egoOpr!(mem, v))
    end
    return mat
end

asymMat(m) = (m - m') / 2im
symMat(m)  = (m + m') / 2

relErr(a, b) = opnorm(a - b) / opnorm(b)
