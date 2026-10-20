# Shared helpers — included before all topic files in runtests.jl
import GilaElectromagnetics: mskRng
import GilaElectromagnetics: uniVol
import GilaElectromagnetics: GlaKerOpt, GlaVacOprMem, GlaCmpOprVac, glaSze

# sym, unlike asym, is not exported; the fine mesh block type is internal as well
const glaSym = GilaElectromagnetics.sym
const GlaSnd = GilaElectromagnetics.GlaSndOprVac

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
# A Hermitian part of a self operator is never mutated by adjoint!, so its memory is shared like the others
const _asyMem4  = GlaVacOprMem(GlaKerOpt{Float64}(; hrmPrt = :asy), _vol4)
const _symMem4  = GlaVacOprMem(GlaKerOpt{Float64}(; hrmPrt = :sym), _vol4)

# Tiny (2,2,2) operator for tests that loop over many mul! calls (linAlg, extOps)
const _vol2s   = mkVol((2,2,2))
const _sus2s   = mkSus((2,2,2))
const _mem2s   = GlaVacOprMem(GlaKerOpt{Float64}(), _vol2s)
const _asyMem2s = GlaVacOprMem(GlaKerOpt{Float64}(; hrmPrt = :asy), _vol2s)
_g0s()     = GlaOprVac(_mem2s)
_asys()    = GlaOprVac(_asyMem2s)
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

_asy()    = GlaOprVac(_asyMem4)
_sym()    = GlaOprVac(_symMem4)

#= A slab and a box sharing interior, which the constructor sends through the
union route, so the operator carries a source and a target mask. =#
const _mskSlb = GlaVol((2,4,4), scl16, stdOrg)
const _mskBox = GlaVol((2,2,2), scl16, (1//16, 0//1, 0//1))
const _mskOpr = GlaOprVac{Float64}(_mskSlb, _mskBox)
# Two (2,2,2) regions side by side, a cheap composite operator
const _cvl2s = GlaCmpVol([GlaVol((2,2,2), scl16, stdOrg),
    GlaVol((2,2,2), scl16, (1//8, 0//1, 0//1))])
const _cmp2s = GlaCmpOprVac{Float64}(_cvl2s)

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

# show runs in both forms, and the compact one is a single line
shwChk(obj) = !occursin('\n', sprint(show, obj)) && !isempty(sprint(show, MIME"text/plain"(), obj))

serByt(obj) = (buf = IOBuffer(); serialize(buf, obj); take!(buf))

#= A round trip either reproduces the stored data or is broken: the read back
operator writes the same bytes, and one probe checks the plans rebuilt on load. =#
function serChk(opr; rnd = serRnd)
    desOpr = rnd(opr)
    @test desOpr isa typeof(opr)
    @test serByt(desOpr) == serByt(opr)
    innVec = rand(eltype(opr), size(opr, 2))
    @test norm(desOpr * innVec - opr * innVec) < 1e4 * eps(real(eltype(opr))) * norm(opr * innVec)
    return desOpr
end

#= Three random probes tell two operators apart as surely as their dense
matrices do, at a fraction of the matvecs. Either side may be an operator, a
LinearMap or a matrix. =#
function prbErr(oprA, oprB)
    inp = randn(eltype(oprA), size(oprA, 2), 3)
    # mul!, since a LinearMap times a matrix is another LinearMap
    return frbErr(oprA * inp, mul!(similar(inp, size(oprA, 1), 3), oprB, inp))
end

# The dot identity dot(W y, A x) = dot(W B y, x) of an adjoint pair, W the target or source cell volumes; adjErr(A, A) measures A against Hermitian
function adjErr(opr, adj; wgt=1, srcWgt=wgt)
    x, y = randn(eltype(opr), size(opr, 2)), randn(eltype(opr), size(opr, 1))
    lhs = dot(wgt .* y, opr * x)
    return abs(lhs - dot(srcWgt .* (adj * y), x)) / abs(lhs)
end

# The transpose identity transpose(y) A x = transpose(x) A y of a complex-symmetric A
function symErr(opr; wgt=1)
    x, y = randn(eltype(opr), size(opr, 2)), randn(eltype(opr), size(opr, 1))
    lhs = transpose(wgt .* y) * (opr * x)
    return abs(lhs - transpose(wgt .* x) * (opr * y)) / abs(lhs)
end

# The anti-Hermitian and Hermitian parts of opr applied to inp
asyPrb(opr, inp) = (opr * inp - adjoint(opr) * inp) / 2im
symPrb(opr, inp) = (opr * inp + adjoint(opr) * inp) / 2

asymMat(m) = (m - m') / 2im
symMat(m)  = (m + m') / 2

relErr(a, b) = opnorm(a - b) / opnorm(b)
# Frobenius version, for vectors and for arrays opnorm does not take
frbErr(a, b) = norm(a .- b) / norm(b)
