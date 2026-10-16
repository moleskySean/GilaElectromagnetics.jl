# Separated blocks of the discretized vacuum Green operator, in closed form.
# T_ab(R) = (1/V_t) int_D w(d) [(da db + dab k^2) g](R+d) dd, g = e^{ikr}/(4 pi f^2 r), k = 2 pi f,
# D = prod_i [-s_i, s_i], w(d) = prod_i (s_i - |d_i|), V_t = s1 s2 s3.  Three routes per offset:
#   1  the whole box by the spherical addition theorem, one expansion about R, even l only;
#   2  the 8 octants of D with the affine weight, same theorem, all l;
#   3  the k-series on the 36 face pairs of glaVacIntMom.jl, where neither expansion converges.
# The l per offset and per sub-box comes from the proven tail bounds of Theorems A (whole box) and
# B (sub-box); the route is the first of 1, 2 whose bound meets tol * est(R), else 3.  The geometry
# table is exact rational, built lazily in l and extended in place; its k-series order N is chosen
# per (shape, frequency) from the n-tail bound, and the library throws rather than
# truncate silently.  Generic in T, every constant typed.
# Unequal cells (target edges sT at R, source edges sS at 0) change only the weight: per axis the
# trapezoid w_i(t) = (b_i - |t|)_+ - (a_i - |t|)_+, a = |sT - sS|/2, b = (sT + sS)/2, D = prod[-b_i,
# b_i], (1/V_t) int_D w = V_s.  Their routes are the whole trapezoid box (Theorem A'), the gcd
# average of equal-cell tensors of g = min(sT, sS), and the k-series on the 36 unequal face pairs.

using Scratch

const QI = Rational{BigInt}
# the six independent entries of the symmetric tensor, in the order every bound and sum uses
const ENTIJ = ((1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3))

# affine weight al + bt t on [-h,h]: nu_n = 2 al h^{n+1}/(n+1) even n, 2 bt h^{n+2}/(n+2) odd n
wgtA(n::Int, al::S, bt::Int, h::S) where {S} =
    n < 0 ? zero(S) : (iseven(n) ? 2 * al * h^(n + 1) / S(n + 1) : 2 * S(bt) * h^(n + 2) / S(n + 2))

# (v0, v1, v2) per axis: v0[n] = moment of t^n, v1[n] = n v0[n-1], v2[n] = n(n-1) v0[n-2]
function momTrp(v0::Vector{S}) where {S}
    dM = length(v0) - 1
    v1 = S[n < 1 ? zero(S) : S(n) * v0[n] for n in 0:dM]
    v2 = S[n < 2 ? zero(S) : S(n * (n - 1)) * v0[n - 1] for n in 0:dM]
    (v0, v1, v2)
end
wgtTrpA(al::S, bt::Int, h::S, dM::Int) where {S} = momTrp(S[wgtA(n, al, bt, h) for n in 0:dM])
# trapezoid of the pair (sT, sS): a = |sT - sS|/2, b = (sT + sS)/2, w = b - a on |t| <= a and b - |t|
# on a <= |t| <= b, i.e. the triangle of half-width b minus the triangle of half-width a, so
# mu_n = 2 (b^{n+2} - a^{n+2})/((n+1)(n+2)) even n, 0 odd; a = 0 is the triangle bit for bit (x - 0 == x)
wgtX(n::Int, a::S, b::S) where {S} =
    (n < 0 || isodd(n)) ? zero(S) : 2 * (b^(n + 2) - a^(n + 2)) / S((n + 1) * (n + 2))
wgtTrpX(a::S, b::S, dM::Int) where {S} = momTrp(S[wgtX(n, a, b) for n in 0:dM])

# C_m^m = (2m-1)!!, C_{m+1}^m = (2m+1)!! z, C_l^m = (2l-1) z C_{l-1}^m - (l+m-1)(l-m-1) r^2 C_{l-2}^m
# with C_l^m = (l-m)! [r^l P_l^m(z/r) stripped of (x+iy)^m]; integers, no Condon-Shortley phase.
function legCof(l::Int, m::Int)
    dfc = one(BigInt)
    for j in 1:m; dfc *= 2j - 1; end
    g0 = [dfc]
    l == m && return g0
    g1 = [dfc * (2m + 1)]
    l == m + 1 && return g1
    for ll in (m + 2):l
        g = zeros(BigInt, div(ll - m, 2) + 1)
        for b in 1:length(g1); g[b] += (2ll - 1) * g1[b]; end
        for b in 1:length(g0); g[b + 1] -= (ll + m - 1) * (ll - m - 1) * g0[b]; end
        g0, g1 = g1, g
    end
    return g1
end

# Re (snf = false) or Im (snf = true) of (x+iy)^m over the basis x^{m-t} y^t
function azmCof(m::Int, snf::Bool)
    cf = zeros(BigInt, m + 1)
    for t in 0:m
        (snf == isodd(t)) || continue
        sg = iseven(t) ? (-1)^div(t, 2) : (-1)^div(t - 1, 2)
        cf[t + 1] = sg * binomial(big(m), big(t))
    end
    return cf
end

# parity of the x, y exponents of the harmonic polynomial of (l, m, family)
prtXY(m::Int, snf::Bool) = snf ? ((m - 1) % 2, 1) : (m % 2, 0)
ylmInd(l::Int, m::Int) = l * l + l + m + 1

# entry [a+1,b+1] of a parity-compact degree-deg array is the coefficient of x^i y^j z^{deg-i-j},
# i = 2a + p1, j = 2b + p2; multiplication by r^2 is a three-point stencil.
# The polynomial matrices hold one distinct BigInt per entry -- zeros(BigInt, ..) would alias a
# single object -- and every write is an in-place GMP call; the allocating spelling leaves BigInt
# garbage the collector does not return.
const MPZ = Base.GMP.MPZ
bigMat(nA::Int) = [BigInt(0) for _ in 1:nA, _ in 1:nA]
bigZro!(M::Matrix{BigInt}) = (for x in M; MPZ.set_si!(x, 0); end; M)

function mulRsq!(dst::Matrix{BigInt}, src::Matrix{BigInt})
    nA = size(src, 1)
    @inbounds for a in nA:-1:1, b in nA:-1:1
        d = dst[a, b]
        MPZ.set!(d, src[a, b])
        a > 1 && MPZ.add!(d, src[a - 1, b])
        b > 1 && MPZ.add!(d, src[a, b - 1])
    end
    dst
end

function hrmPly!(P::Matrix{BigInt}, U::Matrix{BigInt}, V::Matrix{BigInt},
                 l::Int, m::Int, snf::Bool, p1::Int, p2::Int, tmp::BigInt = BigInt(0))
    nA = size(P, 1)
    gam = legCof(l, m); az = azmCof(m, snf)
    bigZro!(P); bigZro!(U)
    for t in 0:m
        az[t + 1] == 0 && continue
        MPZ.set!(U[div(m - t - p1, 2) + 1, div(t - p2, 2) + 1], az[t + 1])
    end
    for b in 1:length(gam)
        if gam[b] != 0
            gb = gam[b]
            @inbounds for a in 1:nA, bb in 1:nA
                u = U[a, bb]; iszero(u) && continue
                MPZ.mul!(tmp, gb, u); MPZ.add!(P[a, bb], tmp)
            end
        end
        if b < length(gam)
            mulRsq!(V, U); U, V = V, U
        end
    end
    return P
end

# N_lm/(l-m)!, the factor turning C_l^m A_m into r^l Y_lm
function nrmCof(l::Int, m::Int, ::Type{T}) where {T}
    num = T(2l + 1) * (m == 0 ? one(T) : T(2))
    den = 4 * T(pi) * T(factorial(big(l - m))) * T(factorial(big(l + m)))
    return sqrt(num / den)
end

# which (v0,v1,v2) each axis takes for the seven integrands da^2 (a=1,2,3), 1, d1d2, d1d3, d2d3
const KND = ((3, 1, 1), (1, 3, 1), (1, 1, 3), (1, 1, 1), (2, 2, 1), (2, 1, 2), (1, 2, 2))

# int_box wgt(d) da db [d^gamma] contracted against the polynomial P; returns (sum, sum|term|).
# ztl is the relative size below which a cancelled sum is an exact zero: it must sit above the
# rounding noise of the accumulation (n_trm 2^-prc) and below the smallest true value
# (cancellation^-1 ~ 2^-40), which 2^-(prc-24) does for every prc >= 128 used here.
function momAcc(P::Matrix{BigInt}, W, kk::Int, deg::Int, p1::Int, p2::Int, ztl::S) where {S}
    k1, k2, k3 = KND[kk]
    nA = size(P, 1)
    acc = zero(S); abs_ = zero(S)
    @inbounds for a in 0:(nA - 1)
        i = 2a + p1; i > deg && break
        w1 = W[1][k1][i + 1]; iszero(w1) && continue
        for b in 0:(nA - 1)
            j = 2b + p2; k = deg - i - j
            k < 0 && break
            v = P[a + 1, b + 1]; iszero(v) && continue
            w23 = W[2][k2][j + 1] * W[3][k3][k + 1]; iszero(w23) && continue
            trm = S(v) * w1 * w23
            acc += trm; abs_ += abs(trm)
        end
    end
    abs(acc) < abs_ * ztl && (acc = zero(S))
    return (acc, abs_)
end

# In-place MPFR for momAcc's inner loop: the generic BigFloat method allocates five limb buffers
# per term, and the loop runs ~3e8 times per shape.  Bit-identical, ~5x faster.
const RND = Base.MPFR.ROUNDING_MODE
mpfMul!(z::BigFloat, x::BigFloat, y::BigFloat) = (ccall((:mpfr_mul, Base.MPFR.libmpfr), Int32,
    (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, Int32), z, x, y, RND[]); z)
mpfAdd!(z::BigFloat, x::BigFloat, y::BigFloat) = (ccall((:mpfr_add, Base.MPFR.libmpfr), Int32,
    (Ref{BigFloat}, Ref{BigFloat}, Ref{BigFloat}, Int32), z, x, y, RND[]); z)
mpfSetZ!(z::BigFloat, x::BigInt) = (ccall((:mpfr_set_z, Base.MPFR.libmpfr), Int32,
    (Ref{BigFloat}, Ref{BigInt}, Int32), z, x, RND[]); z)
mpfAbs!(z::BigFloat, x::BigFloat) = (ccall((:mpfr_abs, Base.MPFR.libmpfr), Int32,
    (Ref{BigFloat}, Ref{BigFloat}, Int32), z, x, RND[]); z)
mpfZro!(z::BigFloat) = (ccall((:mpfr_set_zero, Base.MPFR.libmpfr), Cvoid,
    (Ref{BigFloat}, Cint), z, 1); z)

# the four registers momAcc's BigFloat method works in, at the ambient precision
momReg() = (BigFloat(), BigFloat(), BigFloat(), BigFloat())

# rg is (term, product, acc, abs); the caller must consume the returned pair before the next call
function momAcc(P::Matrix{BigInt}, W, kk::Int, deg::Int, p1::Int, p2::Int, ztl::BigFloat,
                rg::NTuple{4,BigFloat})
    k1, k2, k3 = KND[kk]
    nA = size(P, 1)
    t, u, acc, abs_ = rg
    mpfZro!(acc); mpfZro!(abs_)
    @inbounds for a in 0:(nA - 1)
        i = 2a + p1; i > deg && break
        w1 = W[1][k1][i + 1]; iszero(w1) && continue
        for b in 0:(nA - 1)
            j = 2b + p2; k = deg - i - j
            k < 0 && break
            v = P[a + 1, b + 1]; iszero(v) && continue
            w2 = W[2][k2][j + 1]; iszero(w2) && continue
            w3 = W[3][k3][k + 1]; iszero(w3) && continue
            mpfMul!(u, w2, w3)
            mpfSetZ!(t, v); mpfMul!(t, t, w1); mpfMul!(t, t, u)
            mpfAdd!(acc, acc, t)
            mpfAbs!(t, t); mpfAdd!(abs_, abs_, t)
        end
    end
    mpfAbs!(t, acc); mpfMul!(u, abs_, ztl)
    t < u && mpfZro!(acc)
    return (acc, abs_)
end

# the exact-rational build has no registers to reuse
momAcc(P::Matrix{BigInt}, W, kk::Int, deg::Int, p1::Int, p2::Int, ztl::S, ::Nothing) where {S} =
    momAcc(P, W, kk, deg, p1, p2, ztl)

# Qgeo entries, sorted by l: the three diagonal integrands share one (l,m) list with the k^2 moment.
struct GeoBox{T}
    L::Int
    nMax::Int
    dLm::Vector{Int}
    dLv::Vector{Int}
    Ad::Array{T,3}                  # (3, nMax+1, nd)
    Bd::Matrix{T}                   # (nMax+1, nd)
    oLm::NTuple{3,Vector{Int}}
    oLv::NTuple{3,Vector{Int}}
    Ao::NTuple{3,Matrix{T}}         # xy, xz, yz
    cnc::Float64                    # worst sum|term|/|acc| inside the exact contraction
end

# drop (l,m) columns that vanish identically for every n
function cmpCol(lm::Vector{Int}, lv::Vector{Int}, cols::Vector{<:AbstractMatrix})
    kp = Int[]
    for j in eachindex(lm)
        any(!iszero, (c[n, j] for c in cols for n in axes(c, 1))) && push!(kp, j)
    end
    (lm[kp], lv[kp], kp)
end

# whole box: the triangle weight is even, so only even l and four disjoint (l,m) classes survive
# (diag: cos, m even; xy: sin, m even >= 2; xz: cos, m odd; yz: sin, m odd).
# Only the shells lLo:2:L are built, so a table can be extended in l by concatenating segments.
tabWhl(s::NTuple{3,S}, lLo::Int, L::Int, nMax::Int, ::Type{T}, ztl::S) where {S,T} =
    tabWhl(ntuple(_ -> zero(S), 3), s, lLo, L, nMax, T, ztl)
# unequal cells: trapezoid a = |sT - sS|/2, b = (sT + sS)/2 per axis, stored divided by prod(b)
# (the caller multiplies by prod(b)/V_t); a = 0 is the equal-cell table
function tabWhl(ta::NTuple{3,S}, tb::NTuple{3,S}, lLo::Int, L::Int, nMax::Int, ::Type{T},
                ztl::S) where {S,T}
    @assert iseven(L) && iseven(lLo)
    dM = L + 2nMax; nA = div(dM, 2) + 2
    W = ntuple(d -> wgtTrpX(ta[d], tb[d], dM), 3)
    vt = tb[1] * tb[2] * tb[3]
    msOf(c, l) = c == 1 ? (0:2:l) : c == 2 ? (2:2:l) : (1:2:l - 1)
    snOf = (false, true, false, true)
    lmC = ntuple(c -> Int[], 4); lvC = ntuple(c -> Int[], 4)
    for c in 1:4, l in lLo:2:L, m in msOf(c, l)
        push!(lmC[c], ylmInd(l, snOf[c] ? -m : m)); push!(lvC[c], l)
    end
    nd = length(lmC[1])
    Ad = zeros(T, 3, nMax + 1, nd); Bd = zeros(T, nMax + 1, nd)
    Ao = ntuple(c -> zeros(T, nMax + 1, length(lmC[c + 1])), 3)
    P = bigMat(nA); U = bigMat(nA); V = bigMat(nA); tmp = BigInt(0)
    rg = S === BigFloat ? momReg() : nothing
    cmx = 1.0
    for c in 1:4
        kl = c == 1 ? (1, 2, 3, 4) : (c + 3,)
        for (jj, l) in enumerate(lvC[c])
            m = abs(lmC[c][jj] - ylmInd(l, 0)); snf = snOf[c]
            p1, p2 = prtXY(m, snf)
            hrmPly!(P, U, V, l, m, snf, p1, p2, tmp)
            nc = T(nrmCof(l, m, BigFloat))
            for n in 0:nMax
                deg = l + 2n
                for kk in kl
                    acc, ab = momAcc(P, W, kk, deg, p1, p2, ztl, rg)
                    val = T(acc) * nc / T(vt)
                    iszero(acc) || (cmx = max(cmx, Float64(ab / abs(acc))))
                    kk <= 3 ? (Ad[kk, n + 1, jj] = val) :
                        kk == 4 ? (Bd[n + 1, jj] = val) : (Ao[kk - 4][n + 1, jj] = val)
                end
                if n < nMax
                    mulRsq!(V, P); P, V = V, P
                end
            end
        end
    end
    dLm, dLv, kp = cmpCol(lmC[1], lvC[1], [view(Ad, 1, :, :), view(Ad, 2, :, :), view(Ad, 3, :, :), Bd])
    oL = ntuple(c -> cmpCol(lmC[c + 1], lvC[c + 1], [Ao[c]]), 3)
    GeoBox{T}(L, nMax, dLm, dLv, Ad[:, :, kp], Bd[:, kp],
              ntuple(c -> oL[c][1], 3), ntuple(c -> oL[c][2], 3),
              ntuple(c -> Ao[c][:, oL[c][3]], 3), cmx)
end

# one octant of D: c = h = al = s/2, bt = -1; no parity survives, every (l,m) contributes.
# Only the shells lLo:L are built (lm keeps the global Y index, so segments concatenate).
function tabOct(s::NTuple{3,S}, lLo::Int, L::Int, nMax::Int, ::Type{T}, ztl::S) where {S,T}
    dM = L + 2nMax; nA = div(dM, 2) + 2
    hf = ntuple(d -> s[d] / 2, 3)
    W = ntuple(d -> wgtTrpA(hf[d], -1, hf[d], dM), 3)
    vt = s[1] * s[2] * s[3]
    nlm = (L + 1)^2 - lLo^2
    lm = collect((lLo^2 + 1):(L + 1)^2); lv = Int[l for l in lLo:L for _ in -l:l]
    Ad = zeros(T, 3, nMax + 1, nlm); Bd = zeros(T, nMax + 1, nlm)
    Ao = ntuple(c -> zeros(T, nMax + 1, nlm), 3)
    P = bigMat(nA); U = bigMat(nA); V = bigMat(nA); tmp = BigInt(0)
    rg = S === BigFloat ? momReg() : nothing
    cmx = 1.0
    for l in lLo:L, mm in -l:l
        snf = mm < 0; m = abs(mm); p1, p2 = prtXY(m, snf)
        jj = ylmInd(l, mm) - lLo^2
        hrmPly!(P, U, V, l, m, snf, p1, p2, tmp)
        nc = T(nrmCof(l, m, BigFloat))
        for n in 0:nMax
            deg = l + 2n
            for kk in 1:7
                acc, ab = momAcc(P, W, kk, deg, p1, p2, ztl, rg)
                val = T(acc) * nc / T(vt)
                iszero(acc) || (cmx = max(cmx, Float64(ab / abs(acc))))
                kk <= 3 ? (Ad[kk, n + 1, jj] = val) :
                    kk == 4 ? (Bd[n + 1, jj] = val) : (Ao[kk - 4][n + 1, jj] = val)
            end
            if n < nMax
                mulRsq!(V, P); P, V = V, P
            end
        end
    end
    dLm, dLv, kp = cmpCol(lm, lv, [view(Ad, 1, :, :), view(Ad, 2, :, :), view(Ad, 3, :, :), Bd])
    oL = ntuple(c -> cmpCol(lm, lv, [Ao[c]]), 3)
    GeoBox{T}(L, nMax, dLm, dLv, Ad[:, :, kp], Bd[:, kp],
              ntuple(c -> oL[c][1], 3), ntuple(c -> oL[c][2], 3),
              ntuple(c -> Ao[c][:, oL[c][3]], 3), cmx)
end

struct ShpTab
    s::NTuple{3,QI}                 # b of the pair (the cell itself for equal cells)
    a::NTuple{3,QI}                 # a of the pair, 0 for equal cells
    Lw::Int
    Lo::Int
    nMax::Int
    whl::GeoBox{BigFloat}
    oct::GeoBox{BigFloat}
end

const LMAX = 56                     # largest l the selector may ask for
const LDEF = 24                     # l the table is built to before any offset asks for more
# NMAX is a FLOOR on the table's k-series order, not a cap: farSet raises it to whatever the
# n-tail bound asks for, which grows with |k| r_d.  Beyond NCAP farSet throws rather than
# returning an uncertified value.
const NMAX = 12                     # enough for every cell with a longest edge <= lambda/4
const NCAP = 40                     # the largest N the n-tail bound may ask for, |k| r_d ~ 22
const TABPRC = 192                  # precision at which the tables are stored and served
# the geometry contraction cancels by up to 2^39.4 over at most 2^11 terms, so 256 bits leaves
# a relative 2^-205, well under the 2^-192 the table is stored at
const BLDPRC = 256
const LEXT = 16                     # extra l levels summed so the "tail after L" is a real tail
# Theorem A over-states the remainder by a median 49, so 1e-14 delivers 1e-15 to 1e-14 per entry
const TOL = 1e-14                   # relative to est(R), the a priori scale of the tensor
const SHPC = Dict{Tuple{NTuple{3,QI},NTuple{3,QI},Int},ShpTab}()   # keyed (a, b, nMax)
# empty until the disk cache is asked for, and then a Scratch space
const TABDIR = Ref("")
# Scratch.jl logs every access in a plain Dict, so threads take turns resolving a directory
const SCRLK = ReentrantLock()
tabDir() = @lock SCRLK isempty(TABDIR[]) ? (TABDIR[] = @get_scratch!("shapetab")) : TABDIR[]
const SHPLK = ReentrantLock()

# the exact-zero threshold of the contraction, tied to the build precision (see momAcc)
zroTol(prc::Int) = BigFloat(2)^-(prc - 24)

shpKey(s::NTuple{3,QI}, L::Int, nMax::Int) =
    string("s", s[1].num, "_", s[1].den, "-", s[2].num, "_", s[2].den, "-",
           s[3].num, "_", s[3].den, "_L", L, "_n", nMax, "_p", TABPRC)
# the cache file holds every l-segment ever built for this shape, appended in order; an unequal
# pair (a != 0) extends the name by its a triple, so the equal-cell files keep their names
const ZQ3 = (zero(QI), zero(QI), zero(QI))
shpFil(s::NTuple{3,QI}, nMax::Int, dir::AbstractString = TABDIR[]) = shpFil(ZQ3, s, nMax, dir)
shpFil(ta::NTuple{3,QI}, tb::NTuple{3,QI}, nMax::Int, dir::AbstractString = TABDIR[]) =
    joinpath(dir, shpKey(tb, 0, nMax) *
             (all(iszero, ta) ? "" :
              string("_a", ta[1].num, "_", ta[1].den, "-", ta[2].num, "_", ta[2].den,
                     "-", ta[3].num, "_", ta[3].den)) * "_seg.txt")

function putGeo(io::IO, gb::GeoBox{BigFloat})
    println(io, gb.L, " ", gb.nMax, " ", length(gb.dLm), " ",
            join([length(gb.oLm[c]) for c in 1:3], " "), " ", gb.cnc)
    println(io, join(gb.dLm, " ")); println(io, join(gb.dLv, " "))
    for c in 1:3; println(io, join(gb.oLm[c], " ")); println(io, join(gb.oLv[c], " ")); end
    for x in gb.Ad; println(io, x); end
    for x in gb.Bd; println(io, x); end
    for c in 1:3, x in gb.Ao[c]; println(io, x); end
end

function getGeo(st::Vector{String}, p::Ref{Int})
    nx() = (p[] += 1; st[p[]])
    hd = split(nx())
    L = parse(Int, hd[1]); nM = parse(Int, hd[2]); nd = parse(Int, hd[3])
    no = ntuple(c -> parse(Int, hd[3 + c]), 3); cnc = parse(Float64, hd[7])
    dLm = parse.(Int, split(nx())); dLv = parse.(Int, split(nx()))
    oLm = Vector{Vector{Int}}(undef, 3); oLv = Vector{Vector{Int}}(undef, 3)
    for c in 1:3
        a = split(nx()); b = split(nx())
        oLm[c] = parse.(Int, a); oLv[c] = parse.(Int, b)
    end
    gv(n) = BigFloat[parse(BigFloat, nx()) for _ in 1:n]
    Ad = reshape(gv(3 * (nM + 1) * nd), 3, nM + 1, nd)
    Bd = reshape(gv((nM + 1) * nd), nM + 1, nd)
    Ao = ntuple(c -> reshape(gv((nM + 1) * no[c]), nM + 1, no[c]), 3)
    GeoBox{BigFloat}(L, nM, dLm, dLv, Ad, Bd, Tuple(oLm), Tuple(oLv), Ao, cnc)
end

# the top l a segment carries; -1 for the placeholder written when only the other table grew
segTop(g::GeoBox) = isempty(g.dLm) && all(isempty, g.oLm) ? -1 : g.L

# concatenate two l-segments of one table; both (l,m) column lists are already l-ordered.  Columns
# of b already present in a are dropped, because the cache file is appended without a lock and two
# processes extending the same table write the same segment twice.
function mrgGeo(a::GeoBox{T}, b::GeoBox{T}) where {T}
    sd = Set(a.dLm); kd = [j for j in eachindex(b.dLm) if !(b.dLm[j] in sd)]
    ko = ntuple(c -> (so = Set(a.oLm[c]); [j for j in eachindex(b.oLm[c]) if !(b.oLm[c][j] in so)]), 3)
    GeoBox{T}(max(segTop(a), segTop(b)), a.nMax, vcat(a.dLm, b.dLm[kd]), vcat(a.dLv, b.dLv[kd]),
              cat(a.Ad, b.Ad[:, :, kd]; dims = 3), hcat(a.Bd, b.Bd[:, kd]),
              ntuple(c -> vcat(a.oLm[c], b.oLm[c][ko[c]]), 3), ntuple(c -> vcat(a.oLv[c], b.oLv[c][ko[c]]), 3),
              ntuple(c -> hcat(a.Ao[c], b.Ao[c][:, ko[c]]), 3), max(a.cnc, b.cnc))
end

# geometry tables for shape s, whole box to Lw and octant to Lo, built lazily in l, appended to the
# cache file as segments, and merged back on read
farShp(s::NTuple{3,QI}; kw...) = farShp(ZQ3, s; kw...)
# unequal pair (a, b): the trapezoid whole box goes in the whl slot, and Lo < 0 skips the octant
# slot, which a trapezoid has none of; the skipped slot is written as the empty segment
function farShp(ta::NTuple{3,QI}, tb::NTuple{3,QI}; L::Int = 0, Lw::Int = LDEF, Lo::Int = LDEF,
                nMax::Int = NMAX, disk::Bool = true, exct::Bool = false, prc::Int = BLDPRC,
                dir::AbstractString = TABDIR[])
    L > 0 && (Lw = max(Lw, L); Lo = max(Lo, L))       # the pre-lazy spelling, one L for both
    lock(SHPLK) do
        ky = (ta, tb, nMax)
        tb0 = get(SHPC, ky, nothing)
        if tb0 !== nothing && tb0.Lw >= Lw && tb0.Lo >= Lo
            return tb0
        end
        disk && !exct && isempty(dir) && (dir = tabDir())
        fn = shpFil(ta, tb, nMax, dir)
        if tb0 === nothing && disk && !exct && isfile(fn)
            st = readlines(fn); p = Ref(0)
            setprecision(BigFloat, TABPRC) do
                while p[] < length(st)
                    w = getGeo(st, p); o = getGeo(st, p)
                    tb0 = tb0 === nothing ? ShpTab(tb, ta, segTop(w), segTop(o), nMax, w, o) :
                          ShpTab(tb, ta, max(tb0.Lw, segTop(w)), max(tb0.Lo, segTop(o)), nMax,
                                 mrgGeo(tb0.whl, w), mrgGeo(tb0.oct, o))
                end
            end
            tb0 !== nothing && (SHPC[ky] = tb0)
            tb0 !== nothing && tb0.Lw >= Lw && tb0.Lo >= Lo && return tb0
        end
        lw0 = tb0 === nothing || tb0.Lw < 0 ? 0 : tb0.Lw + 2
        lo0 = tb0 === nothing || tb0.Lo < 0 ? 0 : tb0.Lo + 1
        Lw = Lw < 0 ? Lw : max(Lw, tb0 === nothing ? 0 : tb0.Lw)
        Lo = Lo < 0 ? Lo : max(Lo, tb0 === nothing ? 0 : tb0.Lo)
        (lw0 > Lw && lo0 > Lo) && return tb0
        bld(aB, bB, zt) = (lw0 <= Lw ? tabWhl(aB, bB, lw0, Lw, nMax, BigFloat, zt) : nothing,
                           lo0 <= Lo ? tabOct(bB, lo0, Lo, nMax, BigFloat, zt) : nothing)
        w, o = if exct
            setprecision(() -> bld(ta, tb, zero(tb[1])), BigFloat, TABPRC)
        else
            aB = setprecision(() -> ntuple(d -> BigFloat(ta[d]), 3), BigFloat, prc)
            bB = setprecision(() -> ntuple(d -> BigFloat(tb[d]), 3), BigFloat, prc)
            wo = setprecision(() -> bld(aB, bB, zroTol(prc)), BigFloat, prc)
            setprecision(BigFloat, TABPRC) do
                (wo[1] === nothing ? nothing : cvtGeo(wo[1], BigFloat),
                 wo[2] === nothing ? nothing : cvtGeo(wo[2], BigFloat))
            end
        end
        tb0 = if tb0 === nothing
            ShpTab(tb, ta, w === nothing ? -1 : w.L, o === nothing ? -1 : o.L, nMax,
                   w === nothing ? emtGeo(nMax) : w, o === nothing ? emtGeo(nMax) : o)
        else
            ShpTab(tb, ta, w === nothing ? tb0.Lw : max(tb0.Lw, segTop(w)),
                   o === nothing ? tb0.Lo : max(tb0.Lo, segTop(o)), nMax,
                   w === nothing ? tb0.whl : mrgGeo(tb0.whl, w),
                   o === nothing ? tb0.oct : mrgGeo(tb0.oct, o))
        end
        SHPC[ky] = tb0
        if disk && !exct && !(w === nothing && o === nothing)
            try
                mkpath(dir)
                setprecision(BigFloat, TABPRC) do
                    open(fn, "a") do io
                        putGeo(io, w === nothing ? emtGeo(nMax) : w)
                        putGeo(io, o === nothing ? emtGeo(nMax) : o)
                    end
                end
            catch
            end
        end
        return tb0
    end
end

# an l-segment holding nothing, written when only one of the two tables grew
emtGeo(nMax::Int) =
    GeoBox{BigFloat}(-1, nMax, Int[], Int[], zeros(BigFloat, 3, nMax + 1, 0),
                     zeros(BigFloat, nMax + 1, 0), ntuple(_ -> Int[], 3), ntuple(_ -> Int[], 3),
                     ntuple(_ -> zeros(BigFloat, nMax + 1, 0), 3), 1.0)

cvtGeo(g::GeoBox, ::Type{T}) where {T} =
    GeoBox{T}(g.L, g.nMax, g.dLm, g.dLv, T.(g.Ad), T.(g.Bd), g.oLm, g.oLv,
              ntuple(c -> T.(g.Ao[c]), 3), g.cnc)

# j_l(k|d|) |d|^{-l} = sum_n c_ln |d|^{2n}, c_ln = (-1)^n k^{l+2n}/(2^n n! (2l+2n+1)!!)
function serCof!(c::Vector{C}, l::Int, nMax::Int, k::C) where {C}
    T = real(C)
    df = one(T); for j in 0:l; df *= T(2j + 1); end
    kl = k^l
    @inbounds for n in 0:nMax
        n > 0 && (df *= T(2 * (2l + 2n + 1) * n))
        c[n + 1] = T((-1)^n) * kl * k^(2n) / df
    end
    return c
end

# Q element type: real for real frequency (the complex sum then IS two independent real sums,
# Re from j_l and Im from y_l), complex otherwise.
struct FrqBox{E}
    L::Int
    dLm::Vector{Int}
    dLv::Vector{Int}
    Qd::Matrix{E}
    oLm::NTuple{3,Vector{Int}}
    oLv::NTuple{3,Vector{Int}}
    Qo::NTuple{3,Vector{E}}
    cnc::Float64
end

function frqBox(gb::GeoBox{BigFloat}, frq::Complex{T}, nCut::Vector{Int}) where {T}
    rl = imag(frq) == 0
    E = rl ? T : Complex{T}
    prc = max(TABPRC, precision(T) + 64)
    nd = length(gb.dLm)
    Qd = zeros(E, 3, nd); Qo = ntuple(c -> zeros(E, length(gb.oLm[c])), 3)
    cmx = 1.0
    setprecision(BigFloat, prc) do
        CB = Complex{BigFloat}
        fB = CB(BigFloat(real(frq)), BigFloat(imag(frq)))
        k = 2 * BigFloat(pi) * fB
        cb = Vector{CB}(undef, gb.nMax + 1)
        cvt(x) = rl ? E(real(x)) : E(x)
        for j in 1:nd
            l = gb.dLv[j]; nc = min(nCut[l + 1], gb.nMax)
            serCof!(cb, l, gb.nMax, k)
            sg = iseven(l) ? one(BigFloat) : -one(BigFloat)
            for a in 1:3
                s = zero(CB); sa = zero(BigFloat)
                for n in 0:nc
                    t = cb[n + 1] * (CB(gb.Ad[a, n + 1, j]) + k * k * CB(gb.Bd[n + 1, j]))
                    s += t; sa += abs(t)
                end
                iszero(s) || (cmx = max(cmx, Float64(sa / abs(s))))
                Qd[a, j] = cvt(sg * s)
            end
        end
        for c in 1:3, j in eachindex(gb.oLm[c])
            l = gb.oLv[c][j]; nc = min(nCut[l + 1], gb.nMax)
            serCof!(cb, l, gb.nMax, k)
            sg = iseven(l) ? one(BigFloat) : -one(BigFloat)
            s = zero(CB); sa = zero(BigFloat)
            for n in 0:nc
                t = cb[n + 1] * CB(gb.Ao[c][n + 1, j]); s += t; sa += abs(t)
            end
            iszero(s) || (cmx = max(cmx, Float64(sa / abs(s))))
            Qo[c][j] = cvt(sg * s)
        end
    end
    FrqBox{E}(gb.L, gb.dLm, gb.dLv, Qd, gb.oLm, gb.oLv, Qo, cmx)
end

# log |h_l^{(1)}(z)| <= log((e^{-Im z}/|z|) sum_{s=0}^{l} (l+s)!/(s!(l-s)!(2|z|)^s)).  The bound
# itself overflows at small |z| (l = 54, |z| = 7.9e-5 gives 1e309) exactly where its partner
# bslTalLog underflows, so both have to stay in logarithms throughout.
function hnkBndLog(l::Int, z::C) where {C}
    T = real(C); az = abs(z); s = one(T); lt = zero(T); mx = zero(T)
    for q in 1:l
        lt += log(T(l + q) * T(l - q + 1) / (T(q) * 2 * az))
        lt > mx && (s *= exp(mx - lt); mx = lt)      # log-sum-exp on the running maximum
        s += exp(lt - mx)
    end
    mx + log(s) - imag(z) - log(az)
end
# tail of j_l after n = N at z = |k| r_d, divided by r_d^{l+2N+2} and taken as a logarithm:
#   |sum_{n>N} c_ln z^{l+2n}| / r_d^{l+2N+2} <=
#     |k|^{l+2N+2} e^{z^2/(4l+4N+10)} / (2^{N+1} (N+1)! (2l+2N+3)!!)
# The r_d powers cancel analytically, so the vanishing power is never formed either.
function bslTalLog(l::Int, N::Int, ak::T, rd::T) where {T}
    s = (l + 2N + 2) * log(ak) + (ak * rd)^2 / T(4l + 4N + 10)
    for j in 0:(l + N + 1); s -= log(T(2j + 1)); end
    for j in 1:(N + 1); s -= log(2 * T(j)); end
    s
end

# W_l = int_box wgt(d) |d|^l dd for even l: a positive sum of box moments, no cancellation
function radMom(lTop::Int, m::NTuple{3,Vector{T}}) where {T}
    hM = div(lTop, 2)
    fac = [factorial(big(i)) for i in 0:hM]
    v = zeros(T, hM + 1)
    for h in 0:hM
        s = zero(T)
        for i in 0:h, j in 0:(h - i)
            p = h - i - j
            s += T(div(fac[h + 1], fac[i + 1] * fac[j + 1] * fac[p + 1])) *
                 m[1][i + 1] * m[2][j + 1] * m[3][p + 1]
        end
        v[h + 1] = s
    end
    v
end

# Theorem A (whole box) and Theorem B (one octant).
# Both bound the truncation this file performs: the derivatives sit on the REGULAR factor
# j_l(k|d|) Y_lm(dhat), not on h_l(kR) Y_lm(Rhat).  Two exact identities do the work:
#  (1) integration by parts onto the weight (w'' = delta_{-s} - 2 delta_0 + delta_s, w = 0 on dD),
#      so the harmonics are never differentiated and two derivatives cost 4/(s_a s_b);
#  (2) one joint Cauchy-Schwarz over m with sum_m Y_lm^2 = (2l+1)/(4 pi), which removes the
#      constant 17 and the sqrt(2l+5) of the singular-side bound.
#   |T_ab - T_ab^(L)| <= (|k|/(4 pi |f|^2 V_t)) sum_{l > L, even} (2l+1) |h_l(kR)|
#                        |k|^l e^{(|k| r_d)^2/(4l+6)}/(2l+1)!! V^{ab}_l ,
#   V^{ab}_l = int_D (s_c - |d_c|) |d|^l dd                                          (a != b)
#   V^{aa}_l = 2 int_{|d_a|=s_a} (s_b-|d_b|)(s_c-|d_c|)|d|^l + 2 int_{d_a=0} (...) |d|^l
#              + |k|^2 W_l .
# Every factor is a certified majorant: |h_l| is taken as hb(l, z) (`hnkBndS`), not as the value of
# the upward Hankel recurrence, which is accurate in practice but has no proof; hb costs a median
# 8% (at most 38%) in the selected L and buys a bound with no unproven step in it.

# hb(l, z) |k|^l/(2l+1)!! for l = 0..L, the scaling folded into each term so nothing overflows
function hnkBndS(L::Int, z::C, ak::T) where {T,C}
    az = abs(z); pre = exp(-imag(z)) / az
    out = zeros(T, L + 1)
    for l in 0:L
        trm = one(T); for q in 1:l; trm *= ak / T(2q + 1); end
        acc = trm
        for q in 1:l
            trm *= T(l + q) * T(l - q + 1) / (T(q) * 2 * az)
            acc += trm
        end
        out[l + 1] = pre * acc
    end
    out
end

# V^{ab}_l of Theorem A, split as fc[r][l/2+1] + (a == b ? |k|^2 W_l : 0); even l = 0..lTop
farMomW(lTop::Int, s::NTuple{3,T}) where {T} = farMomW(lTop, ntuple(_ -> zero(T), 3), s)
# Theorem A' (trapezoid a, b): |w'| is the flat measure on the ramps only, lam_n = 2 (b^{n+1} -
# a^{n+1})/(n+1), and |w''| = delta_{+-b} + delta_{+-a} with masses 2 b^{2i} + 2 a^{2i}; a = 0 is
# the triangle bit for bit (0^0 = 1, 0^{2i} = 0 is farMomW's zro vector)
fltMom(n::Int, a::T, b::T) where {T} = 2 * (b^(n + 1) - a^(n + 1)) / T(n + 1)
function farMomW(lTop::Int, ta::NTuple{3,T}, tb::NTuple{3,T}) where {T}
    hM = div(lTop, 2)
    tri = ntuple(d -> T[wgtX(2i, ta[d], tb[d]) for i in 0:hM], 3)
    flt = ntuple(d -> T[fltMom(2i, ta[d], tb[d]) for i in 0:hM], 3)
    pnt = ntuple(d -> T[tb[d]^(2i) for i in 0:hM], 3)
    zro = ntuple(d -> T[ta[d]^(2i) for i in 0:hM], 3)
    fc = ntuple(6) do r
        a, b = ENTIJ[r]
        if a != b
            c = 6 - a - b
            radMom(lTop, ntuple(d -> d == c ? tri[d] : flt[d], 3))
        else
            2 .* (radMom(lTop, ntuple(d -> d == a ? pnt[d] : tri[d], 3)) .+
                  radMom(lTop, ntuple(d -> d == a ? zro[d] : tri[d], 3)))
        end
    end
    (fc, radMom(lTop, tri))
end

# per-l, per-entry factors of the Theorem A tail, everything but |h_l|; even l only
function bndTrm(lTop::Int, mom, rd::T, k::C, frq::C, vt::T) where {T,C}
    fc, wr = mom
    ls = collect(0:2:lTop)
    ak = abs(k); pf = ak / (4 * T(pi) * abs2(frq) * vt)
    t = ntuple(r -> zeros(T, length(ls)), 6)
    for (i, l) in enumerate(ls)
        cf = pf * T(2l + 1) * exp((ak * rd)^2 / T(4l + 6))
        for r in 1:6
            a, b = ENTIJ[r]
            t[r][i] = cf * (fc[r][i] + (a == b ? ak^2 * wr[i] : zero(T)))
        end
    end
    (ls, t)
end

# Theorem B: on a sub-box the weight does not vanish on the boundary, so the same integration by
# parts leaves face and edge terms and, on the diagonal, ONE first derivative of j_l Y_lm, which
# sum_m |grad(f_l Y_lm)|^2 = (|k|^2/4pi)(l |f_{l-1}|^2 + (l+1)|f_{l+1}|^2) controls exactly.
# All l contribute: the octant split destroys the parity of the triangle weight.
# even-order radial moments of the four one-dimensional measures of an octant sub-box
function farMomO(lTop::Int, hf::NTuple{3,T}) where {T}
    pM = div(lTop + 2, 2)          # radQ reaches |d|^{lTop+1}
    aff = ntuple(d -> T[2 * hf[d] * hf[d]^(2i + 1) / T(2i + 1) for i in 0:pM], 3)   # (al + bt t) dt
    fac = ntuple(d -> T[2 * hf[d] * hf[d]^(2i) for i in 0:pM], 3)                   # u^+ + u^- at +-h
    ibp = ntuple(d -> fac[d] .+ T[2 * hf[d]^(2i + 1) / T(2i + 1) for i in 0:pM], 3) # + |bt| dt
    bet = ntuple(d -> T[2 * hf[d]^(2i) for i in 0:pM], 3)                           # 2|bt| at +-h
    off = ntuple(3) do c
        radMom(2pM, ntuple(d -> d == c ? aff[d] : ibp[d], 3))
    end
    gA = ntuple(a -> radMom(2pM, ntuple(d -> d == a ? fac[d] : aff[d], 3)), 3)
    bA = ntuple(a -> radMom(2pM, ntuple(d -> d == a ? bet[d] : aff[d], 3)), 3)
    (off, gA, bA, radMom(2pM, aff))
end

# |d|^q moment from an even-order table; odd q by Cauchy-Schwarz with the non-negative measure
radQ(v::Vector{T}, q::Int) where {T} =
    q < 0 ? zero(T) : iseven(q) ? v[div(q, 2) + 1] :
    sqrt(v[div(q - 1, 2) + 1] * v[div(q + 1, 2) + 1])

# per-l, per-entry factors of the Theorem B octant tail, everything but |h_l|; all l
function bndTrmO(lTop::Int, mom, rj::T, k::C, frq::C, vt::T) where {T,C}
    off, gA, bA, vA = mom
    ls = collect(0:lTop)
    ak = abs(k); ak2 = ak * ak; pf = ak / (4 * T(pi) * abs2(frq) * vt)
    t = ntuple(r -> zeros(T, lTop + 1), 6)
    for l in 0:lTop
        Ej = exp((ak * rj)^2 / T(4l + 6))
        El = exp((ak * rj)^2 / T(4l + 2)); Eu = exp((ak * rj)^2 / T(4l + 10))
        cf = pf * T(2l + 1)
        for r in 1:6
            a, b = ENTIJ[r]
            v = if a != b
                Ej * radQ(off[6 - a - b], l)
            else
                sqrt(T(l) * T(2l + 1)) * El * radQ(gA[a], l - 1) +
                    ak2 * sqrt(T(l + 1) / T(2l + 1)) / T(2l + 3) * Eu * radQ(gA[a], l + 1) +
                    Ej * radQ(bA[a], l) + ak2 * Ej * radQ(vA, l)
            end
            t[r][l + 1] = cf * v
        end
    end
    (ls, t)
end

# cumulative tail after each l, max over the six entries, continued geometrically past lTop
function bndCum(ls::Vector{Int}, t::NTuple{6,Vector{T}}, k::C, rr::T) where {T,C}
    n = length(ls)
    hs = hnkBndS(ls[n], k * rr, abs(k))
    cum = zeros(T, n); acc = zeros(T, 6); ext = zero(T)
    for r in 1:6
        p = t[r][n - 1] * hs[ls[n - 1] + 1]; q = t[r][n] * hs[ls[n] + 1]
        (isnan(q) || isinf(q)) && return fill(T(Inf), n)
        iszero(p) && continue
        rt = q / p
        rt >= 1 && return fill(T(Inf), n)
        ext = max(ext, q * rt / (1 - rt))
    end
    for i in n:-1:1
        cum[i] = maximum(acc) + ext
        for r in 1:6; acc[r] += t[r][i] * hs[ls[i] + 1]; end
    end
    cum
end

# a priori magnitude of the far tensor: the pointwise dyadic scale times (1/V_t) int_D w = V_t
est(rr::T, k::C, frq::C, vt::T) where {T,C} =
    vt * abs(k)^2 * exp(-imag(k) * rr) / (4 * T(pi) * abs2(frq) * rr) *
    (1 + 3 / abs(k * rr) + 3 / abs(k * rr)^2)

# smallest l <= lTop in ls whose cumulative tail is below bud; -1 if none is
function pikCut(ls::Vector{Int}, cum::Vector{T}, bud::T, lTop::Int) where {T}
    for i in eachindex(ls)
        ls[i] > lTop && break
        cum[i] <= bud && return ls[i]
    end
    return -1
end

const SGN8 = ((1, 1, 1), (-1, 1, 1), (1, -1, 1), (1, 1, -1),
              (-1, -1, 1), (-1, 1, -1), (1, -1, -1), (-1, -1, -1))

struct FrqSet{T,E}
    s::NTuple{3,T}
    sQ::NTuple{3,QI}
    frq::Complex{T}
    k::Complex{T}
    prf::Complex{T}
    L::Int                          # largest l the selector may ask for
    Lw::Int                         # l the whole-box table actually holds
    Lo::Int                         # l the octant table actually holds
    whl::FrqBox{E}
    oct::FrqBox{E}
    ctr::NTuple{3,T}                # octant centre (all components positive)
    thr::Vector{Float64}            # thr[i] = smallest |R| for which whole-box L = 2(i-1) suffices
    wLs::Vector{Int}                # l values of the whole-box bound terms
    wT::NTuple{6,Vector{Float64}}   # per-entry l-only factors of the whole-box bound
    oLs::Vector{Int}                # l values of the sub-box bound terms
    oT::NTuple{6,Vector{Float64}}   # per-entry l-only factors of the sub-box bound
    e0::Float64                     # leading factor of the proven lower bound lowVal
    eT::Vector{Float64}             # its l >= 2 tail terms, indexed like wLs
    scl::Symbol                     # :est (a priori scale) or :low (proven lower bound)
    tol::Float64
    nCut::Vector{Int}                # per-l k-series order the n-tail bound certifies
    nNd::Int                         # max(nCut): the order the tables must carry
    nMx::Int                         # the order they do carry
    rNc::Float64                     # radii between which nCut is certified: |R| on route 1,
    rHi::Float64                     # the octant vertex radii on route 2
    ksr::Dict{NTuple{3,Int},Matrix{Complex{T}}}
    ksrI::Dict{NTuple{3,Int},Tuple{Int,Float64}}   # (N, Lambda) of each route-(iii) tensor
    lk::ReentrantLock                # guards ksr, ksrI and the route-(iii) cache file
end

# whole-box tail/est is a function of |R| alone and is monotone in |R|: tabulate the thresholds
function whlThr(ls::Vector{Int}, t::NTuple{6,Vector{T}}, k::C, scf, tol::T,
                rLo::T, rHi::T, lTop::Int) where {T,C}
    n = count(<=(lTop), ls)
    ok(r, i) = (cum = bndCum(ls, t, k, r); cum[i] <= tol * scf(r))
    thr = fill(T(Inf), n)
    for i in 1:n
        ok(rHi, i) || continue
        if ok(rLo, i)
            thr[i] = rLo; continue
        end
        a = rLo; b = rHi
        for _ in 1:200
            c = (a + b) / 2
            ok(c, i) ? (b = c) : (a = c)
            b - a <= eps(T) * b * 8 && break
        end
        thr[i] = b
    end
    for i in (n - 1):-1:1
        thr[i] = max(thr[i], thr[i + 1])
    end
    thr
end

# Truncating j_l's own series after n = N is Theorem A again, with |k|^l e^{..}/(2l+1)!! replaced
# by exp(bslTalLog(l, N, |k|, r_d)) and V^{ab}_l by V^{ab}_{l+2N+2}.  The budget is
# tol est/(NBUD (lTop+1)): tol is spent on the l-truncation, and nCut only certifies that the
# stored orders are enough.  The bound is evaluated at both ends of the radius range the expansion
# is actually used over, since its ratio to est falls with |R| for l >= 2 and rises for l = 0.
const NBUD = 256                    # so the whole n-truncation costs at most tol/256 of est
# per-l n-truncation cut: nCut[l+1] is the smallest N <= nCap meeting the budget, or nCap + 1 when
# none does.  Every entry is an order to carry rather than a status code, so two cuts merge under
# max. and frqBox's min(nCut[l+1], nMax) degrades to "contract every stored order" rather than to
# an empty sum.
cutVec(lTop::Int, mom, s::NTuple{3,T}, k::C, frq::C, kR::C, tol::T, nCap::Int) where {T,C} =
    cutVec(lTop, mom, prod(s), prod(s), sqrt(sum(s[i]^2 for i in 1:3)), k, frq, kR, tol, nCap)
# vt = V_t (the 1/V_t prefactor), vs = (1/V_t) int_D w (the scale of est), rd = analyticity radius
function cutVec(lTop::Int, mom, vt::T, vs::T, rd::T, k::C, frq::C, kR::C, tol::T,
                nCap::Int) where {T,C}
    fc, wr = mom
    ak = abs(k); lpf = log(ak / (4 * T(pi) * abs2(frq) * vt))
    lbud = log(tol * est(abs(kR) / ak, k, frq, vs) / (T(NBUD) * T(lTop + 1)))
    v = fill(nCap + 1, lTop + 1)
    for l in 0:lTop
        lhl = hnkBndLog(l, kR)
        for N in 0:nCap
            q = l + 2N + 2; iq = div(q, 2) + 1
            iq <= length(wr) || break
            vm = zero(T)
            for r in 1:6
                a, b = ENTIJ[r]
                vm = max(vm, fc[r][iq] + (a == b ? ak^2 * wr[iq] : zero(T)))
            end
            lb = lpf + log(T(2l + 1)) + lhl + bslTalLog(l, N, ak, rd) + log(vm)
            if lb <= lbud; v[l + 1] = N; break; end
        end
    end
    v
end

# lattice offsets whose whole-box bound asks for more than lDef shells; the table is sized on them.
# off is |sub-cell shift| per axis, and a shifted offset is smallest -- so worst -- at n s - off
function nerOff(thr::Vector{T}, s::NTuple{3,T}, lDef::Int, nMin::Int, nBlk::Int, rHi::T,
                off::NTuple{3,T}) where {T}
    ic = div(lDef, 2) + 1
    rCr = min(ic <= length(thr) ? thr[ic] : T(Inf), rHi)
    nc = ntuple(d -> min(nBlk - 1, floor(Int, (rCr + off[d]) / s[d]) + 1), 3)
    ds = NTuple{3,Int}[]
    for n3 in 0:nc[3], n2 in 0:nc[2], n1 in 0:nc[1]
        maximum((n1, n2, n3)) >= nMin || continue
        nerRad((n1, n2, n3), s, off) < rCr && push!(ds, (n1, n2, n3))
    end
    ds
end
nerRad(D, s::NTuple{3,T}, off::NTuple{3,T}) where {T} =
    sqrt(sum(max(zero(T), abs(T(D[d])) * s[d] - off[d])^2 for d in 1:3))

# frequency-contracted tables, thresholds and route cache for shape s at frequency f
function farSet(sQ::NTuple{3,QI}, frq::Complex{T}; tol::Float64 = TOL,
                L::Int = LMAX, nMax::Int = NMAX, nMin::Int = 2, nBlk::Int = 128,
                lDef::Int = LDEF, offs = nothing, scl::Symbol = :est, off = ZR3,
                disk::Bool = true, nCap::Int = NCAP, dir::AbstractString = TABDIR[]) where {T}
    s = ntuple(d -> T(sQ[d]), 3)
    k = 2 * T(pi) * frq
    # the bound, the thresholds and the n-cut are selection arithmetic, done in Float64 whatever
    # T is: in Float32 (2l+1)!! overflows at l = 29 and the NaN poisons every tail.
    s6 = ntuple(d -> Float64(sQ[d]), 3); f6 = ComplexF64(frq); k6 = 2 * pi * f6
    vt6 = prod(s6); rd6 = sqrt(sum(s6[i]^2 for i in 1:3))
    rLo = nMin * minimum(s6); rHi = 4 * nBlk * rd6
    momW = farMomW(L + LEXT + 2 * nMax + 2, s6)
    wLs, wT = bndTrm(L + LEXT, momW, rd6, k6, f6, vt6)
    e0, eT = lowScl(wLs, momW, rd6, k6, f6, vt6)
    scf = scl === :low ? (r -> lowVal(wLs, e0, eT, k6, r)) : (r -> est(r, k6, f6, vt6))
    thr = whlThr(wLs, wT, k6, scf, tol, rLo, rHi, L)
    hf6 = ntuple(d -> s6[d] / 2, 3); rj6 = sqrt(sum(hf6[i]^2 for i in 1:3))
    oLs, oT = bndTrmO(L + LEXT, farMomO(L + LEXT, hf6), rj6, k6, f6, vt6)
    Lw = min(lDef, L); Lo = 0; rNc = rHi
    ds = offs === nothing ? nerOff(thr, s6, min(lDef, L), nMin, nBlk, rHi, off) :
         [ntuple(d -> abs(D[d]), 3) for D in offs]
    for D in ds
        R = ntuple(d -> max(0.0, Float64(D[d]) * s6[d] - off[d]), 3)
        rr = sqrt(sum(R[i]^2 for i in 1:3))
        Lc = whlLvl(thr, rr)
        if Lc >= 0
            Lw = max(Lw, Lc); rNc = min(rNc, rr)
        else
            bud = tol * scf(rr) / 8
            for sg in SGN8
                v = ntuple(d -> sg[d] * R[d] + hf6[d], 3)
                vr = sqrt(sum(v[i]^2 for i in 1:3))
                q = pikCut(oLs, bndCum(oLs, oT, k6, vr), bud, L)
                q >= 0 && (Lo = max(Lo, q); rNc = min(rNc, vr))
            end
        end
    end
    rNc == rHi && isfinite(thr[end]) && (rNc = min(rNc, thr[end]))   # no offset certified anything
    # the n-tail grows as (|k| r_d)^{2N+2}, so the order the table must carry rises with cell size
    # in wavelengths.  nMax is only a floor, so a table already on disk at NMAX is reused; the cut
    # is recomputed against nCap, and the moment table widened, only when the floor falls short.
    # Only the shells this set can ask for need certifying: no admissible radius selects more than
    # lCut, since thr is non-increasing and rngChk holds |R| >= rNc, the radius that fixed Lw.
    lCut = max(Lw, Lo)
    cut(mm, nc) = max.(cutVec(L, mm, s6, k6, f6, k6 * rNc, tol, nc),
                       cutVec(L, mm, s6, k6, f6, k6 * rHi, tol, nc))[1:(lCut + 1)]
    nCut = cut(momW, nMax)
    if maximum(nCut) > nMax
        momW = farMomW(max(L + LEXT, lCut + 2 * nCap + 2), s6)
        nCut = cut(momW, nCap)
        # the n-tail stops decreasing only once (|k| r_d)^2 > 2 nCap (2 lCut + 2 nCap)
        maximum(nCut) > nCap &&
            error("farSet: the k-series of j_l is not certified within N = $nCap at |k| r_d = ",
                  abs(k6) * rd6, " (cell ", s6, ", f = ", f6, "); ",
                  abs(k6) * rd6 > sqrt(2 * nCap * (2 * lCut + 2 * nCap)) ?
                  "the cell is too large in wavelengths for this expansion" :
                  "this is not a cell-size limit -- raise nCap")
    end
    nNd = maximum(nCut)
    tb = farShp(sQ; Lw = Lw, Lo = Lo, nMax = max(nMax, nNd), disk = disk, dir = dir)
    tb.nMax >= nNd || error("farSet: the n-tail bound needs N = $nNd, the table holds $(tb.nMax)")
    # a cached table can carry more shells than this selector may ask for (the thresholds stop at
    # L); frqBox contracts every stored column, so nCut is padded to cover them
    lT = max(L, tb.Lw, tb.Lo)
    length(nCut) < lT + 1 && (nCut = vcat(nCut, fill(nCut[end], lT + 1 - length(nCut))))
    hf = ntuple(d -> s[d] / 2, 3)
    prf = imag(frq) == 0 ? Complex(zero(T), 2 * T(pi) / real(frq)) : im * k / frq^2
    E = imag(frq) == 0 ? T : Complex{T}
    FrqSet{T,E}(s, sQ, frq, k, prf, L, tb.Lw, tb.Lo,
                frqBox(tb.whl, frq, nCut), frqBox(tb.oct, frq, nCut),
                hf, thr, wLs, wT, oLs, oT, e0, eT, scl, tol, nCut, nNd, tb.nMax, rNc, rHi,
                Dict{NTuple{3,Int},Matrix{Complex{T}}}(),
                Dict{NTuple{3,Int},Tuple{Int,Float64}}(), ReentrantLock())
end

# throw when the certified k-series order exceeds what the geometry table carries
@inline chkCut(fs::FrqSet) = fs.nNd <= fs.nMx ? nothing :
    error("chkCut: the n-tail bound needs N = $(fs.nNd), the table holds $(fs.nMx)")

# tol = 45 eps, which is 9.992e-15 at Float64: below the 16.4 ulp max-norm floor of the expansion,
# so tightening it only adds terms
farTol(::Type{T}) where {T} = 45 * Float64(eps(T))

# A set is certified only on the radius interval [rNc, rHi = 4 nBlk r_d] the offsets it was BUILT
# for fix, so nBlk and nMin belong in the key: one built for a 32^3 block and reused at 128^3
# returns a wrong tensor under a valid-looking certificate.
const ZR3 = (0.0, 0.0, 0.0)
const FRQC = Dict{Tuple{NTuple{3,QI},DataType,Any,Float64,Int,Int,NTuple{3,Float64}},FrqSet}()
const FRQLK = ReentrantLock()

farSetMem(sQ::NTuple{3,QI}, frq::Complex{T}; tol::Float64 = TOL, nBlk::Int = 128,
          nMin::Int = 2, off = ZR3, disk::Bool = true,
          dir::AbstractString = TABDIR[]) where {T} =
    lock(FRQLK) do
        get!(() -> farSet(sQ, frq; tol = tol, nBlk = nBlk, nMin = nMin, off = off,
                          disk = disk, dir = dir),
             FRQC, (sQ, T, frq, tol, nBlk, nMin, off))::FrqSet{T}
    end

# whole-box l for offset radius rr from the tabulated thresholds; -1 if more than LMAX is needed
function whlLvl(thr::Vector{Float64}, rr::Real)
    r6 = Float64(rr)
    @inbounds for i in eachindex(thr)
        r6 >= thr[i] && return 2 * (i - 1)
    end
    return -1
end
whlLvl(fs, rr::Real) = whlLvl(fs.thr, rr)   # FrqSet or FrqSetX

# the Theorem A bound itself on max_ab |T_ab - T_ab^(L)| at radius rr
bndWhl(fs::FrqSet{T}, rr::T, L::Int) where {T} =
    (c = bndCum(fs.wLs, fs.wT, ComplexF64(fs.k), Float64(rr)); i = div(L, 2) + 1;
     i <= length(c) ? c[i] : 0.0)

# per-sub-box l for the octant split at offset R; -1 in a slot means LMAX is not enough
function octLvl(fs::FrqSet{T}, R::NTuple{3,T}) where {T}
    R6 = ntuple(d -> Float64(R[d]), 3); hf = ntuple(d -> Float64(fs.ctr[d]), 3)
    k6 = ComplexF64(fs.k)
    rr = sqrt(sum(R6[i]^2 for i in 1:3))
    bud = fs.tol * estScl(fs, rr) / 8
    Lc = zeros(Int, 8)
    for (q, sg) in enumerate(SGN8)
        v = ntuple(d -> sg[d] * R6[d] + hf[d], 3)
        vr = sqrt(sum(v[i]^2 for i in 1:3))
        Lc[q] = pikCut(fs.oLs, bndCum(fs.oLs, fs.oT, k6, vr), bud, fs.L)
    end
    Lc
end

# A proven lower bound on max_ab |T_ab|, from the trace.  Helmholtz gives lap g = -k^2 g away
# from the source, so tr T = 2 k^2 S with S = (1/V_t) int_D w g(R+d) dd, and
# max_ab |T_ab| >= |tr T|/3 >= (2|k|^2/3)(|S_0| - tailS), where S_0 is the l = 0 shell and tailS
# the Theorem A sum with V^{ab}_l replaced by W_l and no derivative factor.  It needs
# |k| r_d <= pi (j_0 positive and decreasing there) and the leading shell to survive its tail, and
# returns 0 otherwise -- which it does at every offset of a lambda/4 cell.  Hence scl = :est.
# leading factor and l >= 2 tail terms of the proven lower bound on max_ab |T_ab|
function lowScl(ls::Vector{Int}, mom, rd::T, k::C, frq::C, vt::T) where {T,C}
    fc, wr = mom
    ak = abs(k); pf = ak / (4 * T(pi) * abs2(frq) * vt)
    ak * rd > T(pi) && return (zero(T), zeros(T, length(ls)))
    j0 = ak * rd < eps(T) ? one(T) : sin(ak * rd) / (ak * rd)
    eT = zeros(T, length(ls))
    for (i, l) in enumerate(ls)
        l == 0 && continue
        eT[i] = pf * T(2l + 1) * exp((ak * rd)^2 / T(4l + 6)) * wr[i]
    end
    (ak * vt * j0 / (4 * T(pi) * abs2(frq)), eT)
end

# the lower bound itself at radius rr; 0 when the leading shell does not survive its own tail
function lowVal(ls::Vector{Int}, e0::T, eT::Vector{T}, k::C, rr::T) where {T,C}
    iszero(e0) && return zero(T)
    ak = abs(k)
    hs = hnkBndS(ls[end], k * rr, ak)
    tl = zero(T)
    for i in eachindex(ls); tl += eT[i] * hs[ls[i] + 1]; end
    2 * ak^2 * max(e0 * exp(-imag(k) * rr) / (ak * rr) - tl, zero(T)) / 3
end

# the scale the selector divides by: est by default, else the proven lower bound
estScl(fs::FrqSet, rr::Real) =
    fs.scl === :low ? lowVal(fs.wLs, fs.e0, fs.eT, ComplexF64(fs.k), Float64(rr)) :
    est(Float64(rr), ComplexF64(fs.k), ComplexF64(fs.frq), Float64(prod(fs.s)))

# Terms actually summed, one complex multiply-add each, plus the Y_lm recursion.  Diagnostic only,
# and off by default: routing short-circuits to route 1 whenever route 1 converges, so the two
# costs are never compared, and evaluating them costs 30% of a build.
function cstWhl(fb::FrqBox, L::Int)
    c = (L + 1)^2
    for l in fb.dLv; l <= L && (c += 3); end
    for cc in 1:3, l in fb.oLv[cc]; l <= L && (c += 1); end
    c
end
cstOct(fb::FrqBox, Lc::Vector{Int}) = sum(L -> L < 0 ? 0 : 7 * (L + 1)^2, Lc)

# Throw for a radius outside [rNc, rHi], where this set's k-series cut nCut is certified: a set reused
# outside its interval returned a wrong tensor under a valid-looking certificate.  A set built by
# farTns or farBlk! covers its own offsets, so this can fire only for a hand-built or reused one.
@inline chkRad(fs::FrqSet, rLo::Float64, rUp::Float64, R) =
    (fs.rNc * (1 - 1e-12) <= rLo && rUp <= fs.rHi * (1 + 1e-12)) ? nothing :
    error("farRte: offset $R has radius $rLo..$rUp outside [$(fs.rNc), $(fs.rHi)], where this ",
          "set's n-cut is certified; build the set with this offset in offs")

# route for one offset: (kind, L, Lc, cost); kind 1 = whole box, 2 = octants, 3 = k-series
function farRte(fs::FrqSet{T}, R::NTuple{3,T}; cst::Bool = false) where {T}
    rr = sqrt(sum(R[i]^2 for i in 1:3))
    L = whlLvl(fs, rr)
    if L >= 0
        L <= fs.Lw || error("farRte: offset $R needs whole-box l = $L, table holds $(fs.Lw)")
        chkRad(fs, Float64(rr), Float64(rr), R)
        return (1, L, Int[], cst ? cstWhl(fs.whl, L) : 0)
    end
    Lc = octLvl(fs, R)
    any(<(0), Lc) && return (3, -1, Lc, 0)
    maximum(Lc) <= fs.Lo || error("farRte: offset $R needs octant l = $(maximum(Lc)), table holds $(fs.Lo)")
    vs = extrema(sqrt(sum(Float64(sg[d] * R[d] + fs.ctr[d])^2 for d in 1:3)) for sg in SGN8)
    chkRad(fs, vs[1], vs[2], R)
    (2, -1, Lc, cst ? cstOct(fs.oct, Lc) : 0)
end
farRte(fs::FrqSet{T}, D::NTuple{3,Int}; cst::Bool = false) where {T} =
    farRte(fs, ntuple(d -> T(D[d]) * fs.s[d], 3); cst = cst)

# double-double arithmetic on pairs of Float64, for the phase argument reduction of phsSed
@inline function twoSum(a::Float64, b::Float64)
    s = a + b; bb = s - a
    s, (a - (s - bb)) + (b - bb)
end
@inline function twoPrd(a::Float64, b::Float64)
    p = a * b
    p, fma(a, b, -p)
end
@inline function ddfAdd(x::NTuple{2,Float64}, y::NTuple{2,Float64})
    s, e = twoSum(x[1], y[1]); e += x[2] + y[2]
    twoSum(s, e)
end
@inline function ddfMul(x::NTuple{2,Float64}, y::NTuple{2,Float64})
    p, e = twoPrd(x[1], y[1]); e += x[1] * y[2] + x[2] * y[1]
    twoSum(p, e)
end

# |R| as a double-double from the exact squares of the components
function ddfNrm(R::NTuple{3,Float64})
    q = (0.0, 0.0)
    for i in 1:3; q = ddfAdd(q, twoPrd(R[i], R[i])); end
    r0 = sqrt(q[1])
    p, e = twoPrd(r0, r0)
    res = ddfAdd(q, (-p, -e))
    twoSum(r0, res[1] / (2 * r0) + res[2] / (2 * r0))
end

# (|R|, e^{2 pi i f |R|}); the phase argument is reduced in double-double for Float64
function phsSed(R::NTuple{3,T}, frq::Complex{T}) where {T}
    rr = sqrt(R[1]^2 + R[2]^2 + R[3]^2)
    (rr, exp(im * 2 * T(pi) * frq * rr))
end
function phsSed(R::NTuple{3,Float64}, frq::ComplexF64)
    rho = ddfNrm(R)
    fr = real(frq); fi = imag(frq)
    th = ddfMul((2 * fr, 0.0), rho)
    m = 2 * round(th[1] / 2)
    t = ddfAdd(th, (-m, 0.0))
    ph = cispi(t[1]) * cispi(t[2])
    x = ddfMul((-2 * pi * fi, 0.0), rho)
    (rho[1], exp(x[1]) * (1 + x[2]) * ph)
end
phsSed(R::NTuple{3,Float32}, frq::ComplexF32) =
    (r = phsSed(ntuple(d -> Float64(R[d]), 3), ComplexF64(frq)); (Float32(r[1]), ComplexF32(r[2])))

# h_l^{(1)}(z) by the upward recurrence from h_0 = -i e^{iz}/z; e = e^{iz} is supplied
function hnkRec!(h::Vector{C}, z::C, e::C, L::Int) where {C<:Complex}
    T = real(C)
    iz = inv(z); ez = e * iz
    h[1] = -im * ez
    L >= 1 && (h[2] = -(one(C) + im * iz) * ez)
    @inbounds for l in 1:(L - 1)
        h[l + 2] = T(2l + 1) * iz * h[l + 1] - h[l]
    end
    return h
end

# y_l(x) by the upward recurrence, its dominant solution, from y_0 = -cos(x)/x
function bslY!(y::Vector{T}, x::T, cs::T, sn::T, L::Int) where {T}
    ix = inv(x)
    y[1] = -cs * ix
    L >= 1 && (y[2] = y[1] * ix - sn * ix)
    @inbounds for l in 1:(L - 1)
        y[l + 2] = T(2l + 1) * ix * y[l + 1] - y[l]
    end
    return y
end

# j_l(x) by the downward (Miller) recurrence, normalised on the larger of j_0, j_1
function bslJR!(j::Vector{T}, x::T, cs::T, sn::T, L::Int) where {T}
    dg = 0.30103 * precision(T)
    lt = L + 10 + ceil(Int, 2 * abs(x) + 1.6 * sqrt(dg) * sqrt(Float64(L + 1) + abs(Float64(x))) + dg)
    bg = ldexp(one(T), div(exponent(floatmax(T)), 2))
    ix = inv(x)
    jp = zero(T); jc = one(T) / bg
    @inbounds for l in lt:-1:1
        jm = T(2l + 1) * ix * jc - jp
        jp = jc; jc = jm
        l <= L && (j[l + 1] = jp)
        if abs(jc) > bg
            jc /= bg; jp /= bg
            for q in (l + 1):(L + 1); j[q] /= bg; end
        end
    end
    j[1] = jc
    j0 = sn * ix; j1 = j0 * ix - cs * ix
    sc = abs(j0) >= abs(j1) ? j0 / j[1] : j1 / j[2]
    @inbounds for l in 1:(L + 1); j[l] *= sc; end
    return j
end

# per-thread workspace of one offset evaluation
struct FarWrk{T}
    L::Int
    Y::Vector{T}
    H::Matrix{Complex{T}}
    h::Vector{Complex{T}}
    jr::Vector{T}
    yr::Vector{T}
end
FarWrk(L::Int, ::Type{T}) where {T} =
    FarWrk{T}(L, zeros(T, (L + 1)^2), zeros(Complex{T}, L + 1, L + 1),
              zeros(Complex{T}, L + 1), zeros(T, L + 1), zeros(T, L + 1))

# h_l(kR) into ws.h; for real k the real part is j_l by Miller and the imaginary part y_l upward
function hnkFil!(ws::FarWrk{T}, rr::T, e::Complex{T}, k::Complex{T}, L::Int) where {T}
    h = ws.h
    if imag(k) == 0
        x = real(k) * rr
        cs = real(e); sn = imag(e)
        bslJR!(ws.jr, x, cs, sn, L); bslY!(ws.yr, x, cs, sn, L)
        @inbounds for l in 0:L; h[l + 1] = Complex(ws.jr[l + 1], ws.yr[l + 1]); end
    else
        hnkRec!(h, k * rr, e, L)
    end
    return h
end

# real spherical harmonics Y_lm (no Condon-Shortley phase) up to L, by the normalised recursion
function shmFil!(ws::FarWrk{T}, u::T, v::T, w::T, L::Int) where {T}
    H = ws.H; Y = ws.Y
    s2 = sqrt(T(2))
    H[1, 1] = Complex{T}(1 / sqrt(4 * T(pi)), zero(T))
    uv = Complex{T}(u, v)
    @inbounds for m in 1:L
        H[m + 1, m + 1] = sqrt(T(2m + 1) / T(2m)) * uv * H[m, m]
    end
    @inbounds for m in 0:L
        m + 1 <= L && (H[m + 2, m + 1] = sqrt(T(2m + 3)) * w * H[m + 1, m + 1])
        for l in (m + 2):L
            a = sqrt(T(4l * l - 1) / T(l * l - m * m))
            b = sqrt(T((l - 1)^2 - m * m) / T(4 * (l - 1)^2 - 1))
            H[l + 1, m + 1] = a * (w * H[l, m + 1] - b * H[l - 1, m + 1])
        end
    end
    @inbounds for l in 0:L
        Y[ylmInd(l, 0)] = real(H[l + 1, 1])
        for m in 1:l
            Y[ylmInd(l, m)] = s2 * real(H[l + 1, m + 1])
            Y[ylmInd(l, -m)] = s2 * imag(H[l + 1, m + 1])
        end
    end
    return Y
end

@inline function accOne(h::Vector{H}, Y::Vector{T}, lm::Vector{Int}, lv::Vector{Int},
                        Q::Vector{E}, Lc::Int) where {H,T,E}
    s = zero(promote_type(H, E))
    @inbounds for j in eachindex(lm)
        lv[j] > Lc && break
        s += h[lv[j] + 1] * (Y[lm[j]] * Q[j])
    end
    return s
end

# the six sums sum_{lm} h_l(k|v|) Y_lm(vhat) Q^{ab}_{lm}, for entries 11, 22, 33, 12, 13, 23
function boxAcc(fb::FrqBox{E}, ws::FarWrk{T}, v::NTuple{3,T}, frq::Complex{T}, k::Complex{T},
                L::Int) where {T,E}
    rr, e = phsSed(v, frq)
    shmFil!(ws, v[1] / rr, v[2] / rr, v[3] / rr, L)
    hnkFil!(ws, rr, e, k, L)
    Y = ws.Y; h = ws.h; lm = fb.dLm; lv = fb.dLv; Qd = fb.Qd
    a1 = zero(Complex{T}); a2 = zero(Complex{T}); a3 = zero(Complex{T})
    @inbounds for j in eachindex(lm)
        lv[j] > L && break
        hy = h[lv[j] + 1] * Y[lm[j]]
        a1 += hy * Qd[1, j]; a2 += hy * Qd[2, j]; a3 += hy * Qd[3, j]
    end
    (a1, a2, a3,
     accOne(h, Y, fb.oLm[1], fb.oLv[1], fb.Qo[1], L),
     accOne(h, Y, fb.oLm[2], fb.oLv[2], fb.Qo[2], L),
     accOne(h, Y, fb.oLm[3], fb.oLv[3], fb.Qo[3], L))
end

# route 1: the whole difference box, one expansion about R
function tnsWhl!(G::AbstractMatrix{Complex{T}}, fs::FrqSet{T}, ws::FarWrk{T},
                 R::NTuple{3,T}, L::Int) where {T}
    a = boxAcc(fs.whl, ws, R, fs.frq, fs.k, L)
    p = fs.prf
    @inbounds for r in 1:6
        i, j = ENTIJ[r]
        G[i, j] = p * a[r]; G[j, i] = G[i, j]
    end
    G
end

# route 2: the 8 octants of D, reflected from one table by T^{(sg)}_ab(V) = sg_a sg_b T_ab(sg V)
function tnsOct!(G::AbstractMatrix{Complex{T}}, fs::FrqSet{T}, ws::FarWrk{T},
                 R::NTuple{3,T}, Lc::Vector{Int}) where {T}
    @inbounds for i in 1:9; G[i] = zero(Complex{T}); end
    c = fs.ctr; p = fs.prf
    @inbounds for (q, sg) in enumerate(SGN8)
        L = Lc[q]; L < 0 && continue
        v = (T(sg[1]) * R[1] + c[1], T(sg[2]) * R[2] + c[2], T(sg[3]) * R[3] + c[3])
        a = boxAcc(fs.oct, ws, v, fs.frq, fs.k, L)
        for r in 1:6
            i, j = ENTIJ[r]
            sn = r <= 3 ? 1 : sg[i] * sg[j]
            G[i, j] += T(sn) * p * a[r]
        end
    end
    @inbounds for r in 4:6
        i, j = ENTIJ[r]; G[j, i] = G[i, j]
    end
    G
end

# empty until route 3 is first reached, and then a Scratch space
const KSRDIR = Ref("")
ksrDir() = @lock SCRLK isempty(KSRDIR[]) ? (KSRDIR[] = @get_scratch!("ksrcache")) : KSRDIR[]
const KSRPRC = 128                  # the 36 signed face pairs cancel; route 3 needs the headroom
const MOMC = Dict{Tuple{NTuple{3,Int},NTuple{3,QI}},Tuple{Int,Vector{Vector{BigFloat}}}}()
const KSRLK = ReentrantLock()           # guards MOMC and MOMCX

# max and min |x - y| over an axis-aligned panel pair
function panSpn(A, B)
    T = typeof(A[1][1])
    hi = zero(T); lo = zero(T)
    for d in 1:3
        a1, a2 = A[d]; b1, b2 = B[d]
        hi += max(abs(a1 - b2), abs(a2 - b1))^2
        g = max(b1 - a2, a1 - b2, zero(T))
        lo += g^2
    end
    (sqrt(hi), sqrt(lo))
end

# |R_N|/|I_{-1}/(4 pi f^2)| <= x^{N+1}/(N+1)! min(e^x, 1/(1 - x/(N+2))), x = |k| Dmax
function ksrBnd(x::T, N::Int) where {T}
    lg = (N + 1) * log(x) - sum(log, one(T):T(N + 1))
    tail = x < N + 2 ? -log1p(-x / (N + 2)) : x
    exp(lg + tail)
end
function ksrOrd(x::T, tol::Real, nCap::Int) where {T}
    for N in 0:nCap
        ksrBnd(x, N) <= tol && return N
    end
    return -1
end

# |R_N| <= (I_{-1}/(4 pi |f|^2)) x^{N+1}/(N+1)! B(x,N); the assembled tensor picks up the static
# amplification Lambda = (sum_fp I_{-1})/(4 pi |f|^2 V_t)/est(R), so N is chosen for tol/Lambda.
# route 3: the 36 face-pair k-series in BigFloat(KSRPRC), assembled by the srfSum! signs
function tnsKsr(sQ::NTuple{3,QI}, D::NTuple{3,Int}, frq::Complex{T}; tol::Float64 = 1e-16,
                nCap::Int = 200) where {T}
    out = setprecision(BigFloat, KSRPRC) do
        CB = Complex{BigFloat}
        bf(p) = map(iv -> (BigFloat(iv[1]), BigFloat(iv[2])), p)
        pnB = [map(bf, parFac(D, F, Fp, sQ)) for F in 1:6 for Fp in 1:6]
        vtB = prod(ntuple(d -> BigFloat(sQ[d]), 3))
        fB = CB(BigFloat(real(frq)), BigFloat(imag(frq)))
        kk = 2 * BigFloat(pi) * fB
        dmx = maximum(first(panSpn(p[1], p[2])) for p in pnB)
        lam = sum(parMom(p[1], p[2], -1)[1] for p in pnB)
        rr = sqrt(sum((BigFloat(D[d]) * BigFloat(sQ[d]))^2 for d in 1:3))
        amp = lam / (4 * BigFloat(pi) * abs2(fB) * vtB) / est(rr, kk, fB, vtB)
        N = ksrOrd(abs(kk) * dmx, tol / max(1.0, Float64(amp)), nCap)
        N < 0 && error("tnsKsr: the k-series bound is not met within $nCap terms at D = $D")
        ky = (D, sQ)
        # the moments are frequency independent and expensive; read and write MOMC under a lock,
        # but compute outside it, so two threads on two offsets do not serialize
        mm = lock(() -> get(MOMC, ky, (-1, Vector{Vector{BigFloat}}())), KSRLK)
        if mm[1] < N + 1
            mm = (N + 1, [parMom(p[1], p[2], N + 1) for p in pnB])
            lock(() -> (MOMC[ky] = mm), KSRLK)
        end
        f2 = 4 * BigFloat(pi) * fB^2
        srf = Vector{CB}(undef, 36)
        for i in 1:36
            m = mm[2][i]
            cof = one(CB); acc = zero(CB)
            for n in 0:N
                acc += cof * m[n + 1]
                cof *= im * kk / (n + 1)
            end
            srf[i] = acc / f2 / vtB
        end
        gb = zeros(CB, 3, 3); srfSum!(gb, srf)
        (gb, N, Float64(amp))
    end
    G = Complex{T}[Complex{T}(T(real(out[1][i, j])), T(imag(out[1][i, j]))) for i in 1:3, j in 1:3]
    (G, out[2], out[3], out[1])
end

# route 3 tensor at D from the memory or disk cache, else computed and stored; a cache line is
# "d1 d2 d3 N Lambda re im ..." (23 tokens), and the 21-token lines of an older format read too
function ksrCch!(fs::FrqSet{T}, D::NTuple{3,Int}) where {T}
    lock(fs.lk)
    try
        haskey(fs.ksr, D) && return fs.ksr[D]
    finally
        unlock(fs.lk)
    end
    fn = joinpath(ksrDir(), string("ksr_", shpKey(fs.sQ, 0, 0), "_f",
                                   Float64(real(fs.frq)), "_", Float64(imag(fs.frq)), ".txt"))
    if isfile(fn)
        lock(fs.lk)
        try
            for ln in eachline(fn)
                tk = split(ln); nt = length(tk)
                (nt == 21 || nt == 23) || continue
                nw = nt == 23 ? 2 : 0
                Dk = (parse(Int, tk[1]), parse(Int, tk[2]), parse(Int, tk[3]))
                haskey(fs.ksr, Dk) && continue
                fs.ksr[Dk] = reshape([Complex{T}(T(parse(BigFloat, tk[2 + nw + 2i])),
                                                 T(parse(BigFloat, tk[3 + nw + 2i]))) for i in 1:9], 3, 3)
                fs.ksrI[Dk] = nw == 2 ? (parse(Int, tk[4]), parse(Float64, tk[5])) : (-1, NaN)
            end
            haskey(fs.ksr, D) && return fs.ksr[D]
        finally
            unlock(fs.lk)
        end
    end
    G, N, amp, GB = tnsKsr(fs.sQ, D, fs.frq)
    lock(fs.lk)
    try
        fs.ksr[D] = G; fs.ksrI[D] = (N, amp)
        try
            mkpath(ksrDir())
            setprecision(BigFloat, KSRPRC) do
                open(fn, "a") do io
                    print(io, D[1], " ", D[2], " ", D[3], " ", N, " ", amp)
                    for i in 1:9; print(io, " ", real(GB[i]), " ", imag(GB[i])); end
                    println(io)
                end
            end
        catch
        end
    finally
        unlock(fs.lk)
    end
    G
end

# true for the egoToe indices farBlk! fills: max(i) >= 3, i.e. max-norm separation >= 2 cells
@inline farInd(i::NTuple{3,Int}) = maximum(i) >= 3
@inline farInd(i1::Int, i2::Int, i3::Int) = max(i1, i2, i3) >= 3

# far tensor for one offset D in cells of shape s at frequency f; max-norm separation >= 2
# required.  A test entry point: generation fills whole blocks with farBlk!
function farTns(D::NTuple{3,Int}, s::NTuple{3,<:Rational}, f::Complex{T};
                tol::Float64 = TOL, fs::Union{Nothing,FrqSet} = nothing,
                dir::AbstractString = TABDIR[]) where {T}
    maximum(abs, D) >= 2 || error("farTns: max-norm separation $(maximum(abs,D)) <= 1")
    sQ = ntuple(d -> QI(s[d]), 3)
    st = fs === nothing ?
         farSet(sQ, f; tol = tol, nBlk = max(4, maximum(abs, D)), offs = (D,), dir = dir) : fs
    chkCut(st)
    ws = FarWrk(max(st.Lw, st.Lo), T)
    G = zeros(Complex{T}, 3, 3)
    R = ntuple(d -> T(D[d]) * st.s[d], 3)
    kind, L, Lc, _ = farRte(st, D)
    kind == 1 && return tnsWhl!(G, st, ws, R, L)
    kind == 2 && return tnsOct!(G, st, ws, R, Lc)
    copyto!(G, ksrCch!(st, D))
    G
end

# Fill dst(q), a 3x3 view, with the tensor at the offset Ds[q] cells of shape s plus the sub-cell
# shift off an external pair off the cell lattice carries.  Threaded; route 3 is exact-rational in
# the cell, so an offset that needs it under a nonzero off has no route and throws.
function farFil!(dst::F, Ds::Vector{NTuple{3,Int}}, sQ::NTuple{3,QI}, f::Complex{T},
                 off::NTuple{3,T} = ntuple(_ -> zero(T), 3); tol::Float64 = TOL,
                 fs::Union{Nothing,FrqSet} = nothing, disk::Bool = true,
                 dir::AbstractString = TABDIR[]) where {F,T}
    isempty(Ds) && return nothing
    zro = all(iszero, off)
    st = fs === nothing ?
         farSetMem(sQ, f; tol = tol, nBlk = maximum(D -> maximum(abs, D), Ds) + 1,
                   nMin = zro ? 2 : 1, off = ntuple(d -> abs(Float64(off[d])), 3),
                   disk = disk, dir = dir) : fs
    chkCut(st)
    pos(q) = ntuple(d -> T(Ds[q][d]) * st.s[d] + off[d], 3)
    kn = Vector{Int}(undef, length(Ds)); lw = Vector{Int}(undef, length(Ds))
    lo = Vector{Vector{Int}}(undef, length(Ds))
    Threads.@threads :static for q in eachindex(Ds)
        kind, L, Lc, _ = farRte(st, pos(q))
        kn[q] = kind; lw[q] = L; lo[q] = Lc
    end
    # Route 3 dominates a cold build wherever it is reached, so it is threaded; the state it
    # shares (MOMC, fs.ksr, fs.ksrI, the cache file) is behind KSRLK and fs.lk.  The ambient
    # BigFloat precision is set once here because setprecision is process-global: every nested
    # setprecision inside tnsKsr then saves and restores the same KSRPRC.
    ksq = [q for q in eachindex(Ds) if kn[q] == 3]
    if !isempty(ksq)
        zro || error("farFil!: offset $(Ds[ksq[1]]) .+ $off is off the cell lattice and too ",
                     "close for the shifted expansion, where the k-series route has no ",
                     "exact-rational panel pair; put the two grids on a common lattice, or ",
                     "separate them by a further cell")
        setprecision(BigFloat, KSRPRC) do
            Threads.@threads :static for j in eachindex(ksq)
                ksrCch!(st, Ds[ksq[j]])
            end
        end
    end
    wss = [FarWrk(max(st.Lw, st.Lo), T) for _ in 1:Threads.maxthreadid()]   # threadid() is a global id
    Threads.@threads :static for q in eachindex(Ds)
        ws = wss[Threads.threadid()]
        G = dst(q)
        if kn[q] == 1
            tnsWhl!(G, st, ws, pos(q), lw[q])
        elseif kn[q] == 2
            tnsOct!(G, st, ws, pos(q), lo[q])
        else
            copyto!(G, st.ksr[Ds[q]])
        end
    end
    return nothing
end

# fill egoToe[:, :, i1, i2, i3] for every index with max(i) >= 3, at the offset (i .- 1) .* stp
# cells: stp is the volume's grid step in cells, which exceeds one on a partitioned volume
function farBlk!(egoToe::AbstractArray{Complex{T},5}, s::NTuple{3,<:Rational}, f::Complex{T};
                 tol::Float64 = TOL, fs::Union{Nothing,FrqSet} = nothing, disk::Bool = true,
                 stp::NTuple{3,Int} = (1, 1, 1), dir::AbstractString = TABDIR[]) where {T}
    n1, n2, n3 = size(egoToe, 3), size(egoToe, 4), size(egoToe, 5)
    Ds = NTuple{3,Int}[]
    for i3 in 1:n3, i2 in 1:n2, i1 in 1:n1
        farInd(i1, i2, i3) &&
            push!(Ds, ((i1 - 1) * stp[1], (i2 - 1) * stp[2], (i3 - 1) * stp[3]))
    end
    farFil!(Ds, ntuple(d -> QI(s[d]), 3), f; tol = tol, fs = fs, disk = disk, dir = dir) do q
        D = Ds[q]
        view(egoToe, :, :, div(D[1], stp[1]) + 1, div(D[2], stp[2]) + 1, div(D[3], stp[3]) + 1)
    end
    egoToe
end

# G(R; sT, sS) = (1/V_t) int_D w(d) [(da db + dab k^2) g](R+d) dd with the trapezoid weight of the
# pair, D = prod [-b_i, b_i], V_t = prod sT; (1/V_t) int_D w = V_s, so est takes V_s while the
# Theorem A'/B prefactor keeps V_t.  The whole-box table is keyed on (a, b), symmetric in the
# pair, and stored divided by prod(b), with prod(b)/V_t folded into prf.  Routes per exact offset
# R of the target centre, source at 0:
#   1  the whole trapezoid box (Theorem A', r_d = |b|, ramp and point measures), one expansion;
#   2  the gcd average G(R) = (1/N_t) sum_{j,j'} G(R + c_j - c'_j'; g, g), g = min(sT, sS) per
#      axis: N_t x N_s equal-cell tensors at integer offsets in cells of g, since Gila's grids sit
#      on the common gcd lattice.  farTnsX sums farTns, farBlkX! takes a box sum over one fine
#      egoToe filled by farBlk!;
#   3  the k-series on the 36 unequal face pairs, off the lattice or when a sub-offset would touch.
# Touching pairs (|R_i| <= b_i on every axis) are Gila's contact path and throw here.
trpHlf(sT::NTuple{3,S}, sS::NTuple{3,S}) where {S} =
    (ntuple(d -> abs(sT[d] - sS[d]) / 2, 3), ntuple(d -> (sT[d] + sS[d]) / 2, 3))

struct FrqSetX{T,E}
    sT::NTuple{3,T}
    sS::NTuple{3,T}
    sTQ::NTuple{3,QI}
    sSQ::NTuple{3,QI}
    tbQ::NTuple{3,QI}               # b = (sT + sS)/2 per axis, exact
    b6::NTuple{3,Float64}
    gQ::NTuple{3,QI}                # the gcd cell min(sT, sS) per axis
    g6::NTuple{3,Float64}
    o6::NTuple{3,Float64}           # latPos: R = m g + o, o = (nT - nS) g/2
    nT::NTuple{3,Int}               # target sub-cells per axis
    nS::NTuple{3,Int}               # source sub-cells per axis
    frq::Complex{T}
    k::Complex{T}
    prf::Complex{T}                 # i k/f^2 times prod(b)/V_t
    L::Int
    Lw::Int                         # l the whole-box table holds
    whl::FrqBox{E}
    thr::Vector{Float64}
    wLs::Vector{Int}
    wT::NTuple{6,Vector{Float64}}
    tol::Float64
    nCut::Vector{Int}
    nBlk::Int
    rNc::Float64                    # radii between which the whole-box n-cut is certified
    rHi::Float64
    disk::Bool
    dir::String
    fine::Base.RefValue{Union{Nothing,FrqSet{T,E}}}   # equal-cell set of g, built on first use
    ksr::Dict{NTuple{3,QI},Matrix{Complex{T}}}
    ksrI::Dict{NTuple{3,QI},Tuple{Int,Float64}}
    lk::ReentrantLock                # guards fine, ksr, ksrI and the route-3 cache file
end

# frequency-contracted trapezoid table, thresholds and caches of the pair (sT, sS) at f
function farSetX(sTQ::NTuple{3,QI}, sSQ::NTuple{3,QI}, frq::Complex{T}; tol::Float64 = TOL,
                 L::Int = LMAX, nMax::Int = NMAX, nBlk::Int = 128, lDef::Int = LDEF, offs = (),
                 disk::Bool = true, nCap::Int = NCAP,
                 dir::AbstractString = TABDIR[]) where {T}
    all(isinteger(sTQ[d] / sSQ[d]) || isinteger(sSQ[d] / sTQ[d]) for d in 1:3) ||
        error("farSetX: the edges must be integer multiples per axis (Gila's GlaExtInf rule)")
    taQ, tbQ = trpHlf(sTQ, sSQ)
    gQ = ntuple(d -> min(sTQ[d], sSQ[d]), 3)
    nT = ntuple(d -> Int(sTQ[d] / gQ[d]), 3); nS = ntuple(d -> Int(sSQ[d] / gQ[d]), 3)
    sT = ntuple(d -> T(sTQ[d]), 3); sS = ntuple(d -> T(sSQ[d]), 3)
    k = 2 * T(pi) * frq
    a6 = ntuple(d -> Float64(taQ[d]), 3); b6 = ntuple(d -> Float64(tbQ[d]), 3)
    f6 = ComplexF64(frq); k6 = 2 * pi * f6
    vt6 = prod(ntuple(d -> Float64(sTQ[d]), 3)); vs6 = prod(ntuple(d -> Float64(sSQ[d]), 3))
    rd6 = sqrt(sum(b6[i]^2 for i in 1:3))
    rLo = minimum(b6); rHi = 4 * nBlk * rd6
    momW = farMomW(L + LEXT + 2 * nMax + 2, a6, b6)
    wLs, wT = bndTrm(L + LEXT, momW, rd6, k6, f6, vt6)
    thr = whlThr(wLs, wT, k6, r -> est(r, k6, f6, vs6), tol, rLo, rHi, L)
    # the table is sized on the offsets that take route 1, and the n-cut certified at the nearest of
    # them; without offsets, at the nearest radius route 1 can be taken at all (thr[end])
    Lw = min(lDef, L); rNc = rHi
    for R in offs
        R6 = ntuple(d -> Float64(R[d]), 3); rr = sqrt(sum(R6[i]^2 for i in 1:3))
        Lc = whlLvl(thr, rr)
        Lc >= 0 && (Lw = max(Lw, Lc); rNc = min(rNc, rr))
    end
    rNc == rHi && isfinite(thr[end]) && (rNc = min(rNc, thr[end]))   # no route-1 offset given
    cut(mm, nc) = max.(cutVec(L, mm, vt6, vs6, rd6, k6, f6, k6 * rNc, tol, nc),
                       cutVec(L, mm, vt6, vs6, rd6, k6, f6, k6 * rHi, tol, nc))
    nCut = cut(momW, nMax)
    if maximum(nCut) > nMax
        momW = farMomW(max(L + LEXT, L + 2 * nCap + 2), a6, b6)
        nCut = cut(momW, nCap)
        maximum(nCut) > nCap &&
            error("farSetX: the k-series of j_l is not certified within N = $nCap at |k| r_d = ",
                  abs(k6) * rd6, " (pair ", sT, " / ", sS, ", f = ", f6, ")")
    end
    nNd = maximum(nCut)
    tb = farShp(taQ, tbQ; Lw = Lw, Lo = -1, nMax = max(nMax, nNd), disk = disk, dir = dir)
    tb.nMax >= nNd || error("farSetX: the n-tail bound needs N = $nNd, the table holds $(tb.nMax)")
    lT = max(L, tb.Lw)
    length(nCut) < lT + 1 && (nCut = vcat(nCut, fill(nCut[end], lT + 1 - length(nCut))))
    prf = (imag(frq) == 0 ? Complex(zero(T), 2 * T(pi) / real(frq)) : im * k / frq^2) *
          T(prod(tbQ) / prod(sTQ))
    E = imag(frq) == 0 ? T : Complex{T}
    g6 = ntuple(d -> Float64(gQ[d]), 3)
    o6 = ntuple(d -> Float64((nT[d] - nS[d]) * gQ[d] // 2), 3)
    FrqSetX{T,E}(sT, sS, sTQ, sSQ, tbQ, b6, gQ, g6, o6, nT, nS, frq, k, prf, L, tb.Lw, frqBox(tb.whl, frq, nCut),
                 thr, wLs, wT, tol, nCut, nBlk, rNc, rHi, disk, String(dir),
                 Ref{Union{Nothing,FrqSet{T,E}}}(nothing),
                 Dict{NTuple{3,QI},Matrix{Complex{T}}}(),
                 Dict{NTuple{3,QI},Tuple{Int,Float64}}(), ReentrantLock())
end

# A cross-scale set is certified on [rNc, rHi = 4 nBlk r_d] and carries a table sized to l = Lw,
# both fixed by the offsets it was BUILT for, so nBlk belongs in the key and a hit is handed on
# only where it already covers -- see farSetMem.
const FRQCX = Dict{Tuple{NTuple{3,QI},NTuple{3,QI},DataType,Any,Float64,Int},FrqSetX}()

function covOff(fx::FrqSetX, offs)
    for R in offs
        rr = sqrt(sum(Float64(R[d])^2 for d in 1:3))
        L = whlLvl(fx, rr)
        L < 0 && continue
        (L <= fx.Lw && fx.rNc * (1 - 1e-12) <= rr <= fx.rHi * (1 + 1e-12)) || return false
    end
    true
end

farSetXMem(sTQ::NTuple{3,QI}, sSQ::NTuple{3,QI}, frq::Complex{T}; tol::Float64 = TOL,
           nBlk::Int = 128, offs = (), kw...) where {T} =
    lock(FRQLK) do
        ky = (sTQ, sSQ, T, frq, tol, nBlk)
        fx = get(FRQCX, ky, nothing)
        (fx === nothing || !covOff(fx, offs)) &&
            (fx = farSetX(sTQ, sSQ, frq; tol = tol, nBlk = nBlk, offs = offs, kw...);
             FRQCX[ky] = fx)
        fx::FrqSetX{T}
    end

# the equal-cell set of the gcd cell g, built on first use and shared by every route-2 evaluation
function finSet(fx::FrqSetX{T,E}; nBlk::Int = fx.nBlk) where {T,E}
    lock(fx.lk) do
        fx.fine[] === nothing &&
            (fx.fine[] = farSet(fx.gQ, fx.frq; tol = fx.tol, nBlk = max(fx.nBlk, nBlk),
                                disk = fx.disk, dir = fx.dir))
        fx.fine[]
    end
end

# integer gcd-lattice coordinate m of R, in cells of g; nothing when R is off that lattice
function latInd(fx::FrqSetX, RQ::NTuple{3,QI})
    m = ntuple(d -> RQ[d] / fx.gQ[d] + QI(fx.nS[d] - fx.nT[d]) // 2, 3)
    all(isinteger, m) ? ntuple(d -> Int(m[d]), 3) : nothing
end
# the offset vector of the gcd-lattice coordinate m, R = m g + (nT - nS) g/2; inverse of latInd
latPos(fx::FrqSetX, m::NTuple{3,Int}) = ntuple(d -> m[d] * fx.g6[d] + fx.o6[d], 3)
# multiplicity of the sub-offset difference t = j - j' over j in 1:nT, j' in 1:nS
@inline subMlt(t::Int, nT::Int, nS::Int) = min(nT, t + nS) - max(1, t + 1) + 1
# smallest max-norm over the sub-offsets m + j - j'; the equal-cell routes need it >= 2
function subSep(fx::FrqSetX, m::NTuple{3,Int})
    maximum(ntuple(d -> max(0, m[d] + 1 - fx.nS[d], -(m[d] + fx.nT[d] - 1)), 3))
end

errTchX(x) = error("farRteX: the cells at R = $x touch or overlap (Gila's contact path)")
# the cells touch or overlap when |R_d| <= b_d on every axis; exact in both representations
tchX(fx::FrqSetX, m::NTuple{3,Int}) =
    all(abs(2 * m[d] + fx.nT[d] - fx.nS[d]) <= fx.nT[d] + fx.nS[d] for d in 1:3)
# the rational test is behind a Float64 screen: Rational{BigInt} arithmetic allocates, and a block
# has 1e6 offsets of which a few hundred touch
tchX(fx::FrqSetX, RQ::NTuple{3,QI}) =
    all(abs(Float64(RQ[d])) <= fx.b6[d] * (1 + 1e-9) for d in 1:3) &&
    all(abs(RQ[d]) <= fx.tbQ[d] for d in 1:3)

# route for one offset: (kind, L); 1 whole trapezoid box at l = L, 2 gcd average, 3 k-series
function farRteX(fx::FrqSetX, R6::NTuple{3,Float64}, m::Union{Nothing,NTuple{3,Int}})
    rr = sqrt(sum(R6[i]^2 for i in 1:3))
    L = whlLvl(fx, rr)
    if L >= 0
        L <= fx.Lw || error("farRteX: offset $R6 needs whole-box l = $L, table holds $(fx.Lw)")
        # the k-series cut is certified on [rNc, rHi] only, and a set reused outside that
        # interval is silently wrong; a set from farTnsX or farBlkX! covers its own offsets
        fx.rNc * (1 - 1e-12) <= rr <= fx.rHi * (1 + 1e-12) ||
            error("farRteX: |R| = $rr is outside [$(fx.rNc), $(fx.rHi)], where this set's n-cut is ",
                  "certified; build the set with this offset in offs")
        return (1, L)
    end
    (m !== nothing && subSep(fx, m) >= 2) ? (2, -1) : (3, -1)
end
farRteX(fx::FrqSetX, m::NTuple{3,Int}) =
    (tchX(fx, m) && errTchX(m); farRteX(fx, latPos(fx, m), m))
farRteX(fx::FrqSetX, RQ::NTuple{3,QI}) =
    (tchX(fx, RQ) && errTchX(RQ); farRteX(fx, ntuple(d -> Float64(RQ[d]), 3), latInd(fx, RQ)))
farRteX(fx::FrqSetX, R::NTuple{3,<:Rational}) = farRteX(fx, ntuple(d -> QI(R[d]), 3))

# route 1: the whole trapezoid box, one expansion about R
function tnsWhlX!(G::AbstractMatrix{Complex{T}}, fx::FrqSetX{T}, ws::FarWrk{T}, R::NTuple{3,T},
                  L::Int) where {T}
    a = boxAcc(fx.whl, ws, R, fx.frq, fx.k, L)
    p = fx.prf
    @inbounds for r in 1:6
        i, j = ENTIJ[r]
        G[i, j] = p * a[r]; G[j, i] = G[i, j]
    end
    G
end

# the Theorem A' bound on max_ab |T_ab - T_ab^(L)| of the whole trapezoid box at radius rr
bndWhlX(fx::FrqSetX, rr::Real, L::Int) =
    (c = bndCum(fx.wLs, fx.wT, ComplexF64(fx.k), Float64(rr)); i = div(L, 2) + 1;
     i <= length(c) ? c[i] : 0.0)

const KSRTOL = 1e-16                # route 3's own tolerance: remainder <= KSRTOL est(R)
# certified absolute bound on max_ab |T_ab - G_ab| of the equal-cell tensor at D, as routed
function certEq(fs::FrqSet{T}, D::NTuple{3,Int}, kind::Int, L::Int, Lc::Vector{Int}) where {T}
    R6 = ntuple(d -> Float64(D[d]) * Float64(fs.s[d]), 3); rr = sqrt(sum(R6[i]^2 for i in 1:3))
    k6 = ComplexF64(fs.k)
    kind == 1 && return bndWhl(fs, T(rr), L)
    kind == 3 && return KSRTOL * est(rr, k6, ComplexF64(fs.frq), Float64(prod(fs.s)))
    hf = ntuple(d -> Float64(fs.ctr[d]), 3); c = 0.0
    for (q, sg) in enumerate(SGN8)
        v = ntuple(d -> sg[d] * R6[d] + hf[d], 3); vr = sqrt(sum(v[i]^2 for i in 1:3))
        c += bndCum(fs.oLs, fs.oT, k6, vr)[Lc[q] + 1]
    end
    c
end

# route 2: (1/N_t) sum of the reflected equal-cell tensors at |D|; returns the summed certificate
function tnsGcd!(G::AbstractMatrix{Complex{T}}, fx::FrqSetX{T}, fs::FrqSet{T}, ws::FarWrk{T},
                 m::NTuple{3,Int}, tmp::AbstractMatrix{Complex{T}}, cert::Bool) where {T}
    @inbounds for i in 1:9; G[i] = zero(Complex{T}); end
    nT = fx.nT; nS = fx.nS; c = 0.0
    for t3 in (1 - nS[3]):(nT[3] - 1), t2 in (1 - nS[2]):(nT[2] - 1), t1 in (1 - nS[1]):(nT[1] - 1)
        w = T(subMlt(t1, nT[1], nS[1]) * subMlt(t2, nT[2], nS[2]) * subMlt(t3, nT[3], nS[3]))
        D = (m[1] + t1, m[2] + t2, m[3] + t3)
        Da = ntuple(d -> abs(D[d]), 3)
        kind, L, Lc, _ = farRte(fs, Da)
        if kind == 3
            copyto!(tmp, ksrCch!(fs, Da))
        else
            Ra = ntuple(d -> T(Da[d]) * fs.s[d], 3)
            kind == 1 ? tnsWhl!(tmp, fs, ws, Ra, L) : tnsOct!(tmp, fs, ws, Ra, Lc)
        end
        @inbounds for r in 1:6
            i, j = ENTIJ[r]
            sn = (D[i] < 0 ? -1 : 1) * (D[j] < 0 ? -1 : 1)
            G[i, j] += (w * T(sn)) * tmp[i, j]
        end
        cert && (c += Float64(w) * certEq(fs, Da, kind, L, Lc))
    end
    nt = T(prod(nT))
    @inbounds for r in 1:6
        i, j = ENTIJ[r]; G[i, j] /= nt; G[j, i] = G[i, j]
    end
    c / prod(nT)
end

# face F of the target (edges sT at R) and face F' of the source (edges sS at 0) as panels
parFacX(RQ::NTuple{3,QI}, sTQ::NTuple{3,QI}, sSQ::NTuple{3,QI}, F::Int, Fp::Int) =
    (boxFace(FACES[F]..., RQ, sTQ), boxFace(FACES[Fp]..., ZQ3, sSQ))

const MOMCX = Dict{Tuple{NTuple{3,QI},NTuple{3,QI},NTuple{3,QI}},Tuple{Int,Vector{Vector{BigFloat}}}}()
# route 3: the 36 unequal face-pair k-series in BigFloat(KSRPRC), srfSum! signs, over V_t
function tnsKsrX(sTQ::NTuple{3,QI}, sSQ::NTuple{3,QI}, RQ::NTuple{3,QI}, frq::Complex{T};
                 tol::Float64 = KSRTOL, nCap::Int = 200) where {T}
    out = setprecision(BigFloat, KSRPRC) do
        CB = Complex{BigFloat}
        bf(p) = map(iv -> (BigFloat(iv[1]), BigFloat(iv[2])), p)
        pnB = [map(bf, parFacX(RQ, sTQ, sSQ, F, Fp)) for F in 1:6 for Fp in 1:6]
        vtB = prod(ntuple(d -> BigFloat(sTQ[d]), 3)); vsB = prod(ntuple(d -> BigFloat(sSQ[d]), 3))
        fB = CB(BigFloat(real(frq)), BigFloat(imag(frq)))
        kk = 2 * BigFloat(pi) * fB
        dmx = maximum(first(panSpn(p[1], p[2])) for p in pnB)
        lam = sum(parMom(p[1], p[2], -1)[1] for p in pnB)
        rr = sqrt(sum(BigFloat(RQ[d])^2 for d in 1:3))
        amp = lam / (4 * BigFloat(pi) * abs2(fB) * vtB) / est(rr, kk, fB, vsB)
        N = ksrOrd(abs(kk) * dmx, tol / max(1.0, Float64(amp)), nCap)
        N < 0 && error("tnsKsrX: the k-series bound is not met within $nCap terms at R = $RQ")
        ky = (RQ, sTQ, sSQ)
        mm = lock(() -> get(MOMCX, ky, (-1, Vector{Vector{BigFloat}}())), KSRLK)
        if mm[1] < N + 1
            mm = (N + 1, [parMom(p[1], p[2], N + 1) for p in pnB])
            lock(() -> (MOMCX[ky] = mm), KSRLK)
        end
        f2 = 4 * BigFloat(pi) * fB^2
        srf = Vector{CB}(undef, 36)
        for i in 1:36
            m = mm[2][i]
            cof = one(CB); acc = zero(CB)
            for n in 0:N
                acc += cof * m[n + 1]
                cof *= im * kk / (n + 1)
            end
            srf[i] = acc / f2 / vtB
        end
        gb = zeros(CB, 3, 3); srfSum!(gb, srf)
        (gb, N, Float64(amp))
    end
    G = Complex{T}[Complex{T}(T(real(out[1][i, j])), T(imag(out[1][i, j]))) for i in 1:3, j in 1:3]
    (G, out[2], out[3], out[1])
end

# route 3 tensor at RQ from the memory or disk cache, else computed and stored; a cache line is
# "r1n r1d r2n r2d r3n r3d N Lambda re im ..." (26 tokens), one file per ordered pair
function ksrCchX!(fx::FrqSetX{T}, RQ::NTuple{3,QI}) where {T}
    lock(fx.lk)
    try
        haskey(fx.ksr, RQ) && return fx.ksr[RQ]
    finally
        unlock(fx.lk)
    end
    fn = joinpath(ksrDir(), string("ksrx_", shpKey(fx.sTQ, 0, 0), "_", shpKey(fx.sSQ, 0, 0), "_f",
                                   Float64(real(fx.frq)), "_", Float64(imag(fx.frq)), ".txt"))
    if isfile(fn)
        lock(fx.lk)
        try
            for ln in eachline(fn)
                tk = split(ln); length(tk) == 26 || continue
                Rk = ntuple(d -> QI(parse(BigInt, tk[2d - 1]), parse(BigInt, tk[2d])), 3)
                haskey(fx.ksr, Rk) && continue
                fx.ksr[Rk] = reshape([Complex{T}(T(parse(BigFloat, tk[7 + 2i])),
                                                 T(parse(BigFloat, tk[8 + 2i]))) for i in 1:9], 3, 3)
                fx.ksrI[Rk] = (parse(Int, tk[7]), parse(Float64, tk[8]))
            end
            haskey(fx.ksr, RQ) && return fx.ksr[RQ]
        finally
            unlock(fx.lk)
        end
    end
    G, N, amp, GB = tnsKsrX(fx.sTQ, fx.sSQ, RQ, fx.frq)
    lock(fx.lk)
    try
        fx.ksr[RQ] = G; fx.ksrI[RQ] = (N, amp)
        try
            mkpath(ksrDir())
            setprecision(BigFloat, KSRPRC) do
                open(fn, "a") do io
                    for d in 1:3; print(io, numerator(RQ[d]), " ", denominator(RQ[d]), " "); end
                    print(io, N, " ", amp)
                    for i in 1:9; print(io, " ", real(GB[i]), " ", imag(GB[i])); end
                    println(io)
                end
            end
        catch
        end
    finally
        unlock(fx.lk)
    end
    G
end

# block size in fine cells that covers the offsets Rs and their sub-offsets
function finBlk(Rs, gQ::NTuple{3,QI}, nT::NTuple{3,Int}, nS::NTuple{3,Int})
    g6 = ntuple(d -> Float64(gQ[d]), 3)
    max(4, maximum(ceil(Int, abs(Float64(R[d])) / g6[d]) for R in Rs for d in 1:3; init = 0) +
           maximum(nT) + maximum(nS))
end

# far tensor of the pair (target edges sT centred at R, source edges sS at 0) at f; with
# cert = true it also returns the certified bound on max_ab |T_ab - G_ab|
function farTnsX(R::NTuple{3,<:Rational}, sT::NTuple{3,<:Rational}, sS::NTuple{3,<:Rational},
                 f::Complex{T}; tol::Float64 = TOL, fs::Union{Nothing,FrqSetX} = nothing,
                 cert::Bool = false, dir::AbstractString = TABDIR[]) where {T}
    RQ = ntuple(d -> QI(R[d]), 3); sTQ = ntuple(d -> QI(sT[d]), 3); sSQ = ntuple(d -> QI(sS[d]), 3)
    fx = if fs === nothing
        gQ = ntuple(d -> min(sTQ[d], sSQ[d]), 3)
        # farSetX raises on this too, but only after Int() below would have thrown
        all(isinteger(sTQ[d] / sSQ[d]) || isinteger(sSQ[d] / sTQ[d]) for d in 1:3) ||
            error("farTnsX: the edges must be integer multiples per axis (Gila's GlaExtInf rule)")
        nT = ntuple(d -> Int(sTQ[d] / gQ[d]), 3); nS = ntuple(d -> Int(sSQ[d] / gQ[d]), 3)
        farSetX(sTQ, sSQ, f; tol = tol, nBlk = finBlk((RQ,), gQ, nT, nS), offs = (RQ,), dir = dir)
    else
        fs
    end
    (fx.sTQ == sTQ && fx.sSQ == sSQ && fx.frq == f) ||
        error("farTnsX: fs was built for the pair $(fx.sTQ) / $(fx.sSQ) at f = $(fx.frq)")
    kind, L = farRteX(fx, RQ)
    G = zeros(Complex{T}, 3, 3)
    Rr = ntuple(d -> T(RQ[d]), 3); rr = sqrt(sum(Float64(RQ[i])^2 for i in 1:3))
    c = 0.0
    if kind == 1
        tnsWhlX!(G, fx, FarWrk(fx.Lw, T), Rr, L)
        cert && (c = bndWhlX(fx, rr, L))
    elseif kind == 2
        fg = finSet(fx); chkCut(fg)
        c = tnsGcd!(G, fx, fg, FarWrk(max(fg.Lw, fg.Lo), T), latInd(fx, RQ), zeros(Complex{T}, 3, 3), cert)
    else
        copyto!(G, ksrCchX!(fx, RQ))
        cert && (c = KSRTOL * est(Float64(rr), ComplexF64(fx.k), ComplexF64(fx.frq), Float64(prod(fx.sS))))
    end
    cert || return G
    (G, c + 20 * eps(T) * Float64(maximum(abs, G)))
end

# the box sum (1/N_t) sum_t mult(t) T(m + t) over the fine egoToe, reflected; returns the
# summed certificate from cg
function boxSum!(G::AbstractMatrix{Complex{T}}, ego::AbstractArray{Complex{T},5},
                 cg::Array{Float64,3}, m::NTuple{3,Int}, nT::NTuple{3,Int},
                 nS::NTuple{3,Int}) where {T}
    c = 0.0
    g11 = zero(Complex{T}); g22 = g11; g33 = g11; g12 = g11; g13 = g11; g23 = g11
    @inbounds for t3 in (1 - nS[3]):(nT[3] - 1)
        w3 = T(subMlt(t3, nT[3], nS[3])); d3 = m[3] + t3; a3 = abs(d3) + 1; s3 = d3 < 0 ? -one(T) : one(T)
        for t2 in (1 - nS[2]):(nT[2] - 1)
            w2 = w3 * T(subMlt(t2, nT[2], nS[2])); d2 = m[2] + t2; a2 = abs(d2) + 1; s2 = d2 < 0 ? -one(T) : one(T)
            for t1 in (1 - nS[1]):(nT[1] - 1)
                w = w2 * T(subMlt(t1, nT[1], nS[1])); d1 = m[1] + t1; a1 = abs(d1) + 1; s1 = d1 < 0 ? -one(T) : one(T)
                g11 += w * ego[1, 1, a1, a2, a3]; g22 += w * ego[2, 2, a1, a2, a3]; g33 += w * ego[3, 3, a1, a2, a3]
                g12 += (w * s1 * s2) * ego[1, 2, a1, a2, a3]; g13 += (w * s1 * s3) * ego[1, 3, a1, a2, a3]
                g23 += (w * s2 * s3) * ego[2, 3, a1, a2, a3]
                c += Float64(w) * cg[a1, a2, a3]
            end
        end
    end
    nt = T(prod(nT))
    @inbounds begin
        G[1, 1] = g11 / nt; G[2, 2] = g22 / nt; G[3, 3] = g33 / nt
        G[1, 2] = g12 / nt; G[2, 1] = G[1, 2]; G[1, 3] = g13 / nt; G[3, 1] = G[1, 3]
        G[2, 3] = g23 / nt; G[3, 2] = G[2, 3]
    end
    c / prod(nT)
end

# fill G[:, :, q] for every exact offset Rs[q] of the pair, threaded, the near band as one box
# sum over a fine egoToe; returns the route kinds, and with cert = true the certificates
function farBlkX!(G::AbstractArray{Complex{T},3}, Rs::AbstractVector{NTuple{3,QI}},
                  sT::NTuple{3,<:Rational}, sS::NTuple{3,<:Rational}, f::Complex{T};
                  tol::Float64 = TOL, fs::Union{Nothing,FrqSetX} = nothing, cert::Bool = false,
                  disk::Bool = true, dir::AbstractString = TABDIR[]) where {T}
    n = length(Rs); size(G) == (3, 3, n) || error("farBlkX!: G must be 3 x 3 x $n")
    sTQ = ntuple(d -> QI(sT[d]), 3); sSQ = ntuple(d -> QI(sS[d]), 3)
    fx = if fs === nothing
        gQ = ntuple(d -> min(sTQ[d], sSQ[d]), 3)
        nT = ntuple(d -> Int(sTQ[d] / gQ[d]), 3); nS = ntuple(d -> Int(sSQ[d] / gQ[d]), 3)
        farSetXMem(sTQ, sSQ, f; tol = tol, nBlk = finBlk(Rs, gQ, nT, nS), offs = Rs,
                   disk = disk, dir = dir)
    else
        fs
    end
    (fx.sTQ == sTQ && fx.sSQ == sSQ && fx.frq == f) ||
        error("farBlkX!: fs was built for the pair $(fx.sTQ) / $(fx.sSQ) at f = $(fx.frq)")
    rt = zeros(Int, n); Ls = fill(-1, n); ms = fill((0, 0, 0), n)
    Threads.@threads :static for q in 1:n
        kind, L = farRteX(fx, Rs[q])
        rt[q] = kind; Ls[q] = L
        kind == 2 && (ms[q] = latInd(fx, Rs[q]))
    end
    # the fine block: every sub-offset |m + t| of the route-2 offsets, filled once by farBlk!
    nbx = 1
    for q in 1:n
        rt[q] == 2 || continue
        m = ms[q]
        nbx = max(nbx, maximum(max(abs(m[d] + 1 - fx.nS[d]), abs(m[d] + fx.nT[d] - 1)) for d in 1:3) + 1)
    end
    n2 = count(==(2), rt)
    ego = zeros(Complex{T}, 3, 3, n2 > 0 ? nbx : 0, n2 > 0 ? nbx : 0, n2 > 0 ? nbx : 0)
    cg = zeros(Float64, size(ego, 3), size(ego, 4), size(ego, 5))
    fg = nothing
    if n2 > 0
        fg = finSet(fx; nBlk = nbx); chkCut(fg)
        farBlk!(ego, fx.gQ, f; tol = fx.tol, fs = fg, dir = fx.dir)
        if cert
            Threads.@threads :static for i3 in 1:nbx
                for i2 in 1:nbx, i1 in 1:nbx
                    farInd(i1, i2, i3) || continue
                    D = (i1 - 1, i2 - 1, i3 - 1)
                    kind, L, Lc, _ = farRte(fg, D)
                    cg[i1, i2, i3] = certEq(fg, D, kind, L, Lc)
                end
            end
        end
    end
    # route 3, threaded; the shared state (MOMCX, fx.ksr, the cache file) is behind KSRLK and fx.lk
    ks3 = [q for q in 1:n if rt[q] == 3]
    cs = zeros(Float64, n)
    if !isempty(ks3)
        k6 = ComplexF64(fx.k); f6 = ComplexF64(fx.frq); vs6 = Float64(prod(fx.sS))
        setprecision(BigFloat, KSRPRC) do
            Threads.@threads :static for j in eachindex(ks3)
                q = ks3[j]
                copyto!(view(G, :, :, q), ksrCchX!(fx, Rs[q]))
                cert && (cs[q] = KSRTOL * est(sqrt(sum(Float64(Rs[q][d])^2 for d in 1:3)), k6, f6, vs6))
            end
        end
    end
    # routes 1 and 2, threaded over offsets
    wss = [FarWrk(fx.Lw, T) for _ in 1:Threads.maxthreadid()]
    Threads.@threads :static for q in 1:n
        rt[q] == 3 && continue
        id = Threads.threadid(); Gq = view(G, :, :, q)
        if rt[q] == 1
            Rr = ntuple(d -> T(Rs[q][d]), 3)
            tnsWhlX!(Gq, fx, wss[id], Rr, Ls[q])
            cert && (cs[q] = bndWhlX(fx, sqrt(sum(Rr[i]^2 for i in 1:3)), Ls[q]))
        else
            cs[q] = boxSum!(Gq, ego, cg, ms[q], fx.nT, fx.nS)
        end
    end
    cert || return rt
    for q in 1:n; cs[q] += 20 * eps(T) * Float64(maximum(abs, view(G, :, :, q))); end
    (rt, cs)
end

# The quasistatic tensor S_ab(R) = lim_{f -> 0} f^2 T_ab(R) is the n = 0 slice of the same tables:
# the three powers of k in i k h_l(k|R|) c_l0(k) cancel as k -> 0 and leave
#   S_ab(R) = sum_{l, m} (-1)^l Y_lm(Rhat) A^{ab}[.,0,lm] / ((2l+1) |R|^{l+1}) ,
# so no frequency enters, there is no k-series to cut and no Hankel bound to evaluate.  Theorems
# S and S' replace Theorems A and B,
#   |S_ab - S_ab^(L)|       <= (1/(4 pi V_t)) sum_{l > L, even} V^{ab}_l /|R|^{l+1}   (whole box)
#   |S^(j)_ab - S^(j),L_ab| <= (1/(4 pi V_t)) sum_{l > L}       Xi^{ab}_l/|V_j|^{l+1} (sub-box),
# V^{ab}_l being farMomW's moments without their |k|^2 W_l companion and Xi^{ab}_l farMomO's with
# every Bessel factor set to 1 and sum_m |grad(|d|^l Y_lm)|^2 = l(2l+1)^2|d|^{2l-2}/(4 pi).  Both
# criteria are functions of the single scalar rho = r_d/|R| (sub-boxes: r_j/min_q|V_q|), which is
# why the static band is exactly direction independent and the selector an O(1) table scan.

# Theorem S terms, scaled to be dimensionless: ls[i] = 2(i-1), t[r][i] = V^{ab}_l/(4 pi V_t r_d^l),
# so the tail after L is sum_{l > L} t[r][i] rho^l / |R|
function qssTrmW(lTop::Int, s::NTuple{3,T}) where {T}
    fc, _ = farMomW(lTop, s)
    vt = prod(s); rd = sqrt(sum(s[i]^2 for i in 1:3)); ls = collect(0:2:lTop)
    (ls, ntuple(r -> [fc[r][i] / (4 * T(pi) * vt * rd^ls[i]) for i in eachindex(ls)], 6))
end

# Theorem S' terms for one octant sub-box, scaled the same way with r_j = r_d/2; all l contribute
function qssTrmO(lTop::Int, s::NTuple{3,T}) where {T}
    hf = ntuple(d -> s[d] / 2, 3)
    vt = prod(s); rj = sqrt(sum(hf[i]^2 for i in 1:3)); ls = collect(0:lTop)
    off, gA, bA, _ = farMomO(lTop, hf)
    t = ntuple(6) do r
        a, b = ENTIJ[r]
        [(a != b ? radQ(off[6 - a - b], l) :
          sqrt(T(l) * T(2l + 1)) * radQ(gA[a], l - 1) + radQ(bA[a], l)) /
         (4 * T(pi) * vt * rj^l) for l in ls]
    end
    (ls, t)
end

# cumulative tail after each l, max over the six entries.  Past ls[end] the tail continues at the
# ratio q, bounded above by rho^2 from V_{l+2} <= r_d^2 V_l on the whole box and by
# rho sup_l (N_{l+1}/N_l scaled) on a sub-box.  q >= 1 is a tail that does not converge, and
# rho >= 1 always lands there.
function qssCum(ls::Vector{Int}, t::NTuple{6,Vector{T}}, rho::T, q::T) where {T}
    n = length(ls); q >= 1 && return fill(T(Inf), n)
    cum = zeros(T, n); acc = zeros(T, 6); ext = zero(T)
    for r in 1:6; ext = max(ext, t[r][n] * rho^ls[n] * q / (1 - q)); end
    for i in n:-1:1
        cum[i] = maximum(acc) + ext
        for r in 1:6; acc[r] += t[r][i] * rho^ls[i]; end
    end
    cum
end

# largest rho each order ls[i] <= L still covers, by bisection: the tail against tol * estS(R),
#   whole box  cum <= 3 tol V_t rho^2/(4 pi r_d^2)
#   sub-box    cum <= 3 tol V_t rho^2/(32 pi r_j^2 (1 + rho)^3)  (an eighth of the budget each,
#                                                                 and |R| <= |V_q|(1 + rho))
function qssThr(ls::Vector{Int}, t::NTuple{6,Vector{T}}, s::NTuple{3,T}, tol::T, L::Int,
                oct::Bool) where {T}
    lT = ls[end]
    rr = sqrt(sum(s[i]^2 for i in 1:3)) / (oct ? 2 : 1)
    cq = sqrt(T((lT + 1) * (2lT + 3)) / T(lT * (2lT + 1)))    # sup_{l >= lT} of the sub-box ratio
    pf = 3 * tol * prod(s) / (4 * T(pi) * rr^2)
    ok(rho, i) = qssCum(ls, t, rho, oct ? cq * rho : rho^2)[i] <=
                 pf * rho^2 / (oct ? 8 * (1 + rho)^3 : one(T))
    thr = zeros(T, count(<=(L), ls))
    for i in eachindex(thr)
        a = zero(T); b = one(T)
        for _ in 1:200
            c = (a + b) / 2
            ok(c, i) ? (a = c) : (b = c)
            b - a <= 8 * eps(T) && break
        end
        thr[i] = a
    end
    for i in 2:length(thr); thr[i] = max(thr[i], thr[i - 1]); end
    thr
end

# smallest tabulated order covering rho, where thr[i] belongs to l = stp (i - 1); -1 if none does
function qssLvl(thr::Vector{T}, rho::Real, stp::Int) where {T}
    @inbounds for i in eachindex(thr)
        rho <= thr[i] && return stp * (i - 1)
    end
    return -1
end

# the n = 0 column of a geometry table, signed by (-1)^l.  Bd is the delta_ab k^2 companion and has
# no static part; nothing is contracted here, so the cancellation estimate stays at 1.
function qssBox(gb::GeoBox{BigFloat}, ::Type{T}) where {T}
    sg(l) = iseven(l) ? one(T) : -one(T)
    FrqBox{T}(gb.L, gb.dLm, gb.dLv,
              T[sg(gb.dLv[j]) * T(gb.Ad[a, 1, j]) for a in 1:3, j in eachindex(gb.dLm)],
              gb.oLm, gb.oLv,
              ntuple(c -> T[sg(gb.oLv[c][j]) * T(gb.Ao[c][1, j]) for j in eachindex(gb.oLm[c])], 3),
              1.0)
end

# the six sums sum_{lm} Y_lm(vhat) Q^{ab}_{lm}/((2l+1)|v|^{l+1}), for entries 11, 22, 33, 12, 13, 23
function qssAcc(fb::FrqBox{T}, ws::FarWrk{T}, v::NTuple{3,T}, L::Int) where {T}
    rr = sqrt(sum(v[i]^2 for i in 1:3))
    shmFil!(ws, v[1] / rr, v[2] / rr, v[3] / rr, L)
    rd = ws.jr; p = one(T)              # the radial factor, in place of hnkFil!'s h_l(k|v|)
    @inbounds for l in 0:L
        p /= rr; rd[l + 1] = p / T(2l + 1)
    end
    Y = ws.Y; lm = fb.dLm; lv = fb.dLv; Qd = fb.Qd
    a1 = zero(T); a2 = zero(T); a3 = zero(T)
    @inbounds for j in eachindex(lm)
        lv[j] > L && break
        ry = rd[lv[j] + 1] * Y[lm[j]]
        a1 += ry * Qd[1, j]; a2 += ry * Qd[2, j]; a3 += ry * Qd[3, j]
    end
    (a1, a2, a3,
     accOne(rd, Y, fb.oLm[1], fb.oLv[1], fb.Qo[1], L),
     accOne(rd, Y, fb.oLm[2], fb.oLv[2], fb.Qo[2], L),
     accOne(rd, Y, fb.oLm[3], fb.oLv[3], fb.Qo[3], L))
end

# Fill dst(q), a 3x3 view, with the quasistatic tensor at the offset Ds[q] cells of shape sQ.
# Threaded; the routes are chosen first, because the geometry tables are sized on the orders those
# routes ask for.  Offsets that neither route can certify -- the near band, non-empty for a flat
# enough cell and empty for a cubic one -- are returned rather than filled, since which filler owns
# them is the caller's decision.  Their dst is left NaN, so dropping the return value cannot pass
# for a zero tensor.  This is the one place that decides what the near band is.
function farFilQss!(dst::F, Ds::Vector{NTuple{3,Int}}, sQ::NTuple{3,QI}, ::Type{T} = Float64;
                    tol::Float64 = TOL, L::Int = LMAX, disk::Bool = true,
                    dir::AbstractString = TABDIR[]) where {F,T}
    isempty(Ds) && return NTuple{3,Int}[]
    s6 = ntuple(d -> Float64(sQ[d]), 3); hf6 = ntuple(d -> s6[d] / 2, 3)
    rd6 = sqrt(sum(s6[i]^2 for i in 1:3))
    wThr = qssThr(qssTrmW(L + LEXT, s6)..., s6, tol, L, false)
    oThr = qssThr(qssTrmO(L + LEXT, s6)..., s6, tol, L, true)
    kn = Vector{Int}(undef, length(Ds)); lv = Vector{Int}(undef, length(Ds))
    Threads.@threads :static for q in eachindex(Ds)
        R = ntuple(d -> Float64(Ds[q][d]) * s6[d], 3)
        kn[q] = 1; lv[q] = qssLvl(wThr, rd6 / sqrt(sum(R[i]^2 for i in 1:3)), 2)
        if lv[q] < 0
            mn = minimum(sqrt(sum((sg[d] * R[d] + hf6[d])^2 for d in 1:3)) for sg in SGN8)
            kn[q] = 2; lv[q] = qssLvl(oThr, rd6 / (2 * mn), 1)
        end
    end
    ner = [Ds[q] for q in eachindex(Ds) if lv[q] < 0]
    Lw = 0; Lo = -1
    for q in eachindex(Ds); kn[q] == 1 ? (Lw = max(Lw, lv[q])) : (Lo = max(Lo, lv[q])); end
    tb = farShp(sQ; Lw = Lw, Lo = Lo, nMax = NMAX, disk = disk, dir = dir)
    fw = qssBox(tb.whl, T); fo = qssBox(tb.oct, T)
    hf = ntuple(d -> T(sQ[d]) / 2, 3); zr = ntuple(_ -> zero(T), 3)
    wss = [FarWrk(max(Lw, Lo), T) for _ in 1:Threads.maxthreadid()]
    Threads.@threads :static for q in eachindex(Ds)
        ws = wss[Threads.threadid()]; G = dst(q)
        fill!(G, lv[q] < 0 ? NaN : 0.0)
        lv[q] < 0 && continue
        R = ntuple(d -> T(Ds[q][d]) * T(sQ[d]), 3)
        # route 1 is the same sum over a single sub-box: the whole box, expanded about R itself
        fb = kn[q] == 1 ? fw : fo; ct = kn[q] == 1 ? zr : hf
        for u in 1:(kn[q] == 1 ? 1 : 8)
            sg = SGN8[u]
            a = qssAcc(fb, ws, ntuple(d -> T(sg[d]) * R[d] + ct[d], 3), lv[q])
            for r in 1:6
                i, j = ENTIJ[r]
                G[i, j] += T(r <= 3 ? 1 : sg[i] * sg[j]) * a[r]
            end
        end
        for r in 4:6; i, j = ENTIJ[r]; G[j, i] = G[i, j]; end
    end
    ner
end

# fill egoToe[:, :, i1, i2, i3] with S at the offset (i .- 1) .* stp cells, for every index with
# max(i) >= 3; the static counterpart of farBlk!, and like it frequency is applied by the caller.
# Returns the near band, in the same cell offsets as Ds: those entries hold NaN and are the
# caller's to fill from the closed-form face-pair moments.
function farBlkQss!(egoToe::AbstractArray{Complex{T},5}, s::NTuple{3,<:Rational};
                    tol::Float64 = TOL, disk::Bool = true, stp::NTuple{3,Int} = (1, 1, 1),
                    dir::AbstractString = TABDIR[]) where {T}
    n1, n2, n3 = size(egoToe, 3), size(egoToe, 4), size(egoToe, 5)
    Ds = NTuple{3,Int}[]
    for i3 in 1:n3, i2 in 1:n2, i1 in 1:n1
        farInd(i1, i2, i3) &&
            push!(Ds, ((i1 - 1) * stp[1], (i2 - 1) * stp[2], (i3 - 1) * stp[3]))
    end
    farFilQss!(Ds, ntuple(d -> QI(s[d]), 3), T; tol = tol, disk = disk, dir = dir) do q
        D = Ds[q]
        view(egoToe, :, :, div(D[1], stp[1]) + 1, div(D[2], stp[2]) + 1, div(D[3], stp[3]) + 1)
    end
end
