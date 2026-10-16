# Hermitian and anti-Hermitian kernels of the Bloch-periodic cell-averaged Green tensor at any f (d = 1, 2, 3),
# by one heat-kernel representation with per-axis periodic sums and exact real weights:
#   G = ∫_0^∞ [4π² e^{zt} K_t δ + (e^{zt}/f²) ∂∂K_t] dt,  z = k² = x + iy,  K_t = Π_i κ_i(x_i, t)/V_t,
#   κ_i = box_{sT,i} * box_{sS,i} * H1_t (the trapezoid heat kernel), periodic axes κ_i -> S_i = Σ_n e^{iθn} κ_i(x_i + nL_i).
# Herm = lattice sum of Re G_vac, AH = of Im G_vac: the z-dependence sits only in the scalar weights, so
#   Herm/AH use Re/Im of e^{zt} = e^{xt}(cos yt, sin yt) and of e^{zt}/f² = e^{xt}(cos, sin)(yt - 2 arg f)/|f|².
# Split: [0, τ] real space for all orders (RS). d = 3: t > τ spectral, closed form per q. d < 3: t > τ per periodic
# order G: EV orders (u_G = |q_G|² - x > c_th |y|) on the real t axis (exact cos/sin weights); SP orders
# (u_G <= c_th |y|: propagating, grazing, near-grazing) on a complex ray from τ, T_G = ∫ e^{-p t} h_G, p = |q_G|² - z,
# Herm_G = Re, AH_G = Im of that (analytic continuation; J10 sheet at Im f < 0).

const QI = Rational{BigInt}
q3(x) = ntuple(i -> QI(x[i]), 3)
# e^{2πi ν_m R}, ν_m = (m - t)/L, with the lattice part 2 m R/L reduced exactly (rational mod 2)
phsPer(::Type{T}, m::Integer, t::T, R::QI, L::QI) where {T} = cispi(T(mod(2 * m * R / L, 2))) * cispi(-2 * t * T(R / L))
# exact twist t = θ/2π rational (θ = π is t = 1/2, as Gila's odd branch): the whole phase reduced exactly
phsPer(::Type{T}, m::Integer, t::QI, R::QI, L::QI) where {T} = cispi(T(mod(2 * (m - t) * R / L, 2)))
# ν_m = (m - t)/L, rounded once when t is exact
nuPer(::Type{T}, m::Integer, t::T, L::QI) where {T} = (T(m) - t) / T(L)
nuPer(::Type{T}, m::Integer, t::QI, L::QI) where {T} = T((m - t) / L)
twT(::Type{T}, θ::T) where {T} = θ / (2 * T(π))
twT(::Type{T}, θ::Rational) where {T} = QI(θ)
nmr(s::T, x::T, y::T) where {T} = abs(s) >= abs(x) ? (s - y) + x : (x - y) + s

# exact for the small integers of the geometry; one rounding in the division
toT(::Type{T}, q::Rational) where {T} = (n = numerator(q); d = denominator(q); -2^53 < n < 2^53 && d < 2^53 ? T(Int(n)) / T(Int(d)) : T(BigFloat(n; precision = 320) / BigFloat(d; precision = 320)))
toT(::Type{BigFloat}, q::Rational) = BigFloat(numerator(q)) / BigFloat(denominator(q))

# Gauss-Legendre nodes/weights: QuadGK.gauss (Golub-Welsch-type eigen-Newton, generic) at precision(T) + 64 bits, rounded
# once to T, memoized. A Newton Gauss-Legendre rule in Float64 has weight errors up to 142 eps; it is not used here.
const GLC = Dict{Tuple{DataType,Int,Int},Tuple{Vector,Vector}}(); const GLCK = ReentrantLock()
function glq(::Type{T}, n::Int) where {T}
    key = (T, n, T === BigFloat ? precision(BigFloat) : 0)
    r0 = @lock GLCK get(GLC, key, nothing)
    r0 === nothing || return r0::Tuple{Vector{T},Vector{T}}
    pw = (T === BigFloat ? precision(BigFloat) : precision(T)) + 64
    xb, wb = setprecision(() -> QuadGK.gauss(BigFloat, n), BigFloat, pw)
    r = (T === BigFloat ? BigFloat.(xb) : T.(xb), T === BigFloat ? BigFloat.(wb) : T.(wb))
    @lock GLCK get!(GLC, key, r)::Tuple{Vector{T},Vector{T}}
end

# work budget of one outermost blcTns (or blcFil with a finite wmax) call, shared by its nested calls. Work units: one kap evaluation 1, one
# Poisson term 0.3, one majorant term 0.02, one angular node of the d_o = 2 pole form 0.1, one pDif 0.3, one Gauss bound 1 (measured
# 0.6-6 µs each); the call raises once its budget is spent, and refuses at once when an a priori estimate already exceeds it. The count
# sits in one cache line per thread (w[8i]); nothing is counted while no finite budget is active (NBUD = 0: the default fill)
struct Bud; w::Vector{Float64}; lim::Float64; mu::Float64; end
Bud(lim, mu = 1.0) = Bud(zeros(8 * Threads.maxthreadid()), Float64(lim), Float64(mu))
const BUD = ScopedValue{Union{Nothing,Bud}}(nothing); const NBUD = Threads.Atomic{Int}(0)
wrkTot(b::Bud) = sum(b.w[8i] for i in 1:Threads.maxthreadid())
@inline function wrk!(n)
    NBUD[] == 0 && return nothing
    b = BUD[]; b === nothing && return nothing
    i = Threads.threadid()
    @inbounds b.w[8i] += n * b.mu
    @inbounds b.w[8i] > b.lim && wrkErr(b)
    nothing
end
@inline function wrkChk(est, what)
    NBUD[] == 0 && return nothing
    b = BUD[]; b === nothing && return nothing
    wrkTot(b) + est * b.mu > b.lim && error("blcTns: a priori cost of $what, $(round(est * b.mu; sigdigits = 3)) work units, exceeds the budget left ($(round(b.lim - wrkTot(b); sigdigits = 3))); refused")
    nothing
end
wrkErr(b) = error("blcTns: cost budget exceeded ($(round(wrkTot(b); sigdigits = 3)) > $(round(b.lim; sigdigits = 3)) work units); refused")
# f() under a fresh budget of wmax units counted at mu per unit (an infinite one is only the mark of the outermost call: nothing is counted)
function withBud(f, wmax, mu = 1.0)
    isfinite(wmax) || return with(f, BUD => Bud(Inf, mu))
    Threads.atomic_add!(NBUD, 1)
    try
        with(f, BUD => Bud(wmax, mu))
    finally
        Threads.atomic_sub!(NBUD, 1)
    end
end

# Rational{BigInt} memo keys hashed per lookup, and every BigInt made in a loop, cost heap that Julia 1.12.6 never returns (8-40 B per BigInt
# after GC; 2.2 MB per trpMom call): a fill leaked 20-90 MB per call, which is what made the GC
# time of a θ sweep grow 0.7 -> 5-8 s per phase (not the KapCache, which is isbits-typed). Memos are keyed by qk(x) (two Int128),
# the geometry moments are kept per (T, edges, order), and the hot loops make no BigInt
qk(x::QI) = (Int128(numerator(x)), Int128(denominator(x)))
const QI0 = QI(0)
function trpMom(a::QI, b::QI, mMx::Int)
    M(c, k) = isodd(k) ? zero(QI) : c^(k + 1) / ((k + 1) * QI(2)^k)
    [sum(binomial(big(m), big(j)) * M(a, j) * M(b, m - j) for j in 0:m) for m in 0:mMx]
end

# one axis: the trapezoid w = box_a * box_b, its exact moments normalized μ_m/(m! cp^m) (|.| <= w0/m!): the series runs in
# (s cp)^m <= 2^{-m} (moment regime |t| >= cp²), so neither factor leaves the exponent range (μ_m/m! underflowed and s^m
# overflowed for edges below ~λ/740 in Double64: NaN)
struct AxK{T}
    a::QI; b::QI; cp::QI; cm::QI
    w0::T
    μf::Vector{T}
    # per-call memo of the exact-x data of kap (|x|, x, tri, dtr, |x + ps_j|, sign(x + ps_j) rounded once to T): kap ran this
    # Rational{BigInt} arithmetic at every quadrature node, and the GMP temporaries (freed only by finalizers) drove the RSS of long runs;
    # an AxK lives for one blcTns/blcFil call, so the memo is bounded by that call's distinct image positions
    xm::TMemo{Tuple{Int128,Int128},Tuple{T,T,T,T,NTuple{4,T},NTuple{4,Int}}}
end
AxK(::Type{T}, a::QI, b::QI; MM::Int = precision(T) + 48) where {T} =
    AxK{T}(a, b, (a + b) / 2, abs(a - b) / 2, toT(T, a * b), muF(T, a, b, MM), TMemo{Tuple{Int128,Int128},Tuple{T,T,T,T,NTuple{4,T},NTuple{4,Int}}}())
# the normalized moments are frequency and phase independent: kept per (T, edges, order, BigFloat precision), at most MUFMAX entries
const MUF = Dict{Any,Any}(); const MUFK = ReentrantLock(); const MUFMAX = 256
function muF(::Type{T}, a::QI, b::QI, MM::Int) where {T}
    key = (T, qk(a), qk(b), MM, T === BigFloat ? precision(BigFloat) : 0)
    v = @lock MUFK get(MUF, key, nothing)
    v === nothing || return v::Vector{T}
    μf = [toT(T, μ / (factorial(big(m)) * ((a + b) / 2)^m)) for (m, μ) in zip(0:MM, trpMom(a, b, MM))]
    @lock MUFK (length(MUF) < MUFMAX ? get!(MUF, key, μf) : μf)::Vector{T}
end
function kapX(A::AxK{T}, x::QI) where {T}
    get!(A.xm, qk(x)) do
        ps = (A.cp, A.cm, -A.cm, -A.cp); sg = (1, -1, -1, 1)
        tri = zero(QI); dtr = zero(QI)
        for j in 1:4
            y = x + ps[j]
            tri += sg[j] * abs(y) / 2
            dtr += sg[j] * sign(y) / 2
        end
        (toT(T, abs(x)), toT(T, x), toT(T, tri), toT(T, dtr), ntuple(j -> toT(T, abs(x + ps[j])), 4), ntuple(j -> Int(sign(x + ps[j])), 4))
    end
end

erfcC(v::Real) = erfc(v)
erfcC(v::Complex) = erfcG(v)
# Float64 at complex argument, |v| < 3, Re v >= 0: the Maclaurin series of erfc in Complex{Double64} (amplification <= e^{|v|²}/|erfc| <= 2^29
# of Double64's 106 bits), rounded once: <= 0.49 eps against 256 bits in 4-9 µs; the BigFloat series of blcSfn cost 43-55 µs and
# left heap garbage (route C: 53 GB allocated in 147 s, d3 c8 L = 4 1+0.01i). ierfc from the same numbers
const SQPID = sqrt(Double64(π))
function erfcS(zd::Complex{Double64})
    z2 = zd * zd; t = zd; S = zd; k = 0
    while k < 400
        k += 1; t *= -z2 / k; d = t / (2k + 1); S += d
        abs(d) < 1e-34 * abs(S) && break
    end
    1 - 2 * S / SQPID
end
erfcC(v::Complex{Float64}) = abs(v) < 3 && real(v) >= 0 ? ComplexF64(erfcS(Complex{Double64}(v))) : erfcG(v)
# i erfc(v) = e^{-v²}/√π - v erfc(v); for |v| >= 1/2 the S-fraction e^{-v²}/√π / (2 v C + 1), C = v + 1/(v + (3/2)/(v + 2/(v + ...)))
function ierfcC(v::Complex{Float64})
    abs(v) < 3 && real(v) >= 0 || return ierfcCF(v)
    vd = Complex{Double64}(v)
    ComplexF64(exp(-vd * vd) / SQPID - vd * erfcS(vd))
end
ierfcC(v) = ierfcCF(v)
function ierfcCF(v::S) where {S}
    T = real(S)
    abs(v) < T(1) / 2 && return exp(-v^2) / sqrt(T(π)) - v * erfcC(v)
    real(v) > 0 || error("ierfcC: Re v <= 0 at v = $v")
    cf(n) = (c = v; for j in n:-1:2; c = v + (T(j) / 2) / c; end; c)
    # the doubling starts lower at large |v| (|v| >= 3 is all that reaches here in Float64; converged within 16-64 terms there)
    n = abs(v) >= 6 ? 8 : 16; c1 = cf(n)
    while true
        n *= 2; c2 = cf(n)
        # e^{-v²} with v² formed exactly (expM2): the plain exp(-v^2) cost |v|² eps (33 eps at |v| = 6)
        abs(c2 - c1) <= eps(T) * abs(c2) && return (S <: Complex ? expM2(v) : exp(-v^2)) / sqrt(T(π)) / (2 * v * c2 + 1)
        c1 = c2
        n > 1 << 18 && error("ierfcC: no convergence at v = $v")
    end
end

# (ierfc v, erfc v, e^{-v²}) at once, the three numbers of one kink of kap. Float64 complex: |v| < 3 all three from one Double64 evaluation;
# |v| >= 3 the backward continued fraction C (ierfc = E/(2vC + 1), erfc = 2C ierfc, E = e^{-v²}/√π with v² exact): gsf's forward Lentz
# erfc lost 18 eps there
erfTrio(v) = (ierfcC(v), erfcC(v), exp(-v^2))
function erfTrio(v::Complex{Float64})
    if abs(v) < 3 && real(v) >= 0
        vd = Complex{Double64}(v); E = exp(-vd * vd); er = erfcS(vd)
        return ComplexF64(E / SQPID - vd * er), ComplexF64(er), ComplexF64(E)
    end
    real(v) > 0 || return (ierfcC(v), erfcC(v), exp(-v^2))
    cf(n) = (c = v; for j in n:-1:2; c = v + (j / 2) / c; end; c)
    n = abs(v) >= 6 ? 8 : 16; c1 = cf(n)
    while true
        n *= 2; c2 = cf(n)
        if abs(c2 - c1) <= eps() * abs(c2)
            E = expM2(v); ie = E / sqrt(π) / (2 * v * c2 + 1)
            return ie, 2 * c2 * ie, E
        end
        c1 = c2
        n > 1 << 18 && error("erfTrio: no convergence at v = $v")
    end
end


# terms of the moment series: Cauchy on |ζ - x| = 2cp with |μ_m| <= w0 cp^m gives, in the moment regime
# |t| >= cp(|x| + cp), |tail_k| <= e w0 |g e^{-w²}| Σ_{m>M} (m+k)^k 2^{-m} (2cp)^{-k}; reference scale w0 |g e^{-w²}| (2cp)^{-k}
# memoized per (T, tol) (a few distinct tolerances): the loop ran at every kap call (8% of route C's time)
const MOMMC = TMemo{Tuple{DataType,Float64},Int}()
momM(::Type{T}, tol::T) where {T} = T === BigFloat ? momM0(T, tol) : get!(() -> momM0(T, tol), MOMMC, (T, Float64(tol)))
function momM0(::Type{T}, tol::T) where {T}
    M = 2
    while 3 * (M + 3)^2 * (T(1) / 2)^M * 2 > tol
        M += 2
    end
    M
end
momReg(cp, ax, t) = abs(t) >= cp * (ax + cp)

"""
    kap(A, x, t, tol) -> (κ, κ', κ'')

κ(x,t) = (box_a * box_b * H1_t)(x) and its x-derivatives at real or complex t (Re √t > 0), x exact.
"""
kap(A::AxK{T}, x::QI, t::S, tol::T) where {T,S} = kapV(A, kapX(A, x), t, tol)
function kapV(A::AxK{T}, dx, t::S, tol::T) where {T,S}
    wrk!(1)
    cp = toT(T, A.cp); ax, xT, triT, dtrT, ay, sy = dx
    if momReg(cp, ax, t)
        u = sqrt(t); w = xT / (2 * u); s = 1 / (2 * u); g = 1 / sqrt(4 * T(π) * t); sc = s * cp
        M = min(momM(T, tol), length(A.μf) - 3)
        # H_m e^{-w²} by the recurrence H_{n+1} = 2w H_n - 2n H_{n-1}, three at a time (no vector per call)
        ha = exp(-w^2); hb = 2 * w * ha; hc = 2 * w * hb - 2 * ha
        k0 = zero(S); k1 = zero(S); k2 = zero(S); sm = one(S)
        for m in 0:2:M
            c = A.μf[m + 1] * sm * g
            k0 += c * ha
            k1 -= c * s * hb
            k2 += c * s^2 * hc
            sm *= sc^2
            hd = 2 * w * hc - 2 * (m + 2) * hb; he = 2 * w * hd - 2 * (m + 3) * hc
            ha, hb, hc = hc, hd, he
        end
        return k0, k1, k2
    end
    sg = (1, -1, -1, 1)
    u = sqrt(t)
    k0 = S(triT); k1 = S(dtrT); k2 = zero(S)
    for j in 1:4
        v = ay[j] / (2 * u)
        ie, er, ev = erfTrio(v)
        k0 += sg[j] * u * ie
        sy[j] != 0 && (k1 -= (sy[j] > 0 ? sg[j] : -sg[j]) * er / 2)
        k2 += sg[j] * ev / (2 * sqrt(T(π)) * u)
    end
    return k0, k1, k2
end

# periodic axis: L, twist tw = θ/2π (QI exact or T), θ (T or QI)
# per-call memos (an PerAx lives for one blcTns/blcFil call): phN, phX the twist phases; im the kapX data of the images x + nL of each
# x met (Int keys: imgSum hashed a Rational{BigInt} and did BigInt arithmetic per image and node); rq x/L as a reduced Int128 pair
const KXT{T} = Tuple{T,T,T,T,NTuple{4,T},NTuple{4,Int}}
struct PerAx{T,W}
    A::AxK{T}; L::QI; tw::W; θ
    phN::TMemo{Int,Complex{T}}; phX::TMemo{Tuple{Int128,Int128},Complex{T}}
    im::TMemo{Tuple{Int128,Int128},Dict{Int,KXT{T}}}; rq::TMemo{Tuple{Int128,Int128},Tuple{Int128,Int128}}
end
PerAx(A::AxK{T}, L::QI, tw::W, θ) where {T,W} = PerAx{T,W}(A, L, tw, θ, TMemo{Int,Complex{T}}(), TMemo{Tuple{Int128,Int128},Complex{T}}(), TMemo{Tuple{Int128,Int128},Dict{Int,KXT{T}}}(), TMemo{Tuple{Int128,Int128},Tuple{Int128,Int128}}())
const I53 = Int128(2)^53
# x/L as an Int128 pair, or nothing when it does not fit
function rqP(P::PerAx, x::QI)
    get!(P.rq, qk(x)) do
        r = x / P.L; n = numerator(r); d = denominator(r)
        abs(n) < I53 && d < I53 ? (Int128(n), Int128(d)) : (Int128(0), Int128(0))
    end
end
# e^{iθq} for a binary θ (exact dyadic rational) and an exact rational q, with θq reduced mod 2π in extended precision before the one
# rounding to T: cis(n θ) in T erred by ~n·eps (0.37 eps per period, 371 eps at n = 1000)
function cisQ(::Type{T}, θ, q::QI) where {T}
    (q == 0 || θ == 0) && return one(Complex{T})
    pr = precision(T) + 64 + ndigits(numerator(q); base = 2) + max(0, exponent(BigFloat(θ)))
    r = setprecision(BigFloat, pr) do
        v = BigFloat(θ) * BigFloat(numerator(q)) / BigFloat(denominator(q)); tp = 2 * BigFloat(π)
        v - tp * round(v / tp)
    end
    cis(T(r))
end
# (P.θ is an untyped field: the annotation keeps the callers' sums type-stable; without it imgSum boxed every term, 0.9 GB per route-C call)
phsN(::Type{T}, n::Int, P::PerAx) where {T} = (P.tw isa QI ? cispi(toT(T, mod(2 * n * P.tw, 2))) :
    abs(n) <= 2 ? cis(n * T(P.θ)) : get!(() -> cisQ(T, P.θ, QI(n)), P.phN, n))::Complex{T}
# the Poisson phase e^{2πi(m - tw)x/L} = e^{2πi m x/L} e^{-iθ x/L}: with a binary θ the second factor from θ itself (the form
# e^{-2πi fl(θ/2π) x/L} carried the rounding of θ/2π times x/L)
phsP(::Type{T}, m::Integer, P::PerAx, x::QI) where {T} = iszero(x) ? one(Complex{T}) : P.tw isa QI ? phsPer(T, m, P.tw, x, P.L) :
    cispi(T(mod(2 * m * x / P.L, 2))) * get!(() -> cisQ(T, -P.θ, x / P.L), P.phX, qk(x))
nuP(::Type{T}, m::Int, P::PerAx) where {T} = nuPer(T, m, P.tw, P.L)
# Float64: the same numbers from Int128 arithmetic on x/L = rn/rd, tw = tn/td, L = Ln/Ld (T(k)/T(d) with k, d < 2^53 is the correctly
# rounded k/d, as T(k//d) is): no Rational{BigInt} per term and node (GMP garbage)
function phsP(::Type{Float64}, m::Integer, P::PerAx, x::QI)
    iszero(x) && return one(ComplexF64)
    rn, rd = rqP(P, x)
    if rd != 0
        if P.tw isa QI
            tn = Int128(numerator(P.tw)); td = Int128(denominator(P.tw))
            D = 2 * td * rd; k = mod(2 * (m * td - tn) * rn, D)
            (abs(td) < I53 && D < I53) && return cispi(Float64(k) / Float64(td * rd))
        else
            k = mod(2 * m * rn, 2 * rd)
            return cispi(Float64(k) / Float64(rd)) * get!(() -> cisQ(Float64, -P.θ, x / P.L), P.phX, qk(x))
        end
    end
    P.tw isa QI ? phsPer(Float64, m, P.tw, x, P.L) : cispi(Float64(mod(2 * m * x / P.L, 2))) * get!(() -> cisQ(Float64, -P.θ, x / P.L), P.phX, qk(x))
end
function nuP(::Type{Float64}, m::Int, P::PerAx)
    if P.tw isa QI
        tn = Int128(numerator(P.tw)); td = Int128(denominator(P.tw)); Ln = Int128(numerator(P.L)); Ld = Int128(denominator(P.L))
        a = (m * td - tn) * Ld; b = td * Ln
        abs(a) < I53 && b < I53 && return Float64(a) / Float64(b)
        return nuPer(Float64, m, P.tw, P.L)
    end
    (Float64(m) - P.tw) / toT(Float64, P.L)
end
# one Poisson term without e^{-4π²ν²t}: (w0/L) sinc(ν a) sinc(ν b) e^{2πiνx} (2πiν)^k, k = 0,1,2
function poiPre(::Type{T}, m::Int, P::PerAx, x::QI) where {T}
    wrk!(0.3)
    ν = nuP(T, m, P)
    c = P.A.w0 / toT(T, P.L) * sinc(ν * toT(T, P.A.a)) * sinc(ν * toT(T, P.A.b)) * phsP(T, m, P, x)
    iq = complex(zero(T), 2 * T(π) * ν)
    return (c, c * iq, c * iq^2), ν
end
function poiTrm(::Type{T}, m::Int, P::PerAx, x::QI, t::Number) where {T}
    pr, ν = poiPre(T, m, P, x)
    e = exp(-4 * T(π)^2 * ν^2 * t)
    return (pr[1] * e, pr[2] * e, pr[3] * e)
end
# the orders ever needed at t: 4π²(ν² - ν_min²) t <= α
function poiRng(::Type{T}, P::PerAx, t::T, α::T) where {T}
    tw = P.tw isa QI ? toT(T, P.tw) : P.tw
    L = toT(T, P.L)
    νr = sqrt(α / t) / (2 * T(π)) + 2 / L
    (floor(Int, tw - νr * L) - 1):(ceil(Int, tw + νr * L) + 1)
end
# the orders of an EV-only Poisson sum (the SP/SL orders removed): 4π²(ν² - ν2)t <= α around ν2 = the smallest ν² LEFT in the sum.
# poiRng's reference is the smallest |ν| overall (<= 1/2L), which is a removed propagating order when fL is large: every EV order was then
# dropped beyond t ~ α/(4π²ν_EV²) and the EV part lost its tail, under a certificate built on the same truncated sum (
# d1 c32 f = 6 L = 1 θ = 0.7, 3.5e6 eps "certified"); the same absolute cut made κ_up vanish at far open offsets
function poiRngE(::Type{T}, P::PerAx, t::T, α::T, ν2::T) where {T}
    tw = P.tw isa QI ? toT(T, P.tw) : P.tw
    L = toT(T, P.L)
    νr = sqrt(max(ν2, zero(T)) + α / (4 * T(π)^2 * t)) + 2 / L
    (floor(Int, tw - νr * L) - 1):(ceil(Int, tw + νr * L) + 1)
end
# smallest ν² over the orders not in excl (one periodic axis): the excluded set is a run of orders around the twist
function nu2Ex(::Type{T}, P::PerAx, excl) where {T}
    m0 = round(Int, P.tw isa QI ? toT(T, P.tw) : P.tw)
    lo = min(m0, minimum(excl; init = m0)) - 2; hi = max(m0, maximum(excl; init = m0)) + 2
    minimum(m -> nuP(T, m, P)^2, (m for m in lo:hi if !(m in excl)))
end
useReal(::Type{T}, P::PerAx, t::T) where {T} = t <= toT(T, P.L)^2 / (4 * T(π))

# the two forms of one periodic sum S^{(k)}(x, t) = Σ_n e^{iθn} κ^{(k)}(x + nL, t) (real or complex t, Re t > 0), each with its
# Σ|terms|: images with (d_n² - d_min²) Re(1/t)/4 <= α, Poisson orders with 4π²(ν² - ν_min²) Re t <= α
function imgSum(P::PerAx{T}, x::QI, t, tol::T) where {T}
    α = -log(tol) + 3; S = zero(Complex{T})
    Sv = [S, S, S]; Av = zeros(T, 3)
    L = P.L; ri = real(1 / t); cpT = toT(T, P.A.cp)
    imx = get!(() -> Dict{Int,KXT{T}}(), P.im, qk(x))
    kx(n) = get!(() -> kapX(P.A, x + n * L), imx, n)
    dn(n) = max(kx(n)[1] - cpT, zero(T))
    n0 = round(Int, toT(T, -x / L)); dmin = min(dn(n0), dn(n0 - 1), dn(n0 + 1))
    D2 = dmin^2 + 4 * α / ri
    nlo = n0; while dn(nlo - 1)^2 <= D2; nlo -= 1; end
    nhi = n0; while dn(nhi + 1)^2 <= D2; nhi += 1; end
    for n in nlo:nhi
        k = kapV(P.A, kx(n), t, tol); ph = phsN(T, n, P)
        for j in 1:3
            v = ph * k[j]; Sv[j] += v; Av[j] += abs(v)
        end
    end
    (Sv[1], Sv[2], Sv[3]), (Av[1], Av[2], Av[3])
end
function poiSum(P::PerAx{T}, x::QI, t, tol::T) where {T}
    α = -log(tol) + 3; S = zero(Complex{T})
    Sv = [S, S, S]; Av = zeros(T, 3)
    for m in poiRng(T, P, real(t), α)
        p = poiTrm(T, m, P, x, t)
        for j in 1:3
            Sv[j] += p[j]; Av[j] += abs(p[j])
        end
    end
    (Sv[1], Sv[2], Sv[3]), (Av[1], Av[2], Av[3])
end
# the full periodic sum: the form of fewer terms first (switch at 1/Re(1/t) = L²/4π, loss <= e^{π/4} (cos arg t)^{-1/2} when the
# value is set by the nearest image); where its Σ|terms|/|S^{(k)}| > 4 (an odd derivative at a symmetry point: the nearest image
# contributes 0 and the value comes from images at distance L, e.g. x = 0: 1.1e4 eps on H_xy, d2 c8 L = 2λ 1+i) the other
# form is evaluated too and, per derivative order, the one with the smaller Σ|terms| is kept
# perSkA also returns the chosen form's Σ|terms| per derivative order: the node accumulators count it (not |S|), so that a periodic sum that
# cancels inside (x ≡ 0 with small θ: G_xy ∝ sin θ) raises the a posteriori λ of its entries (|S| hid it: 9e4 eps per entry at λ 30)
function perSkA(P::PerAx{T}, x::QI, t, tol::T) where {T}
    img = 1 / real(1 / t) <= toT(T, P.L)^2 / (4 * T(π))
    S, A = img ? imgSum(P, x, t, tol) : poiSum(P, x, t, tol)
    any(j -> A[j] > 4 * abs(S[j]), 1:3) || return S, A
    S2, A2 = img ? poiSum(P, x, t, tol) : imgSum(P, x, t, tol)
    ntuple(j -> A2[j] < A[j] ? S2[j] : S[j], 3), ntuple(j -> min(A2[j], A[j]), 3)
end
useRealC(::Type{T}, P::PerAx, t::Complex) where {T} = 1 / real(1 / t) <= toT(T, P.L)^2 / (4 * T(π))
negAx(P::PerAx) = PerAx(P.A, P.L, -P.tw, P.θ isa Rational ? -P.θ : -P.θ)

# derivative order of axis i in entry (a,b) of ∂a∂b
dOrd(a, b, i) = (a == i) + (b == i)

# u = |q|² - x = 4π²(|ν|² - a² + b²), the largest of |ν_i|, |a| factored: the subtraction left is intrinsic
function uOrd(ν::NTuple{N,T}, fa::T, fb::T) where {N,T}
    am = maximum(abs, ν; init = zero(T))
    if am >= abs(fa)
        return 4 * T(π)^2 * ((am - abs(fa)) * (am + abs(fa)) + (sum(abs2, ν; init = zero(T)) - am^2) + fb^2)
    end
    return 4 * T(π)^2 * ((sum(abs2, ν; init = zero(T)) - fa^2) + fb^2)
end

# the smallest u = |q|² - Re k² over the orders of a d = 3 lattice (θ radians in T or Rational twists): > 0 when every order is evanescent
function uEv3(::Type{T}, L, θ, fa::T, fb::T) where {T}
    ν = ntuple(i -> (tw = θ[i] isa Rational ? toT(T, mod(QI(θ[i]), 1)) : mod(T(θ[i]) / (2 * T(π)), one(T)); min(tw, 1 - tw) / toT(T, QI(L[i]))), 3)
    uOrd(ν, fa, fb)
end
# u of one order (orders ms on the periodic axes Ps) formed in extended precision from the exact inputs (the binary θ, f; exact twist and
# period) and rounded once: near grazing u is small against 4π²ν² and 4π²f², and the roundings of ν = (m - tw)/L and of the products, each
# a few eps of the large terms, were a fixed multiple of the f-ulp sensitivity of u, i.e. of κ (per entry c ≈ 6 eps(1 + κ_e) at
# d1 grazing, flat from 1e-4 to 1e-6 and also in Double64: a method constant). Float64 -> Double64, else BigFloat at precision + 64 bits
nuX(::Type{X}, m::Int, P::PerAx) where {X} = P.tw isa QI ? toT(X, (m - P.tw) / P.L) : (X(m) - X(P.θ) / (2 * X(π))) / toT(X, P.L)
uOrdX(::Type{Float64}, ms::NTuple{N,Int}, Ps, fa::Float64, fb::Float64) where {N} =
    Float64(uOrd(ntuple(i -> nuX(Double64, ms[i], Ps[i]), N), Double64(fa), Double64(fb)))
uOrdX(::Type{T}, ms::NTuple{N,Int}, Ps, fa::T, fb::T) where {T,N} = setprecision(BigFloat, (T === BigFloat ? precision(BigFloat) : precision(T)) + 64) do
    T(uOrd(ntuple(i -> nuX(BigFloat, ms[i], Ps[i]), N), BigFloat(fa), BigFloat(fb)))
end

# the diagonal of one d = 3 spectral order, W = e^{-pτ}/p = E/p, S_a = 4π²Σ_{d≠a}ν_d², ν_a² as 4π²ν_a²: the exact rewrite (S_a W - E)/f²
# (4π²W and 4π²ν_a²W/f² each carry the pole and cancel it near a mode along a) when its terms are smaller than the plain form's 4π²W - 4π²ν_a²W/f²,
# which is kept where |q|² >> |k|² (there S_a W - E cancels: S_a W -> E S_a/|q|²; 1000-key regression 0.20 -> 19.6 eps on a d3 H entry before)
dgE(Sa, νa, aW, aE, af2) = (Sa * aW + aE) * af2 < aW * (4 * oftype(aW, π)^2 + νa * af2)

# an axis whose twist is exact with 2tw integer (Rational tw ∈ Z/2, or the binary θ = 0): the orders m and 2tw - m have ν and -ν, the same
# |ν|-dependent factors and exactly conjugate Poisson phases e^{±2πiνx}, so their sum is 2 cos(2πνx) (entries even in ν_i) or 2i sin(2πνx)
# (odd) times one term (the ± pairs of the d = 3 empty-lattice mode cancel exactly at x = ±L/4, Σ|terms| counted them apart, λ 6e20)
prTw(P::PerAx) = P.tw isa QI ? isinteger(2 * P.tw) : iszero(P.tw)
# (cos πq, sin πq) of an exact rational q, each with relative accuracy and exact zeros: reduced exactly to sin(πb), |b| <= 1/2
function csPiQ(::Type{T}, q::QI) where {T}
    r = mod(q, 2); sg = one(T)
    r >= 1 && (r -= 1; sg = -sg)
    sb(b) = iszero(b) ? zero(T) : T === Float64 ? sinpi(Float64(b)) : csn(T(π) * toT(T, b))[2]
    (sg * sb(QI(1, 2) - r), sg * sb(min(r, 1 - r)))
end
# the same at q = n/d from Int128 parts (Float64, |n| and 2d below 2^53): the same exact reduction and the same single rounding of b (a correctly
# rounded quotient of exact integers is the correctly rounded rational), without Rational{BigInt} arithmetic (the fill's prMat: +0.07 s, 13%, per
# 32³ fill at θ = 0 with the BigInt path)
function csPiQ(::Type{Float64}, n::Int128, d::Int128)
    r = mod(n, 2 * d); sg = 1.0
    r >= d && (r -= d; sg = -sg)
    sb(bn, bd) = iszero(bn) ? 0.0 : sinpi(Float64(bn) / Float64(bd))
    (sg * sb(d - 2 * r, 2 * d), sg * sb(min(r, d - r), d))
end
# per-axis order tables of a d = 3 spectral sum: phe/pho the phase factors of the even/odd entries (a paired order carries its partner),
# jr the order indices summed (paired axes: ν_i >= 0 only)
function prTab(::Type{T}, Pv3, rngs, pht, R) where {T}
    prd = ntuple(i -> prTw(Pv3[i]), 3)
    twq = ntuple(i -> Pv3[i].tw isa QI ? Pv3[i].tw : QI0, 3)
    cs(i, m) = csPiQ(T, 2 * (m - twq[i]) * R[i] / Pv3[i].L)
    phe = [Complex{T}[prd[i] && m != twq[i] ? 2 * cs(i, m)[1] : pht[i][j] for (j, m) in enumerate(rngs[i])] for i in 1:3]
    pho = [Complex{T}[prd[i] && m != twq[i] ? complex(zero(T), 2 * cs(i, m)[2]) : pht[i][j] for (j, m) in enumerate(rngs[i])] for i in 1:3]
    jr = [[j for (j, m) in enumerate(rngs[i]) if !prd[i] || m >= twq[i]] for i in 1:3]
    phe, pho, jr
end
# d < 3 orders: on an axis with an exact twist the orders m and 2tw - m share every |ν|-dependent factor (u, the open-axis integrals,
# e^{-ut}) and carry conjugate phases; prRol(P, m, inS) = :lead (m > tw, partner in the set: its term carries the pair), :skip (m < tw, partner in
# the set) or :none (unpaired axis, ν = 0, or partner absent)
function prRol(P::PerAx, m::Int, inS)
    prTw(P) || return :none
    tw = P.tw isa QI ? P.tw : QI0; m == tw && return :none
    m2 = 2 * tw - m; (isinteger(m2) && inS(Int(m2))) || return :none
    m > tw ? :lead : :skip
end
# poiPre of the pair (m, 2tw - m): (c P₀, c iq P₁, c iq² P₀), P₀ = 2 cos(2πνx), P₁ = 2i sin(2πνx), c = (w0/L) sinc sinc without the phase
function poiPrePr(::Type{T}, m::Int, P::PerAx, x::QI) where {T}
    wrk!(0.6)
    ν = nuP(T, m, P); tw = P.tw isa QI ? P.tw : QI0
    c = P.A.w0 / toT(T, P.L) * sinc(ν * toT(T, P.A.a)) * sinc(ν * toT(T, P.A.b))
    cs, sn = csPiQ(T, 2 * (m - tw) * x / P.L)
    iq = complex(zero(T), 2 * T(π) * ν)
    return (complex(2 * c * cs), c * iq * complex(zero(T), 2 * sn), c * iq^2 * (2 * cs)), ν
end
# a d < 3 order's prefactors with the pairing of the axes where it leads (rl per axis: :lead pairs, else the plain term)
poiPreR(::Type{T}, m::Int, P::PerAx, x::QI, rl::Symbol) where {T} = rl == :lead ? poiPrePr(T, m, P, x) : poiPre(T, m, P, x)

# Neumaier step: running sum s with compensation c
@inline kadd(s, c, x) = (t = s + x; (t, c + (abs(s) >= abs(x) ? (s - t) + x : (x - t) + s)))

mutable struct Acc{T}
    H::Matrix{Complex{T}}; A::Matrix{Complex{T}}; cH::Matrix{Complex{T}}; cA::Matrix{Complex{T}}
    sH::Matrix{T}; sA::Matrix{T}
end
Acc(::Type{T}) where {T} = Acc{T}(zeros(Complex{T}, 3, 3), zeros(Complex{T}, 3, 3), zeros(Complex{T}, 3, 3), zeros(Complex{T}, 3, 3), zeros(T, 3, 3), zeros(T, 3, 3))
@inline function nAdd!(S, C, i, j, x)
    y = S[i, j] + x
    C[i, j] += complex(nmr(real(S[i, j]), real(x), real(y)), nmr(imag(S[i, j]), imag(x), imag(y)))
    S[i, j] = y
end
function add!(a::Acc, i, j, h, an, mh = abs(h), ma = abs(an))
    nAdd!(a.H, a.cH, i, j, h); nAdd!(a.A, a.cA, i, j, an)
    a.sH[i, j] += mh; a.sA[i, j] += ma
end
fin(a::Acc) = (a.H .+ a.cH, a.A .+ a.cA)


# sin(ξc/2)/(ξc/2) = sinc(ξc/2π). Double64: Base's sinc goes through sinpi(Double64) and Double64(π) (a BigFloat conversion per call): 2.8 µs, 80%
# of a Double64 blcTns rerun; sin(z)/z is 260 ns, within 1.13 epsD(1 + κ_sinc) of 300 bits; Float64 unchanged
sncH(ξ, c::T) where {T} = sinc(ξ * c / (2 * T(π)))
sncH(ξ::Double64, c::Double64) = (z = ξ * c / 2; abs(HI(z)) < 1e-3 ? (z2 = z * z; 1 - z2 / 6 * (1 - z2 / 20 * (1 - z2 / 42 * (1 - z2 / 72 * (1 - z2 / 110 * (1 - z2 / 156)))))) : sincosDD(z)[1] / z)
# (cos v, sin v); Double64 beyond |v| = π/4 by sincosDD (DoubleFloats goes through Quadmath Float128 there: 1.2 µs against 0.41 µs)
csn(v) = (cos(v), sin(v))
csn(v::Double64) = abs(HI(v)) < 0.785 ? (cos(v), sin(v)) : ((s_, c_) = sincosDD(v); (c_, s_))
# sin and cos of a Double64: n = round(x/(π/2)), r = x - n π/2 with π/2 in three Float64 parts and exact products (two-product by fma), then
# Taylor on |r| <= π/4 + 1e-16 to (π/4)^29/29! < 1e-34; quadrant by n mod 4. Within 0.44 epsD absolute, 0.8 epsD relative of 300 bits for
# |x| <= 1e3 (40000 random arguments); |x| > 2^40 falls back to DoubleFloats
const PIO2P = let b = setprecision(() -> big(π) / 2, BigFloat, 400); p1 = Float64(b); p2 = Float64(b - p1); p3 = Float64(b - p1 - p2); (p1, p2, p3) end
const ITAY = [Double64(setprecision(() -> 1 / factorial(big(n)), BigFloat, 400)) for n in 0:31]
function sincosDD(x::Double64)
    hx = HI(x)
    abs(hx) > 2.0^40 && return (sin(x), cos(x))
    n = round(hx / PIO2P[1])
    p1 = n * PIO2P[1]; p2 = n * PIO2P[2]
    r = ((x - Double64(p1, fma(n, PIO2P[1], -p1))) - Double64(p2, fma(n, PIO2P[2], -p2))) - Double64(n * PIO2P[3])
    m = -(r * r); sv = ITAY[30]; cv = ITAY[29]
    for k in 13:-1:0
        sv = ITAY[2k + 2] + m * sv; cv = ITAY[2k + 1] + m * cv
    end
    sv *= r
    q = mod(Int(n), 4)
    q == 0 ? (sv, cv) : q == 1 ? (cv, -sv) : q == 2 ? (-sv, -cv) : (-cv, sv)
end

# parity-projected open-axis factor: [κ̂(ξ)(iξ)^k e^{iξx} + (ξ -> -ξ)]/2, κ̂ = w0 sinc(ξa/2π) sinc(ξb/2π); real for real ξ
# all three k at once, the edges already in T (aT = toT(T, A.a), bT): the same operations as pFac, one sinc pair and one sin/cos per (ξ, x)
# (pFac per k with toT of the Rational{BigInt} edges was 40% of a d1 fill)
function pFac3(A::AxK{T}, ξ::S, x::T, aT::T, bT::T) where {T,S}
    kh = A.w0 * sncH(ξ, aT) * sncH(ξ, bT)
    c, sn = csn(ξ * x)
    (kh * c, -kh * ξ * sn, -kh * ξ^2 * c)
end
# pFac3 at ξ = ξW given in the wider type W (wideT), with the phase ξW x reduced mod 2π in W before its one rounding to T. In T the argument ξ x
# of the d_o = 2 angular sums carries ~2.5 ulp |ξ x| (the roundings of the node s + δ, of √, of the angular table and of the products), which
# the open-axis oscillation turns into ~6 eps |k|ρ on the small entries of far open offsets (2.2-3.5 eps(1 + κ_e) at 4-32λ); used where the
# phase exceeds PHX. BigFloat W runs inside wideDo's precision scope
const PHX = 8
# off inside the wider reruns of flagged entries (peD64, filD64): their rounding is eps of the wider type
const WPH = Base.ScopedValues.ScopedValue(true)
wideT(::Type{Float64}) = Double64
wideT(::Type{T}) where {T} = BigFloat
wideDo(f, ::Type{Float64}, s0) = f()
wideDo(f, ::Type{T}, s0) where {T} = setprecision(f, BigFloat, 2 * precision(real(s0)) + 16)
function pFac3W(A::AxK{T}, ξW, x::T, aT::T, bT::T) where {T}
    RW = real(typeof(ξW)); tp = 2 * RW(π); toS(v) = v isa Complex ? Complex{T}(v) : T(v)
    φ = ξW * RW(x); φ -= tp * round(T(real(φ)) * (1 / (2 * T(π))))
    ξ = toS(ξW); kh = A.w0 * sncH(ξ, aT) * sncH(ξ, bT)
    c, sn = csn(toS(φ))
    (kh * c, -kh * ξ * sn, -kh * ξ^2 * c)
end
# Float64: the same with ξ = ξh + ξl and the reduction by error-free products (fma) against 2π = TP2[1] + TP2[2] + TP2[3]
const TP2 = let b = setprecision(() -> 2 * big(π), BigFloat, 400); p1 = Float64(b); p2 = Float64(b - p1); (p1, p2, Float64(b - p1 - p2)) end
function pFac3W(A::AxK{Float64}, ξh::Float64, ξl::Float64, x::Float64, aT::Float64, bT::Float64)
    φh = ξh * x; φl = fma(ξh, x, -φh) + ξl * x
    k = round(φh * (1 / TP2[1]))
    φ = ((fma(-k, TP2[1], φh) - k * TP2[2]) + φl) - k * TP2[3]
    kh = A.w0 * sncH(ξh, aT) * sncH(ξh, bT)
    sn, c = sincos(φ)
    (kh * c, -kh * ξh * sn, -kh * ξh^2 * c)
end
# r = √(s0 + δ0) as rh + rl, and ξ = r c (c = ch + cl) as ξh + ξl, by error-free transformations
function sqrt2(s0::Float64, δ0::Float64)
    sh = s0 + δ0; bb = sh - s0; sl = (s0 - (sh - bb)) + (δ0 - bb)
    rh = sqrt(sh); rh, (fma(-rh, rh, sh) + sl) / (2 * rh)
end
mul2(rh, rl, ch, cl) = (p = rh * ch; (p, fma(rh, ch, -p) + rh * cl + rl * ch))
# a T number (real or complex) in wideT(T)
toW(::Type{T}, v::Complex) where {T} = Complex{wideT(T)}(v)
toW(::Type{T}, v) where {T} = wideT(T)(v)
# the angular tables cos, sin(2πj/M) in W (at the current BigFloat precision for BigFloat), by rotation from one cospi/sinpi pair: the error grows
# by a few eps(W) per step (<= 1e4 eps(W) at the largest M, far below eps(T)); one W cospi per entry cost seconds per order
function csWide(::Type{W}, M) where {W}
    c1 = cospi(2 * one(W) / M); s1 = sinpi(2 * one(W) / M)
    c = Vector{W}(undef, M); sn = Vector{W}(undef, M); c[1] = one(W); sn[1] = zero(W)
    for j in 2:M; c[j] = c[j - 1] * c1 - sn[j - 1] * s1; sn[j] = sn[j - 1] * c1 + c[j - 1] * s1; end
    c, sn
end
function pFac(A::AxK{T}, ξ::S, x::T, k::Int) where {T,S}
    kh = A.w0 * sinc(ξ * toT(T, A.a) / (2 * T(π))) * sinc(ξ * toT(T, A.b) / (2 * T(π)))
    k == 0 && return kh * cos(ξ * x)
    k == 1 && return -kh * ξ * sin(ξ * x)
    return -kh * ξ^2 * cos(ξ * x)
end
# sinc(w0 + ε) - sinc(w0) (Julia's sinc(w) = sin πw/(πw)) with absolute error O(eps |ε|), real or complex arguments: the plain
# difference errs by eps, which the pole weight 1/v ~ 1/(2ξ_c δ) of the propagating-order remainder turns into eps/ξ_c ~ eps·|AH| in H.
# Both |w0|, |w| <= 1/2: Maclaurin series, W^m - W0^m = (W - W0) h_{m-1}(W, W0) (terms fall by >= 2 per m, |sum| >= 1.2);
# |ε| >= max/2: plain difference (|Δ| is O(1) there); else sin πw - sin πw0 = 2 cos(π(w+w0)/2) sin(πε/2) with |w|, |w0| >= 1/4
function sincD(w0::S, ε::S) where {S}
    T = real(S); w = w0 + ε; m = max(abs(w0), abs(w))
    if m <= T(1) / 2
        W = w^2; W0 = w0^2; dW = ε * (2 * w0 + ε)
        s = zero(S); c = one(T); h = one(S); W0m = one(S); pp = T(π)^2
        for k in 1:200
            c = -c * pp / ((2k) * (2k + 1))
            tr = c * h; s += tr
            abs(tr) <= eps(T) * abs(s) / 8 && break
            W0m *= W0; h = W * h + W0m
        end
        return dW * s
    end
    abs(ε) >= m / 2 && return sinc(w) - sinc(w0)
    (2 * w0 * cospi((w + w0) / 2) * sinpi(ε / 2) - ε * sinpi(w0)) / (T(π) * w * w0)
end
# pFac(A, ξc + δ, x, k) - pFac(A, ξc, x, k) with absolute error O(eps |δ|): telescoped over the factors sinc, sinc, trig, each difference
# in a form carrying δ as a factor (cos A - cos B = -2 sin((A+B)/2) sin((A-B)/2), ξ² - ξc² = δ(2ξc + δ), sincD)
function pDif(A::AxK{T}, ξc::S, δ::S, x::T, k::Int) where {T,S}
    wrk!(0.3)
    ξ = ξc + δ; ca = toT(T, A.a) / (2 * T(π)); cb = toT(T, A.b) / (2 * T(π))
    f1c = sinc(ξc * ca); f1 = sinc(ξ * ca); d1 = sincD(ξc * ca, δ * ca)
    f2c = sinc(ξc * cb); f2 = sinc(ξ * cb); d2 = sincD(ξc * cb, δ * cb)
    sh = sin(δ * x / 2); mA = (2 * ξc + δ) * x / 2
    if k == 0
        g = cos(ξ * x); d3 = -2 * sin(mA) * sh
    elseif k == 1
        g = -ξ * sin(ξ * x); d3 = -(δ * sin(ξ * x) + ξc * 2 * cos(mA) * sh)
    else
        g = -ξ^2 * cos(ξ * x); d3 = -(δ * (2 * ξc + δ) * cos(ξ * x) - ξc^2 * 2 * sin(mA) * sh)
    end
    A.w0 * (d1 * f2 * g + f1c * d2 * g + f1c * f2c * d3)
end
# E(w) = erfi(√w)/√w = (2/√π) Σ_n w^n/(n!(2n+1)), entire; for |w| <= 1 the terms fall by 1/n, tail <= 2|last| <= eps|s|/2, |s| >= 0.53
function erfiQ(w::S) where {S}
    T = real(S); t = one(S); s = one(S)
    for n in 1:400
        t *= w / n; d = t / (2n + 1); s += d
        abs(d) <= eps(T) * abs(s) / 4 && break
    end
    2 * s / sqrt(T(π))
end
# Ein(w) = Σ_{n>=1} (-1)^{n+1} w^n/(n n!) = E_1(w) + γ + log w, entire; |w| <= 1: Σ|terms| <= e^{|w|} - 1 <= 1.8|w|, |Ein| >= 0.7|w|
function einQ(w::S) where {S}
    T = real(S); t = one(S); s = zero(S)
    for n in 1:400
        t *= -w / n; d = -t / n; s += d
        abs(d) <= eps(T) * abs(s) / 4 && break
    end
    s
end
# the grazing order's singular term, split into its Hermitian and anti-Hermitian parts. Propagating order in the anomaly window
# (u < 0, |p|τ <= 1), p = u - iy, b = √(-p) (principal; b = √|u| > 0 at y = 0):
# d_o = 1: C1 = π/(2√p) erfc(√(pτ)) = iπ/(2b) - (π√τ/2) E(-pτ) (√p = -ib at y >= 0; at y < 0 (J10, continued) the continuation term
#            iπP(b)/b turns -iπ/(2b) into +iπ/(2b) plus iπ(P(b) - P(ξc))/b, added by the caller)
#   d_o = 2: E_1(pτ) = -γ - log|pτ| - i arg(pτ) + Ein(pτ), arg = -π at y = 0 (the limit from Im f > 0)
# so Re and Im of the singular terms are each formed from exact lattice data without the other's size: Re(iπ/(2b)) = π Im b/(2|b|²)
anmWin(up::T, p, τ::T) where {T} = up < 0 && abs(p) * τ <= 1
c1Win(p::Complex{T}, τ::T) where {T} = (b = sqrt(-p); (complex(zero(T), T(π)) / (2 * b) - T(π) * sqrt(τ) / 2 * erfiQ(-p * τ), b))
function e1Win(p::Complex{T}, y::T, τ::T) where {T}
    w = p * τ
    ag = y == 0 ? -T(π) : angle(w)
    complex(-T(BigFloat(Base.MathConstants.eulergamma)) - log(abs(w)), -ag) + einQ(w)
end

# log of the strip bound of |pFac| on |Im φ| <= σ at radius r (ξ = r cos φ or r sin φ): |sinc(w)| <= e^{π|Im w|}
lpBnd(A::AxK{T}, r::T, x::T, k::Int, σ::T) where {T} = log(A.w0) + r * sinh(σ) * (abs(x) + toT(T, A.cp)) + k * log(max(r * cosh(σ), eps(T)))
# trapezoid nodes on the circle of radius <= r for the product of the open factors (Trefethen-Weideman, target tol of the scale)
function circM(axs, xs, ks, r::T, tol::T) where {T}
    best = typemax(Int)
    for σ in T[1 // 32, 1 // 16, 1 // 8, 1 // 4, 1 // 2, 1, 2, 3]
        lB = sum(lpBnd(axs[i], r, xs[i], ks[i], σ) for i in eachindex(axs))
        lS = sum(log(axs[i].w0) + ks[i] * log(max(r, one(T)) / 10) for i in eachindex(axs))
        Mf = (lB - lS + log(8 * T(π) / tol)) / σ
        Mf < best && (best = ceil(Int, Mf))
    end
    4 * cld(max(best, 16), 4)
end
# graded panel breakpoints on [lo, hi] around c (geometric from width w at c), plus a graded start at lo
function grdPts(lo::T, hi::T, c::T, w::T) where {T}
    pts = T[lo, hi]
    c > lo && c < hi && push!(pts, c)
    d = w
    while c - d > lo
        push!(pts, c - d); d *= 2
    end
    d = w
    while c + d < hi
        push!(pts, c + d); d *= 2
    end
    # grade the start too
    d = max(w, (hi - lo) * eps(T))
    while lo + d < min(c, hi) && d < (min(c, hi) - lo) / 2
        push!(pts, lo + d); d *= 2
    end
    sort!(unique!(pts))
end
# the pole-form panels split so that the open-axis phase ξX (ξ = √(sc + δ) for d_o = 2, ξ = ξc + δ for d_o = 1) turns by <= 4π per panel:
# the remainder's integrand oscillates like e^{iξX}, and the certification otherwise bisected the wide outer panels round after round (84
# bisections in 3 rounds, most of the 100 s of a 64³ d1 fill)
function oscSplit(pts, X, sq::Bool, sc)
    X > 0 || return pts
    out = [pts[1]]
    for q in 2:length(pts)
        a = pts[q - 1]; b = pts[q]
        ph = sq ? (sqrt(max(sc + b, zero(b))) - sqrt(max(sc + a, zero(a)))) * X : (b - a) * X
        k = max(1, ceil(Int, ph / (4 * oftype(X, π))))
        for r in 1:k; push!(out, a + (b - a) * r / k); end
    end
    out
end
# geometric subdivision of [a, b] (a > 0) so that each piece has log-width w with w·sqrt(sh) <= C at its ends:
# sh(t) = u t + d²/(4t) is -d²/d(log t)² of the exponent -u t - d²/4t, i.e. the peak of an integrand e^{-ut-d²/4t} spans
# log-width ~ 1/sqrt(sh) (sh ~ γρ at the saddle); a 26-point Gauss rule is at the floor for w sqrt(sh) <= 4.
# pieces where the exponent E is below its global maximum EM by more than αs carry nothing and are not subdivided
function shpSub(a::T, b::T, sh, C::T; kmin::Int = 1, E = nothing, EM::T = zero(T), αs::T = T(Inf)) where {T}
    E !== nothing && max(E(a), E(b)) < EM - αs && return kmin == 1 ? T[a, b] : [a * (b / a)^(T(j) / kmin) for j in 0:kmin]
    k = max(kmin, ceil(Int, log(b / a) * sqrt(max(sh(a), sh(b), zero(T))) / C))
    [a * (b / a)^(T(j) / k) for j in 0:k]
end

# a priori Gauss-Legendre bounds (Trefethen ATAP Thm 19.3) with rigorous majorants on the Bernstein ellipse, used to
# bisect the RS (u = √t) and EV (t) panels until each is certified
function ellT(m::T, h::T, ρ::T, sq::Bool) where {T}
    A = (ρ + 1 / ρ) / 2; B = (ρ - 1 / ρ) / 2
    ulo = m - h * A; uhi = m + h * A; ui = h * B
    ulo <= 0 && return nothing
    if sq
        tlo = ulo^2; thi = uhi^2; rlo = ulo^2 - ui^2; ihi = 2 * uhi * ui
    else
        tlo = ulo; thi = hypot(uhi, ui); rlo = ulo; ihi = ui
    end
    rlo <= 0 && return nothing
    (tlo = tlo, thi = thi, rlo = rlo, rinv = rlo / thi^2, ihi = ihi, uabs = uhi + ui)
end

gk(k, ξ, ta) = k == 0 ? one(ξ) : k == 1 ? ξ / (2 * ta) : ξ^2 / (4 * ta^2) + 1 / (2 * ta)

# majorant of |κ^{(k)}| (open) or |S^{(k)}| (periodic) on the ellipse data e
function axMaj(A::AxK{T}, x::QI, P, k::Int, e, tol::T) where {T}
    cp = toT(T, A.cp)
    # |H1_t(y)| = e^{-y² Re(1/t)/4}/√(4π|t|) over the support |y - x| <= cp: largest at the nearest point if Re(1/t) >= 0, at the
    # farthest (|x| + cp) if Re(1/t) < 0 (ray panels whose ellipse leaves Re t > 0, ellC neg = true)
    # image positions |x + nL| in T (one rounding each: the majorant's exponent d² Re(1/t)/4 <= α + 40 moves by <= 1e-13 relative); the
    # Rational{BigInt} x + nL per image and panel was the certification's GMP garbage
    one_(xa::T) = (d = e.rinv >= 0 ? max(xa - cp, zero(T)) : xa + cp; A.w0 * gk(k, xa + cp, e.tlo) / sqrt(4 * T(π) * e.tlo) * exp(-d^2 * e.rinv / 4))
    xT = toT(T, x)
    P === nothing && return one_(abs(xT))
    e.rinv > 0 || return T(Inf)
    # real form over images (until the Gaussian factor is below tol of the nearest)
    L = P.L; LT = toT(T, L); n0 = round(Int, -xT / LT); sr = one_(abs(xT + n0 * LT))
    for dir in (-1, 1)
        n = n0 + dir
        while true
            v = one_(abs(xT + n * LT)); sr += v; wrk!(0.02)
            v <= eps(T)^2 * sr && break
            abs(n - n0) > 2000 && (sr = T(Inf); break)
            n += dir
        end
    end
    # Poisson form
    sp = zero(T)
    Lt = toT(T, L); tw = P.tw isa QI ? toT(T, P.tw) : P.tw
    m0 = round(Int, tw)
    for dir in (-1, 1)
        m = dir < 0 ? m0 : m0 + 1
        while true
            ν = (m - tw) / Lt
            v = A.w0 / Lt * (2 * T(π) * abs(ν))^k * exp(-4 * T(π)^2 * ν^2 * e.rlo); sp += v; wrk!(0.02)
            v <= eps(T)^2 * sp && abs(ν) > 1 / Lt && break
            abs(m - m0) > 2000 && (sp = T(Inf); break)
            m += dir
        end
    end
    min(sr, sp)
end

# majorants of the Hermitian and anti-Hermitian weights on a Bernstein ellipse of a real panel (|Im t| <= ihi, |t| <= ta), without e^{x Re t}:
# δ part 4π²(cos, sin)(yt), ∂∂ part (cos yt cr - sin yt ci, sin yt cr + cos yt ci), 1/f² = cr + i ci; |cos w| <= cosh Im w,
# |sin w| <= min(cosh Im w, sinh|w|): the anti-Hermitian weights are O(y t), so AH is certified against its own size (AH << H at small Im f)
function wHA(y::T, cr::T, ci::T, ihi::T, ta::T) where {T}
    C = cosh(y * ihi); Sn = min(C, sinh(abs(y) * ta))
    (4 * T(π)^2 * C, C * abs(cr) + Sn * abs(ci), 4 * T(π)^2 * Sn, Sn * abs(cr) + C * abs(ci))
end
# bound of the Gauss error of one panel, (H, AH) entry majorants (max over the 3x3 entries), n nodes
function panBnd(R, sT, sS, f::Complex{T}, pers, axk, m, h, n, sq, tol) where {T}
    x = 4 * T(π)^2 * (real(f) - imag(f)) * (real(f) + imag(f)); y = 8 * T(π)^2 * real(f) * imag(f)
    af4 = abs2(f)^2; cr = (real(f) - imag(f)) * (real(f) + imag(f)) / af4; ci = -2 * real(f) * imag(f) / af4
    Vt = prod(i -> toT(T, sT[i]), 1:3)
    bH = T(Inf); bA = T(Inf)
    for ρ in T[1.05, 1.1, 1.2, 1.4, 1.7, 2, 2.5, 3, 4, 6, 8]
        e = ellT(m, h, ρ, sq)
        e === nothing && continue
        Ms = [[axMaj(axk[i], R[i], get(pers, i, nothing), k, e, tol) for k in 0:2] for i in 1:3]
        dH, pH, dA, pA = wHA(y, cr, ci, e.ihi, e.thi)
        ex = exp(max(x, zero(T)) * e.thi)
        Mδ = Ms[1][1] * Ms[2][1] * Ms[3][1]
        M∂ = maximum(prod(Ms[i][dOrd(a, b, i) + 1] for i in 1:3) for a in 1:3, b in 1:3)
        jac = sq ? 2 * e.uabs : one(T)
        g = (64 / T(15)) * h * ρ^(-2n) / (ρ^2 - 1) * jac * ex / Vt
        bH = min(bH, g * (dH * Mδ + pH * M∂)); bA = min(bA, g * (dA * Mδ + pA * M∂))
    end
    bH, bA
end

# certification budgets: bisection depth and leaf panels per certify pass (beyond them the panel is uncertified and blcTns raises)
const MAXDP = 40
const MAXLEAF = 200_000
# compensated complex 3x3 accumulator with Σ|terms|
mutable struct CAcc{T}
    S::Matrix{Complex{T}}; C::Matrix{Complex{T}}; A::Matrix{T}
end
CAcc(::Type{T}) where {T} = CAcc{T}(zeros(Complex{T}, 3, 3), zeros(Complex{T}, 3, 3), zeros(T, 3, 3))
cadd!(a::CAcc, i, j, v, m = abs(v)) = (nAdd!(a.S, a.C, i, j, v); a.A[i, j] += m)
cfin(a::CAcc) = a.S .+ a.C

# Bernstein-ellipse data of the complex panel t = tm + Δ w, w ∈ E_ρ = {A cos ψ + i B sin ψ}: min |t|, max |t|, min Re t,
# min Re(1/t) >= min Re t / max|t|², and the exact max over the ellipse of Re(c t) (a linear functional)
# neg = true (open-axis factors only, as on the SP/order rays): the ellipse may leave Re t > 0 as long as it avoids t = 0; then
# min Re(1/t) >= min Re t / min|t|² < 0 and axMaj takes the growing Gaussian at the far edge of the support
function ellC(tm::Complex{T}, Δ::Complex{T}, ρ::T; neg::Bool = false) where {T}
    A = (ρ + 1 / ρ) / 2; B = (ρ - 1 / ρ) / 2
    ad = abs(Δ)
    tlo = abs(tm) - ad * A
    tlo <= 0 && return nothing
    thi = abs(tm) + ad * A
    rmax(c) = real(c * tm) + hypot(A * real(c * Δ), B * imag(c * Δ))
    rlo = -rmax(-one(Complex{T}))
    rlo <= 0 && !neg && return nothing
    (tlo = tlo, thi = thi, rlo = rlo, rinv = rlo > 0 ? rlo / thi^2 : rlo / tlo^2, rmax = rmax)
end
const RHOS = (1.05, 1.1, 1.2, 1.4, 1.7, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0)
# ATAP Thm 19.3 for one panel of half-vector Δ, n nodes, with a majorant function of the ellipse data
function gBnd(tm::Complex{T}, Δ::Complex{T}, n::Int, maj; neg::Bool = false) where {T}
    wrk!(1)
    best = T(Inf)
    for ρ in RHOS
        e = ellC(tm, Δ, T(ρ); neg); e === nothing && continue
        M = maj(e)
        isfinite(M) || continue
        best = min(best, (64 / T(15)) * abs(Δ) * M * T(ρ)^(-2n) / (T(ρ)^2 - 1))
    end
    best
end

struct CPan{T}
    kind::Int; tm::Complex{T}; Δ::Complex{T}; g::Int
end

"""
    cPart(R, sT, sS, axk, pers, ax, f, τ1, φ, sc, U, tol, nG) -> (C, info, acc, integ!, bnd)

Complex part of G^per along Γ: τ1 → τc = τ1 + sc e^{iφ} (lattice sum, per-axis image/Poisson form at complex t), then the
tail from τc: d = 3 closed form per q, W = e^{-(Q-z)τc}/(Q-z); d < 3 every order with u_G = |q_G|² - x <= U on its own ray
t = τc + s e^{iφ_p}, φ_p = -arg(p)/2 (p = u_G - iy, through the saddle of e^{-pt - ρ²/4t}), the other orders together on
t = τc + s. Raw normalization; Σ_n e^{iθ·n} ∫_Γ [4π² e^{zt} K δ + (e^{zt}/f²) ∂∂K](R + Λn) dt, the continuation of the real axis.
integ!(acc, pan) integrates one panel, bnd(pan) is its a priori Gauss bound (every panel kind: segment, order ray, EV ray).
"""
function cPart(R, sT, sS, axk, pers, ax::NTuple{D,Int}, f::Complex{T}, τ1::T, φ::T, sc::T, U::T, tol::T, nG::Int; Csh::T = T(4)) where {T,D}
    fa = real(f); fb = imag(f)
    x = 4 * T(π)^2 * (fa - fb) * (fa + fb); y = 8 * T(π)^2 * fa * fb; z = complex(x, y)
    af4 = abs2(f)^2; f2i = complex((fa - fb) * (fa + fb) / af4, -2 * fa * fb / af4); if2 = 1 / abs2(f)
    Vt = prod(i -> toT(T, sT[i]), 1:3); Vs = prod(i -> toT(T, sS[i]), 1:3)
    α = -log(tol); αp = α + 12
    op = Tuple(setdiff(1:3, ax)); isper(i) = haskey(pers, i)
    xg, wg = glq(T, nG)
    ac = CAcc(T); acT = CAcc(T)
    pans = CPan{T}[]
    # fold: the periodic factor already carries e^{xt} (EV-only Poisson terms e^{-u_G t}), the weight only e^{iyt}
    # av: per-axis Σ|terms| (as blcTns's addNode!: the node's |value| hid the cancellation inside the periodic sums)
    function addC!(a_, vals, wq, t, fold::Bool, av)
        ew_ = (fold ? exp(complex(zero(T), y) * t) : exp(z * t)) * wq; wd = 4 * T(π)^2 * ew_; wp = ew_ * f2i
        k0 = vals[1][1] * vals[2][1] * vals[3][1] / Vt; k0a = av[1][1] * av[2][1] * av[3][1] / Vt
        awd = abs(wd); awp = abs(wp)
        for b in 1:3, a in 1:3
            kk = one(Complex{T}); ka = one(T)
            for i in 1:3
                kk *= vals[i][dOrd(a, b, i) + 1]; ka *= av[i][dOrd(a, b, i) + 1]
            end
            v = wp * kk / Vt; m = awp * ka / Vt
            a == b && (v += wd * k0; m += awd * k0a)
            cadd!(a_, a, b, v, m)
        end
    end
    axValCA(i, t) = isper(i) ? perSkA(pers[i], R[i], t, tol) : (k_ = Complex{T}.(kap(axk[i], R[i], t, tol)); (k_, abs.(k_)))
    dAx(i) = isper(i) ? minimum(n -> max(toT(T, abs(R[i] + n * pers[i].L) - axk[i].cp), zero(T)), round(Int, toT(T, -R[i] / pers[i].L)) .+ (-1:1)) :
                        max(toT(T, abs(R[i]) - axk[i].cp), zero(T))
    d2 = sum(i -> dAx(i)^2, 1:3)
    eφ = cis(φ)
    # majorants of the per-axis factors on an ellipse (axMaj: image and Poisson forms, min of the two)
    majLat(e, cz) = begin
        Ms = [[axMaj(axk[i], R[i], get(pers, i, nothing), k, e, tol) for k in 0:2] for i in 1:3]
        Mδ = Ms[1][1] * Ms[2][1] * Ms[3][1]
        M∂ = maximum(prod(Ms[i][dOrd(a, b, i) + 1] for i in 1:3) for a in 1:3, b in 1:3)
        exp(e.rmax(cz)) * (4 * T(π)^2 * Mδ + if2 * M∂) / Vt
    end
    if sc > 0
        az = abs(z)
        np = max(1, ceil(Int, az * sc / T(3 // 2)))
        np > 10^5 && error("cPart: segment of $np panels (Im f too small for route C)")
        br = [sc * T(j) / np for j in 0:np]
        for q in 1:np
            sub = shpSub(τ1 + br[q], τ1 + br[q + 1], tt -> az * tt + d2 / (4 * tt), Csh) .- τ1
            for r in 1:length(sub)-1
                push!(pans, CPan{T}(1, τ1 + (sub[r] + sub[r + 1]) / 2 * eφ, (sub[r + 1] - sub[r]) / 2 * eφ, 0))
            end
        end
    end
    τc = τ1 + sc * eφ
    info = Dict{Symbol,Any}(:τc => τc)
    # ordP typed (a Vector{Any} made every order-ray node of ordC! dynamic: 10 GB of boxes per d1 route-C call)
    # typed declarations: these are reassigned in the d < 3 branch and captured by the node closures (untyped they were boxed as Any)
    SL = NTuple{D,Int}[]; spProj::Vector{Vector{Int}} = Vector{Vector{Int}}(); Pv::Vector{valtype(pers)} = valtype(pers)[]
    ordP = NamedTuple{(:p, :prs),Tuple{Complex{T},Vector{NTuple{3,Complex{T}}}}}[]; uEV::T = U; uSL::T = U; νE2::T = zero(T)
    if D == 3
        VΛ = prod(i -> toT(T, pers[i].L), 1:3)
        Pv3 = [pers[i] for i in 1:3]
        rτ = real(τc)
        rngs = [begin
            tw = Pv3[i].tw isa QI ? toT(T, Pv3[i].tw) : Pv3[i].tw
            νm = sqrt(max(x, zero(T)) + αp / rτ) / (2 * T(π)) + 1 / toT(T, Pv3[i].L)
            (floor(Int, tw - νm * toT(T, Pv3[i].L)) - 1):(ceil(Int, tw + νm * toT(T, Pv3[i].L)) + 1)
        end for i in 1:3]
        nSp = 0
        uMin = T(Inf)
        νt = [[nuP(T, m, Pv3[i]) for m in rngs[i]] for i in 1:3]
        for ν1 in νt[1], ν2 in νt[2], ν3 in νt[3]
            uMin = min(uMin, uOrd((ν1, ν2, ν3), fa, fb))
        end
        sct = [[sinc(νt[i][j] * toT(T, sT[i])) * sinc(νt[i][j] * toT(T, sS[i])) for j in eachindex(rngs[i])] for i in 1:3]
        pht = [[phsP(T, m, Pv3[i], R[i]) for m in rngs[i]] for i in 1:3]
        phe, pho, jr = prTab(T, Pv3, rngs, pht, R)
        for j1 in jr[1], j2 in jr[2], j3 in jr[3]
            jj = (j1, j2, j3)
            ν = (νt[1][j1], νt[2][j2], νt[3][j3])
            uq = uOrd(ν, fa, fb)
            (uq - uMin) * rτ > αp && continue
            nSp += 1; wrk!(0.5)
            p = complex(uq, -y)
            E = exp(-p * τc); W = E / p
            c0 = Vs / VΛ * prod(i -> sct[i][jj[i]], 1:3)
            # diagonal as (-E + 4π² S_a W)/f² (blcTns route R d = 3)
            for b in 1:3, a in 1:3
                c = c0 * prod(i -> isodd(dOrd(a, b, i)) ? pho[i][jj[i]] : phe[i][jj[i]], 1:3)
                if a == b
                    Sa = 4 * T(π)^2 * sum(d -> d == a ? zero(T) : ν[d]^2, 1:3); νa = 4 * T(π)^2 * ν[a]^2
                    dgE(Sa, νa, abs(W), abs(E), abs(f2i)) ? cadd!(acT, a, b, c * (Sa * W - E) * f2i, abs(c) * (Sa * abs(W) + abs(E)) * abs(f2i)) :
                        cadd!(acT, a, b, c * (4 * T(π)^2 * W - νa * W * f2i), abs(c) * abs(W) * (4 * T(π)^2 + νa * abs(f2i)))
                else
                    cadd!(acT, a, b, -c * 4 * T(π)^2 * ν[a] * ν[b] * W * f2i)
                end
            end
        end
        info[:nSp] = nSp
    else
        Pv = [pers[ax[i]] for i in 1:D]
        ordR = [begin
            tw = Pv[i].tw isa QI ? toT(T, Pv[i].tw) : Pv[i].tw
            νm = sqrt(max(x + U, zero(T))) / (2 * T(π)) + 1 / toT(T, Pv[i].L)
            (floor(Int, tw - νm * toT(T, Pv[i].L)) - 1):(ceil(Int, tw + νm * toT(T, Pv[i].L)) + 1)
        end for i in 1:D]
        for mm in Iterators.product(ordR...)
            uOrd(ntuple(i -> nuP(T, mm[i], Pv[i]), D), fa, fb) <= U && push!(SL, mm)
        end
        ρo2 = sum(o -> max(toT(T, abs(R[o]) - axk[o].cp), zero(T))^2, op; init = zero(T))
        ri = real(1 / τc)
        for (ig, g) in enumerate(SL)
            ν = ntuple(i -> nuP(T, g[i], Pv[i]), D)
            p = complex(uOrdX(T, g, Pv, fa, fb), -y)
            abs(p) <= 64 * eps(T) * max(abs(x), one(T)) && error("cPart: Rayleigh-Wood anomaly at rounding distance, order $g")
            φp = -angle(p) / 2; ep = cis(φp)
            rate = abs(p) * cos(angle(p) / 2)
            smax = max((αp + ρo2 * ri / 4) / rate, ((sqrt(αp) + sqrt(αp + 2 * sqrt(rate) * sqrt(ρo2))) / (2 * sqrt(rate)))^2)
            prs = [poiPre(T, g[i], Pv[i], R[ax[i]])[1] for i in 1:D]
            push!(ordP, (p = p, prs = prs))
            s0 = zero(T); s1 = min(smax, max(abs(τc), 1 / rate) / 4)
            while s0 < smax
                s1 = min(s1, smax)
                osc = max(1, ceil(Int, abs(p) * (s1 - s0) / T(3 // 2)))
                length(pans) + osc > CPMAX && error("cPart: panel budget: more than $CPMAX complex-path panels (order rays at open distance ρ = $(sqrt(ρo2)))")
                for q in 1:osc
                    lo = s0 + (s1 - s0) * T(q - 1) / osc; hi = s0 + (s1 - s0) * T(q) / osc
                    push!(pans, CPan{T}(2, τc + (lo + hi) / 2 * ep, (hi - lo) / 2 * ep, ig))
                end
                s0 = s1; s1 = 2 * s1
            end
        end
        spProj = [unique(g[i] for g in SL) for i in 1:D]
        ρo = sqrt(ρo2)
        # smallest u of the orders on the EV ray (box scan), for the cancellation switch of the EV factor (as blcTns route R)
        boxE = [begin
            tw = Pv[i].tw isa QI ? toT(T, Pv[i].tw) : Pv[i].tw
            νm = sqrt(max(x + U, zero(T))) / (2 * T(π)) + 3 / toT(T, Pv[i].L)
            (floor(Int, tw - νm * toT(T, Pv[i].L)) - 2):(ceil(Int, tw + νm * toT(T, Pv[i].L)) + 2)
        end for i in 1:D]
        uEV = minimum((uOrd(ntuple(i -> nuP(T, mm[i], Pv[i]), D), fa, fb) for mm in Iterators.product(boxE...) if !(mm in SL)); init = U)
        uSL = minimum((uOrd(ntuple(i -> nuP(T, g[i], Pv[i]), D), fa, fb) for g in SL); init = U)
        νE2 = (uEV + x) / (4 * T(π)^2)
        smaxE = max((αp + ρo2 * ri / 4) / U, ((sqrt(αp) + sqrt(αp + 2 * sqrt(U) * ρo)) / (2 * sqrt(U)))^2)
        s0 = zero(T); s1 = min(smaxE, max(abs(τc), 1 / U) / 4)
        while s0 < smaxE
            s1 = min(s1, smaxE)
            osc = max(1, ceil(Int, abs(y) * (s1 - s0) / T(3 // 2)))
            length(pans) + osc > CPMAX && error("cPart: panel budget: more than $CPMAX complex-path panels (EV ray at open distance ρ = $ρo)")
            for q in 1:osc
                lo = s0 + (s1 - s0) * T(q - 1) / osc; hi = s0 + (s1 - s0) * T(q) / osc
                push!(pans, CPan{T}(3, τc + (lo + hi) / 2, complex((hi - lo) / 2), 0))
            end
            s0 = s1; s1 = 2 * s1
        end
        info[:SL] = SL
    end
    pats = unique(vcat([ntuple(i -> i in op ? dOrd(a, b, i) : 0, 3) for a in 1:3, b in 1:3][:], [(0, 0, 0)]))
    function evC!(a_, t, wq)
        vals = Vector{NTuple{3,Complex{T}}}(undef, 3); avs = Vector{NTuple{3,T}}(undef, 3)
        for o in op
            vals[o] = Complex{T}.(kap(axk[o], R[o], t, tol)); avs[o] = abs.(vals[o])
        end
        allP = all(i -> !useRealC(T, Pv[i], t), 1:D) || (uEV - uSL) * real(t) > log(T(4))
        if D == 1
            i1 = ax[1]
            if allP
                # EV-only Poisson sum with e^{xt} folded into each term, e^{-u_m t} (e^{zt} times e^{-4π²ν²t} overflowed: Inf·0 = NaN at
                # d2 c8 L = λ/2 R = (0,0,8) 1+0.1i)
                α2 = -log(tol) + 3; S0 = zero(Complex{T}); S1 = zero(Complex{T}); S2 = zero(Complex{T}); A0 = zero(T); A1 = zero(T); A2 = zero(T)
                for m in poiRngE(T, Pv[1], real(t), α2, νE2)
                    m in spProj[1] && continue
                    pr, ν = poiPre(T, m, Pv[1], R[i1]); e_ = exp(-uOrd((ν,), fa, fb) * t)
                    S0 += pr[1] * e_; S1 += pr[2] * e_; S2 += pr[3] * e_
                    ae = abs(e_); A0 += abs(pr[1]) * ae; A1 += abs(pr[2]) * ae; A2 += abs(pr[3]) * ae
                end
                vals[i1] = (S0, S1, S2); avs[i1] = (A0, A1, A2)
            else
                sv, sa = perSkA(Pv[1], R[i1], t, tol)
                for m in spProj[1]
                    pt = poiTrm(T, m, Pv[1], R[i1], t); sv = sv .- pt; sa = sa .+ abs.(pt)
                end
                vals[i1] = sv; avs[i1] = sa
            end
            addC!(a_, vals, wq, t, allP, avs)
        else
            E = zeros(Complex{T}, 3, 3); EA = zeros(T, 3, 3)
            i1, i2 = ax
            if allP
                α2 = -log(tol) + 3
                for m1 in poiRngE(T, Pv[1], real(t), α2, νE2)
                    p1, ν1 = poiPre(T, m1, Pv[1], R[i1])
                    for m2 in poiRngE(T, Pv[2], real(t), α2, νE2)
                        (m1, m2) in SL && continue
                        p2, ν2 = poiPre(T, m2, Pv[2], R[i2]); e_ = exp(-uOrd((ν1, ν2), fa, fb) * t)
                        for k1 in 0:2, k2 in 0:2-k1
                            v_ = p1[k1 + 1] * p2[k2 + 1] * e_; E[k1 + 1, k2 + 1] += v_; EA[k1 + 1, k2 + 1] += abs(v_)
                        end
                    end
                end
            else
                s1v, a1v = perSkA(Pv[1], R[i1], t, tol); s2v, a2v = perSkA(Pv[2], R[i2], t, tol)
                for k1 in 0:2, k2 in 0:2-k1
                    E[k1 + 1, k2 + 1] = s1v[k1 + 1] * s2v[k2 + 1]; EA[k1 + 1, k2 + 1] = a1v[k1 + 1] * a2v[k2 + 1]
                end
                for g in SL
                    p1 = poiTrm(T, g[1], Pv[1], R[i1], t); p2 = poiTrm(T, g[2], Pv[2], R[i2], t)
                    for k1 in 0:2, k2 in 0:2-k1
                        E[k1 + 1, k2 + 1] -= p1[k1 + 1] * p2[k2 + 1]; EA[k1 + 1, k2 + 1] += abs(p1[k1 + 1] * p2[k2 + 1])
                    end
                end
            end
            ew = (allP ? exp(complex(zero(T), y) * t) : exp(z * t)) * wq; wd = 4 * T(π)^2 * ew; wp = ew * f2i
            o = op[1]; ko = vals[o]; koa = avs[o]; awd = abs(wd); awp = abs(wp)
            for b in 1:3, a in 1:3
                v = wp * E[dOrd(a, b, i1) + 1, dOrd(a, b, i2) + 1] * ko[dOrd(a, b, o) + 1] / Vt
                m = awp * EA[dOrd(a, b, i1) + 1, dOrd(a, b, i2) + 1] * koa[dOrd(a, b, o) + 1] / Vt
                a == b && (v += wd * E[1, 1] * ko[1] / Vt; m += awd * EA[1, 1] * koa[1] / Vt)
                cadd!(a_, a, b, v, m)
            end
        end
    end
    function ordC!(a_, g::Int, t, wq)
        pr = ordP[g]
        ks = ntuple(j -> kap(axk[op[j]], R[op[j]], t, tol), length(op))
        ew = exp(-pr.p * t) * wq
        for b in 1:3, a in 1:3
            pper = one(Complex{T}); vo = ew
            for i in 1:D
                pper *= pr.prs[i][dOrd(a, b, ax[i]) + 1]
            end
            for (j, o) in enumerate(op)
                vo *= ks[j][dOrd(a, b, o) + 1]
            end
            v = pper * vo * f2i / Vt
            if a == b
                p0 = one(Complex{T}); v0 = ew
                for i in 1:D
                    p0 *= pr.prs[i][1]
                end
                for j in eachindex(op)
                    v0 *= ks[j][1]
                end
                v += p0 * 4 * T(π)^2 * v0 / Vt
            end
            cadd!(a_, a, b, v)
        end
    end
    function integ!(a_, pn::CPan)
        for j in 1:nG
            t = pn.tm + pn.Δ * xg[j]; wq = pn.Δ * wg[j]
            pn.kind == 1 ? (va = [axValCA(i, t) for i in 1:3]; addC!(a_, [v[1] for v in va], wq, t, false, [v[2] for v in va])) : pn.kind == 2 ? ordC!(a_, pn.g, t, wq) : evC!(a_, t, wq)
        end
    end
    # majorants: kind 1 the lattice sum; kind 3 the EV part |S - Σ_SL| <= min(img + Σ_SL|p_m|, Poisson over all);
    # kind 2 one order e^{-pt} times the open factors
    function majEV(e)
        Mo = Dict(o => [axMaj(axk[o], R[o], nothing, k, e, tol) for k in 0:2] for o in op)
        Mp = Dict{Int,Vector{T}}()
        for (i, ia) in enumerate(ax)
            P = Pv[i]
            full = [axMaj(axk[ia], R[ia], P, k, e, tol) for k in 0:2]
            ex = [sum(m -> (w0 = P.A.w0 / toT(T, P.L); ν = nuP(T, m, P); w0 * (2 * T(π) * abs(ν))^k * exp(-4 * T(π)^2 * ν^2 * e.rlo)), spProj[i]; init = zero(T)) for k in 0:2]
            Mp[ia] = full .+ ex
        end
        Ms = [haskey(Mo, i) ? Mo[i] : Mp[i] for i in 1:3]
        Mδ = Ms[1][1] * Ms[2][1] * Ms[3][1]
        M∂ = maximum(prod(Ms[i][dOrd(a, b, i) + 1] for i in 1:3) for a in 1:3, b in 1:3)
        exp(e.rmax(z)) * (4 * T(π)^2 * Mδ + if2 * M∂) / Vt
    end
    # the EV integrand as evaluated in Poisson form (e^{xt} folded): Σ_{G ∉ SL} Π_i |pre_i| (2π|ν_i|)^{k_i} e^{-u_G min Re t}, weight
    # |e^{iyt}| <= e^{max Re(iyt)}; majEV's e^{max Re(zt)} times the full Poisson majorant overflows where x t > 709 (Inf/NaN bound)
    function majEVf(e)
        Mo = Dict(o => [axMaj(axk[o], R[o], nothing, k, e, tol) for k in 0:2] for o in op)
        rgs = [poiRngE(T, Pv[i], e.rlo, α + 40, νE2) for i in 1:D]
        Mp = zeros(T, 3, 3)
        for mm in Iterators.product(rgs...)
            (D == 2 ? mm in SL : mm[1] in spProj[1]) && continue
            ν = ntuple(i -> nuP(T, mm[i], Pv[i]), D)
            c = exp(-uOrd(ν, fa, fb) * e.rlo) * prod(i -> abs(poiPre(T, mm[i], Pv[i], R[ax[i]])[1][1]), 1:D)
            for k1 in 0:2, k2 in 0:(D == 2 ? 2 - k1 : 0)
                Mp[k1 + 1, k2 + 1] += c * (2 * T(π) * abs(ν[1]))^k1 * (D == 2 ? (2 * T(π) * abs(ν[2]))^k2 : one(T))
            end
        end
        M = zero(T)
        for b in 1:3, a in 1:3
            kp(i) = dOrd(a, b, ax[i])
            mp = Mp[kp(1) + 1, D == 2 ? kp(2) + 1 : 1] * prod(o -> Mo[o][dOrd(a, b, o) + 1], op) * if2 / Vt
            a == b && (mp += 4 * T(π)^2 * Mp[1, 1] * prod(o -> Mo[o][1], op) / Vt)
            M = max(M, mp)
        end
        exp(e.rmax(complex(zero(T), y))) * M
    end
    majEVm(e) = (a = majEV(e); b = majEVf(e); isnan(a) ? b : min(a, b))
    function majOrd(e, g)
        pr = ordP[g]
        Mo = ntuple(j -> ntuple(k -> axMaj(axk[op[j]], R[op[j]], nothing, k - 1, e, tol), 3), length(op))
        m = zero(T)
        for b in 1:3, a in 1:3
            pper = prod(i -> abs(pr.prs[i][dOrd(a, b, ax[i]) + 1]), 1:D)
            v = pper * prod(j -> Mo[j][dOrd(a, b, op[j]) + 1], eachindex(op)) * if2
            a == b && (v += prod(i -> abs(pr.prs[i][1]), 1:D) * prod(j -> Mo[j][1], eachindex(op)) * 4 * T(π)^2)
            m = max(m, v)
        end
        exp(e.rmax(-pr.p)) * m / Vt
    end
    bnd(pn::CPan) = gBnd(pn.tm, pn.Δ, nG, pn.kind == 1 ? (e -> majLat(e, z)) : pn.kind == 2 ? (e -> majOrd(e, pn.g)) : majEVm; neg = pn.kind == 2)
    for pn in pans
        integ!(ac, pn)
    end
    info[:npan] = length(pans); info[:pans] = pans; info[:tail] = cfin(acT); info[:acT] = acT
    return cfin(ac) .+ cfin(acT), info, ac, integ!, bnd
end
kaddC(s, c, v) = (t = s + v; (t, c + complex(nmr(real(s), real(v), real(t)), nmr(imag(s), imag(v), imag(t)))))

# input validation shared by blcTns, blcFil, blcCmp: axes distinct in 1:3, one period and one phase per axis, periods > 0,
# f finite and nonzero, phases finite
function inpChk(nm, f, ax, L, θ)
    length(L) == length(ax) == length(θ) || error("$nm: $(length(ax)) periodic axes but $(length(L)) periods and $(length(θ)) phases")
    all(i -> i in 1:3, ax) || error("$nm: periodic axes $ax are not in 1:3")
    allunique(ax) || error("$nm: periodic axes $ax repeat an axis")
    for j in eachindex(ax)
        L[j] > 0 || error("$nm: period $(L[j]) along axis $(ax[j]) is not positive")
        θ[j] isa Rational || isfinite(θ[j]) || error("$nm: Bloch phase $(θ[j]) along axis $(ax[j]) is not finite")
    end
    isfinite(f) && f != 0 || error("$nm: frequency f = $f is zero or not finite")
    nothing
end

# reflection zeros: per axis i the sign σ_i(a, b) = (-1)^{(a==i)+(b==i)} of the entries that vanish exactly at the offset
# x_i, or 0. Open axis at x_i = 0: G(-x) = σG(x) kills σ = -1. Periodic axis with an exact twist tw ∈ {0, 1/2} (θ ≡ -θ): at x_i ≡ 0
# (mod L_i) (1 - σ)G = 0 kills σ = -1; at x_i ≡ L_i/2, G(x - L) = e^{iθ}G(x) = σG(x) gives (e^{iθ} - σ)G = 0: σ = -1 at tw = 0, σ = +1 at
# tw = 1/2. Entries are exact zeros of G, H and AH alike; unequal cells too (both boxes are centred)
function zKill(x::QI, per::Bool, L, tw)
    per || return x == 0 ? -1 : 0
    (tw isa QI && (mod(tw, 1) == 0 || mod(tw, 1) == 1 // 2)) || return 0
    r = mod(x, L)
    r == 0 && return -1
    2r == L && return mod(tw, 1) == 0 ? -1 : 1
    0
end
zSig(a, b, i) = isodd((a == i) + (b == i)) ? -1 : 1
function symZ(R, ax, L, tw)
    kl = [(j = findfirst(==(i), ax); j === nothing ? zKill(QI(R[i]), false, 0, nothing) : zKill(QI(R[i]), true, QI(L[j]), tw[j])) for i in 1:3]
    [any(i -> kl[i] != 0 && zSig(a, b, i) == kl[i], 1:3) for a in 1:3, b in 1:3]
end

# a bound relative to its scale (Inf for a positive bound on a zero scale)
rlB(b, S) = S > 0 ? b / S : (b > 0 ? oftype(b, Inf) : zero(b))

# raise on any uncertified panel, naming the stage
function chkCert(info)
    for (k, nm) in ((:qcert, "real-axis (RS/EV)"), (:qcertC, "complex-path (route C)"), (:qcertSP, "propagating-order (SP)"))
        haskey(info, k) || continue
        v = info[k]
        for w in (v isa Tuple ? v : (v,))
            w.uncertified > 0 && error("blcTns: $nm quadrature not certified at the target: $(w.uncertified) panels at the depth/panel budget, relative bound $(Float64(w.rel))")
        end
    end
    nothing
end

# the brief's refusals: every period a whole multiple of both cell edges along its axis, both cells on their common
# gcd lattice (corner offset R - (sT - sS)/2 a multiple of gcd(sT, sS) per axis), and no image of an unequal source overlapping the target
# lat = :any keeps only the overlap refusal (sensitivity probes at perturbed periods, and centre-aligned unequal pairs off Gila's grid)
function lstChk(R, sT, sS, ax, L; lat::Symbol = :gila)
    lat == :any && @goto ovl
    for (j, i) in enumerate(ax)
        (isinteger(L[j] / sT[i]) && isinteger(L[j] / sS[i])) || error("blcTns: period $(L[j]) along axis $i is not a whole multiple of the cell edges $(sT[i]) and $(sS[i])")
    end
    for i in 1:3
        isinteger((R[i] - (sT[i] - sS[i]) / 2) / gcd(sT[i], sS[i])) || error("blcTns: cells $(sT) and $(sS) at R = $(R) are off their common lattice along axis $i")
    end
    @label ovl
    # lat = :any (off-lattice): the smallest nonzero offset δ of a box-convolution kink (x ± (sT ± sS)/2, x = R + nL over the nearest
    # images) sets the RS resolution u ~ δ; below 2^-12 of the smaller edge the certification ran > 100 s and grew GMP garbage without
    # bound (gap 2^-14 of the edge 10 s, 2^-15 > 100 s), so it is refused instead of hanging
    if lat == :any
        for i in 1:3
            cp = (sT[i] + sS[i]) / 2; cm = abs(sT[i] - sS[i]) / 2; j = findfirst(==(i), ax)
            xs = j === nothing ? (R[i],) : (n0 = round(Int, -R[i] / L[j]); Tuple(R[i] + n * L[j] for n in n0-1:n0+1))
            for x in xs, c in (cp, -cp, cm, -cm)
                δ = abs(x + c)
                0 < δ < min(sT[i], sS[i]) / 4096 && error("blcTns: off-lattice kink offset $(δ) along axis $i is below 2^-12 of the cell edge (cost unbounded; refused)")
            end
        end
    end
    sT == sS && return nothing
    nr = [i in ax ? (j = findfirst(==(i), ax); n0 = round(Int, -R[i] / L[j]); (n0-1:n0+1) .* L[j]) : (QI(0):QI(1):QI(0)) for i in 1:3]
    for s1 in nr[1], s2 in nr[2], s3 in nr[3]
        sh = (s1, s2, s3)
        all(i -> abs(R[i] + sh[i]) < (sT[i] + sS[i]) / 2, 1:3) && error("blcTns: cells $(sT) and $(sS) at R = $(R) overlap (image shift $(sh))")
    end
    nothing
end

# the pole-form (propagating-order) remainder majorants of blcTns, top level so the fill certifies with them per open offset:
# op the open axes, w0o their trapezoid masses, Xo = |x_o| + cp_o (d_o = 2: only X1 + X2 enters), τ the split, y = Im k²
struct PoleDat{T}
    op::Vector{Int}; w0o::Vector{T}; Xo::Vector{T}; τ::T; y::T
end
# Gauss bound (ATAP 19.3) of a real panel [lo, hi] with a majorant maj(δm, hA, hB) on the disk |δ - δm| <= hA containing E_ρ
function gBndR(lo::T, hi::T, n::Int, maj) where {T}
    wrk!(1)
    δm = (lo + hi) / 2; h = (hi - lo) / 2; best = T(Inf)
    for ρ in RHOS
        A_ = (T(ρ) + 1 / T(ρ)) / 2 * h; B_ = (T(ρ) - 1 / T(ρ)) / 2 * h
        M = maj(δm, A_, B_)
        isfinite(M) || continue
        best = min(best, (64 / T(15)) * h * M * T(ρ)^(-2n) / (T(ρ)^2 - 1))
    end
    best
end
# sup of the angular average ā_pt (d_o = 2) on |s - c| <= r: (1/4π) w0_1 w0_2 R^{(k1+k2)/2} e^{|Im √s|(X1+X2)}, R = |c| + r,
# |Im √s| = √((|s| - Re s)/2); |sinc(w)| <= cosh(Im w) (sin w/w = ∫_0^1 cos(tw)dt), |cos, sin(ξx)| <= e^{|Im ξ||x|}
majA(d::PoleDat{T}, pt, c, r) where {T} = (Rr = abs(c) + r; d.w0o[1] * d.w0o[2] * Rr^((pt[d.op[1]] + pt[d.op[2]]) / T(2)) * exp(sqrt(max(Rr - real(c) + r, zero(T)) / 2) * (d.Xo[1] + d.Xo[2])) / (4 * T(π)))
majP(d::PoleDat{T}, pt, c, r) where {T} = (Rr = abs(c) + r; d.w0o[1] * Rr^pt[d.op[1]] * exp((abs(imag(c)) + r) * d.Xo[1]))
majQ(d::PoleDat{T}, pt, c, r) where {T} = (Rr = abs(c) + r; d.w0o[1] * Rr^(pt[d.op[1]] / T(2)) * exp(sqrt(max(Rr - real(c) + r, zero(T)) / 2) * d.Xo[1]))
# sup of the derivative on |s - c| <= r by Cauchy on a larger circle
dMaj(d::PoleDat{T}, mf, c, r) where {T} = minimum(r0 -> mf(c, r + r0) / r0, (r / 2, r, 2r, 4r, max(r, one(T)), 1 / max(sum(d.Xo)^2, eps(T))) )
function poleMaj(d::PoleDat{T}, pt, p, up, δm, hA, hB) where {T}
    op = d.op; w0o = d.w0o; Xo = d.Xo; τ = d.τ; y = d.y
    sr = -up; sc = max(sr, zero(T)); gap = sc - sr
    tot = zero(T)
    for sg in (y == 0 ? (1,) : (1, -1))
        if length(op) == 2
            pδ = complex(-gap, sg * y)
            dst = abs(δm - pδ) - hA
            ew = exp(-τ * (δm - hA + gap))
            mf = (c, r) -> majA(d, pt, c, r)
            q1 = dMaj(d, mf, complex(sc + δm), hA + abs(δm - pδ))
            # pδ = 0 (real f, open order): D/δ is entire, the quotient bound needs no distance
            if pδ == 0
                # the quotient's Cauchy disk reaches back to the pole; away from it the crude bound is far smaller (1e175 -> finite)
                dst0 = abs(δm) - hA
                tot += ew * (dst0 > 0 ? min(q1, (mf(complex(sc + δm), hA) + mf(complex(sc), zero(T))) / dst0) : q1); continue
            end
            dst > 0 || return T(Inf)
            crude = (mf(complex(sc + δm), hA) + mf(complex(sc), zero(T))) / dst
            q2 = abs(pδ) * dMaj(d, mf, sc + pδ / 2, abs(pδ) / 2) / dst
            tot += ew * min(crude, q1 + q2)
        else
            ξc = sqrt(sc)
            r1 = -ξc + sqrt(complex(ξc^2 - gap, sg * y)); r2 = -ξc - sqrt(complex(ξc^2 - gap, sg * y))
            abs(r2) < abs(r1) && ((r1, r2) = (r2, r1))
            # real f, propagating (gap = 0): the roots are exactly 0 and -2ξc (sqrt(ξc²) != ξc in Double64 left r1 ~ 1e-33: Inf bounds)
            y == 0 && gap == 0 && (r1 = zero(r1); r2 = complex(-2 * ξc))
            d1 = abs(δm - r1) - hA; d2 = abs(δm - r2) - hA
            ulo = δm - hA; uhi = δm + hA
            qmin = -ξc >= ulo && -ξc <= uhi ? -ξc^2 : min(ulo^2 + 2ξc * ulo, uhi^2 + 2ξc * uhi)
            ew = exp(-τ * (qmin - hB^2 + gap))
            mf = (c, r) -> majP(d, pt, c, r)
            if r1 == 0
                # real f, propagating: both roots are removable, (P(ξ) - P(ξc))/(ξ² - ξc²) = Q[s, sc] with Q(s) = P(√s) entire (P even), so
                # |integrand| <= sup|Q'| over the hull of s(disk) and sc (Cauchy, |Q(u)| <= w0 |u|^{k/2} e^{|Im √u| X}); the pole form's
                # "root" -2ξc is not a singularity (it forced panels of width ~ξc next to ξ = 0: SP uncertified in Double64, t_anmD)
                c0 = (ξc + δm)^2; R0 = 2 * abs(ξc + δm) * hA + hA^2 + abs(c0 - sc)
                qQ = dMaj(d, (c, r) -> majQ(d, pt, c, r), complex(c0), R0)
                bq = d2 > 0 ? (q1_ = dMaj(d, mf, complex(ξc + δm), hA + abs(δm - r1)); d1 > 0 ? min(q1_ / d2, (mf(complex(ξc + δm), hA) + mf(complex(ξc), zero(T))) / (d1 * d2)) : q1_ / d2) : T(Inf)
                tot += ew * min(bq, qQ) / T(π); continue
            end
            d2 > 0 || return T(Inf)
            d1 > 0 || return T(Inf)
            q1 = dMaj(d, mf, complex(ξc + δm), hA + abs(δm - r1))
            crude = (mf(complex(ξc + δm), hA) + mf(complex(ξc), zero(T))) / (d1 * d2)
            q2 = abs(r1) * dMaj(d, mf, ξc + r1 / 2, abs(r1) / 2) / d1
            tot += ew * min(crude, (q1 + q2) / d2) / T(π)
        end
    end
    tot
end

# a posteriori route selection. Each route's rounding is <= c eps Σ|terms| entrywise (compensated sums, per-term relative
# errors of a few eps), and both routes track Σ|terms| of every entry of H and AH, including the internal sums of the propagating-order
# rays and pole forms (ordAdd! Ia) where route R's e^{Im(k) r} lives. λ_e = Σ|terms_e| / |X_e| per entry (entries > entT of the kernel max).
#  1. Im f <= 0 or Re f <= 0: route R (route C needs the saddle direction of Im f > 0).
# 2. route R's real-space part alone costs √2 e^{1 + 2 Im(k)² τ_R} (a priori): above LAM0, route C directly.
#  3. otherwise route R; if every λ_e <= LAM0 it is returned; else route C too, and every entry of H and AH takes the route with the
#     smaller λ_e (route R's small entries built from cancelling order sums, e.g. H_yz = 1.2e-4 of max at d2 L = 4λ, λ_e 1292, 162 eps; route
# C 2.2, 4.1 eps).
# The returned entry's rounding is <= c eps min(λ_R,e, λ_C,e) |X_e|; info[:rte] reports the worst λ of each route and the entries taken from C.
const LAM0 = 16
function lamE(sX, X, entT)
    m = maximum(abs, X); λ = zeros(eltype(sX), size(X))
    m == 0 && return λ
    for i in eachindex(X); abs(X[i]) > entT * m && (λ[i] = sX[i] / abs(X[i])); end
    λ
end
const NPCMAX = 10^4
# complex-path panels of one cPart call (at open offsets of 300λ the order rays alone built millions of panels, 8.7 GB); beyond it
# route C is skipped (rteAuto) or refused (route = :C)
const CPMAX = 200_000
function rteAuto(R, sT, sS, f::Complex{T}, ax, L, θ, kw, Lmx, Z = falses(3, 3)) where {T}
    fa = real(f); fb = imag(f); D = length(ax)
    x = 4 * T(π)^2 * (fa - fb) * (fa + fb); y = 8 * T(π)^2 * fa * fb
    τR = min(1 / (max(x, zero(T)) + 1), 8 / (abs(y) + eps(T))); D == 3 && (τR = min(τR, Lmx^2 / (4 * T(π))))
    ΛRS = sqrt(T(2)) * exp(1 + 2 * (2 * T(π) * fb)^2 * τR)
    cOK = fb > 0 && fa > 0; eT = T(kw.entT)
    # (work units of each route and the route of every entry, for peD64's Double64 rerun of the route the flagged entries come from)
    bw_ = BUD[]; wtot() = bw_ === nothing ? 0.0 : wrkTot(bw_); w0 = wtot()
    # d = 3 with every order evanescent (u_S > 4|y|): route R's real axis with the evanescent extension is tried first whatever
    # Im f; route C's saddle ray (the image scale e^{-Im(k) r}) carries tail terms far above an entry e^{-√u_S r} (all orders evanescent, 1+i)
    evR = D == 3 && uEv3(T, L, θ, fa, fb) > 4 * abs(y)
    if cOK && ΛRS > LAM0 && !evR
        H, A, inf = blcTns(R, sT, sS, f, ax, L, θ; kw..., route = :C)
        λH = lamE(inf[:sH], H, eT); λA = lamE(inf[:sA], A, eT); λH[Z] .= 0; λA[Z] .= 0
        inf[:rte] = (ΛRS = ΛRS, λC = (maximum(λH), maximum(λA)), nC = 18); inf[:λe] = (λH, λA)
        inf[:wrt] = (R = 0.0, C = wtot() - w0); inf[:fromC] = (trues(3, 3), trues(3, 3))
        return H, A, inf
    end
    H, A, inf = blcTns(R, sT, sS, f, ax, L, θ; kw..., route = :R)
    inf[:wrt] = (R = wtot() - w0, C = 0.0); inf[:fromC] = (falses(3, 3), falses(3, 3))
    λH = lamE(inf[:sH], H, eT); λA = lamE(inf[:sA], A, eT)
    # entries that vanish by reflection (symZ, zeroed by the caller) carry no λ. Zeroing λ of any kernel at route R's rounding
    # floor (max|X| <= 64 eps max Σ|terms|), meant for the reflection zeros (θ = π, R = L/2); the same rule passed a genuine 1e-46 tensor
    # computed as 6.5e-34 of rounding noise with λ = 0 (d2 unequal, open 293λ, 10+0.5i). The reflection zeros are now symZ's, so the
    # floor rule is kept only for a kernel with no terms at all (AH at real f without propagating orders)
    λH[Z] .= 0; λA[Z] .= 0
    vH = maximum(inf[:sH]) == 0; vA = maximum(inf[:sA]) == 0
    if !cOK || max(maximum(λH), maximum(λA)) <= LAM0
        inf[:rte] = (ΛRS = ΛRS, λR = (maximum(λH), maximum(λA)), nC = 0, vnsh = (vH, vA)); inf[:λe] = (λH, λA)
        return H, A, inf
    end
    # route C's complex path needs ~|k²|·sc/(3/2) panels per segment with sc ~ (Im(k) r + c2)/(|k| Im k) (cPart): at small Im f it is
    # unaffordable (d1 c4 L = 4 1+1e-6i R = (0,-5/2,0): 8.7e6 panels, or > 1 h), so route R is returned with its λ in info[:rte]
    kk = 2 * T(π) * abs(f); ik = 2 * T(π) * fb
    npC = abs(complex(x, y)) * (T(get(kw, :c2, 12)) + x * τR) / (kk * ik) / T(3 // 2)
    # d = 3 below Im f = Re f/20 (the range of route C): no open offset carries route R's e^{Im(k) r}, the large λ_e left are the spectral
    # sum's own cancellations, which route C's closed-form tail repeats, while its segment of ~npC panels costs three periodic sums per node
    # (d3 c4 L = 4 1+1e-3i, route R 6.5 s, the route C attempt spent the whole 1e8-unit budget); the fill keeps route R there too
    if npC > NPCMAX || (D == 3 && fb < fa / 20)
        inf[:rte] = (ΛRS = ΛRS, λR = (maximum(λH), maximum(λA)), nC = 0, cSkip = npC); inf[:λe] = (λH, λA)
        return H, A, inf
    end
    # route C runs on at most half of the budget left: it only replaces entries route R already has, so running out of its share skips it
    # (d3 c4 L = 4 at 1+1e-3i, route R 6.5 s, route C's segment at Im f = 1e-3 alone spent the whole 1e8 units: "cost budget
    # exceeded" on an input route R answers); the entries route R flags go to the Double64 rerun (peD64)
    w1 = wtot()
    rC = try
        bw_ === nothing ? blcTns(R, sT, sS, f, ax, L, θ; kw..., route = :C) : withBud((bw_.lim - wrkTot(bw_)) / 2, bw_.mu) do
            try
                blcTns(R, sT, sS, f, ax, L, θ; kw..., route = :C)
            finally
                bw_.w[8 * Threads.threadid()] += wrkTot(BUD[])
            end
        end
    catch e
        (e isa ErrorException && (occursin("panel budget", e.msg) || occursin("cost budget exceeded", e.msg) || occursin("a priori cost", e.msg))) || rethrow()
        e
    end
    if rC isa Exception
        inf[:rte] = (ΛRS = ΛRS, λR = (maximum(λH), maximum(λA)), nC = 0, cSkip = rC.msg); inf[:λe] = (λH, λA)
        return H, A, inf
    end
    Hc, Ac, ic = rC
    λHc = lamE(ic[:sH], Hc, eT); λAc = lamE(ic[:sA], Ac, eT); λHc[Z] .= 0; λAc[Z] .= 0
    inf[:λR] = (copy(λH), copy(λA))
    Ho = copy(H); Ao = copy(A); nC = 0; fH = falses(3, 3); fA = falses(3, 3)
    for i in eachindex(H)
        λHc[i] < λH[i] && (Ho[i] = Hc[i]; nC += 1; fH[i] = true)
        λAc[i] < λA[i] && (Ao[i] = Ac[i]; nC += 1; fA[i] = true)
    end
    inf[:wrt] = (R = inf[:wrt].R, C = wtot() - w1); inf[:fromC] = (fH, fA)
    inf[:λe] = (min.(λH, λHc), min.(λA, λAc))
    inf[:rte] = (ΛRS = ΛRS, λR = (maximum(λH), maximum(λA)), λC = (maximum(λHc), maximum(λAc)), nC = nC)
    inf[:route] = nC == 0 ? :R : nC == 18 ? :C : :mixed; inf[:infoC] = ic
    return Ho, Ao, inf
end

# the outermost call sets the budget wmax (work units) and runs the per-entry Double64 fallback; nested calls (rteAuto's routes, the principal
# sheet) share the budget
const WMAX0 = 1.0e8

"""
    blcTns(R, sT, sS, f, ax, L, θ; tol, nG, cth, cg) -> (H, A, info)

Herm[G^per] = Σ_n e^{iθ·n} Re G(R + Λn) and AH[G^per] = Σ_n e^{iθ·n} Im G(R + Λn), raw normalization (1/V_t),
Gila's full entry (−δ/f² inside the n = 0 self image). ax: periodic axes (1..3 of them), L: periods, θ: T radians or
Rational twists θ/2π. Complex f with any sign of Im f (J10 continuation at Im f < 0).
"""
function blcTns(R, sT, sS, f::Complex{T}, ax::NTuple{D,Int}, L::NTuple{D,<:Rational}, θ::NTuple{D,Any}; wmax::Real = WMAX0, pe::Bool = true, kw...) where {T,D}
    blasMt() && return blas1(() -> blcTns(R, sT, sS, f, ax, L, θ; wmax, pe, kw...))
    if BUD[] !== nothing
        r = blcTns0(R, sT, sS, f, ax, L, θ; kw...)
        r[3][:wrk] = wrkTot(BUD[])
        return r
    end
    withBud(wmax) do
        r = blcTns0(R, sT, sS, f, ax, L, θ; kw...)
        r[3][:wrk] = wrkTot(BUD[])
        pe && (r = peWid(r, R, sT, sS, f, ax, L, θ; kw...))
        r
    end
end
# per-entry fallback (an entry is judged against eps(1 + κ_e), its own κ): an entry above entT of its kernel's
# max whose a posteriori rounding factor λ_e = Σ|terms_e|/|X_e| exceeds LAMD in the route chosen (small entries built by cancellation of
# order sums, e.g. x-odd entries at x ≡ 0 on a periodic axis with small θ: G_xy ∝ sin θ, λ_e 6e6, 1.4e7 eps) is recomputed with the same
# representation in the wider type (Float64 -> Double64, Double64 -> BigFloat 212 bits, BigFloat p -> 2p bits) at the same inputs (exact
# conversions) and tol = eps(T): rounding eps(wider)·λ_e, truncation unchanged, cost ~4x for Double64. The rerun's a priori cost is checked
# against its budget first; if it does not fit, the call refuses in one line (raise wmax)
const LAMD = 32; const LAMH = 256
# time per work unit of a Double64 run over a Float64 one (d1 c8 L = 4 1-0.1i R = (1/8,8,8), 1.46 against 30 µs per unit, the pole forms' angular sums; 20% margin)
const CD64 = 24.0
# the Double64 rerun's budget in units of the call's wmax (counted at CD64 per unit): a call ends within (1 + DBUD) wmax Float64-equivalent units
# (the rerun's own a priori pole-form estimate, 24 x 5.8e6, exceeded 1 wmax while the whole rerun took 123 s)
const DBUD = 2.0
# the rerun's type: Float64 -> Double64; Double64 -> BigFloat, BigFloat (p bits) -> BigFloat (2p bits), at the precision peWid sets
widT(::Type{Float64}) = Double64
widT(::Type{T}) where {T} = BigFloat
# time per work unit of the wider run over the T run (Double64 -> BigFloat at 212 bits and BigFloat p -> 2p, 20% margin)
# (D64PE input, route R, warm, 1 thread: Double64 197 µs/unit, BigFloat 212 bits 6113, BigFloat 128 bits 2757, 256 bits 9604)
const CWDB = 37.0; const CWBB = 4.2
cWid(::Type{Float64}) = CD64
cWid(::Type{Double64}) = CWDB
cWid(::Type{BigFloat}) = CWBB
# the precision of the wider rerun of a T = Double64 or BigFloat call (p0 bits): 2 p0, or more where the Float inputs span more bits (a
# Double64's hi and lo parts may lie far apart), so that f and θ convert exactly
function widPrc(p0, f, θ)
    spn(v::Double64) = iszero(LO(v)) ? 53 : exponent(HI(v)) - exponent(LO(v)) + 53
    spn(v) = v isa Rational ? 0 : precision(v)
    max(2 * p0, maximum(spn, (real(f), imag(f), (t for t in θ if !(t isa Rational))...)) + 8)
end
# the rerun of the flagged entries in the wider type: Float64 in Double64 (peD64); Double64 and BigFloat in BigFloat at widPrc bits, the
# merged entries rounded back to T (a BigFloat result to its own precision)
function peWid(r, R, sT, sS, f::Complex{T}, ax, L, θ; kw...) where {T}
    T === Float64 && return peD64(r, R, sT, sS, f, ax, L, θ; kw...)
    p0 = T === BigFloat ? precision(real(f)) : precision(T)
    pW = widPrc(p0, f, θ)
    H, A, inf = try
        setprecision(BigFloat, pW) do
            peD64(r, R, sT, sS, f, ax, L, θ; kw...)
        end
    catch e
        e isa ErrorException ? error(replace(e.msg, "Double64 rerun" => "BigFloat ($pW-bit) rerun", "beyond Double64 too" => "beyond BigFloat ($pW bits) too", "even in Double64" => "even in BigFloat ($pW bits)")) : rethrow()
    end
    rd(x) = T === BigFloat ? BigFloat(x; precision = p0) : T(x)
    H = map(z -> complex(rd(real(z)), rd(imag(z))), H); A = map(z -> complex(rd(real(z)), rd(imag(z))), A)
    haskey(inf, :pe) && inf[:pe] isa NamedTuple && haskey(inf[:pe], :d64) && (inf[:pe] = merge(inf[:pe], (wide = "BigFloat($pW)",)))
    return H, A, inf
end
function peD64(r, R, sT, sS, f::Complex{T}, ax, L, θ; tol = eps(T), entT = 1e-8, kw...) where {T}
    H, A, inf = r
    # an explicit route = :R or :C carries no λ_e yet: Σ|terms|/|X| of its own entries, every entry and the call's work from that route
    if !haskey(inf, :λe)
        haskey(inf, :sH) || return r
        inf[:λe] = (lamE(inf[:sH], H, T(entT)), lamE(inf[:sA], A, T(entT)))
        c = inf[:route] == :C; w = Float64(get(inf, :wrk, 0.0))
        inf[:fromC] = (fill(c, 3, 3), fill(c, 3, 3)); inf[:wrt] = c ? (R = 0.0, C = w) : (R = w, C = 0.0)
    end
    λH, λA = copy.(inf[:λe])
    haskey(inf, :symZ) && (λH[inf[:symZ]] .= 0; λA[inf[:symZ]] .= 0)
    fl = count(>(LAMD), λH) + count(>(LAMD), λA)
    if fl == 0
        inf[:pe] = (flagged = 0,)
        return r
    end
    b = BUD[]; wtot() = b === nothing ? 0.0 : wrkTot(b)
    λmx = max(maximum(λH), maximum(λA))
    D = widT(T); CD64 = cWid(T)
    # eps of the call's own precision (eps(BigFloat) would read peWid's wider scope)
    epsT = T === BigFloat ? ldexp(BigFloat(1), 1 - precision(real(f))) : eps(T)
    # (compared through log2: eps(BigFloat) at a high precision underflows Float64)
    lg2(v) = Float64(log2(big(v)))
    # the true λ is >= the Float64 one (|X_F64| is rounding noise at least as large as |X|): past eps/(4 eps(D64)) the Double64 rerun
    # would refuse too, so refuse now (λ 1e30, 214 s spent before the Double64 refusal)
    lg2(λmx) + lg2(eps(D)) > lg2(epsT) - 2 && error("blcTns: rounding amplification Σ|terms|/|X| >= $(round(Float64(λmx); sigdigits = 3)) is beyond Double64 too; refused")
    # the flagged entries are recomputed or the call is refused, never returned flagged (444 eps(1+κ) returned under "budget"). The
    # rerun repeats the Float64 run's work (same nodes, same tol) at CD64 times the time per unit; it gets a budget of its own, DBUD times the
    # call's wmax, counted at CD64 per unit, and is refused a priori if CD64 times the Float64 run's work does not fit
    # only the route the flagged entries come from is rerun (rteAuto's fromC), and only those entries are replaced: the d1 grazing point
    # d1 sl R = (1/2,0,8) 1+1e-3i took 2.6e7 units in both routes, its flagged entries all from route C
    FH = λH .> LAMD; FA = λA .> LAMD
    fC = get(inf, :fromC, (falses(3, 3), falses(3, 3))); wrt = get(inf, :wrt, (R = wtot(), C = 0.0))
    nC_ = any(FH .& fC[1]) || any(FA .& fC[2]); nR_ = any(FH .& .!fC[1]) || any(FA .& .!fC[2])
    rtD = nC_ && nR_ ? :auto : nC_ ? :C : :R
    wm = b === nothing ? Inf : b.lim; wF = wtot(); wD = rtD == :auto ? wrt.R + wrt.C : rtD == :C ? wrt.C : wrt.R
    # route R answers every entry: where its Double64 rerun is cheaper than that of the route(s) the flagged entries come from, rerun route R
    # (d1 L = 1 θ ≈ ±1.6e-6 1+1e-3i, the θ-odd entries cancel ~1/θ in both routes, λ_R 1.7e5..3.8e5 against λ_C 5.5e4..3.6e5;
    # route R took 2.7e4 units, route C 1e7, so the route C rerun was refused at 2.4e8 > 2 wmax while route R's costs 6.8e5)
    # (only where route R's own Float64 λ on the flagged entries leaves Double64 a margin of 16: route R's e^{Im(k) r} at far open offsets)
    λRf = haskey(inf, :λR) ? max(maximum(inf[:λR][1][FH]; init = 0.0), maximum(inf[:λR][2][FA]; init = 0.0)) : Inf
    swR = rtD != :R && wrt.R > 0 && wrt.R < wD && lg2(λRf) + lg2(eps(D)) <= lg2(epsT) - 6
    swR && (rtD = :R; wD = wrt.R)
    # entries with LAMD < λ_e <= LAMH whose rerun exceeds the budget are returned as computed, reported in info[:pe] (returned = :budget):
    # below LAMH the measured rounding is <= 4.2 λ_e eps, the pre-LAMD-32 statement; above LAMH the call refuses
    CD64 * wD > DBUD * wm && λmx <= LAMH && (inf[:pe] = (flagged = fl, λmax = λmx, returned = :budget); return r)
    CD64 * wD > DBUD * wm && error("blcTns: $fl entries need the $(D === BigFloat ? "BigFloat ($(precision(BigFloat))-bit)" : "Double64") rerun (rounding amplification Σ|terms|/|X| up to $(round(λmx; sigdigits = 3)) > $LAMD in the route taken), whose a priori cost $(round(CD64 * wD; sigdigits = 3)) work units ($(CD64)x the $T run's of route $rtD) exceeds its budget $(DBUD) wmax = $(round(DBUD * wm; sigdigits = 3)); refused (raise wmax)")
    fD = Complex{D}(D(real(f)), D(imag(f))); θD = Tuple(t isa Rational ? t : D(t) for t in θ)
    # keywords carried in T (τ, ψmx from peWid) in the rerun's type
    kw = NamedTuple{keys(kw)}(map(v -> v isa AbstractFloat ? D(v) : v, values(kw)))
    Hd, Ad, infD, wDr = try
        withBud(DBUD * wm, CD64) do
            # (the rerun's own rounding is eps(D): the wide phase of the pole form is not needed there)
            r_ = with(() -> blcTns0(R, sT, sS, fD, ax, L, θD; tol = D(tol), entT, kw..., route = rtD), WPH => false)
            (r_..., wrkTot(BUD[]))
        end
    catch e
        # (the soft tier also when the rerun itself runs out of budget: entries with λ_e <= LAMH are returned as computed)
        e isa ErrorException && λmx <= LAMH && occursin("budget", e.msg) && (inf[:pe] = (flagged = fl, λmax = λmx, returned = :budget); return r)
        e isa ErrorException ? error("blcTns, Double64 rerun of $fl flagged entries (λ_e up to $(round(λmx; sigdigits = 3))): " * e.msg) : rethrow()
    end
    b === nothing || (b.w[8 * Threads.threadid()] += wDr)
    # the Double64 run must itself be clear of its rounding: its own λ (Σ|terms|/|X| of the Double64 result) times eps(Double64) below eps/4;
    # otherwise the entries are below both precisions' floors (route R at open offsets of 1e3 λ with Im f > 0: terms ~ e^{-Im(k_z)ρ/3} cancelling
    # to e^{-Im(k_z)ρ}, λ ~ 1e200, with route C beyond its panel budget) and blcTns refuses rather than return rounding noise
    # (on the flagged entries)
    lH, lA = haskey(infD, :λe) ? (Float64.(infD[:λe][1]), Float64.(infD[:λe][2])) : (Float64.(lamE(infD[:sH], Hd, D(entT))), Float64.(lamE(infD[:sA], Ad, D(entT))))
    λD = max(maximum(lH[FH]; init = 0.0), maximum(lA[FA]; init = 0.0))
    lg2(λD) + lg2(eps(D)) > lg2(epsT) - 2 && error("blcTns: rounding amplification Σ|terms|/|X| = $(round(λD; sigdigits = 3)) even in Double64 (route $(infD[:route])); refused")
    # every entry the rerun's route supplied in the Float64 result takes the Double64 value (the same representation, more accurate), the
    # flagged ones included; entries of the other route keep their Float64 values (their λ_e <= LAMD)
    RH = rtD == :auto ? trues(3, 3) : rtD == :C ? fC[1] : .!fC[1]; RA = rtD == :auto ? trues(3, 3) : rtD == :C ? fC[2] : .!fC[2]
    # switched to route R: also every route C entry whose Double64 route R rounding eps(D64) λ_D,e is below its Float64 one eps λ_e
    if swR
        eD = Float64(eps(D) / epsT)
        RH = RH .| FH .| (eD .* lH .< λH); RA = RA .| FA .| (eD .* lA .< λA)
    end
    haskey(inf, :symZ) && (RH = RH .& .!inf[:symZ]; RA = RA .& .!inf[:symZ])
    H2 = copy(H); A2 = copy(A); H2[RH] .= Complex{T}.(Hd[RH]); A2[RA] .= Complex{T}.(Ad[RA])
    # (the rerun's explicit route bypasses blcTns0's auto path and its realness projection at exact twists)
    all(j -> θ[j] isa Rational ? isinteger(2 * θ[j]) : iszero(θ[j]), eachindex(θ)) && (H2 .= real.(H2); A2 .= real.(A2))
    inf[:pe] = (flagged = fl, λmax = Float64(max(maximum(λH), maximum(λA))), d64 = true, swR = swR, routeD = infD[:route], wrkD = wDr)
    return H2, A2, inf
end
function blcTns0(R0, sT0, sS0, f::Complex{T}, ax::NTuple{D,Int}, L0::NTuple{D,<:Rational}, θ::NTuple{D,Any};
               tol::T = eps(T), nG::Int = 0, cth::Real = 1, cg::Real = 1 / 2, τ::T = zero(T), ψmx::T = T(13) / 10,
               nGt::Int = 0, sheet::Symbol = :J10, Csh::Real = 4, qcert::Bool = true, route::Symbol = :auto,
               c1::Real = 1 // 8, c2::Real = 12, cU::Real = 4, lat::Symbol = :gila, entT::Real = 1e-8) where {T,D}
    # principal sheet at Im f < 0 (keyword): principal sqrt/E_1 commute with conj, so G_prin(f̄; θ) = conj G(f; -θ):
    # H_prin(f̄; θ) = H(f; θ), AH_prin(f̄; θ) = -AH(f; θ) (checked against the direct principal evaluation)
    if sheet == :principal && imag(f) < 0
        H_, A_, inf_ = blcTns(R0, sT0, sS0, conj(f), ax, L0, θ; tol, nG, cth, cg, τ, ψmx, nGt, sheet = :J10, Csh, qcert, route, c1, c2, cU, lat, entT)
        inf_[:sheet] = :principal
        return H_, -A_, inf_
    end
    sheet in (:J10, :principal) || error("blcTns: sheet = $sheet is neither :J10 nor :principal")
    sheetv = sheet
    inpChk("blcTns", f, ax, L0, θ)
    R = q3(R0); sT = q3(sT0); sS = q3(sS0); L = ntuple(i -> QI(L0[i]), D)
    all(>(0), sT) && all(>(0), sS) || error("blcTns: cell edges $sT, $sS are not all positive")
    lstChk(R, sT, sS, ax, L; lat)
    fa = real(f); fb = imag(f); cthv = T(cth); cgv = T(cg)
    x = 4 * T(π)^2 * (fa - fb) * (fa + fb); y = 8 * T(π)^2 * fa * fb
    # 1/f² = ((a-b)(a+b) - 2abi)/|f|⁴: both parts with relative accuracy
    af4 = abs2(f)^2; cr = (fa - fb) * (fa + fb) / af4; ci = -2 * fa * fb / af4
    Vt = prod(i -> toT(T, sT[i]), 1:3); Vs = prod(i -> toT(T, sS[i]), 1:3)
    # nGv, nGtv, L, umin, ... each bound once: reassigned variables captured by the closures were boxed (every node allocated)
    nGv = nG == 0 ? ceil(Int, -log(tol) / T(2)) + 8 : nG
    # every truncation relative to the smallest reported entry (1e-8 of the tensor max), not to the max: at tol alone a
    # 5e-8 entry carried 112 eps of spectral cutoff (d3 c8 L = 2λ 1+i); the Gauss order stays set by tol
    tol0 = tol; tolv = tol * T(entT)
    α = -log(tolv)
    nGtv = nGt == 0 ? nGv : nGt
    axk = [AxK(T, sT[i], sS[i]) for i in 1:3]
    # op with a length known from D (Tuple(setdiff(...)) was Tuple{Vararg{Int}}: every closure over the open axes was dynamic)
    op = ntuple(j -> setdiff(1:3, ax)[j], Val(3 - D))
    pers = Dict(ax[i] => PerAx(axk[ax[i]], L[i], twT(T, θ[i]), θ[i]) for i in 1:D)
    isper(i) = haskey(pers, i)
    Lmx = maximum(i -> toT(T, L[i]), 1:D)
    if route == :auto
        # the caller's tolv: the routes apply entT themselves (passing the reduced tolv applied it twice, 1e-16 of max: a runaway)
        kw = (; tol = tol0, nG = nGv, cth = cthv, cg = cgv, τ, ψmx, nGt = nGtv, sheet = sheetv, Csh, qcert, c1, c2, cU, lat, entT)
        Z = symZ(R, ax, L, ntuple(j -> twT(T, θ[j]), D))
        # a tensor that vanishes by reflection is returned as exact zeros at once: its certification target was the per-node rounding
        # noise of the cancelling image sums, which no bisection reaches (anisotropic d2 at θ = π, m = (0,0,±1): budget exceeded after 1e8 units)
        all(Z) && return zeros(Complex{T}, 3, 3), zeros(Complex{T}, 3, 3), Dict{Symbol,Any}(:route => :symzero, :symZ => Z, :sH => zeros(T, 3, 3), :sA => zeros(T, 3, 3), :λe => (zeros(T, 3, 3), zeros(T, 3, 3)), :pe => (flagged = 0,))
        Ha, Aa, infa = rteAuto(R, sT, sS, f, ax, L, θ, kw, Lmx, Z)
        Ha[Z] .= 0; Aa[Z] .= 0; infa[:symZ] = Z
        # every twist exact 0 or 1/2: -θ ≡ θ, H and AH real entry by entry; their imaginary parts are the route's rounding
        all(j -> θ[j] isa Rational ? isinteger(2 * θ[j]) : iszero(θ[j]), 1:D) && (Ha .= real.(Ha); Aa .= real.(Aa))
        return Ha, Aa, infa
    end
    rteInf = nothing
    route == :C && !(fb > 0 && fa > 0) && error("blcTns: route C needs Re f > 0 and Im f > 0")
    # τ is bound once (τv): a keyword reassigned in the body and captured by the closures below was boxed, and every use in the
    # order loops allocated (d3 L = 16: 32 GB in 26 s)
    τv = let τ = τ
        if τ == 0
            # hazard 2 (e^{xτ} <= e), and yτ <= 8: the e^{iyt} weight of the real-space part cancels far offsets beyond that
            τ = min(1 / (max(x, zero(T)) + 1), 8 / (abs(y) + eps(T)))
            # route C: the real segment ends where e^{Im(k) r - r²/4τ} <= e^{c1} for every r
            route == :C && (τ = min(1 / (max(x, zero(T)) + 1), T(c1) / (2 * T(π) * fb)^2))
            D == 3 && (τ = min(τ, Lmx^2 / (4 * T(π))))
        end
        τ
    end
    acc = Acc(T)
    # nox: the e^{xt} factor already folded into the terms (EV Poisson form: e^{-u_G t}, no overflow of e^{xt} at large t)
    # no default arguments on these closures: each default adds a method, i.e. a second assignment of the binding, which boxes it
    wts(t, nox) = (ex = nox ? one(T) : exp(x * t); (4 * T(π)^2 * ex * cos(y * t), 4 * T(π)^2 * ex * sin(y * t),
                                ex * (cos(y * t) * cr - sin(y * t) * ci), ex * (sin(y * t) * cr + cos(y * t) * ci)))
    # per-axis values at real t: full sums (periodic) or κ (open)
    # (values, Σ|terms|) per axis
    axValA(i, t) = isper(i) ? perSkA(pers[i], R[i], t, tolv) : (k_ = Complex{T}.(kap(axk[i], R[i], t, tolv)); (k_, abs.(k_)))
    function addNode!(vals, wq, t, ac, nox, av)
        wHd, wAd, wHp, wAp = wts(t, nox)
        aw = abs(wq)
        for b in 1:3, a in 1:3
            kk = one(Complex{T}); ka = one(T)
            for i in 1:3
                kk *= vals[i][dOrd(a, b, i) + 1]; ka *= av[i][dOrd(a, b, i) + 1]
            end
            kk /= Vt; ka /= Vt
            h = wq * wHp * kk; an = wq * wAp * kk
            mh = aw * abs(wHp) * ka; ma = aw * abs(wAp) * ka
            if a == b
                k0 = vals[1][1] * vals[2][1] * vals[3][1] / Vt; k0a = av[1][1] * av[2][1] * av[3][1] / Vt
                h += wq * wHd * k0; an += wq * wAd * k0
                mh += aw * abs(wHd) * k0a; ma += aw * abs(wAd) * k0a
            end
            add!(ac, a, b, h, an, mh, ma)
        end
    end
    function addNodeA!(ax_, wq, t, ac)
        va = [axValA(i, t) for i in 1:3]
        addNode!([v[1] for v in va], wq, t, ac, false, [v[2] for v in va])
    end
    # a priori work of the propagating orders' d_o = 2 pole forms (d = 1, route R): per order the angular nodes M (circM on radius 2√S) times the
    # graded s-panels times nGtv; refused here, before any other work, when it alone exceeds the budget (M ~ 2√S·ρ at open offsets of
    # 1e2-1e3 λ; F10 at 20λ periods and |k| = 63). The runtime counter bounds everything else
    if route == :R && D == 1
        ordR0 = begin
            P0 = pers[ax[1]]; tw0 = P0.tw isa QI ? toT(T, P0.tw) : P0.tw
            νm0 = max(sqrt(max(x + T(cthv) * abs(y), zero(T))) / (2 * T(π)), abs(fa)) + 1 / toT(T, P0.L)
            (floor(Int, tw0 - νm0 * toT(T, P0.L)) - 1):(ceil(Int, tw0 + νm0 * toT(T, P0.L)) + 1)
        end
        axO0 = [axk[o] for o in op]; xsO0 = [toT(T, R[o]) for o in op]
        est = zero(T); τe = τv
        for m in ordR0
            up = uOrd((nuP(T, m, pers[ax[1]]),), fa, fb); up < 0 || continue
            S0 = -up + (-log(tolv) + 12) / τe
            M0 = maximum(circM(axO0, xsO0, [dOrd(a, b, o) for o in op], sqrt(S0), tolv) for a in 1:3, b in 1:3)
            # the certified rule needs about one panel per oscillation of a(s) ~ e^{i√s X}: X√S/π panels (X = Σ|x_o| + cp)
            npn = max(length(grdPts(up, S0 + up, zero(T), abs(y) > eps(T)^2 * abs(x) ? abs(y) / 4 : max(-up, 1 / τe) / 8)) + 1,
                      ceil(Int, sum(k -> abs(xsO0[k]) + toT(T, axO0[k].cp), eachindex(op)) * sqrt(S0) / T(π)))
            est += T(0.1) * M0 * nGtv * npn
        end
        wrkChk(est, "the propagating orders' d_o = 2 pole forms ($(length(ordR0)) orders scanned)")
        info0 = est
    end
    # (the node loops below run over nGv/nGtv: with the keywords nG = nGt = 0 of a direct route = :R/:C call they ran over 0 nodes, the RS,
    # EV and propagating-order parts were silently missing while the certificates, built on nGv, passed)
    # RS: u = √t panels [0, u0], dyadic to √τv; below u0 every Gaussian of a nonzero kink distance is < e^{-(α+8)}
    # the nearest kinks sit on the images next to R (cp <= L): n0 + (-2:2), n0 = round(-R/L) (a scan of -3:3 around 0,
    # so for |R| > 3L u0 came from the wrong image: 2.7e6 eps on the twist relation at n = -7)
    dmin = T(Inf)
    for i in 1:3
        A = axk[i]
        ns = isper(i) ? (round(Int, toT(T, -R[i] / pers[i].L)) .+ (-2:2)) : (0:0)
        for n in ns, pj in (A.cp, A.cm, -A.cm, -A.cp)
            yv = abs(R[i] + (isper(i) ? n * pers[i].L : QI(0)) + pj)
            yv != 0 && (dmin = min(dmin, toT(T, yv)))
        end
    end
    u0 = dmin / (2 * sqrt(α + 8)); uτ = sqrt(τv)
    edges = T[0]; ue = min(u0, uτ); push!(edges, ue)
    while ue < uτ
        ue = min(2 * ue, uτ); push!(edges, ue)
    end
    xg, wg = glq(T, nGv); xt, wt = glq(T, nGtv)
    Cshv = T(Csh)
    # sharpness data: Poisson decay rate of the periodic factors and the squared distance of the offset from the supports
    νmn(P) = (tw = P.tw isa QI ? toT(T, mod(P.tw, 1)) : mod(P.tw, one(T)); min(tw, 1 - tw) / toT(T, P.L))
    # the Poisson decay e^{-4π²ν_min² t} of a periodic factor only holds where that axis is in Poisson form (t > L²/4π)
    ueRS(t) = max(zero(T), 4 * T(π)^2 * sum(i -> isper(i) && !useReal(T, pers[i], t) ? νmn(pers[i])^2 : zero(T), 1:3) - x)
    dAx(i) = isper(i) ? minimum(n -> max(toT(T, abs(R[i] + n * pers[i].L) - axk[i].cp), zero(T)), round(Int, toT(T, -R[i] / pers[i].L)) .+ (-1:1)) :
                        max(toT(T, abs(R[i]) - axk[i].cp), zero(T))
    d2RS = sum(i -> dAx(i)^2, 1:3)
    shRS(t) = ueRS(t) * t + d2RS / (4 * t)
    ERS(u) = -ueRS(u^2) * u^2 - d2RS / (4 * u^2)
    EMRS = maximum(ERS, edges[2:end])
    d2RS > 0 && (EMRS = max(EMRS, -sqrt(ueRS(τv) * d2RS)))
    nNod = 0; rsPan = Tuple{T,T}[]
    for p in 1:length(edges)-1
        lo0, hi0 = edges[p], edges[p + 1]
        kph = max(1, ceil(Int, abs(y) * (hi0^2 - lo0^2) / T(3 // 2)))
        sub = lo0 == 0 ? [lo0 + (hi0 - lo0) * T(j) / kph for j in 0:kph] : shpSub(lo0, hi0, v -> 4 * shRS(v^2), Cshv; E = ERS, EM = EMRS, αs = α + 20, kmin = kph)
        for q in 1:length(sub)-1
            lo, hi = sub[q], sub[q + 1]
            push!(rsPan, (lo, hi))
            for j in 1:nGv
                u = (hi - lo) / 2 * xg[j] + (hi + lo) / 2; wq = (hi - lo) / 2 * wg[j] * 2 * u
                t = u^2
                addNodeA!(nothing, wq, t, acc); nNod += 1
            end
        end
    end
    info = Dict{Symbol,Any}(:τ => τv, :nRS => nNod, :d => D, :rsPan => rsPan, :u0 => u0, :rte => rteInf); info[:RS] = (copy(acc.H .+ acc.cH), copy(acc.A .+ acc.cA))
    # check hook: n-point Gauss on the u-panel [lo, hi] into a fresh accumulator
    function rsInt(lo, hi, n)
        a_ = Acc(T); xq, wq_ = glq(T, n)
        for j in 1:n
            u = (hi - lo) / 2 * xq[j] + (hi + lo) / 2
            addNodeA!(nothing, (hi - lo) / 2 * wq_[j] * 2 * u, u^2, a_)
        end
        (a_.H .+ a_.cH, a_.A .+ a_.cA)
    end
    # Gauss bound of the first RS panel u ∈ [0, hi], hi <= u0 = dmin/(2√(α+8)). There t = u² < cp², so
    # every axis factor is in erf form: κ = P + ε with P the part of the kinks at distance 0 and the trapezoid itself
    # (P0 = tri + (Σ_zero sg) u/√π, P1 = dtr, 2u·P2 = Σ_zero sg/√π: polynomials in u) and ε the Gaussian part of the kinks at
    # distance |y| >= dmin, v = |y|/2u >= √(α+8): |u ierfc v| <= u e^{-v²}/(2v²√π), |erfc(v)/2| <= e^{-v²}/(2v√π),
    # |2u e^{-v²}/(2√π u)| <= e^{-v²}/√π (erfc v <= e^{-v²}/(v√π)), all increasing in u. Integrand = 2u W(u²) Π_i κ_i^{(k_i)}:
    # the polynomial-times-entire part by ATAP 19.3 on E_ρ (|u| <= m + hA_ρ, |e^{zu²}| <= e^{|z||u|²}), the rest bounded on the
    # real panel by Π(|P|+ε) - Π|P| and counted twice (integral and rule, positive weights of sum hi)
    function rs0Bnd(hi::T)
        tri(A, yq) = sum(sg * abs(yq + pj) / 2 for (pj, sg) in zip((A.cp, A.cm, -A.cm, -A.cp), (1, -1, -1, 1)))
        dat = map(1:3) do i
            A = axk[i]
            ns = isper(i) ? (round(Int, toT(T, -R[i] / pers[i].L)) .+ (-4:4)) : (0:0)
            T0 = zero(T); C0 = zero(T); D1 = zero(T); e0 = zero(T); e1 = zero(T); e2 = zero(T)
            for n in ns
                xq = R[i] + (isper(i) ? n * pers[i].L : QI(0))
                T0 += abs(toT(T, tri(A, xq)))
                dt = zero(QI); zs = 0
                for (pj, sg) in zip((A.cp, A.cm, -A.cm, -A.cp), (1, -1, -1, 1))
                    yq = xq + pj
                    dt += sg * sign(yq) / 2
                    if yq == 0
                        zs += sg
                    else
                        v = toT(T, abs(yq)) / (2 * hi); ev = exp(-v^2) / sqrt(T(π))
                        e0 += hi * ev / (2 * v^2); e1 += ev / (2 * v); e2 += ev
                    end
                end
                C0 += abs(zs) / sqrt(T(π)); D1 += abs(toT(T, dt))
            end
            (T0 = T0, C0 = C0, D1 = D1, Z2 = C0, e = (e0, e1, e2))
        end
        # |P^{(k)}| at |u| <= ua (k = 2 with the Jacobian 2u folded in)
        Pk(d, k, ua) = k == 0 ? d.T0 + d.C0 * ua : k == 1 ? d.D1 : d.Z2
        az = abs(complex(x, y))
        function entMaj(ua, epsOn, wδ, w∂)
            m = zero(T)
            for b in 1:3, a in 1:3
                v = zero(T)
                for (w, kk) in ((w∂, ntuple(i -> dOrd(a, b, i), 3)), (a == b ? wδ : zero(T), (0, 0, 0)))
                    w == 0 && continue
                    j2 = any(==(2), kk) ? one(T) : 2 * ua
                    pP = j2; pE = j2
                    for i in 1:3
                        pk = Pk(dat[i], kk[i], ua)
                        pP *= pk; pE *= pk + (epsOn ? dat[i].e[kk[i] + 1] : zero(T))
                    end
                    v += w * (epsOn ? pE - pP : pP)
                end
                m = max(m, v)
            end
            m
        end
        # H: |e^{zu²}| <= e^{|z||u|²}, |1/f²|; AH: |e^{xu²} sin(yu²)| <= e^{|x||u|²} sinh(|y||u|²) and |sin cr + cos ci| <= sinh|cr| + cosh|ci|;
        # on the real panel |sin(y t)| <= min(1, |y| t)
        h = hi / 2; bH = T(Inf); bA = T(Inf)
        for ρ in RHOS
            ua = h + h * (T(ρ) + 1 / T(ρ)) / 2
            g = (64 / T(15)) * h * T(ρ)^(-2 * nGv) / (T(ρ)^2 - 1)
            bH = min(bH, g * exp(az * ua^2) * entMaj(ua, false, 4 * T(π)^2 / Vt, 1 / (abs2(f) * Vt)))
            sy = sinh(abs(y) * ua^2); cy = cosh(abs(y) * ua^2)
            bA = min(bA, g * exp(abs(x) * ua^2) * entMaj(ua, false, 4 * T(π)^2 * sy / Vt, (sy * abs(cr) + cy * abs(ci)) / Vt))
        end
        sr = min(one(T), abs(y) * hi^2); ex = 2 * hi * exp(max(x, zero(T)) * hi^2)
        (bH + ex * entMaj(hi, true, 4 * T(π)^2 / Vt, 1 / (abs2(f) * Vt)), bA + ex * entMaj(hi, true, 4 * T(π)^2 * sr / Vt, (sr * abs(cr) + abs(ci)) / Vt))
    end
    # certification pass: every RS (u) and EV (t) panel whose a priori Gauss bounds exceed tolv·max|H|/(10·#panels) (H weights) or
    # tolv·max|AH|/(10·#panels) (AH weights) is bisected until they do not (depth <= MAXDP, the first u-panel [0, hi] by halving with rs0Bnd);
    # the difference of the refined and the production rule is added. A panel still above target at the depth or panel budget is counted
    # uncertified; blcTns then raises (uncertified truncations were returned silently)
    function certify!(pans, sq; bf = nothing, evf = nothing)
        Hc, Ac = fin(acc)
        # scale floored at Σ|terms|/64: a kernel that vanishes by symmetry (θ = π, R_i = L_i/2) is rounding, and a target below its
        # rounding floor eps·Σ|terms| cannot be certified (runaway to MAXLEAF); the floor keeps truncation <= Σ|terms|·tolv/64
        SH = max(maximum(abs, Hc), maximum(acc.sH) / 64); SA = max(maximum(abs, Ac), maximum(acc.sA) / 64)
        SH == 0 && SA == 0 && return (bound = zero(T), bisected = 0, uncertified = 0, rel = zero(T))
        # targets as ratios (tolv·S underflowed below S ~ 1e-284: AH ∝ Im f at Im f = 1e-300)
        np = max(length(pans), 1); tq = tolv / (10 * np)
        totH = zero(T); totA = zero(T); nb = 0; nu = 0; nleaf = 0
        integ(lo, hi, ac) = for j in 1:(sq ? nGv : nGtv)
            if sq
                u = (hi - lo) / 2 * xg[j] + (hi + lo) / 2; wq = (hi - lo) / 2 * wg[j] * 2 * u
                addNodeA!(nothing, wq, u^2, ac)
            else
                t = (hi - lo) / 2 * xt[j] + (hi + lo) / 2; wq = (hi - lo) / 2 * wt[j]
                evf(ac, t, wq)
            end
        end
        bnd(a, b) = a == 0 ? (sq ? rs0Bnd(b) : (T(Inf), T(Inf))) : bf === nothing ? panBnd(R, sT, sS, f, pers, axk, (a + b) / 2, (b - a) / 2, sq ? nGv : nGtv, sq, tolv) : bf(a, b)
        ok(bb, dp) = rlB(bb[1], SH) <= tq / 2^dp && rlB(bb[2], SA) <= tq / 2^dp
        for (lo, hi) in pans
            b0 = bnd(lo, hi)
            if ok(b0, 0)
                totH += b0[1]; totA += b0[2]; continue
            end
            nb += 1
            old = Acc(T); integ(lo, hi, old)
            new = Acc(T)
            stack = [(lo, hi, 0, b0)]
            while !isempty(stack)
                a_, b_, dp, bb = pop!(stack)
                # no progress from an infinite majorant after 12 halvings: give up on this panel
                if !ok(bb, dp) && dp < MAXDP && nleaf < MAXLEAF && (all(isfinite, bb) || dp < 12)
                    c = a_ == 0 ? b_ / 2 : sqrt(a_ * b_)
                    push!(stack, (a_, c, dp + 1, bnd(a_, c))); push!(stack, (c, b_, dp + 1, bnd(c, b_)))
                else
                    integ(a_, b_, new); totH += bb[1]; totA += bb[2]; nleaf += 1
                    ok(bb, dp) || (nu += 1)
                end
            end
            Hn, An = fin(new); Ho, Ao = fin(old)
            for j in 1:3, i in 1:3
                add!(acc, i, j, Hn[i, j] - Ho[i, j], An[i, j] - Ao[i, j])
            end
        end
        return (bound = (totH, totA), bisected = nb, uncertified = nu, rel = max(SH > 0 ? totH / SH : zero(T), SA > 0 ? totA / SA : zero(T)))
    end
    info[:rsChk] = (rs0Bnd = rs0Bnd, rsInt = rsInt)
    if route == :C
        qcert && (info[:qcert] = certify!(rsPan, true))
        # Γ: τv → τc = τv + scC e^{iφ*}, φ* = -arg(-z)/2 (every image's saddle t* = r/(2√-z) lies on this ray), scC so that
        # Re(-z τc) >= Im(k) r_n + c2 (the spectral/order terms at τc below the entry scale e^{-Im(k) r_n})
        φ = -angle(complex(-x, -y)) / 2
        # kkC, not kk: an outer kk made addNode!'s and evNode!'s local kk one boxed variable shared by every node (the whole route-R
        # accumulation dynamically typed: d3 L = 16 f = 1 allocated 32 GB in 26 s)
        kkC = 2 * T(π) * abs(f); ik = 2 * T(π) * fb
        dA(i) = isper(i) ? minimum(n -> max(toT(T, abs(R[i] + n * pers[i].L) - axk[i].cp), zero(T)), round(Int, toT(T, -R[i] / pers[i].L)) .+ (-1:1)) : zero(T)
        rn = sqrt(sum(i -> dA(i)^2, 1:3))
        scC = max(zero(T), (ik * rn + T(c2) + x * τv) / (kkC * ik))
        # d < 3: the image segment only while every periodic factor is in image form (|t| <= L_min²/4π); beyond it the orders
        # each on their own saddle ray (an evanescent order's e^{-Re√p ρ} is far below the image scale e^{-Im(k) r})
        D < 3 && (scC = min(scC, max(zero(T), minimum(i -> toT(T, L[i])^2, 1:D) / (4 * T(π)) - τv)))
        U = T(cU) * abs(y)
        Cp, ip, ap_, igP, bdP = cPart(R, sT, sS, axk, pers, ax, f, τv, φ, scC, U, tolv, nGtv)
        pm = Dict(k => negAx(v) for (k, v) in pers)
        Cm, im_, am_, igM, bdM = cPart(R, sT, sS, axk, pm, ax, f, τv, φ, scC, U, tolv, nGtv)
        if qcert
            # every complex panel certified by its Bernstein bound against tolv·SC_/(10·#panels), SC_ = the smaller of the
            # two kernels' max-norms (both must meet it); bisection (depth <= 6) replaces the production rule where needed
            H0_, A0_ = fin(acc)
            Hx = H0_ .+ (Cp .+ conj.(Cm)) ./ 2; Ax_ = A0_ .+ (Cp .- conj.(Cm)) ./ (2im)
            mHx = max(maximum(abs, Hx), maximum(acc.sH .+ (ap_.A .+ am_.A) ./ 2) / 64); mAx = max(maximum(abs, Ax_), maximum(acc.sA .+ (ap_.A .+ am_.A) ./ 2) / 64)
            SC_ = min(mHx, mAx); SC_ == 0 && (SC_ = max(mHx, mAx))
            np_ = length(ip[:pans]) + length(im_[:pans])
            tqC = tolv / (10 * max(np_, 1))
            tot = zero(T); nbC = 0; nuC = 0
            for (w, (pl, ig, bd, ia)) in enumerate(((ip[:pans], igP, bdP, ip), (im_[:pans], igM, bdM, im_)))
                fl = CPan{T}[]; nb0C = nbC
                for pn in pl
                    st = [(pn, 0)]
                    while !isempty(st)
                        q, dp = pop!(st)
                        bb = bd(q)
                        if rlB(bb, SC_) > tqC / 2^dp && dp < MAXDP && length(fl) + length(st) < MAXLEAF && (isfinite(bb) || dp < 12)
                            dp == 0 && (nbC += 1)
                            h = q.Δ / 2
                            push!(st, (CPan{T}(q.kind, q.tm - h, h, q.g), dp + 1)); push!(st, (CPan{T}(q.kind, q.tm + h, h, q.g), dp + 1))
                        else
                            push!(fl, q); tot += bb
                            rlB(bb, SC_) > tqC / 2^dp && (nuC += 1)
                        end
                    end
                end
                # the certified rule, integrated from scratch (a correction added onto the production sum would carry its rounding)
                if nbC > nb0C
                    a2 = CAcc(T)
                    for q in fl
                        ig(a2, q)
                    end
                    Cn = cfin(a2) .+ ia[:tail]
                    w == 1 ? (Cp = Cn; ap_ = a2) : (Cm = Cn; am_ = a2)
                end
            end
            info[:qcertC] = (bound = tot, rel = tot / SC_, bisected = nbC, uncertified = nuC, npan = np_)
        end
        for b in 1:3, a in 1:3
            h = (Cp[a, b] + conj(Cm[a, b])) / 2; an = (Cp[a, b] - conj(Cm[a, b])) / (2 * im)
            nAdd!(acc.H, acc.cH, a, b, h); nAdd!(acc.A, acc.cA, a, b, an)
            # (+ the closed-form tail's Σ|terms|, acT: it was left out of λ)
            tS = (ap_.A[a, b] + am_.A[a, b] + ip[:acT].A[a, b] + im_[:acT].A[a, b]) / 2
            acc.sH[a, b] += tS; acc.sA[a, b] += tS
        end
        info[:route] = :C; info[:φ] = φ; info[:sc] = scC; info[:cP] = ip; info[:cM] = im_; info[:U] = U; info[:cChk] = (igP, bdP)
        qcert && chkCert(info)
        info[:sH] = acc.sH; info[:sA] = acc.sA
        Hf, Af = fin(acc)
        return Hf, Af, info
    end
    info[:route] = :R
    if D == 3
        # spectral tail, closed form per q: (Vs/V_Λ) F e^{2πiν·R} [4π² δ W − 4π² νν W/f²], W = e^{-(Q-z)τv}/(Q-z)
        VΛ = prod(i -> toT(T, L[i]), 1:3)
        kS3 = α + 12
        Pv3 = [pers[i] for i in 1:3]
        # cutoff relative to the largest term e^{-u_S τ}, u_S = max(0, smallest u over all orders): with every order evanescent (a short period
        # at θ near π) the absolute cutoff u τ <= α + 12 kept only the orders within e^{-5} of the largest and dropped the rest unseen
        # (2e-3 of Σ|terms| truncated, λ_e = 1)
        uS0 = uOrd(ntuple(i -> νmn(Pv3[i]), 3), fa, fb); uS = max(uS0, zero(T))
        function tail3!(ac, τq)
            rngs = [begin
                tw = Pv3[i].tw isa QI ? toT(T, Pv3[i].tw) : Pv3[i].tw
                νm = sqrt(max(x, zero(T)) + uS + kS3 / τq) / (2 * T(π)) + 1 / toT(T, Pv3[i].L)
                (floor(Int, tw - νm * toT(T, Pv3[i].L)) - 1):(ceil(Int, tw + νm * toT(T, Pv3[i].L)) + 1)
            end for i in 1:3]
            local nSp = 0
            # per-axis tables of ν, the form factor and the Poisson phase (the same numbers; formed once per order index instead of once per
            # order triple: d3 L = 16 f = 1 allocated 52 GB in 52 s)
            νt = [[nuP(T, m, Pv3[i]) for m in rngs[i]] for i in 1:3]
            sct = [[sinc(νt[i][j] * toT(T, sT[i])) * sinc(νt[i][j] * toT(T, sS[i])) for j in eachindex(rngs[i])] for i in 1:3]
            pht = [[phsP(T, m, Pv3[i], R[i]) for m in rngs[i]] for i in 1:3]
            phe, pho, jr = prTab(T, Pv3, rngs, pht, R)
            for j1 in jr[1], j2 in jr[2], j3 in jr[3]
                mm = (rngs[1][j1], rngs[2][j2], rngs[3][j3]); jj = (j1, j2, j3)
                ν = (νt[1][j1], νt[2][j2], νt[3][j3])
                uq = uOrd(ν, fa, fb)
                (uq - uS) * τq > kS3 && continue
                nSp += 1; wrk!(0.5)
                abs(uq) <= 64 * eps(T) * abs(x) && abs(y) <= 64 * eps(T) * abs(x) && error("blcTns: empty-lattice mode (3D) at rounding distance, |q|² − k² = $(uq) + $(y)i, orders $mm")
                e = exp(-uq * τq); den = uq^2 + y^2
                β0 = y * τq
                ReW = e * (uq * cos(β0) - y * sin(β0)) / den; ImW = e * (uq * sin(β0) + y * cos(β0)) / den
                ReWf = ReW * cr - ImW * ci; ImWf = ReW * ci + ImW * cr
                c0 = Vs / VΛ * prod(i -> sct[i][jj[i]], 1:3)
                # diagonal: 4π²W - 4π²ν_a²W/f² = (-E + 4π² S_a W)/f², E = e^{-pτ} = pW, S_a = Σ_{d≠a} ν_d² (p = 4π²(|ν|² - f²) exactly): the δ part and the
                # ν_a² part each carry the pole 1/p and cancel it where q_G is near a mode along axis a (AH_xx at q = (2π,0,0), 1+1e-3i,
                # 5.8e5 eps with λ_e 1-8.6, |h| hid the two terms); off-diagonal entries have no δ part. Σ|terms| counts every product apart
                ReE = e * cos(β0); ImE = e * sin(β0)
                for b in 1:3, a in 1:3
                    c = c0 * prod(i -> isodd(dOrd(a, b, i)) ? pho[i][jj[i]] : phe[i][jj[i]], 1:3); ac_ = abs(c)
                    if a == b
                        Sa = 4 * T(π)^2 * sum(d -> d == a ? zero(T) : ν[d]^2, 1:3); νa = 4 * T(π)^2 * ν[a]^2
                        if dgE(Sa, νa, hypot(ReW, ImW), e, hypot(cr, ci))
                            zr = Sa * ReW - ReE; zi = Sa * ImW - ImE; azr = Sa * abs(ReW) + abs(ReE); azi = Sa * abs(ImW) + abs(ImE)
                            h = c * (zr * cr - zi * ci); an = c * (zr * ci + zi * cr)
                            add!(ac, a, b, h, an, ac_ * (azr * abs(cr) + azi * abs(ci)), ac_ * (azr * abs(ci) + azi * abs(cr)))
                        else
                            h = c * (4 * T(π)^2 * ReW - νa * ReWf); an = c * (4 * T(π)^2 * ImW - νa * ImWf)
                            add!(ac, a, b, h, an, ac_ * (4 * T(π)^2 * abs(ReW) + νa * (abs(ReW * cr) + abs(ImW * ci))), ac_ * (4 * T(π)^2 * abs(ImW) + νa * (abs(ReW * ci) + abs(ImW * cr))))
                        end
                    else
                        νν = 4 * T(π)^2 * ν[a] * ν[b]
                        h = -c * νν * ReWf; an = -c * νν * ImWf
                        add!(ac, a, b, h, an, ac_ * abs(νν) * (abs(ReW * cr) + abs(ImW * ci)), ac_ * abs(νν) * (abs(ReW * ci) + abs(ImW * cr)))
                    end
                end
            end
            nSp
        end
        # evanescent extension: with every order evanescent, u >= u_S > 4|y|, the integrand decays like e^{-u_S t} and the tail's terms
        # (Poisson along every axis) cancel to the entry's e^{-√u_S d}; the real-space part (per-axis image/Poisson forms, no cancellation: every
        # factor is a sum tracked by Σ|terms|, |e^{xt}| <= e^{π/4} while the short axis is in image form) is continued to τE, where the tail's
        # Σ|terms| of every entry is <= |X_RS,e|/2 (entries above entT of the max), so the tail adds <= 1 to λ_e
        τq = τv; nExt = 0
        if uS0 > 4 * abs(y) && max(x, zero(T)) * τv < 200
            Zs = symZ(R, ax, L, ntuple(j -> twT(T, θ[j]), D))
            for it in 1:12
                aT = Acc(T); tail3!(aT, τq)
                Hr, Ar = fin(acc); r = zero(T)
                for (X, S) in ((Hr, aT.sH), (Ar, aT.sA))
                    # (an RS part still exactly 0, its peak far beyond τ: the tail is all there is, r = Inf, f = 10 at x = 1: λ_e 1.95e18)
                    mX = maximum(abs, X); mX == 0 && (maximum(S) > 0 && (r = T(Inf)); continue)
                    for e in eachindex(X); Zs[e] || (r = max(r, S[e] / max(abs(X[e]), T(entT) * mX))); end
                end
                r <= 1 / 2 && break
                # (at most 4x per step: the RS part grows as the peak is reached, r of the first steps overestimates)
                τn = min(τq + (log(2 * r) + 1) / uS0, 4 * τq)
                x > 0 && (τn = min(τn, 200 / x))
                τn <= τq && break
                ue = sqrt(τq); uN = sqrt(τn); eg = T[ue]
                while ue < uN; ue = min(2 * ue, uN); push!(eg, ue); end
                EMx = max(EMRS, maximum(ERS, eg))
                for p in 1:length(eg)-1
                    lo0, hi0 = eg[p], eg[p + 1]
                    kph = max(1, ceil(Int, abs(y) * (hi0^2 - lo0^2) / T(3 // 2)))
                    sub = shpSub(lo0, hi0, v -> 4 * shRS(v^2), Cshv; E = ERS, EM = EMx, αs = α + 20, kmin = kph)
                    for q in 1:length(sub)-1
                        lo, hi = sub[q], sub[q + 1]
                        push!(rsPan, (lo, hi))
                        for j in 1:nGv
                            u = (hi - lo) / 2 * xg[j] + (hi + lo) / 2
                            addNodeA!(nothing, (hi - lo) / 2 * wg[j] * 2 * u, u^2, acc); nExt += 1
                        end
                    end
                end
                τq = τn
            end
        end
        info[:τE] = τq; info[:nExt] = nExt
        nSp = tail3!(acc, τq)
        info[:nSp] = nSp
        qcert && (info[:qcert] = certify!(rsPan, true))
        qcert && chkCert(info)
        info[:sH] = acc.sH; info[:sA] = acc.sA
        Hf, Af = fin(acc)
        return Hf, Af, info
    end
    # d < 3: the orders G, SP set
    Pv = [pers[ax[i]] for i in 1:D]
    kS = α + 12
    ordR = [begin
        tw = Pv[i].tw isa QI ? toT(T, Pv[i].tw) : Pv[i].tw
        νm = max(sqrt(max(x + cthv * abs(y), zero(T))) / (2 * T(π)), abs(fa)) + 1 / toT(T, Pv[i].L)
        (floor(Int, tw - νm * toT(T, Pv[i].L)) - 1):(ceil(Int, tw + νm * toT(T, Pv[i].L)) + 1)
    end for i in 1:D]
    SP = NTuple{D,Int}[]
    for mm in Iterators.product(ordR...)
        ν = ntuple(i -> nuP(T, mm[i], Pv[i]), D)
        (uOrd(ν, fa, fb) <= cthv * abs(y) || (sheetv == :J10 && fb < 0 && abs(fa) >= sqrt(sum(abs2, ν)))) && push!(SP, mm)
    end
    info[:SP] = SP
    # EV: real t axis from τv; smallest EV decay u_min over the orders not in SP (scan a generous box)
    box = [begin
        tw = Pv[i].tw isa QI ? toT(T, Pv[i].tw) : Pv[i].tw
        νm = sqrt(max(x, zero(T)) + abs(y) * (cthv + 1)) / (2 * T(π)) + 3 / toT(T, Pv[i].L)
        (floor(Int, tw - νm * toT(T, Pv[i].L)) - 2):(ceil(Int, tw + νm * toT(T, Pv[i].L)) + 2)
    end for i in 1:D]
    umin = minimum((uOrd(ntuple(i -> nuP(T, mm[i], Pv[i]), D), fa, fb) for mm in Iterators.product(box...) if !(mm in SP)); init = T(Inf))
    umin > 0 || error("blcTns: internal, EV order with u <= 0")
    abs(umin) <= 64 * eps(T) * max(abs(x), 1) && error("blcTns: Rayleigh-Wood anomaly (evanescent side) at rounding distance, u = $umin")
    # beyond t_max the EV integrand is below e^{-(α+12)} of its maximum, which for open distances ρ sits near the saddle
    # t* = ρ/(2√u) of e^{-u t - ρ²/4t}: (√(u t) - ρ/(2√t))² >= α + 12
    ρo = sqrt(sum(o -> max(toT(T, abs(R[o]) - axk[o].cp), zero(T))^2, op))
    αp = α + 12
    sq = (sqrt(αp) + sqrt(αp + 2 * sqrt(umin) * ρo)) / (2 * sqrt(umin))
    tmax = max(τv + αp / umin, sq^2)
    info[:umin] = umin; info[:tmax] = tmax
    spProj = [unique(g[i] for g in SP) for i in 1:D]
    # ±m pairs at exact twists (prRol): the leading order of each orbit carries the paired prefactors and its partners are skipped, in the
    # SP orders (their p, open-axis integrals and rays are the same numbers) and in the EV Poisson sums (the same e^{-ut}); SP is symmetric
    # under each axis' reflection (u depends on ν_j²), so the orbit test is per axis
    SPs = Set(SP)
    rlSP(g) = ntuple(j -> prRol(Pv[j], g[j], v -> ntuple(k -> k == j ? v : g[k], D) in SPs), D)
    poiTrmR(m, P, x, t, rl) = ((pr, ν) = poiPreR(T, m, P, x, rl); e = exp(-4 * T(π)^2 * ν^2 * t); (pr[1] * e, pr[2] * e, pr[3] * e))
    # the EV factor as (image form - SP Poisson terms) cancels by e^{(u_min - u_SP,min) t} (the SP terms do not decay, or grow, while
    # the EV remainder decays as e^{-u_min t}): beyond (u_min - u_SP,min) t > ln 4 the EV-only Poisson sum is used even at t < L²/4π
    # (the image form up to L²/4π gave 122 eps at d1 c8 L = 2λ 1+0.1i R = 0)
    uSPm = minimum((uOrd(ntuple(i -> nuP(T, g[i], Pv[i]), D), fa, fb) for g in SP); init = umin)
    cancEV(t) = (umin - uSPm) * t > log(T(4))
    νe2 = (umin + x) / (4 * T(π)^2)
    # Gauss bound of an EV panel [lo, hi] (real t, d < 3): the EV integrand is the EV-only Poisson sum times the open factors,
    # every order with e^{xt} folded in, |e^{-u_G t}| <= e^{-u_G min Re t} on E_ρ (u_G > 0 off SP) and |cos, sin(yt)| <= cosh(y max|Im t|);
    # panBnd's e^{x⁺ max Re t} times the full Poisson form overflowed (Inf, 93 panels uncertified at d2 c8 L = 3λ f = 1)
    function evBnd(lo::T, hi::T)
        m = (lo + hi) / 2; h = (hi - lo) / 2; bH = T(Inf); bA = T(Inf)
        for ρ in RHOS
            e = ellT(m, h, T(ρ), false); e === nothing && continue
            Mo = Dict(o => [axMaj(axk[o], R[o], nothing, k, e, tolv) for k in 0:2] for o in op)
            rgs = [poiRngE(T, Pv[i], e.rlo, α + 40, νe2) for i in 1:D]
            Mp = zeros(T, 3, 3)   # Σ over EV orders of Π_i |pre_i| (2π|ν_i|)^{k_i} e^{-u rlo}, per derivative pair (k1, k2)
            for mm in Iterators.product(rgs...)
                mm in SP && continue
                ν = ntuple(i -> nuP(T, mm[i], Pv[i]), D)
                c = exp(-uOrd(ν, fa, fb) * e.rlo) * prod(i -> abs(poiPre(T, mm[i], Pv[i], R[ax[i]])[1][1]), 1:D)
                for k1 in 0:2, k2 in 0:(D == 2 ? 2 - k1 : 0)
                    Mp[k1 + 1, k2 + 1] += c * (2 * T(π) * abs(ν[1]))^k1 * (D == 2 ? (2 * T(π) * abs(ν[2]))^k2 : one(T))
                end
            end
            dH, pH, dA, pA = wHA(y, cr, ci, e.ihi, e.thi)
            MH = zero(T); MA = zero(T)
            for b in 1:3, a in 1:3
                kp(i) = dOrd(a, b, ax[i])
                mp = Mp[kp(1) + 1, D == 2 ? kp(2) + 1 : 1] * prod(o -> Mo[o][dOrd(a, b, o) + 1], op) / Vt
                m0 = a == b ? Mp[1, 1] * prod(o -> Mo[o][1], op) / Vt : zero(T)
                MH = max(MH, mp * pH + m0 * dH); MA = max(MA, mp * pA + m0 * dA)
            end
            g = (64 / T(15)) * h * T(ρ)^(-2 * nGtv) / (T(ρ)^2 - 1)
            bH = min(bH, g * MH); bA = min(bA, g * MA)
        end
        bH, bA
    end
    function evNode!(ac, t, wq)
        vals = Vector{NTuple{3,Complex{T}}}(undef, 3); avs = Vector{NTuple{3,T}}(undef, 3)
        for o in op
            vals[o] = Complex{T}.(kap(axk[o], R[o], t, tolv)); avs[o] = abs.(vals[o])
        end
        allP = all(i -> !useReal(T, Pv[i], t), 1:D) || cancEV(t)
        if D == 1
            i1 = ax[1]
            if allP
                # Σ_{m ∉ SP} pre_m e^{-u_m t}, u_m = 4π²ν_m² - x (EV: u_m > 0): e^{xt} folded in (it overflowed at x t > 709: NaN)
                α2 = -log(tolv) + 3; S0 = zero(Complex{T}); S1 = zero(Complex{T}); S2 = zero(Complex{T}); A0 = zero(T); A1 = zero(T); A2 = zero(T)
                local rgE = poiRngE(T, Pv[1], t, α2, νe2)
                for m in rgE
                    m in spProj[1] && continue
                    local rl = prRol(Pv[1], m, v -> v in rgE && !(v in spProj[1])); rl == :skip && continue
                    pr, ν = poiPreR(T, m, Pv[1], R[i1], rl); e_ = exp(-uOrd((ν,), fa, fb) * t)
                    S0 += pr[1] * e_; S1 += pr[2] * e_; S2 += pr[3] * e_
                    ae = abs(e_); A0 += abs(pr[1]) * ae; A1 += abs(pr[2]) * ae; A2 += abs(pr[3]) * ae
                end
                vals[i1] = (S0, S1, S2); avs[i1] = (A0, A1, A2)
                addNode!(vals, wq, t, ac, true, avs)
            else
                s, sa = perSkA(Pv[1], R[i1], t, tolv)
                for m in spProj[1]
                    local rl = prRol(Pv[1], m, v -> v in spProj[1]); rl == :skip && continue
                    pt = poiTrmR(m, Pv[1], R[i1], t, rl); s = s .- pt; sa = sa .+ abs.(pt)
                end
                vals[i1] = s; avs[i1] = sa
                addNode!(vals, wq, t, ac, false, avs)
            end
        else
            # d = 2: periodic factor per derivative pair (k1, k2), EV only
            E = zeros(Complex{T}, 3, 3); EA = zeros(T, 3, 3)
            i1, i2 = ax
            if allP
                α2 = -log(tolv) + 3
                local rg1 = poiRngE(T, Pv[1], t, α2, νe2); local rg2 = poiRngE(T, Pv[2], t, α2, νe2)
                for m1 in rg1
                    local rl1 = prRol(Pv[1], m1, v -> v in rg1); rl1 == :skip && continue
                    p1, ν1 = poiPreR(T, m1, Pv[1], R[i1], rl1)
                    for m2 in rg2
                        (m1, m2) in SP && continue
                        local rl2 = prRol(Pv[2], m2, v -> v in rg2); rl2 == :skip && continue
                        p2, ν2 = poiPreR(T, m2, Pv[2], R[i2], rl2)
                        e_ = exp(-uOrd((ν1, ν2), fa, fb) * t)
                        for k1 in 0:2, k2 in 0:2-k1
                            v_ = p1[k1 + 1] * p2[k2 + 1] * e_
                            E[k1 + 1, k2 + 1] += v_; EA[k1 + 1, k2 + 1] += abs(v_)
                        end
                    end
                end
            else
                s1, a1 = perSkA(Pv[1], R[i1], t, tolv); s2, a2 = perSkA(Pv[2], R[i2], t, tolv)
                for k1 in 0:2, k2 in 0:2-k1
                    E[k1 + 1, k2 + 1] = s1[k1 + 1] * s2[k2 + 1]; EA[k1 + 1, k2 + 1] = a1[k1 + 1] * a2[k2 + 1]
                end
                for g in SP
                    local rl = rlSP(g); any(==(:skip), rl) && continue
                    p1 = poiTrmR(g[1], Pv[1], R[i1], t, rl[1]); p2 = poiTrmR(g[2], Pv[2], R[i2], t, rl[2])
                    for k1 in 0:2, k2 in 0:2-k1
                        E[k1 + 1, k2 + 1] -= p1[k1 + 1] * p2[k2 + 1]; EA[k1 + 1, k2 + 1] += abs(p1[k1 + 1] * p2[k2 + 1])
                    end
                end
            end
            wHd, wAd, wHp, wAp = wts(t, allP)
            o = op[1]; ko = vals[o]; koa = avs[o]; aw = abs(wq)
            for b in 1:3, a in 1:3
                kk = E[dOrd(a, b, i1) + 1, dOrd(a, b, i2) + 1] * ko[dOrd(a, b, o) + 1] / Vt
                ka = EA[dOrd(a, b, i1) + 1, dOrd(a, b, i2) + 1] * koa[dOrd(a, b, o) + 1] / Vt
                h = wq * wHp * kk; an = wq * wAp * kk; mh = aw * abs(wHp) * ka; ma = aw * abs(wAp) * ka
                if a == b
                    k0 = E[1, 1] * ko[1] / Vt; k0a = EA[1, 1] * koa[1] / Vt
                    h += wq * wHd * k0; an += wq * wAd * k0; mh += aw * abs(wHd) * k0a; ma += aw * abs(wAd) * k0a
                end
                add!(ac, a, b, h, an, mh, ma)
            end
        end
    end
    nEV = 0; evPan = Tuple{T,T}[]
    tlo = τv
    while tlo < tmax
        thi = min(2 * tlo, tmax)
        nsub = max(1, ceil(Int, abs(y) * (thi - tlo) / 2))
        subE = shpSub(tlo, thi, t -> umin * t + ρo^2 / (4 * t), Cshv; kmin = nsub, E = t -> -umin * t - ρo^2 / (4 * t), EM = -sqrt(umin) * ρo, αs = α + 20)
        for sb in 1:length(subE)-1
            lo = subE[sb]; hi = subE[sb + 1]
            push!(evPan, (lo, hi))
            for j in 1:nGtv
                t = (hi - lo) / 2 * xt[j] + (hi + lo) / 2; wq = (hi - lo) / 2 * wt[j]
                evNode!(acc, t, wq)
                nEV += 1
            end
        end
        tlo = thi
    end
    function evInt(lo, hi, n)
        a_ = Acc(T); xq, wq_ = glq(T, n)
        for j in 1:n
            evNode!(a_, (hi - lo) / 2 * xq[j] + (hi + lo) / 2, (hi - lo) / 2 * wq_[j])
        end
        (a_.H .+ a_.cH, a_.A .+ a_.cA)
    end
    info[:evChk] = (evBnd = evBnd, evInt = evInt)
    info[:nEV] = nEV; info[:evPan] = evPan; info[:EV] = (copy(acc.H .+ acc.cH), copy(acc.A .+ acc.cA))
    # SP orders (propagating, grazing, continued): spectral form along the open axes with the pole part in closed form,
    # T_pt = ∫ d^{d_o}ξ/(2π)^{d_o} Π_o κ̂_o (iξ_o)^{k_o} e^{iξ_o x_o} W(|ξ|²), W(s) = e^{-(s+p)τv}/(s+p), parity-projected factors a(s):
    #   d_o = 2: T = a(s_c) E_1(pτ) + ∫_0^S (a(s) - a(s_c)) W ds,   a(s) = (1/8π²) ∫ dφ Π_o P_o(√s (cos φ, sin φ))
    #   d_o = 1: T = (1/π)[P(ξ_c)(π/2√p) erfc(√(pτ)) + ∫_0^Ξ (P(ξ) - P(ξ_c)) W(ξ²) dξ]
    # s_c = max(-u, 0); Re and Im of the remainder with the exact real weights Re/Im W = e^{-vτ}(v cos yτ ∓ y sin yτ, v sin yτ + y cos yτ)/(v²+y²),
    # v = s + u formed from the node offset to s_c. At y = 0 the Im remainder vanishes and AH_G = a(s_r) π (d_o = 2) or
    # P(ξ_r)/(2ξ_r) (d_o = 1): the finite propagating sum. Continued sheetv (J10): + 2πi a(-p) (d_o = 2), + i P(√(-p))/√(-p) (d_o = 1).
    xsO = [toT(T, R[o]) for o in op]; axO = [axk[o] for o in op]
    Xo = [abs(xsO[k]) + toT(T, axO[k].cp) for k in eachindex(op)]; w0o = [axO[k].w0 for k in eachindex(op)]
    # |y| <= eps²|x| (Im f = 1e-300 hung blcTns and gave NaN in blcFil): the SP orders are evaluated at y = 0 (real f). A
    # propagating order refused below |p| = 64 eps|x| moves by |ΔT| <= |y| |∂_p T| <= |y| |a| (π + 1)/|p| <= eps |a|/16 (d_o = 1: T ~ iπ/(2b),
    # d_o = 2: a E_1(pτ)), i.e. below 1/16 of eps times that order's Σ|terms|, which the returned λ already counts; RS, EV and the weights keep y
    ysp = abs(y) <= eps(T)^2 * abs(x) ? zero(T) : y
    pdat = PoleDat{T}(collect(op), w0o, Xo, τv, ysp)
    function spcSP(p, up, cont, pats; pans = nothing)
        sr = -up; sc = max(sr, zero(T)); gap = sc - sr
        S = sc + (α + 12) / τv
        nn = 0
        Is = Dict(pt => zero(Complex{T}) for pt in pats)
        yspτ = ysp * τv; cy = cos(yspτ); sy = sin(yspτ)
        wRI(v) = (e = exp(-v * τv) / (v^2 + ysp^2); (e * (v * cy - ysp * sy), e * (v * sy + ysp * cy)))
        if length(op) == 2
            # angular nodes per radius: circM at the panel's largest radius √(s_c + hi), the same rule M_in = M(√(2 s_c)) on every
            # panel with s <= 2 s_c and for the closed part a(s_c) (the remainder (a(s) - a(s_c)) W(s) must vanish at the pole with one rule),
            # M(2√S) only for the continuation a(-p) at complex radius; M(2√S) at every node cost 3-4x the angular work
            Mr(r_) = maximum(circM(axO, xsO, [pt[o] for o in op], r_, tolv) for pt in pats)
            M2 = Mr(sqrt(S) * 2); Min = Mr(sqrt(2 * sc))
            csT = Dict{Int,Tuple{Vector{T},Vector{T}}}()
            abT = [(toT(T, axO[k].a), toT(T, axO[k].b)) for k in 1:2]
            # (the node s0 + δ0 as two numbers; the wide phase per axis where r_g|x_o| > PHX, r_g = √(2 s_c) on s <= 2 s_c: the closed part and the
            # remainder's nodes next to the pole take the same arithmetic, which their difference quotient needs)
            csTW = Dict{Int,Tuple{Vector{wideT(T)},Vector{wideT(T)}}}()
            function aV(s0, δ0, M)
                wrk!(0.1 * M)
                local cM, sM = get!(() -> ([cospi(2 * T(j) / M) for j in 0:M-1], [sinpi(2 * T(j) / M) for j in 0:M-1]), csT, M)
                local sv = s0 + δ0; local r = sqrt(sv)
                out = Dict(pt => zero(typeof(r)) for pt in pats)
                local rg = sv isa Real ? (sv <= 2 * sc ? sqrt(2 * sc) : r) : abs(r)
                local w1 = WPH[] && rg * abs(xsO[1]) > PHX; local w2 = WPH[] && rg * abs(xsO[2]) > PHX
                if (w1 || w2) && T === Float64 && sv isa Real
                    local rh, rl = sqrt2(s0, δ0)
                    local cW, sW = get!(() -> csWide(wideT(T), M), csTW, M)
                    for j in 1:M
                        P1 = w1 ? pFac3W(axO[1], mul2(rh, rl, HI(cW[j]), LO(cW[j]))..., xsO[1], abT[1][1], abT[1][2]) : pFac3(axO[1], r * cM[j], xsO[1], abT[1][1], abT[1][2])
                        P2 = w2 ? pFac3W(axO[2], mul2(rh, rl, HI(sW[j]), LO(sW[j]))..., xsO[2], abT[2][1], abT[2][2]) : pFac3(axO[2], r * sM[j], xsO[2], abT[2][1], abT[2][2])
                        for pt in pats
                            out[pt] += P1[pt[op[1]] + 1] * P2[pt[op[2]] + 1]
                        end
                    end
                elseif sv isa Complex && (w1 || w2)
                    # the continuation a(-p) at complex radius: |e^{iξx}| varies by e^{2|Im r||x|} over the circle and the sum cancels, so the factors
                    # and the sum are formed in W (one such call per continued order)
                    wideDo(T, s0) do
                        local rW = sqrt(toW(T, s0) + toW(T, δ0))
                        local cW, sW = get!(() -> csWide(wideT(T), M), csTW, M)
                        local oW = Dict(pt => zero(rW) for pt in pats)
                        for j in 1:M
                            P1 = pFac3(axO[1], rW * cW[j], xsO[1], abT[1][1], abT[1][2]); P2 = pFac3(axO[2], rW * sW[j], xsO[2], abT[2][1], abT[2][2])
                            for pt in pats
                                oW[pt] += P1[pt[op[1]] + 1] * P2[pt[op[2]] + 1]
                            end
                        end
                        for pt in pats; out[pt] = Complex{T}(oW[pt]); end
                    end
                elseif w1 || w2
                    wideDo(T, s0) do
                        local rW = sqrt(toW(T, s0) + toW(T, δ0))
                        local cW, sW = get!(() -> csWide(wideT(T), M), csTW, M)
                        for j in 1:M
                            P1 = w1 ? pFac3W(axO[1], rW * cW[j], xsO[1], abT[1][1], abT[1][2]) : pFac3(axO[1], r * cM[j], xsO[1], abT[1][1], abT[1][2])
                            P2 = w2 ? pFac3W(axO[2], rW * sW[j], xsO[2], abT[2][1], abT[2][2]) : pFac3(axO[2], r * sM[j], xsO[2], abT[2][1], abT[2][2])
                            for pt in pats
                                out[pt] += P1[pt[op[1]] + 1] * P2[pt[op[2]] + 1]
                            end
                        end
                    end
                else
                    for j in 1:M
                        P1 = pFac3(axO[1], r * cM[j], xsO[1], abT[1][1], abT[1][2]); P2 = pFac3(axO[2], r * sM[j], xsO[2], abT[2][1], abT[2][2])
                        for pt in pats
                            out[pt] += P1[pt[op[1]] + 1] * P2[pt[op[2]] + 1]
                        end
                    end
                end
                for pt in pats; out[pt] /= 4 * T(π) * M; end
                out
            end
            ac = aV(sc, zero(T), Min)
            win2 = anmWin(up, p, τv)
            E1 = win2 ? e1Win(p, ysp, τv) : (ysp == 0 && up < 0) ? complex(real(expintG(1, complex(up * τv, zero(T)))), T(π)) : expintG(1, p * τv)
            for pt in pats; Is[pt] = ac[pt] * E1; end
            # Σ|terms| of the Hermitian / anti-Hermitian parts (a posteriori rounding bound, route selection)
            e1R = win2 ? T(0.58) + abs(log(abs(p) * τv)) + 2 * abs(p) * τv : abs(E1); e1I = win2 ? T(π) + 2 * abs(p) * τv : abs(E1)
            Ia = Dict(pt => [abs(ac[pt]) * e1R, abs(ac[pt]) * e1I] for pt in pats)
            # the remainder is summed on its own, compensated: added one by one to the closed-form part it drifted by sqrt(N) eps (9 eps)
            RS_ = Dict(pt => zeros(T, 4) for pt in pats)
            w = ysp != 0 ? abs(ysp) / 4 : max(sc, 1 / τv) / 8
            pts = oscSplit(grdPts(-sc, S - sc, zero(T), w), sum(k -> abs(xsO[k]) + toT(T, axO[k].cp), eachindex(op)), true, sc)
            pl = pans === nothing ? [(pts[q], pts[q + 1]) for q in 1:length(pts)-1] : pans
            Mpl = [hi <= sc ? Min : max(Min, Mr(sqrt(sc + hi))) for (lo, hi) in pl]
            wrkChk(0.1 * nGtv * (sum(Mpl; init = 0) + M2), "the d_o = 2 pole form (angular nodes up to M = $(maximum(Mpl; init = Min)))")
            for (q_, (lo, hi)) in enumerate(pl)
                for j in 1:nGtv
                    δ = (hi - lo) / 2 * xt[j] + (hi + lo) / 2; wq = (hi - lo) / 2 * wt[j]
                    av = aV(sc, δ, Mpl[q_]); rw, iw = wRI(δ + gap)
                    ysp == 0 && (iw = zero(T))
                    for pt in pats
                        d = (av[pt] - ac[pt]) * wq
                        r_ = RS_[pt]; r_[1], r_[2] = kadd(r_[1], r_[2], d * rw); r_[3], r_[4] = kadd(r_[3], r_[4], d * iw)
                        Ia[pt][1] += abs(d * rw); Ia[pt][2] += abs(d * iw)
                    end
                    nn += 1
                end
            end
            for pt in pats; Is[pt] += complex(RS_[pt][1] + RS_[pt][2], RS_[pt][3] + RS_[pt][4]); end
            if cont
                a0 = aV(-p, zero(T), M2)
                for pt in pats; Is[pt] += 2 * T(π) * im * a0[pt]; Ia[pt] .+= 2 * T(π) * abs(a0[pt]); end
            end
        else
            o = op[1]; A = axk[o]; xo = xsO[1]
            ξc = sqrt(sc); Ξ = sqrt(S)
            win = anmWin(up, p, τv)
            if win
                C1, bw = c1Win(p, τv)
            else
                sp = (ysp == 0 && up < 0) ? complex(zero(T), -sqrt(sr)) : sqrt(p)
                C1 = T(π) / (2 * sp) * erfcG(sp * sqrt(τv))
            end
            Pc = Dict(pt => pFac(A, ξc, xo, pt[o]) for pt in pats)
            for pt in pats; Is[pt] = Pc[pt] * C1; end
            # Σ|terms| per part: in the window each part of C1 is a sum of two terms (the singular one and (π√τv/2)E)
            if win
                Ew = erfiQ(-p * τv); ab2 = abs2(bw)
                c1R = T(π) * abs(imag(bw)) / (2 * ab2) + T(π) * sqrt(τv) / 2 * abs(real(Ew)); c1I = T(π) * abs(real(bw)) / (2 * ab2) + T(π) * sqrt(τv) / 2 * abs(imag(Ew))
            else
                c1R = abs(C1); c1I = abs(C1)
            end
            Ia = Dict(pt => [abs(Pc[pt]) * c1R, abs(Pc[pt]) * c1I] for pt in pats)
            RS_ = Dict(pt => zeros(T, 4) for pt in pats)
            w = ysp != 0 ? abs(ysp) / (4 * max(ξc, sqrt(abs(ysp)))) : max(ξc, 1 / sqrt(τv)) / 8
            pts = oscSplit(grdPts(-ξc, Ξ - ξc, zero(T), w), abs(xo) + toT(T, A.cp), false, zero(T))
            pl = pans === nothing ? [(pts[q], pts[q + 1]) for q in 1:length(pts)-1] : pans
            for (lo, hi) in pl
                for j in 1:nGtv
                    δ = (hi - lo) / 2 * xt[j] + (hi + lo) / 2; wq = (hi - lo) / 2 * wt[j]
                    ξ = ξc + δ
                    rw, iw = wRI(δ * (2 * ξc + δ) + gap)
                    ysp == 0 && (iw = zero(T))
                    for pt in pats
                        d = pDif(A, ξc, δ, xo, pt[o]) * wq
                        r_ = RS_[pt]; r_[1], r_[2] = kadd(r_[1], r_[2], d * rw); r_[3], r_[4] = kadd(r_[3], r_[4], d * iw)
                        Ia[pt][1] += abs(d * rw); Ia[pt][2] += abs(d * iw)
                    end
                    nn += 1
                end
            end
            for pt in pats; Is[pt] += complex(RS_[pt][1] + RS_[pt][2], RS_[pt][3] + RS_[pt][4]); Is[pt] /= T(π); Ia[pt] ./= T(π); end
            if cont && win
                # δ = b - ξc = (b² - ξc²)/(b + ξc) = iy'/(b + ξc) (b² = -p = -u + iy', ξc² = -u), no cancellation
                δb = (-p - sc) / (bw + ξc)
                for pt in pats; v = im * pDif(A, complex(ξc), δb, xo, pt[o]) / bw; Is[pt] += v; Ia[pt] .+= abs(v); end
            elseif cont
                s0 = sqrt(-p)
                for pt in pats; v = im * pFac(A, s0, xo, pt[o]) / s0; Is[pt] += v; Ia[pt] .+= abs(v); end
            end
        end
        return Is, nn, pl, Ia
    end
    # the ray route of one order on given s-panels: t = τv + s e
    function rayIs(p, e, pats, pl)
        Ic = Dict(pt => zeros(T, 4) for pt in pats); nn = 0; Ia = Dict(pt => zeros(T, 2) for pt in pats)
        for (lo, hi) in pl, j in 1:nGtv
            sn = (hi - lo) / 2 * xt[j] + (hi + lo) / 2; wq = (hi - lo) / 2 * wt[j]
            t = τv + sn * e
            ks = ntuple(j -> kap(axk[op[j]], R[op[j]], t, tolv), length(op))
            ep = exp(-p * t) * e * wq
            for pt in pats
                v = ep
                for (j, o) in enumerate(op)
                    v *= ks[j][pt[o] + 1]
                end
                c_ = Ic[pt]; c_[1], c_[2] = kadd(c_[1], c_[2], real(v)); c_[3], c_[4] = kadd(c_[3], c_[4], imag(v))
                Ia[pt] .+= abs(v)
            end
            nn += 1
        end
        Dict(pt => complex(Ic[pt][1] + Ic[pt][2], Ic[pt][3] + Ic[pt][4]) for pt in pats), nn, Ia
    end
    # entry contribution of one order from its pattern integrals
    # Ia: the (Hermitian, anti-Hermitian) Σ|terms| of each pattern integral (nothing: add no magnitude, the certification swap)
    function ordAdd!(ac_, prs, Is, sgn, Ia = nothing)
        for b in 1:3, a in 1:3
            pper = one(Complex{T})
            for i in 1:D
                pper *= prs[i][dOrd(a, b, ax[i]) + 1]
            end
            pt = ntuple(i -> i in op ? dOrd(a, b, i) : 0, 3)
            Wp = Is[pt] * complex(cr, ci)
            h = pper * real(Wp) / Vt; an = pper * imag(Wp) / Vt
            if a == b
                p0 = one(Complex{T})
                for i in 1:D
                    p0 *= prs[i][1]
                end
                W0 = 4 * T(π)^2 * Is[(0, 0, 0)]
                h += p0 * real(W0) / Vt; an += p0 * imag(W0) / Vt
            end
            if Ia === nothing
                add!(ac_, a, b, sgn * h, sgn * an, zero(T), zero(T))
            else
                aR, aI = Ia[pt]; mh = abs(pper) * (aR * abs(cr) + aI * abs(ci)) / Vt; ma = abs(pper) * (aR * abs(ci) + aI * abs(cr)) / Vt
                if a == b
                    p0a = abs(prod(i -> prs[i][1], 1:D)) * 4 * T(π)^2 / Vt
                    mh += p0a * Ia[(0, 0, 0)][1]; ma += p0a * Ia[(0, 0, 0)][2]
                end
                add!(ac_, a, b, sgn * h, sgn * an, mh, ma)
            end
        end
    end
    # entry-level weight of pattern pt's error: max over (a,b) of |prefactor| (the δ part with 4π²)
    function patW(prs)
        wd = Dict{NTuple{3,Int},T}()
        for b in 1:3, a in 1:3
            pt = ntuple(i -> i in op ? dOrd(a, b, i) : 0, 3)
            wv = prod(i -> abs(prs[i][dOrd(a, b, ax[i]) + 1]), 1:D) * sqrt(cr^2 + ci^2) / Vt
            a == b && (wd[(0, 0, 0)] = max(get(wd, (0, 0, 0), zero(T)), 4 * T(π)^2 * prod(i -> abs(prs[i][1]), 1:D) / Vt))
            wd[pt] = max(get(wd, pt, zero(T)), wv)
        end
        wd
    end
    # SP orders
    X2 = sum(o -> (toT(T, abs(R[o]) + axk[o].cp))^2, op)
    nSPn = 0
    φs = T[]; amps = T[]
    spRec = []
    for g in SP
        rl = rlSP(g); any(==(:skip), rl) && continue
        ν = ntuple(i -> nuP(T, g[i], Pv[i]), D)
        up = uOrdX(T, g, Pv, fa, fb)
        p = complex(up, -ysp)
        abs(p) <= 64 * eps(T) * max(abs(x), one(T)) && error("blcTns: Rayleigh-Wood anomaly at rounding distance, order $g, |q∥|² − k² = $p")
        # sheetv: continued when Im f < 0 and Re k > |q∥| (J10)
        cont = sheetv == :J10 && fb < 0 && ysp != 0 && abs(fa) > sqrt(sum(abs2, ν))
        sheetv == :J10 && fb < 0 && abs(abs(fa) - sqrt(sum(abs2, ν))) <= 64 * eps(T) * abs(fa) && error("blcTns: Im f < 0 on the cut Re k = |q∥| of order $(g) (J10 continuation undefined there)")
        ap = angle(p)
        ysp == 0 && up < 0 && (ap = -T(π))   # real f: the limit from Im f > 0
        cont && (ap -= 2 * T(π))
        prs = [poiPreR(T, g[i], Pv[i], R[ax[i]], rl[i])[1] for i in 1:D]
        pats = unique(vcat([ntuple(i -> i in op ? dOrd(a, b, i) : 0, 3) for a in 1:3, b in 1:3][:], [(0, 0, 0)]))
        # route: the spectral pole form for propagating and continued orders unless the open-axis decay e^{-Re√p ρ} makes its
        # oscillatory remainder cancel (loss ~ e^{Re√p ρ} > 4); otherwise the steepest-descent ray φ = -arg p, capped at
        # π/2 + ψ_g where the open-axis Gaussians grow by at most e^{c_g}
        useSpc = cont || (up < 0 && exp(real(sqrt(p)) * ρo) <= 4)
        if !useSpc
            φ = -ap
            if abs(φ) > T(π) / 2
                cψ = 1 / (1 + 8 * τv * cgv / max(X2, eps(T)))
                φ = sign(φ) * min(abs(φ), T(π) / 2 + min(acos(cψ), ψmx))
            end
            rate = abs(p) * cos(ap + φ)
            rate > 0 || error("blcTns: order $g: no decaying ray, p = $p")
            e = cis(φ)
            smax = (α + 12 + ρo^2 / (4 * τv)) / rate
            pl = Tuple{T,T}[]
            s0 = zero(T); s1 = τv
            while s0 < smax
                s1 = min(s1, smax)
                subS = shpSub(τv + s0, τv + s1, t -> rate * t + ρo^2 / (4 * t), Cshv; E = t -> -rate * t - ρo^2 / (4 * t), EM = -sqrt(rate) * ρo, αs = α + 20) .- τv
                for sb in 1:length(subS)-1
                    push!(pl, (subS[sb], subS[sb + 1]))
                end
                s0 = s1; s1 = 2 * s1
            end
            Is, nn, Ia = rayIs(p, e, pats, pl)
            nSPn += nn; push!(φs, φ)
            push!(spRec, (kind = :ray, p = p, up = up, cont = cont, prs = prs, pats = pats, e = e, pl = pl, Is = Is))
        else
            Is, nn, pl, Ia = spcSP(p, up, cont, pats)
            nSPn += nn; push!(φs, T(NaN))
            push!(spRec, (kind = :pole, p = p, up = up, cont = cont, prs = prs, pats = pats, e = one(Complex{T}), pl = pl, Is = Is))
        end
        ordAdd!(acc, prs, Is, 1, Ia)
    end
    # certification of the SP Gauss rules (ray: Bernstein bound with the e^{-pt} Π κ_o majorants; pole remainder: disk majorants
    # of the difference quotient, both poles of W and W*): every panel whose entry-level bound exceeds tolv·S/(10·#panels) is bisected
    # (depth <= 8) and that order recomputed on the certified panels
    function mkBnd(r)
        wd = patW(r.prs)
        if r.kind == :ray
            return (lo, hi) -> begin
                tm = τv + (lo + hi) / 2 * r.e; Δ = (hi - lo) / 2 * r.e
                gBnd(tm, Δ, nGtv, e_ -> begin
                    Mo = ntuple(j -> ntuple(k -> axMaj(axk[op[j]], R[op[j]], nothing, k - 1, e_, tolv), 3), length(op))
                    exp(e_.rmax(-r.p)) * maximum(pt -> wd[pt] * prod(j -> Mo[j][pt[op[j]] + 1], eachindex(op)), keys(wd))
                end; neg = true)
            end
        end
        (lo, hi) -> gBndR(lo, hi, nGtv, (δm, hA, hB) -> maximum(pt -> wd[pt] * poleMaj(pdat, pt, r.p, r.up, δm, hA, hB), keys(wd)))
    end
    info[:spChk] = (spRec = spRec, mkBnd = mkBnd, rayIs = rayIs, spcSP = spcSP, ordAdd! = ordAdd!)
    if qcert && !isempty(spRec)
        Hc_, Ac_ = fin(acc)
        mH = max(maximum(abs, Hc_), maximum(acc.sH) / 64); mA = max(maximum(abs, Ac_), maximum(acc.sA) / 64)
        S_ = mH > 0 && mA > 0 ? min(mH, mA) : max(mH, mA)
        npl = sum(r -> length(r.pl), spRec)
        tqS = tolv / (10 * max(npl, 1))
        totB = zero(T); nbis = 0; nunc = 0
        for r in spRec
            bnd = mkBnd(r)
            fl = Tuple{T,T}[]; ref = false
            for (lo, hi) in r.pl
                st = [(lo, hi, 0)]
                while !isempty(st)
                    a_, b_, dp = pop!(st)
                    bb = bnd(a_, b_)
                    if rlB(bb, S_) > tqS / 2^dp && dp < MAXDP && length(fl) + length(st) < MAXLEAF && (isfinite(bb) || dp < 12)
                        dp == 0 && (nbis += 1); ref = true
                        c_ = (a_ + b_) / 2; push!(st, (a_, c_, dp + 1)); push!(st, (c_, b_, dp + 1))
                    else
                        push!(fl, (a_, b_)); totB += bb
                        rlB(bb, S_) > tqS / 2^dp && (nunc += 1)
                    end
                end
            end
            if ref
                In = r.kind == :ray ? rayIs(r.p, r.e, r.pats, fl)[1] : spcSP(r.p, r.up, r.cont, r.pats; pans = fl)[1]
                ordAdd!(acc, r.prs, r.Is, -1); ordAdd!(acc, r.prs, In, 1)
            end
        end
        info[:qcertSP] = (bound = totB, rel = totB / S_, bisected = nbis, uncertified = nunc, npan = npl)
    end
    info[:nSPnod] = nSPn; info[:φ] = φs; info[:ampSP] = amps
    qcert && (info[:qcert] = (certify!(rsPan, true), certify!(evPan, false; bf = evBnd, evf = evNode!)))
    qcert && chkCert(info)
    info[:sH] = acc.sH; info[:sA] = acc.sA
    Hf, Af = fin(acc)
    return Hf, Af, info
end


# argument shapes the typed method does not take (periods or phases not one per axis, real f): the one-line refusal of inpChk
blcTns(R, sT, sS, f::Number, ax::Tuple, L::Tuple, θ::Tuple; kw...) = (inpChk("blcTns", f, ax, L, θ); error("blcTns: f must be complex (got $(typeof(f))) and the periods Rational"))
