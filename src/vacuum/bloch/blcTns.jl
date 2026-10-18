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
# e^{2πi ν_m R}, ν_m = (m - t)/L, at an exact twist t = θ/2π (θ = π is t = 1/2, as Gila's odd branch): the whole phase reduced exactly
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
const GLC = Dict{Tuple{DataType,Int},Tuple{Vector,Vector}}(); const GLCK = ReentrantLock()
function glq(::Type{T}, n::Int) where {T}
    key = (T, n)
    r0 = @lock GLCK get(GLC, key, nothing)
    r0 === nothing || return r0::Tuple{Vector{T},Vector{T}}
    pw = precision(T) + 64
    xb, wb = setprecision(() -> QuadGK.gauss(BigFloat, n), BigFloat, pw)
    r = (T.(xb), T.(wb))
    @lock GLCK get!(GLC, key, r)::Tuple{Vector{T},Vector{T}}
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
    # per-call memo of the exact x of kap (|x|, x rounded once to T): kap ran this Rational{BigInt} arithmetic at every quadrature
    # node, and the GMP temporaries (freed only by finalizers) drove the RSS of long runs; an AxK lives for one blcFil call, so the
    # memo is bounded by that call's distinct offsets
    xm::TMemo{Tuple{Int128,Int128},Tuple{T,T}}
end
AxK(::Type{T}, a::QI, b::QI; MM::Int = precision(T) + 48) where {T} =
    AxK{T}(a, b, (a + b) / 2, abs(a - b) / 2, toT(T, a * b), muF(T, a, b, MM), TMemo{Tuple{Int128,Int128},Tuple{T,T}}())
# the normalized moments are frequency and phase independent: kept per (T, edges, order), at most MUFMAX entries
const MUF = Dict{Any,Any}(); const MUFK = ReentrantLock(); const MUFMAX = 256
function muF(::Type{T}, a::QI, b::QI, MM::Int) where {T}
    key = (T, qk(a), qk(b), MM)
    v = @lock MUFK get(MUF, key, nothing)
    v === nothing || return v::Vector{T}
    μf = [toT(T, μ / (factorial(big(m)) * ((a + b) / 2)^m)) for (m, μ) in zip(0:MM, trpMom(a, b, MM))]
    @lock MUFK (length(MUF) < MUFMAX ? get!(MUF, key, μf) : μf)::Vector{T}
end
kapX(A::AxK{T}, x::QI) where {T} = get!(() -> (toT(T, abs(x)), toT(T, x)), A.xm, qk(x))

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
    first(ierfcC3(v))
end
# the backward continued fraction C at Re v > 0: (ierfc v = E/√π/(2vC + 1), C, E = e^{-v²})
function ierfcC3(v::S) where {S}
    T = real(S)
    cf(n) = (c = v; for j in n:-1:2; c = v + (T(j) / 2) / c; end; c)
    # the doubling starts lower at large |v| (|v| >= 3 is all that reaches here in Float64; converged within 16-64 terms there)
    n = abs(v) >= 6 ? 8 : 16; c1 = cf(n)
    while true
        n *= 2; c2 = cf(n)
        # e^{-v²} with v² formed exactly (expM2): the plain exp(-v^2) cost |v|² eps (33 eps at |v| = 6)
        if abs(c2 - c1) <= eps(T) * abs(c2)
            E = S <: Complex ? expM2(v) : exp(-v^2)
            return E / sqrt(T(π)) / (2 * v * c2 + 1), c2, E
        end
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
    ie, c2, E = ierfcC3(v)
    ie, 2 * c2 * ie, E
end


# terms of the moment series: Cauchy on |ζ - x| = 2cp with |μ_m| <= w0 cp^m gives, in the moment regime
# |t| >= cp(|x| + cp), |tail_k| <= e w0 |g e^{-w²}| Σ_{m>M} (m+k)^k 2^{-m} (2cp)^{-k}; reference scale w0 |g e^{-w²}| (2cp)^{-k}
# memoized per (T, tol) (a few distinct tolerances): the loop ran at every kap call (8% of route C's time)
const MOMMC = TMemo{Tuple{DataType,Float64},Int}()
momM(::Type{T}, tol::T) where {T} = get!(() -> momM0(T, tol), MOMMC, (T, Float64(tol)))
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

κ(x,t) = (box_a * box_b * H1_t)(x) and its x-derivatives at real or complex t (Re √t > 0), x exact, in the moment regime
momReg(cp, |x|, t) where every caller takes it.
"""
kap(A::AxK{T}, x::QI, t::S, tol::T) where {T,S} = kapV(A, kapX(A, x), t, tol)
function kapV(A::AxK{T}, dx, t::S, tol::T) where {T,S}
    cp = toT(T, A.cp); _, xT = dx
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
poiRng(::Type{T}, P::PerAx, t::T, α::T) where {T} = ordRng(T, P, sqrt(α / t) / (2 * T(π)) + 2 / toT(T, P.L), 1)
# a float phase θ as w = θ/2π mod 1: its distance to the nearer of the twists 0 and 1/2, and that twist (0 or 1 for 1/2)
twD(θ) = (w = mod(Float64(θ) / (2 * π), 1.0); min(w, 1 - w, abs(w - 0.5)))
twI(θ) = (w = mod(Float64(θ) / (2 * π), 1.0); abs(w - 0.5) < min(w, 1 - w) ? 1 : 0)
# the orders m with |m - tw| <= νm L on axis P, widened by k on each side
function ordRng(::Type{T}, P::PerAx, νm, k::Int) where {T}
    tw = P.tw isa QI ? toT(T, P.tw) : P.tw
    L = toT(T, P.L)
    (floor(Int, tw - νm * L) - k):(ceil(Int, tw + νm * L) + k)
end
# the orders of an EV-only Poisson sum (the SP/SL orders removed): 4π²(ν² - ν2)t <= α around ν2 = the smallest ν² LEFT in the sum.
# poiRng's reference is the smallest |ν| overall (<= 1/2L), which is a removed propagating order when fL is large: every EV order was then
# dropped beyond t ~ α/(4π²ν_EV²) and the EV part lost its tail, under a certificate built on the same truncated sum (
# d1 c32 f = 6 L = 1 θ = 0.7, 3.5e6 eps "certified"); the same absolute cut made κ_up vanish at far open offsets
poiRngE(::Type{T}, P::PerAx, t::T, α::T, ν2::T) where {T} =
    ordRng(T, P, sqrt(max(ν2, zero(T)) + α / (4 * T(π)^2 * t)) + 2 / toT(T, P.L), 1)
useReal(::Type{T}, P::PerAx, t::T) where {T} = t <= toT(T, P.L)^2 / (4 * T(π))
useRealC(::Type{T}, P::PerAx, t::Complex) where {T} = 1 / real(1 / t) <= toT(T, P.L)^2 / (4 * T(π))

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
uOrdX(::Type{T}, ms::NTuple{N,Int}, Ps, fa::T, fb::T) where {T,N} = setprecision(BigFloat, precision(T) + 64) do
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
# pFac3 of a Float64 fill at ξ = ξh + ξl, with the phase ξ x reduced mod 2π by error-free products (fma) against 2π = TP2[1] + TP2[2] + TP2[3]
# before its one rounding. In Float64 the argument ξ x of the d_o = 2 angular sums carries ~2.5 ulp |ξ x| (the roundings of the node s + δ, of
# √, of the angular table and of the products), which the open-axis oscillation turns into ~6 eps |k|ρ on the small entries of far open
# offsets (2.2-3.5 eps(1 + κ_e) at 4-32λ); used where the phase exceeds PHX
const PHX = 8
# off inside the Double64 reruns of flagged entries (filD64): their rounding is eps(Double64)
const WPH = Base.ScopedValues.ScopedValue(true)
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
# a Float64 number (real or complex) in Double64
toW(v::Complex) = Complex{Double64}(v)
toW(v) = Double64(v)
# the angular tables cos, sin(2πj/M) in W, by rotation from one cospi/sinpi pair: the error grows
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

# majorants of the Hermitian and anti-Hermitian weights on a Bernstein ellipse of a real panel (|Im t| <= ihi, |t| <= ta), without e^{x Re t}:
# δ part 4π²(cos, sin)(yt), ∂∂ part (cos yt cr - sin yt ci, sin yt cr + cos yt ci), 1/f² = cr + i ci; |cos w| <= cosh Im w,
# |sin w| <= min(cosh Im w, sinh|w|): the anti-Hermitian weights are O(y t), so AH is certified against its own size (AH << H at small Im f)
function wHA(y::T, cr::T, ci::T, ihi::T, ta::T) where {T}
    C = cosh(y * ihi); Sn = min(C, sinh(abs(y) * ta))
    (4 * T(π)^2 * C, C * abs(cr) + Sn * abs(ci), 4 * T(π)^2 * Sn, Sn * abs(cr) + C * abs(ci))
end

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

# the pole-form (propagating-order) remainder majorants of blcTns, top level so the fill certifies with them per open offset:
# op the open axes, w0o their trapezoid masses, Xo = |x_o| + cp_o (d_o = 2: only X1 + X2 enters), τ the split, y = Im k²
struct PoleDat{T}
    op::Vector{Int}; w0o::Vector{T}; Xo::Vector{T}; τ::T; y::T
end
# Gauss bound (ATAP 19.3) of a real panel [lo, hi] with a majorant maj(δm, hA, hB) on the disk |δ - δm| <= hA containing E_ρ
function gBndR(lo::T, hi::T, n::Int, maj) where {T}
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
# The returned entry's rounding is <= c eps min(λ_R,e, λ_C,e) |X_e|.
const LAM0 = 16
const NPCMAX = 10^4
# complex-path panels of one cPart call (at open offsets of 300λ the order rays alone built millions of panels, 8.7 GB); beyond it
# route C is skipped (rteAuto) or refused (route = :C)
const CPMAX = 200_000
