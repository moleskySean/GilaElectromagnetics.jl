# block accumulation kernels of the fill, threaded over fixed blocks of output columns (independent of the thread count, so the result is
# bitwise the same on any number of threads), each block keeping its slice of the output in cache across the node chunks; per-block buffers
# instead of per-chunk temporaries. BLAS is called with its own threads at 1 from the Julia threads (J27).
const KCB = 256
# tFor over n slabs of a loop of w elements in all, run serially below 2^15 elements (the thread fan-out cost more than the loop in small fills)
tForW(f, n::Int, w::Int) = w < 32768 ? (for i in 1:n; f(i); end; nothing) : tFor(f, n)
# acc[a,b] += Σ_t w[t] V1[:,t] ⊗ V2[:,t] ⊗ V3[:,t]: chunks of KCH nodes, each chunk's BLAS partial sum added with Neumaier compensation (Cc)
function krAddB!(C::AbstractArray{Complex{T},3}, Cc::AbstractArray{Complex{T},3}, V1, V2, V3, w) where {T}
    n1, n2, n3 = size(V1, 1), size(V2, 1), size(V3, 1); nt = size(V1, 2); nc = n2 * n3
    nt == 0 && return C
    W1 = V1 .* transpose(w)
    Cm = reshape(C, n1, nc); Ccm = reshape(Cc, n1, nc)
    tFor(cld(nc, KCB)) do b
        c0 = (b - 1) * KCB; m = min(KCB, nc - c0)
        B = Matrix{Complex{T}}(undef, m, KCH); X = Matrix{Complex{T}}(undef, n1, m)
        for t0 in 1:KCH:nt
            r = t0:min(t0 + KCH - 1, nt); nq = length(r)
            @inbounds for (q, t) in enumerate(r), ic in 1:m
                col = c0 + ic; j = (col - 1) % n2 + 1; k = (col - 1) ÷ n2 + 1
                B[ic, q] = V2[j, t] * V3[k, t]
            end
            mul!(X, view(W1, :, r), transpose(view(B, :, 1:nq)))
            @inbounds for ic in 1:m, i in 1:n1
                Cm[i, c0 + ic], Ccm[i, c0 + ic] = kaddC(Cm[i, c0 + ic], Ccm[i, c0 + ic], X[i, ic])
            end
        end
    end
    C
end
# M[a,b,c] += Σ_t w[t] A1[a,t] A2[b,t] A3[c,t], nonnegative real (Σ|terms|), chunks of 256 nodes
function krAddAB!(M::AbstractArray{T,3}, A1, A2, A3, w) where {T}
    n1, n2, n3 = size(A1, 1), size(A2, 1), size(A3, 1); nt = size(A1, 2); nc = n2 * n3
    nt == 0 && return M
    W1 = A1 .* transpose(w)
    Mm = reshape(M, n1, nc)
    tFor(cld(nc, KCB)) do b
        c0 = (b - 1) * KCB; m = min(KCB, nc - c0)
        B = Matrix{T}(undef, m, 256)
        for t0 in 1:256:nt
            r = t0:min(t0 + 255, nt); nq = length(r)
            @inbounds for (q, t) in enumerate(r), ic in 1:m
                col = c0 + ic; j = (col - 1) % n2 + 1; k = (col - 1) ÷ n2 + 1
                B[ic, q] = A2[j, t] * A3[k, t]
            end
            mul!(view(Mm, :, c0+1:c0+m), view(W1, :, r), transpose(view(B, :, 1:nq)), true, true)
        end
    end
    M
end
# X * Y over the inner index in chunks of KCH, the chunk products added with Neumaier compensation, threaded over fixed row blocks of X
function cMulB(X::AbstractMatrix{Complex{T}}, Y::AbstractMatrix) where {T}
    n, k = size(X); p = size(Y, 2)
    S = zeros(Complex{T}, n, p); C = zeros(Complex{T}, n, p)
    tFor(cld(n, 512)) do b
        rows = ((b - 1) * 512 + 1):min(b * 512, n)
        P = Matrix{Complex{T}}(undef, length(rows), p)
        for c0 in 1:KCH:k
            r = c0:min(c0 + KCH - 1, k)
            mul!(P, view(X, rows, r), view(Y, r, :))
            @inbounds for jj in 1:p, (ii, i) in enumerate(rows)
                S[i, jj], C[i, jj] = kaddC(S[i, jj], C[i, jj], P[ii, jj])
            end
        end
    end
    S .+ C
end
# real product threaded over fixed row blocks
function rMulB(X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where {T}
    n = size(X, 1); p = size(Y, 2); S = Matrix{T}(undef, n, p)
    tFor(cld(n, 512)) do b
        rows = ((b - 1) * 512 + 1):min(b * 512, n)
        mul!(view(S, rows, :), view(X, rows, :), Y)
    end
    S
end
# D .+= permutedims(X, perm) without the temporary (threaded over the last dimension of D)
function addPerm!(D::AbstractArray{S,3}, X::AbstractArray{S,3}, perm) where {S}
    Xp = PermutedDimsArray(X, perm)
    tForW(size(D, 3), length(D)) do k
        @inbounds for j in axes(D, 2), i in axes(D, 1)
            D[i, j, k] += Xp[i, j, k]
        end
    end
    D
end
# C, Cc = kaddC(C, Cc, permutedims(X, perm)) elementwise, without the temporary (threaded over the last dimension of C)
function kaddPerm!(C::AbstractArray{S,3}, Cc::AbstractArray{S,3}, X::AbstractArray{S,3}, perm) where {S}
    Xp = PermutedDimsArray(X, perm)
    tForW(size(C, 3), length(C)) do k
        @inbounds for j in axes(C, 2), i in axes(C, 1)
            C[i, j, k], Cc[i, j, k] = kaddC(C[i, j, k], Cc[i, j, k], Xp[i, j, k])
        end
    end
    C
end
# C .+= f.(args...) for 3-d arrays broadcastable to C (singleton dimensions allowed) and scalars, fused per slab, threaded over C's last dimension
function tAdd!(f::F, C::AbstractArray{S,3}, args...) where {F,S}
    tForW(size(C, 3), length(C)) do k
        vs = map(a -> a isa AbstractArray ? view(a, :, :, size(a, 3) == 1 ? (1:1) : (k:k)) : a, args)
        vC = view(C, :, :, k:k)
        vC .+= f.(vs...)
    end
    C
end
# S, Cc += X * Y over the inner index in chunks of KCH, each chunk product added with Neumaier compensation; threaded over fixed row blocks
function cMulAcc!(S::AbstractMatrix{Complex{T}}, Cc::AbstractMatrix{Complex{T}}, X::AbstractMatrix{Complex{T}}, Y::AbstractMatrix) where {T}
    n, k = size(X); p = size(Y, 2)
    tFor(cld(n, 512)) do b
        rows = ((b - 1) * 512 + 1):min(b * 512, n)
        P = Matrix{Complex{T}}(undef, length(rows), p)
        for c0 in 1:KCH:k
            r = c0:min(c0 + KCH - 1, k)
            mul!(P, view(X, rows, r), view(Y, r, :))
            @inbounds for jj in 1:p, (ii, i) in enumerate(rows)
                S[i, jj], Cc[i, jj] = kaddC(S[i, jj], Cc[i, jj], P[ii, jj])
            end
        end
    end
    S
end
# axMajV's Poisson-form majorant of a periodic axis (independent of x): Σ_m w0/L |2πν_m|^k e^{-4π²ν_m² rlo}, ν_m = (m - tw)/L
function axPoi(A::AxK{T}, Lt::T, tw::T, k::Int, rlo::T) where {T}
    sp = zero(T)
    m0 = round(Int, tw)
    for dir in (-1, 1)
        m = dir < 0 ? m0 : m0 + 1
        while true
            ν = (m - tw) / Lt
            v = A.w0 / Lt * (2 * T(π) * abs(ν))^k * exp(-4 * T(π)^2 * ν^2 * rlo); sp += v
            v <= eps(T)^2 * sp && abs(ν) > 1 / Lt && break
            abs(m - m0) > 2000 && (sp = T(Inf); break)
            m += dir
        end
    end
    sp
end
# S = X * Y (overwritten, any element type), threaded over fixed row blocks
function mMulB!(S::AbstractMatrix, X::AbstractMatrix, Y::AbstractMatrix)
    n = size(X, 1)
    tFor(cld(n, 512)) do b
        rows = ((b - 1) * 512 + 1):min(b * 512, n)
        mul!(view(S, rows, :), view(X, rows, :), Y)
    end
    S
end
# M += X * Y (real), threaded over fixed row blocks
function rMulAcc!(M::AbstractMatrix{T}, X::AbstractMatrix{T}, Y::AbstractMatrix{T}) where {T}
    n = size(X, 1)
    tFor(cld(n, 512)) do b
        rows = ((b - 1) * 512 + 1):min(b * 512, n)
        mul!(view(M, rows, :), view(X, rows, :), Y, true, true)
    end
    M
end
# axMaj over a vector of offsets xs (QI) at once: the Poisson-form majorant of a periodic axis does not depend on x and is formed once;
# per offset the same operations as axMaj, so the same numbers
function axMajV(A::AxK{T}, xs::AbstractVector, P, k::Int, e, tol::T) where {T}
    cp = toT(T, A.cp)
    one_(xa::T) = (d = e.rinv >= 0 ? max(xa - cp, zero(T)) : xa + cp; A.w0 * gk(k, xa + cp, e.tlo) / sqrt(4 * T(π) * e.tlo) * exp(-d^2 * e.rinv / 4))
    P === nothing && return [one_(abs(toT(T, x))) for x in xs]
    e.rinv > 0 || return fill(T(Inf), length(xs))
    # (the Poisson sum in its own function: an accumulator the map closure below captured was boxed, one heap Float64 per term)
    sp = axPoi(A, toT(T, P.L), P.tw isa QI ? toT(T, P.tw) : T(P.tw), k, e.rlo)
    LT = toT(T, P.L)
    map(xs) do x
        xT = toT(T, x)
        n0 = round(Int, -xT / LT); sr = one_(abs(xT + n0 * LT))
        for dir in (-1, 1)
            n = n0 + dir
            while true
                v = one_(abs(xT + n * LT)); sr += v
                v <= eps(T)^2 * sr && break
                abs(n - n0) > 2000 && (sr = T(Inf); break)
                n += dir
            end
        end
        min(sr, sp)
    end
end

# cCert!'s scale S (route C): per offset I and entry, hx = Σ_p [cur(CH_p) + (Cq + sg conj(Cr))/2], ax = Σ_p [cur(CA_p) + (Cq - sg conj(Cr))/(2i)]
# with Cq = cur(CC_p)(I), Cr = cur(CC_p)(I reflected on the axes rv), cur(X) = X + its compensation (if any); SH = max(SH, |hx|/V), SA likewise.
# The operations of the broadcasts it replaces, element by element in the same order (the term tuples unrolled); threaded over the last dimension.
@inline cAt(X, ::Nothing, i, j, k) = @inbounds X[i, j, k]
@inline cAt(X, c, i, j, k) = @inbounds X[i, j, k] + c[i, j, k]
@inline accC(::Tuple{}, hx, axx, i, j, k, ir, jr, kr) = (hx, axx)
@inline function accC(ps::Tuple, hx, axx, i, j, k, ir, jr, kr)
    CH, cH, CA, cA, CC, cC, sg = first(ps)
    cq = cAt(CC, cC, i, j, k); cr = cAt(CC, cC, ir, jr, kr)
    accC(Base.tail(ps), hx + (cAt(CH, cH, i, j, k) + (cq + sg * conj(cr)) / 2), axx + (cAt(CA, cA, i, j, k) + (cq - sg * conj(cr)) / (2im)), i, j, k, ir, jr, kr)
end
function certS!(SH::AbstractArray{T,3}, SA::AbstractArray{T,3}, ps::Tuple, rv::NTuple{3,Bool}, V) where {T}
    n1, n2, n3 = size(SH)
    tForW(n3, length(SH)) do k
        kr = rv[3] ? n3 + 1 - k : k
        @inbounds for j in 1:n2, i in 1:n1
            ir = rv[1] ? n1 + 1 - i : i; jr = rv[2] ? n2 + 1 - j : j
            hx, axx = accC(ps, zero(Complex{T}), zero(Complex{T}), i, j, k, ir, jr, kr)
            SH[i, j, k] = max(SH[i, j, k], abs(hx) / V); SA[i, j, k] = max(SA[i, j, k], abs(axx) / V)
        end
    end
    SH
end
# certFil!'s scale S (route R): per offset and entry pH = Σ_q cur(X_q)/V, pA = Σ_q cur(Y_q)/V over the terms (X, cX, Y, cY) in order (unrolled);
# SH = max(SH, |pH|), SA likewise; the operations of the broadcasts it replaces, threaded over the last dimension
@inline accR(::Tuple{}, pH, pA, i, j, k, V) = (pH, pA)
@inline function accR(ts::Tuple, pH, pA, i, j, k, V)
    X, cX, Y, cY = first(ts)
    accR(Base.tail(ts), pH + cAt(X, cX, i, j, k) / V, pA + cAt(Y, cY, i, j, k) / V, i, j, k, V)
end
function certSR!(SH::AbstractArray{T,3}, SA::AbstractArray{T,3}, ts::Tuple, V) where {T}
    n1, n2, n3 = size(SH)
    tForW(n3, length(SH)) do k
        @inbounds for j in 1:n2, i in 1:n1
            X, cX, Y, cY = first(ts)
            pH, pA = accR(Base.tail(ts), cAt(X, cX, i, j, k) / V, cAt(Y, cY, i, j, k) / V, i, j, k, V)
            SH[i, j, k] = max(SH[i, j, k], abs(pH)); SA[i, j, k] = max(SA[i, j, k], abs(pA))
        end
    end
    SH
end

# memo of a certification's per-panel bounds across its rounds (the bound of an unchanged panel is the same value): a locked Dict, the value
# computed outside the lock (two threads may compute the same key; the first stored is kept, both equal)
struct BMemo
    d::Dict{Any,Any}; lk::ReentrantLock
end
BMemo() = BMemo(Dict{Any,Any}(), ReentrantLock())
function bmo(m::BMemo, k, f)
    v = lock(() -> get(m.d, k, m), m.lk)
    v === m || return v
    w = f()
    lock(() -> get!(m.d, k, w), m.lk)
end

# perVec's image sums at one node it: per offset m of axis i, Σ_n e^{iθn} κ((m + nP) s, t) over the images within the Gaussian reach, with
# Σ|terms|; the loop of perVec behind a function barrier (same operations)
function perRS!(M, Ma, it::Int, msi, P_::Int, sT, t, αq, kv::F, Pq) where {F}
    T = typeof(sT)
    for (j, m) in enumerate(msi)
        dn(n) = max(abs(m + n * P_) - 1, 0) * sT
        n0 = round(Int, -m / P_); dmin = min(dn(n0), dn(n0 - 1), dn(n0 + 1))
        Dq = dmin^2 + 4 * t * αq
        nlo = n0; while dn(nlo - 1)^2 <= Dq; nlo -= 1; end
        nhi = n0; while dn(nhi + 1)^2 <= Dq; nhi += 1; end
        a0 = zero(Complex{T}); a1 = zero(Complex{T}); a2 = zero(Complex{T}); s0 = zero(T); s1 = zero(T); s2 = zero(T)
        for n in nlo:nhi
            k = kv(m + n * P_); ph = phsN(T, n, Pq)
            a0 += ph * k[1]; a1 += ph * k[2]; a2 += ph * k[3]; s0 += abs(k[1]); s1 += abs(k[2]); s2 += abs(k[3])
        end
        @inbounds M[1][j, it] = a0; M[2][j, it] = a1; M[3][j, it] = a2; Ma[1][j, it] = s0; Ma[2][j, it] = s1; Ma[3][j, it] = s2
    end
    nothing
end
# perVec's Poisson column at one node: M_k[:, it] += c_k col (function barrier)
function colAdd!(M, it::Int, col, c, c1, c2)
    M1, M2, M3 = M[1], M[2], M[3]
    @inbounds for j in eachindex(col)
        M1[j, it] += c * col[j]; M2[j, it] += c1 * col[j]; M3[j, it] += c2 * col[j]
    end
    nothing
end

# cCert!'s per-offset ratio bound/target (E(m) + E(-m))/2 / Smin / tol (0 where Smin = 0), behind a function barrier
certRR(E::AbstractArray{T}, Er, Smin, tol) where {T} = [Smin[j] > 0 ? (E[j] + Er[j]) / 2 / Smin[j] / tol : zero(T) for j in eachindex(E)]

# rs0Terms' exact kink data per axis and offset (certFil!: T0, C0, D1 and the kink distances), the same exact rational operations in
# Rational{Int128} (no BigInt temporaries, whose heap Julia 1.12 does not return), in QI where Int128 overflows; one rounding
# per value as before, so the same numbers
function rs0Dat(::Type{T}, axk, xq, isper, pers) where {T}
    try
        rs0DatQ(T, Rational{Int128}, axk, xq, isper, pers)
    catch e
        (e isa OverflowError || e isa InexactError) || rethrow()
        rs0DatQ(T, QI, axk, xq, isper, pers)
    end
end
function rs0DatQ(::Type{T}, ::Type{Q}, axk, xq, isper, pers) where {T,Q}
    map(1:3) do i
        A = axk[i]; cp = Q(A.cp); cm = Q(A.cm)
        L = isper(i) ? Q(pers[i].L) : zero(Q)
        pjs = (cp, cm, -cm, -cp); sgs = (1, -1, -1, 1)
        tri(yq) = sum(sg * abs(yq + pj) / 2 for (pj, sg) in zip(pjs, sgs))
        map(xq[i]) do x0q
            x0 = Q(x0q)
            ns = isper(i) ? (round(Int, toT(T, -x0 / L)) .+ (-4:4)) : (0:0)
            T0 = zero(T); C0 = zero(T); D1 = zero(T); ys = T[]
            for n in ns
                xx = x0 + (isper(i) ? n * L : zero(Q))
                T0 += abs(toT(T, tri(xx))); dt = zero(Q); zs = 0
                for (pj, sg) in zip(pjs, sgs)
                    yq = xx + pj; dt += sg * sign(yq) / 2
                    yq == 0 ? (zs += sg) : push!(ys, toT(T, abs(yq)))
                end
                C0 += abs(zs) / sqrt(T(π)); D1 += abs(toT(T, dt))
            end
            (T0, C0, D1, ys)
        end
    end
end

# acc[a,b] += Σ_t w[t] V1[:,t] ⊗ V2[:,t] ⊗ V3[:,t], in chunks of KCH nodes whose BLAS partial sums are added with Neumaier
# compensation (kept in CMP until fold!): the plain sum over thousands of nodes cost up to 13 eps against blcTns's compensated one
# (route C)
const CMPS = ScopedValue(IdDict{Any,Any}())
const KCH = 32
function krAdd!(C::AbstractArray{Complex{T},3}, V1, V2, V3, w) where {T}
    Cc = get!(() -> zeros(Complex{T}, size(C)), CMPS[], C)::Array{Complex{T},3}
    krAddB!(C, Cc, V1, V2, V3, w)
end
fold!(C) = (c = pop!(CMPS[], C, nothing); c === nothing || (C .+= c); C)
cmpGet(C) = get(CMPS[], C, nothing)
# M[a,b,c] += Σ_t w[t] A1[a,t] A2[b,t] A3[c,t] for nonnegative real arrays (the Σ|terms| companion of krAdd!; no compensation needed)
krAddA!(M::AbstractArray{T,3}, A1, A2, A3, w) where {T} = krAddAB!(M, A1, A2, A3, w)
# X * Y summed over the inner index in chunks of KCH, the chunk products added with Neumaier compensation
cMul(X::AbstractMatrix{Complex{T}}, Y::AbstractMatrix) where {T} = cMulB(X, Y)

# phase matrices E[row, order] of one axis of a d = 3 spectral tail for the entries even / odd in ν (prTab on a block of rows): on an axis with
# an exact twist 0 or 1/2 (prTw) the order m > tw carries its partner 2tw - m as 2 cos(2πνx) / 2i sin(2πνx) and the partner's column is 0 (the ± pairs
# of the 6-fold d3 mode cancel exactly at x = ±L/4, and their separate rounded phases left an imaginary residue of 9.8e-6 on the real AH
# relative); rows are lists of (x, weight); E0 the plain matrix, returned (twice) where nothing pairs
# (cos, sin)(2π(m - tw)x/L) = csPiQ(2(m - tw)x/L); Float64: from the Int128 parts of the twist tn/td and of x/L (rqP), the same numbers bit for bit
# without Rational{BigInt} arithmetic per element (which costs +0.06 s per 32³ fill at θ = 0)
csPr(::Type{T}, P::PerAx, m, tw, x::QI, rq, tq) where {T} = csPiQ(T, 2 * (m - tw) * x / P.L)
function csPr(::Type{Float64}, P::PerAx, m, tw, x::QI, rq, tq)
    rn, rd = rq; tn, td = tq
    if rd != 0 && td != 0
        n = 2 * (m * td - tn) * rn; d = td * rd
        abs(n) < I53 && 2 * d < I53 && return csPiQ(Float64, n, d)
    end
    csPiQ(Float64, 2 * (m - tw) * x / P.L)
end
function prMat(::Type{T}, P::PerAx, rng, rows, E0) where {T}
    prTw(P) || return E0, E0
    tw = P.tw isa QI ? P.tw : QI0; tw2 = Int(2 * tw); rs = Set(rng)
    tq = abs(denominator(tw)) < I53 ? (Int128(numerator(tw)), Int128(denominator(tw))) : (Int128(0), Int128(0))
    xr = [[(QI(x), w, rqP(P, QI(x))) for (x, w) in r] for r in rows]
    Ee = similar(E0); Eo = similar(E0)
    for (k, m) in enumerate(rng)
        if 2 * m == tw2 || !((tw2 - m) in rs)
            Ee[:, k] .= view(E0, :, k); Eo[:, k] .= view(E0, :, k)
        elseif 2 * m < tw2
            Ee[:, k] .= 0; Eo[:, k] .= 0
        else
            for j in eachindex(xr)
                ce = zero(T); so = zero(T)
                for (x, w, rq) in xr[j]
                    c_, s_ = csPr(T, P, m, tw, x, rq, tq); ce += w * 2 * c_; so += w * 2 * s_
                end
                Ee[j, k] = ce; Eo[j, k] = complex(zero(T), so)
            end
        end
    end
    Ee, Eo
end

# θ-independent per-frequency cache of the lattice-point κ(k s, t) values, keyed on the cell edge, k and the exact node t (blcSet)
const KCS = ScopedValue{Union{Nothing,KapCache}}(nothing)
kapL(A::AxK{T}, sT::T, k::Int, t::S, tol::T) where {T,S} = (c = KCS[]; c === nothing ? kapL0(A, sT, k, t, tol) :
    kcGet!(() -> kapL0(A, sT, k, t, tol), c, (sT, k, t))::NTuple{3,S})
# κ at the lattice point x = k·s of an equal-cell axis (cp = s, cm = 0), integer kink bookkeeping, same formulas as kap
function kapL0(A::AxK{T}, sT::T, k::Int, t::S, tol::T) where {T,S}
    cp = sT; ax = abs(k) * sT
    # moment regime: x = k a formed as toT does (one rounding of k·num/den), no Rational{BigInt} per node
    xq = T(k * Int(numerator(A.a))) / T(Int(denominator(A.a)))
    momReg(cp, abs(xq), t) && return kapV(A, (abs(xq), xq), t, tol)
    u = sqrt(t)
    tri = sT * (abs(k + 1) - 2 * abs(k) + abs(k - 1)) / 2
    dtr = T(sign(k + 1) - 2 * sign(k) + sign(k - 1)) / 2
    k0 = S(tri); k1 = S(dtr); k2 = zero(S)
    for (kk, sg) in ((k + 1, 1), (k, -2), (k - 1, 1))
        v = abs(kk) * sT / (2 * u)
        ie, er, ev = erfTrio(v)
        k0 += sg * u * ie
        kk != 0 && (k1 -= (kk > 0 ? sg : -sg) * er / 2)
        k2 += sg * ev / (2 * sqrt(T(π)) * u)
    end
    return k0, k1, k2
end

"""
    blcFil(s, f, ax, N, P, θ; tol) -> (H, A, info)

H/AH kernels of the Bloch operator of a unit cell of N = (N1,N2,N3) cells of edges s, periodic along ax with periods
P (in cells), phases θ (T radians or Rational twists). Toeplitz offsets as in the module header.
"""
# true inside an outermost blcFil call: the nested ones (route splits, the Double64 rerun) do not rerun their flagged entries
const NESTED = ScopedValue(false)
function blcFil(s, f::Complex{T}, ax::NTuple{D,Int}, N::NTuple{3,Int}, P::NTuple{D,Int}, θ::NTuple{D,Any}; kw...) where {T,D}
    blasMt() && return blas1(() -> blcFil(s, f, ax, N, P, θ; kw...))
    FILACC[] === nothing && return with(() -> blcFil(s, f, ax, N, P, θ; kw...), FILACC => Threads.Atomic{Float64}(0.0))
    NESTED[] && return blcFil0(s, f, ax, N, P, θ; kw...)
    with(NESTED => true) do
        r = blcFil0(s, f, ax, N, P, θ; kw...)
        filD64(r, s, f, ax, N, P, θ; kw...)
    end
end

# per-entry fallback of the fill (peD64 for blcTns): every entry above rmin of its offset's kernel max whose a posteriori rounding
# factor λ_e = Σ|terms_e|/|X_e| exceeds LAMF (x-odd entries on the planes x ≡ 0, L/2 of a periodic axis with θ near 0 or π, where the image
# and order sums cancel by 1/|θ - θ_0|: 1.85e5 eps at θ = 1.26e-5) is recomputed by the same fill in Double64 at the same
# Float64 inputs, restricted to per-axis offset lists (sel) that cover the flagged offsets: slabs m_i ∈ Z_i chosen greedily by flagged
# offsets per cell of fill, each slab restricted on the other axes to the coordinates of its flagged offsets. Truncation: tol = eps(Float64)
# and entT lowered to a quarter of the smallest flagged entry ratio, so every flagged entry keeps its eps target. Rounding becomes
# eps(D64)·λ_e; a flagged entry whose Double64 λ_e·eps(D64) exceeds eps/4 is refused (the D64 run's own λ)
# Only near-symmetric entries are recomputed: entry (a, b) on a plane x_i ≡ 0 or L_i/2 of a periodic axis i with the parity that
# vanishes exactly at the nearer of the twists 0, 1/2 (zKill: x ≡ 0: σ_i = -1; x ≡ L/2: σ_i = -1 near twist 0, +1 near 1/2). Their κ_e is
# O(1) (G ∝ θ - θ_0), so λ_e is method loss. Every other large λ_e measured so far is an intrinsic cancellation of the entry itself (d3 c32
# L = λ/4 θ = 0.7: λ_e 264, error 274 eps, κ_up,e 2013), covered by its own κ, and a Double64 rerun there would cost the
# whole block (185 s at 32³)
# certification stall: bisection shrinks the worst bound/target geometrically when the target is genuine; when it is the rounding noise of
# an entry that vanishes exactly by a symmetry the fill does not mark, the ratio stalls while every round doubles the panels (a composite
# symAx row at x ≡ L/2: 2.7e12, 1.2e10, 7.3e9 over rounds of 9, 17, 35 s, refusal after 12 rounds and 3193 s). From round 3 on the call refuses once
# the decrease of two consecutive rounds cannot reach the target within the rounds left (one round alone is not a stall: the worst panel can
# change between rounds, e.g. 8.1, 3.6e18, 3.9e6, then certified); st holds the previous round's worst bound/target and whether it failed
function crtStl(st, w, round_, what)
    wp, fp = st[]; r = wp / w
    bad = round_ >= 3 && w > 1 && isfinite(wp) && (r <= 1 || log(w) / log(r) > 12 - round_)
    st[] = (w, bad)
    (bad && fp) && error("blcFil: $what quadrature certification stalls at worst bound/target $(Float64(w)) after $round_ bisection rounds (last round x$(round(Float64(r); sigdigits = 3))): a target at the rounding floor, i.e. an entry that vanishes exactly by a symmetry the fill does not mark; refused")
    nothing
end
const LAMF = 256; const RMINF = 1.0e-15; const DHOT = 1 / 32; const LAMC3 = 1024
function nearSym(ms, s, ax, P, θ, I, a, b)
    for (j, i) in enumerate(ax)
        θ[j] isa Rational && continue
        mod(ms[i][I[i]], P[j]) == 0 || continue
        w = mod(Float64(θ[j]) / (2 * π), 1.0); d0 = min(w, 1 - w, abs(w - 0.5))
        0 < d0 < DHOT && zSig(a, b, i) == -1 && return true
    end
    false
end
function filD64(r, s, f::Complex{T}, ax, N, P, θ; entT = 1e-8, tol = eps(T), sel = nothing, kw...) where {T}
    H, A, inf = r
    λH, λA = inf[:λe]; ms = inf[:ms]; nn = length.(ms)
    ents = ((1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3))
    flg, rmn, nfl = flgScan(H, A, λH, λA, ms, s, ax, P, θ)
    nfl == 0 && return r
    per = [i in ax for i in 1:3]; sa = get(kw, :symAx, 0)
    # greedy slab cover: the slab (i, v) of most flagged offsets per offset filled
    F = Set(findall(flg)); Z = [Set{Int}() for i in 1:3]
    while !isempty(F)
        best = (0.0, 0, 0)
        for i in 1:3
            cnt = Dict{Int,Int}()
            for I in F; cnt[I[i]] = get(cnt, I[i], 0) + 1; end
            for (v, c) in cnt; sc = c * nn[i] / prod(nn); sc > best[1] && (best = (sc, i, v)); end
        end
        _, i, v = best
        # periodic axes: the mirror offset too (route C's reflection needs symmetric lists)
        vs = per[i] && i != sa ? (v, nn[i] + 1 - v) : (v,)
        for w in vs; push!(Z[i], w); end
        filter!(I -> !(I[i] in vs), F)
    end
    # group per axis: the slab set Z_i on axis i, the flagged coordinates of its offsets on the others (symmetrized on periodic axes)
    grp = []
    cov = falses(nn...)
    for i in 1:3
        isempty(Z[i]) && continue
        Fi = [I for I in findall(flg) if I[i] in Z[i] && !cov[I]]
        isempty(Fi) && continue
        sl = ntuple(j -> j == i ? sort(collect(Z[i])) : begin
                c = Set(I[j] for I in Fi); per[j] && j != sa && (c = union(c, Set(nn[j] + 1 .- collect(c)))); sort(collect(c))
            end, 3)
        for I in Iterators.product(sl...); cov[CartesianIndex(I)] = true; end
        push!(grp, sl)
    end
    # the wider type: Double64 for Float64; BigFloat at widPrc bits for Double64 and BigFloat (wp: its precision scope); results rounded back to T
    Dd = widT(T); p0 = T === BigFloat ? precision(real(f)) : precision(T); pW = T === Float64 ? 0 : widPrc(p0, f, θ)
    wp(g) = T === Float64 ? g() : setprecision(g, BigFloat, pW)
    epsT = T === BigFloat ? ldexp(BigFloat(1), 1 - p0) : eps(T)
    rd(x) = T === BigFloat ? BigFloat(x; precision = p0) : T(x)
    fD, θD, epsD, tolD, kwD = wp(() -> (Complex{Dd}(Dd(real(f)), Dd(imag(f))), Tuple(t isa Rational ? t : Dd(t) for t in θ), eps(Dd), Dd(tol),
                                        NamedTuple{keys(kw)}(map(v -> v isa AbstractFloat ? Dd(v) : v, values(kw)))))
    entD = min(entT, max(rmn / 4, RMINF / 4))
    for sl in grp
        selD = ntuple(j -> ms[j][sl[j]], 3)
        Hd, Ad, iD = wp(() -> with(() -> blcFil(s, fD, ax, N, P, θD; kwD..., tol = tolD, entT = entD, sel = selD), WPH => false))
        # the Double64 run's own rounding factor on the flagged entries
        lH, lA = iD[:λe]; msD = iD[:ms]
        for I in Iterators.product(sl...)
            J = CartesianIndex(I); Jd = CartesianIndex(ntuple(j -> findfirst(==(ms[j][I[j]]), msD[j]), 3))
            if flg[J]
                for (X, lam) in ((Hd, lH), (Ad, lA))
                    mx = maximum(abs, view(X, :, :, Jd)); mx == 0 && continue
                    for (q, (a, b)) in enumerate(ents)
                        abs(X[a, b, Jd]) >= entT * mx && lam[q, Jd] * epsD > epsT / 4 && error("blcFil: rounding amplification Σ|terms|/|X| = $(lam[q, Jd]) at offset $(ntuple(j -> ms[j][J[j]], 3)) even in $(T === Float64 ? "Double64" : "BigFloat ($pW bits)"); refused")
                    end
                end
            end
            H[:, :, J] .= (z -> complex(rd(real(z)), rd(imag(z)))).(Hd[:, :, Jd]); A[:, :, J] .= (z -> complex(rd(real(z)), rd(imag(z)))).(Ad[:, :, Jd])
            λH[:, J] .= Float32.(lH[:, Jd] .* (epsD / epsT)); λA[:, J] .= Float32.(lA[:, Jd] .* (epsD / epsT))
        end
    end
    r
end
# the per-frequency cache and a fresh compensation store scoped to one fill (nested fills of the same call share the cache)
blcFil0(s, f, ax, N, P, θ; cache = nothing, kw...) = with(() -> filBlk(s, f, ax, N, P, θ; cache, kw...), KCS => cache, CMPS => IdDict{Any,Any}())
# Block fill of the Herm/AH Bloch kernels for one equal-cell shape: all Toeplitz offsets at once.
# Same representation as blcTns; every piece is a sum of rank-1 tensors over the axes (the integrand of K_t is a
# product of per-axis factors), accumulated by matrix products. Offsets: periodic axes m ∈ -(N-1):(N-1), open 0:(N-1).
# Returns H, A as (3,3,n1,n2,n3) arrays, raw normalization, Gila's full entry.
function filBlk(s, f::Complex{T}, ax::NTuple{D,Int}, N::NTuple{3,Int}, P::NTuple{D,Int}, θ::NTuple{D,Any};
               tol::T = eps(T), route::Symbol = :auto, cache = nothing, entT::Real = 1e-8, sel = nothing, symAx::Int = 0) where {T,D}
    tol0 = tol
    inpChk("blcFil", f, ax, P, θ)
    all(>=(1), N) || error("blcFil: unit cell N = $N has an empty axis")
    for j in 1:D
        P[j] >= N[ax[j]] || error("blcFil: period P = $(P[j]) cells along axis $(ax[j]) is shorter than the unit cell's N = $(N[ax[j]]) cells (cells m and m + P coincide)")
    end
    s = q3(s)
    all(>(0), s) || error("blcFil: cell edges $s are not positive")
    L = ntuple(i -> P[i] * s[ax[i]], D)
    fa = real(f); fb = imag(f)
    x = 4 * T(π)^2 * (fa - fb) * (fa + fb); y = 8 * T(π)^2 * fa * fb
    # propagating orders at |y| <= eps²|x| evaluated at y = 0 (blcTns)
    ysp = abs(y) <= eps(T)^2 * abs(x) ? zero(T) : y
    af4 = abs2(f)^2; cr = (fa - fb) * (fa + fb) / af4; ci = -2 * fa * fb / af4
    V = prod(i -> toT(T, s[i]), 1:3)
    nG = ceil(Int, -log(tol) / T(2)) + 8
    # every truncation relative to the smallest reported entry (1e-8 of the tensor max), not to the max: at tol alone a
    # 5e-8 entry carried 112 eps of spectral cutoff (d3 c8 L = 2λ 1+i); the Gauss order stays set by tol
    tol = tol * T(entT)
    α = -log(tol)
    axk = [AxK(T, s[i], s[i]) for i in 1:3]
    sTa = [toT(T, s[i]) for i in 1:3]
    op = Tuple(setdiff(1:3, ax))
    pers = Dict(ax[i] => PerAx(axk[ax[i]], L[i], twT(T, θ[i]), θ[i]) for i in 1:D)
    isper(i) = haskey(pers, i)
    # sel: explicit per-axis offset lists (a sub-block of the Toeplitz block; periodic lists symmetric, as route C's reflection needs)
    ms = [isper(i) ? collect(-(N[i] - 1):(N[i] - 1)) : collect(0:(N[i] - 1)) for i in 1:3]
    if sel !== nothing
        for i in 1:3
            sq = sort(unique(collect(Int, sel[i])))
            (isempty(sq) || !issubset(sq, ms[i])) && error("blcFil: sel along axis $i is empty or outside the block")
            isper(i) && i != symAx && sq != sort(-sq) && error("blcFil: sel along the periodic axis $i is not symmetric")
            ms[i] = sq
        end
    end
    # symAx (for blcCmp): along that periodic axis the rows k = 0..N-1 are the symmetric sums Σ(k) = G(k) + G(-k) (Σ(0) = 2G(0)),
    # every per-axis factor of the axis formed as V(x_k) + V(-x_k) in Double64; the fill is linear in each axis' factor, so the output is Σ with
    # its θ-odd part (x-odd entries ∝ θ - θ_0 at small twists) free of the ±k cancellation. A composite block between cells whose centres are
    # aligned on that axis (modulo the period) is a positive box sum of such pairs
    if symAx > 0
        isper(symAx) || error("blcFil: symAx = $symAx is not a periodic axis")
        ms[symAx] = sel === nothing ? collect(0:(N[symAx] - 1)) : all(>=(0), ms[symAx]) ? ms[symAx] : error("blcFil: sel along symAx $symAx has negative rows")
    end
    xs = [[m * s[i] for m in ms[i]] for i in 1:3]
    nn = length.(ms)
    # θ = 0.0 exactly is the exact twist 0 (e^{iθn} = 1): its reflection zeros are exact too (θ = fl(π) is not π and is left alone)
    twZ(tw) = tw isa QI ? tw : tw == 0 ? QI(0) : tw
    zKl = [[zKill(QI(xq), isper(i), isper(i) ? pers[i].L : 0, isper(i) ? twZ(pers[i].tw) : nothing) for xq in xs[i]] for i in 1:3]
    # Σ(k) at an exact twist 0 or 1/2 (G(-x) = σ G(x) there): the σ = -1 entries vanish on every row
    symAx > 0 && (zKl[symAx] .= (twZ(pers[symAx].tw) isa QI && (mod(twZ(pers[symAx].tw), 1) == 0 || mod(twZ(pers[symAx].tw), 1) == 1 // 2)) ? -1 : 0)
    # and at the exact twist 1/2 on a row x_k ≡ L/2 the σ = +1 entries vanish as well (G(-x_k) = G(x_k - L) = -G(x_k) by the twist): the row is
    # zero, every offset on it an exact zero without a certification target (its rounding noise would be the target of 12 bisection rounds)
    zRw = symAx > 0 && twZ(pers[symAx].tw) isa QI && mod(twZ(pers[symAx].tw), 1) == 1 // 2 ? [2 * mod(QI(xq), pers[symAx].L) == pers[symAx].L for xq in xs[symAx]] : nothing
    # the lattice points (m, x, weight) whose per-axis factors make row j of axis i (symAx: V(x_k) + V(-x_k))
    rowC(i, j) = i == symAx ? (ms[i][j] == 0 ? ((0, QI(0), 2),) : ((ms[i][j], xs[i][j], 1), (-ms[i][j], -xs[i][j], 1))) : ((ms[i][j], xs[i][j], 1),)
    # hot rows: x_i ≡ 0 (mod L_i) on a periodic axis whose twist lies within DHOT of 0 or 1/2 without being exact. There the x_i-odd
    # entries are ∝ sin(θ_i - θ_0) (κ_e O(1)) while the per-axis image and Poisson sums and the propagating-order pairs (ν, -ν) that
    # form them are O(1): those factors are formed in Double64 (work per row, not per offset) and rounded once
    hot = Dict{Int,Vector{Int}}()
    if T === Float64
        for (jj, i) in enumerate(ax)
            tw = pers[i].tw; tw isa QI && continue
            w = mod(Float64(θ[jj]) / (2 * π), 1.0); d0 = min(w, 1 - w, abs(w - 0.5))
            (tw == 0 || d0 >= DHOT) && continue
            # x ≡ 0 and x ≡ L/2 (the L/2 plane alone is covered by κ, a one-ulp shift of R off the plane, but an entry on the L/2 planes of two
            # hot axes is ∝ (θ₁ - θ₀)(θ₂ - θ₀) while κ covers one factor: d2 c4 P = (4,4), H_xy at (L/2, L/2), 4.6e3 eps(1+κ_up))
            js = findall(m -> mod(m, P[jj]) == 0 || 2 * mod(m, P[jj]) == P[jj], ms[i]); isempty(js) || (hot[i] = js)
        end
    end
    symAx > 0 && (hot[symAx] = collect(1:length(ms[symAx])))
    TD = T === Float64 ? Double64 : T; tolD = TD(tol); faD = TD(real(f)); fbD = TD(imag(f))
    hax = isempty(hot) ? Int[] : collect(ax)
    axkD = Dict(i => AxK(TD, s[i], s[i]) for i in hax); sTaD = Dict(i => toT(TD, s[i]) for i in hax)
    # (an exact twist, θ Rational, stays exact: its PerAx phases are cispi of the twist; a Float θ is the radian value)
    persD = Dict(i => (j = findfirst(==(i), ax); PerAx(axkD[i], L[j], θ[j] isa Rational ? twT(TD, θ[j]) : twT(TD, TD(θ[j])), θ[j])) for i in hax)
    # one periodic-axis row (offset index j) at the nodes tv in Double64: the three derivative orders and their Σ|terms| (perVec's forms)
    # (every name assigned in these closures is declared local: an assignment to a name that filBlk also assigns, e.g. V, would rebind filBlk's)
    function perRowD(i, j, tv)
        local Vr = zeros(Complex{TD}, 3, length(tv))
        local Vra = zeros(TD, 3, length(tv))
        for (m_, xq_, c_) in rowC(i, j)
            local V1, V1a = perRowD1(i, m_, xq_, tv)
            Vr .+= c_ .* V1; Vra .+= c_ .* V1a
        end
        Vr, Vra
    end
    # (memo of the Double64 rows per (axis, m, node): RS, EV and their refinements ask for the same rows at the same nodes several times; the
    # Double64 κ of the images was a quarter of a 64³ d2 hot fill)
    rowMemo = TMemo{Tuple{Int,Int,T},Tuple{NTuple{3,Complex{TD}},NTuple{3,TD}}}()
    function perRowD1(i, m_, xq_, tv)
        local Vr = zeros(Complex{TD}, 3, length(tv))
        local Vra = zeros(TD, 3, length(tv))
        for (it, t0) in enumerate(tv)
            local v_, a_ = get!(() -> perRowD1n(i, m_, xq_, t0), rowMemo, (i, m_, T(t0)))
            for q in 1:3; Vr[q, it] = v_[q]; Vra[q, it] = a_[q]; end
        end
        Vr, Vra
    end
    function perRowD1n(i, m_, xq_, t0s)
        local V1, V1a = perRowD1x(i, m_, xq_, [t0s])
        (V1[1, 1], V1[2, 1], V1[3, 1]), (V1a[1, 1], V1a[2, 1], V1a[3, 1])
    end
    function perRowD1x(i, m_, xq_, tv)
        local Pq_ = persD[i]
        local A_ = axkD[i]
        local sT_ = sTaD[i]
        local P__ = P[findfirst(==(i), ax)]
        local Vr = zeros(Complex{TD}, 3, length(tv))
        local Vra = zeros(TD, 3, length(tv))
        local αD = -log(tolD) + 3
        for (it, t0) in enumerate(tv)
            local tt = TD(t0)
            if useReal(TD, Pq_, tt)
                dnD(n) = max(abs(m_ + n * P__) - 1, 0) * sT_
                local n0_ = round(Int, -m_ / P__)
                local dm_ = min(dnD(n0_), dnD(n0_ - 1), dnD(n0_ + 1))
                local D2_ = dm_^2 + 4 * tt * αD
                local nl_ = n0_
                local nh_ = n0_
                while dnD(nl_ - 1)^2 <= D2_; nl_ -= 1; end
                while dnD(nh_ + 1)^2 <= D2_; nh_ += 1; end
                for n in nl_:nh_
                    local kk_ = kapL(A_, sT_, m_ + n * P__, tt, tolD)
                    local ph_ = phsN(TD, n, Pq_)
                    for q in 1:3; Vr[q, it] += ph_ * kk_[q]; Vra[q, it] += abs(kk_[q]); end
                end
            else
                for mq in poiRng(TD, Pq_, tt, αD)
                    local pr_, ν_ = poiPre(TD, mq, Pq_, xq_)
                    local e_ = exp(-4 * TD(π)^2 * ν_^2 * tt)
                    for q in 1:3; Vr[q, it] += pr_[q] * e_; Vra[q, it] += abs(pr_[q] * e_); end
                end
            end
        end
        Vr, Vra
    end
    phED = Dict(i => TMemo{Int,Vector{Complex{TD}}}() for i in hax)
    pcolD(i, m) = get!(() -> [sum(c_ * phsP(TD, m, persD[i], x_) for (_, x_, c_) in rowC(i, j)) for j in eachindex(xs[i])], phED[i], m)
    # the periodic prefactors (c, c iq, c iq²) of order v at row j of axis i in Double64 (sums over rowC)
    ppD(v, i, j) = (r_ = rowC(i, j); z_ = poiPre(TD, v, persD[i], r_[1][2])[1] .* r_[1][3]; length(r_) == 2 ? z_ .+ poiPre(TD, v, persD[i], r_[2][2])[1] .* r_[2][3] : z_)
    # a Double64 row rounded into the Float64 vectors: value, and Σ|terms| as |value| + (eps(D64)/eps(F64)) Σ|terms|
    function putRow!(M, Ma, j, V, Va)
        for q in 1:3, it in axes(V, 2)
            M[q][j, it] = Complex{T}(V[q, it]); Ma[q][j, it] = abs(M[q][j, it]) + T(Va[q, it]) * T(eps(TD) / eps(T))
        end
    end
    # offsets whose whole tensor vanishes by reflection: exact zeros, no certification target (the target was their rounding noise:
    # an anisotropic d2 cell at θ = π was refused "RS quadrature not certified" at m = (0,0,±1))
    zAll = [all(ab -> any(i -> zKl[i][I[i]] != 0 && zSig(ab[1], ab[2], i) == zKl[i][I[i]], 1:3), ((1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3))) for I in CartesianIndices(Tuple(nn))]
    zRw === nothing || (for I in CartesianIndices(zAll); zRw[I[symAx]] && (zAll[I] = true); end)
    Lmx = maximum(i -> toT(T, L[i]), 1:D)
    # nearest-image support distance of every offset, per axis (periodic: over images; open: |x| - s)
    dAx(i, xq) = isper(i) ? minimum(n -> max(toT(T, abs(xq + n * pers[i].L) - axk[i].cp), zero(T)), round(Int, toT(T, -xq / pers[i].L)) .+ (-1:1)) :
                           max(toT(T, abs(xq) - axk[i].cp), zero(T))
    rmx = sqrt(sum(i -> maximum(xq -> dAx(i, xq), xs[i])^2, 1:3))
    rpx = sqrt(sum(i -> isper(i) ? maximum(xq -> dAx(i, xq), xs[i])^2 : zero(T), 1:3))
    if route == :auto
        # a posteriori per entry (as rteAuto): route R unless its real-space part alone costs more than LAM0 (then route C); where
        # route R's propagating-order Σ|terms| exceed LAM0 times the entry (info MH/MA), route C is filled too and each entry and kernel
        # takes the route with the smaller Σ|terms|/|X|
        τR = min(1 / (max(x, zero(T)) + 1), 8 / (abs(y) + eps(T))); D == 3 && (τR = min(τR, Lmx^2 / (4 * T(π))))
        ΛRS = sqrt(T(2)) * exp(1 + 2 * (2 * T(π) * fb)^2 * τR)
        cOK = fb > 0 && fa > 0
        # d = 3 with every order evanescent: route R with the evanescent extension (rteAuto)
        if cOK && ΛRS > LAM0 && !(D == 3 && uEv3(T, L, θ, fa, fb) > 4 * abs(y))
            route = :C
        else
            kw = (; tol = tol0, cache, entT, sel = Tuple(ms), symAx)
            # where route C may be needed (d < 3, Im f > 0, affordable), it starts on its own task while route R runs (PARRC); its
            # result, or the error it raised, is used below exactly as the sequential call's; it is discarded when route R alone suffices
            npC0 = abs(complex(x, y)) * (T(12) + x * τR) / (2 * T(π) * abs(f) * 2 * T(π) * fb) / T(3 // 2)
            tC = (PARRC && cOK && D < 3 && npC0 <= NPCMAX && Threads.nthreads() > 1) ?
                Threads.@spawn(try blcFil(s, f, ax, N, P, θ; kw..., route = :C, sel = Tuple(ms)) catch e_; e_ end) : nothing
            HR, AR, iR = try
                blcFil(s, f, ax, N, P, θ; kw..., route = :R)
            catch
                tC === nothing || wait(tC)
                rethrow()
            end
            cOK || (tC === nothing || wait(tC); return HR, AR, iR)
            # route C over the whole block whenever an entry of route R has λ_e > LAM0 (d < 3; d = 3 above LAMC3), then per entry the route of
            # the smaller λ_e (at 1+0.1i most offsets carry such an entry, and the whole-block route C costs no more than
            # the tail sub-block did: 64³ d2 28 s whole against 60 s mixed); the tail sub-block of the per-offset criterion only
            # when the whole block exceeds route C's panel budget
            λemx = max(maximum(x -> isfinite(x) ? x : 0f0, iR[:λe][1]), maximum(x -> isfinite(x) ? x : 0f0, iR[:λe][2]))
            # (d = 3 keeps route R: the whole-block route C at 1+0.1i cost 85 s and 6 GB at 64³; its entries with λ_R > 1e3 are few,
            # 34 of 2.5e5 at 32³)
            (D < 3 && λemx > LAM0) || (tC === nothing || wait(tC); return HR, AR, iR)
            mx(M) = dropdims(maximum(abs, M; dims = (1, 2)); dims = (1, 2))
            if haskey(iR, :MH)
                λRH = iR[:MH] ./ mx(HR); λRA = iR[:MA] ./ mx(AR)
                need = (λRH .> LAM0) .| (λRA .> LAM0)
            else
                need = falses(length.(ms)...)
            end
            # route C unaffordable at small Im f (rteAuto NPCMAX): route R with its λ reported
            npC = abs(complex(x, y)) * (T(12) + x * τR) / (2 * T(π) * abs(f) * 2 * T(π) * fb) / T(3 // 2)
            npC > NPCMAX && (tC === nothing || wait(tC); return HR, AR, iR)
            op_ = Tuple(setdiff(1:3, ax))
            lo_ = (0, 0, 0)
            rC = try
                tC === nothing ? blcFil(s, f, ax, N, P, θ; kw..., route = :C, sel = Tuple(ms)) : (v_ = fetch(tC); v_ isa Exception ? throw(v_) : v_)
            catch e
                (e isa ErrorException && occursin("panel budget", e.msg)) || rethrow()
                any(need) || return HR, AR, iR
                # route C only over the open offsets at or beyond the first needed one per open axis (the need sits at far open offsets)
                lo_ = ntuple(i -> i in op_ ? minimum(j[i] for j in CartesianIndices(need) if need[j]) - 1 : 0, 3)
                selC = ntuple(i -> ms[i][lo_[i] + 1:end], 3)
                try
                    blcFil(s, f, ax, N, P, θ; kw..., route = :C, sel = selC)
                catch e2
                    (e2 isa ErrorException && occursin("panel budget", e2.msg)) || rethrow()
                    return HR, AR, iR
                end
            end
            HC, AC, iC = rC
            # per entry (as rteAuto): the route of the smaller λ_e = Σ|terms_e|/|X_e| (not per offset and kernel by
            # route C's |C(m)| + |C(-m)| against route R's propagating-order Σ|terms|, both relative to the offset's max)
            eR = iR[:λe]; eC = iC[:λe]
            # every offset of the route C sub-block takes, entry by entry, the route of the smaller λ_e (not only the offsets the per-offset
            # criterion asked for: a small entry of a near offset, e.g. G_yx at y = 0 with L = 8 at 0.37+0.1i, λ_R 1.8e4 (872 eps, 4.6-12.6
            # eps(1+κ_up)) against λ_C 74 (7-14 eps))
            selEnt!(HR, AR, HC, AC, eR[1], eR[2], eC[1], eC[2], lo_)
            return HR, AR, iR
        end
    end
    route == :C && !(fb > 0 && fa > 0) && error("blcFil: route C needs Re f > 0 and Im f > 0")
    # hazard 2 (e^{xτ} <= e), and yτ <= 8: the e^{iyt} weight of the real-space part cancels far offsets beyond that
    τ = min(1 / (max(x, zero(T)) + 1), 8 / (abs(y) + eps(T)))
    route == :C && (τ = min(1 / (max(x, zero(T)) + 1), T(1 // 8) / (2 * T(π) * fb)^2))
    D == 3 && (τ = min(τ, Lmx^2 / (4 * T(π))))
    xg, wg = glq(T, nG)
    Csh = T(4)
    # a priori: the propagating orders' d_o = 2 pole forms (d = 1, route R) estimated before any other stage, so that a block beyond
    # FILMAX is refused at once (refused after 361 s of RS and EV work); per order panels ~ max(graded, X√S/π) times nG nodes
    # times M(√S) angular nodes times the per-node angular factors and products over the open-offset grid
    if D == 1 && route == :R
        P0_ = pers[ax[1]]; tw0_ = P0_.tw isa QI ? toT(T, P0_.tw) : P0_.tw; L0_ = toT(T, P0_.L)
        νm0_ = max(sqrt(max(x + abs(y), zero(T))) / (2 * T(π)), abs(fa)) + 1 / L0_
        Xs_ = [maximum(abs, toT.(T, xs[o])) for o in op]; X_ = sum(k -> Xs_[k] + sTa[op[k]], 1:2)
        for m_ in (floor(Int, tw0_ - νm0_ * L0_) - 1):(ceil(Int, tw0_ + νm0_ * L0_) + 1)
            up_ = uOrd((nuP(T, m_, P0_),), fa, fb); up_ < 0 || continue
            S0_ = -up_ + (α + 12) / τ
            M0_ = maximum(circM([axk[o] for o in op], Xs_, [p_[1], p_[2]], sqrt(S0_), tol) for p_ in ((0, 0), (1, 0), (0, 1), (2, 0), (0, 2), (1, 1)))
            npn_ = max(length(grdPts(-S0_, zero(T), zero(T), max(-up_, 1 / τ) / 8)), ceil(Int, X_ * sqrt(S0_) / T(π)))
            fl0_ = npn_ * nG * M0_ * (3 * (nn[op[1]] + nn[op[2]]) + 12 * nn[op[1]] * nn[op[2]])
            fl0_ > FILMAX && error("blcFil: a priori cost of the propagating order $m_'s pole form, $(round(Float64(fl0_); sigdigits = 3)) flops (M = $M0_ angular nodes, ~$npn_ panels, open extent $(Xs_[1])λ x $(Xs_[2])λ), exceeds $(FILMAX); refused")
        end
    end
    pats = [(0, 0, 0), (2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
    ents = [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)]
    patOf(a, b) = ntuple(i -> dOrd(a, b, i), 3)
    CH = Dict(p => zeros(Complex{T}, nn...) for p in pats); CA = Dict(p => zeros(Complex{T}, nn...) for p in pats)
    # per-axis distances of the offsets from the support (sharpness of the Gaussian factors)
    dd = [unique(sort([dAx(i, xq)^2 for xq in xs[i]])) for i in 1:3]
    D2 = unique(sort(vec([a + b + c for a in dd[1], b in dd[2], c in dd[3]])))
    D2o = unique(sort(vec(D == 1 ? [a + b for a in dd[op[1]], b in dd[op[2]]] : D == 2 ? dd[op[1]] : [zero(T)])))
    νmn(Pq) = (tw = Pq.tw isa QI ? toT(T, mod(Pq.tw, 1)) : mod(Pq.tw, one(T)); min(tw, 1 - tw) / toT(T, Pq.L))
    ueRS(t) = max(zero(T), 4 * T(π)^2 * sum(i -> isper(i) && !useReal(T, pers[i], t) ? νmn(pers[i])^2 : zero(T), 1:3) - x)
    # panel refinement against the sharpest offset that is not negligible there
    function shpUnion(a, b, ue, D2s, EMf)
        pts = T[a, b]
        for d2 in D2s
            E(t) = -ue(t) * t - d2 / (4 * t)
            max(E(a), E(b)) < EMf(d2) - (α + 20) && continue
            q = shpSub(a, b, t -> ue(t) * t + d2 / (4 * t), Csh)
            length(q) > length(pts) && (pts = q)
        end
        pts
    end
    # RS nodes in u = √t
    u0 = minimum(i -> toT(T, s[i]), 1:3) / (2 * sqrt(α + 8)); uτ = sqrt(τ)
    edges = T[0]; ue_ = min(u0, uτ); push!(edges, ue_)
    while ue_ < uτ
        ue_ = min(2 * ue_, uτ); push!(edges, ue_)
    end
    EMRS(d2) = max(maximum(e -> -ueRS(e^2) * e^2 - d2 / (4 * e^2), edges[2:end]), d2 > 0 ? -sqrt(ueRS(τ) * d2) : -T(Inf))
    rsP = Tuple{T,T}[]
    for p in 1:length(edges)-1
        lo0, hi0 = edges[p], edges[p + 1]
        kph = max(1, ceil(Int, abs(y) * (hi0^2 - lo0^2) / T(3 // 2)))
        sub = lo0 == 0 ? [lo0 + (hi0 - lo0) * T(j) / kph for j in 0:kph] : sqrt.(shpUnion(lo0^2, hi0^2, ueRS, D2, EMRS))
        length(sub) - 1 < kph && (sub = [lo0 * (hi0 / lo0)^(T(j) / kph) for j in 0:kph])
        for q in 1:length(sub)-1; push!(rsP, (sub[q], sub[q + 1])); end
    end
    # Gauss nodes of u-panels (t = u², weight with the Jacobian 2u) and of t-panels
    rsNod(pl) = (tv = T[]; wv = T[]; for (lo, hi) in pl, j in 1:nG; u = (hi - lo) / 2 * xg[j] + (hi + lo) / 2; push!(tv, u^2); push!(wv, (hi - lo) / 2 * wg[j] * 2 * u); end; (tv, wv))
    tNod(pl) = (tv = T[]; wv = T[]; for (lo, hi) in pl, j in 1:nG; push!(tv, (hi - lo) / 2 * xg[j] + (hi + lo) / 2); push!(wv, (hi - lo) / 2 * wg[j]); end; (tv, wv))
    tn, tw_ = rsNod(rsP)
    # Σ|terms| per pattern and offset (MH, MA; raw normalization as CH, CA): the a posteriori rounding factor λ_e = Σ|terms_e|/|X_e| of every
    # entry (as blcTns's sH/sA): each node adds |w| Π_i A_i with A_i the Σ|terms| of axis i's own sum (image or Poisson), not |value|
    MH = Dict(p => zeros(T, nn...) for p in pats); MA = Dict(p => zeros(T, nn...) for p in pats)
    # per axis: 3 matrices (n_i × nt), derivative orders 0,1,2, and their Σ|terms| (open axis: |κ|)
    function nodeVecs(i, tv, val)
        M = [zeros(Complex{T}, nn[i], length(tv)) for k in 1:3]
        tFor(length(tv)) do it
            local t = tv[it]
            for (j, xq) in enumerate(xs[i])
                local v = val(i, j, t)
                M[1][j, it] = v[1]; M[2][j, it] = v[2]; M[3][j, it] = v[3]
            end
        end
        M, [abs.(Mk) for Mk in M]
    end
    function addPats!(Vs, tv, wv, ex, Va)
        wHd = 4 * T(π)^2 .* ex .* cos.(y .* tv) .* wv; wAd = 4 * T(π)^2 .* ex .* sin.(y .* tv) .* wv
        wHp = ex .* (cos.(y .* tv) .* cr .- sin.(y .* tv) .* ci) .* wv; wAp = ex .* (sin.(y .* tv) .* cr .+ cos.(y .* tv) .* ci) .* wv
        for p in pats
            wh, wa = p == (0, 0, 0) ? (wHd, wAd) : (wHp, wAp)
            krAdd!(CH[p], Vs[1][p[1] + 1], Vs[2][p[2] + 1], Vs[3][p[3] + 1], complex.(wh))
            krAdd!(CA[p], Vs[1][p[1] + 1], Vs[2][p[2] + 1], Vs[3][p[3] + 1], complex.(wa))
            krAddA!(MH[p], Va[1][p[1] + 1], Va[2][p[2] + 1], Va[3][p[3] + 1], abs.(wh))
            krAddA!(MA[p], Va[1][p[1] + 1], Va[2][p[2] + 1], Va[3][p[3] + 1], abs.(wa))
        end
        # the (0,0,0) pattern also carries the ∂∂ weight for the diagonal: K00 with wHp is not needed (the diagonal ∂a² term is pattern 2e_a)
    end
    # periodic axes: the images of all offsets are lattice points k·s, so κ(k s, t) is evaluated once per (k, node); with Σ|terms|
    function perVec(i, tv)
        Pq = pers[i]; P_ = P[findfirst(==(i), ax)]
        M = [zeros(Complex{T}, nn[i], length(tv)) for k in 1:3]; Ma = [zeros(T, nn[i], length(tv)) for k in 1:3]
        αq = -log(tol) + 3
        tFor(length(tv)) do it
            local t = tv[it]
            if useReal(T, Pq, t)
                local cache = Dict{Int,NTuple{3,T}}()
                kv(k) = get!(() -> kapL(axk[i], sTa[i], k, t, tol), cache, k)
                # the offsets' image sums behind a function barrier (perRS!; the closure's captured names were boxed)
                perRS!(M, Ma, it, ms[i], P_, sTa[i], t, αq, kv, Pq)
            else
                # Poisson form: S^{(k)}(x_j) = Σ_m c_m^{(k)}(t) e^{2πiν_m x_j}, the phase columns cached per order m
                s0 = zero(T); s1 = zero(T); s2 = zero(T)
                for m in poiRng(T, Pq, t, αq)
                    col = get!(() -> [phsP(T, m, Pq, xq) for xq in xs[i]], phC[i], m)
                    ν = nuP(T, m, Pq)
                    c = Pq.A.w0 / toT(T, Pq.L) * sinc(ν * toT(T, Pq.A.a)) * sinc(ν * toT(T, Pq.A.b)) * exp(-4 * T(π)^2 * ν^2 * t)
                    iq = complex(zero(T), 2 * T(π) * ν)
                    local c1 = c * iq; local c2 = c1 * iq
                    s0 += abs(c); s1 += abs(c1); s2 += abs(c2)
                    colAdd!(M, it, col, c, c1, c2)   # (function barrier)
                end
                Ma[1][:, it] .= s0; Ma[2][:, it] .= s1; Ma[3][:, it] .= s2
            end
        end
        if haskey(hot, i)
            for j in hot[i]; putRow!(M, Ma, j, perRowD(i, j, tv)...); end
        end
        M, Ma
    end
    phC = Dict(i => TMemo{Int,Vector{Complex{T}}}() for i in 1:3)
    openV(i, tv) = nodeVecs(i, tv, (ii, j, t) -> Complex{T}.(kapL(axk[ii], sTa[ii], ms[ii][j], t, tol)))
    rsAdd!(tv, wv) = (VV = [isper(i) ? perVec(i, tv) : openV(i, tv) for i in 1:3]; addPats!([v[1] for v in VV], tv, wv, exp.(x .* tv), [v[2] for v in VV]))
    rsAdd!(tn, tw_)
    evP = Tuple{T,T}[]; evAdd! = nothing; SPset = NTuple{D,Int}[]; spRecF = Any[]; spAdd! = nothing; νe2 = zero(T)
    spPlF = Dict{Int,Any}(); pairOf = nothing
    # d = 2: the closed parts of the SP orders' open-axis diagonal, formed exactly (QZ = (QH, QA, ZH, ZA, o)): Q the pattern-0 closed
    # parts (for the diagonal entries of the periodic axes), Z the combined 4π² I_0 + I_oo/f² closed parts (the open axis o)
    QZ = Any[nothing]
    # (QZ[6:9]: the Σ|terms| of Q (H, A) and of Z (H, A))
    qzGet(o_) = (QZ[1] === nothing && (QZ[1] = (zeros(Complex{T}, nn...), zeros(Complex{T}, nn...), zeros(Complex{T}, nn...), zeros(Complex{T}, nn...), o_,
                                                 zeros(T, nn...), zeros(T, nn...), zeros(T, nn...), zeros(T, nn...))); QZ[1])
    function certFil!()
        SH = zeros(T, nn...); SA = zeros(T, nn...)
        # one threaded pass per entry (certSR!), the same operations per element as the broadcasts it replaces
        for (a, b) in ents
            local tH = ((CH[patOf(a, b)], cmpGet(CH[patOf(a, b)]), CA[patOf(a, b)], cmpGet(CA[patOf(a, b)])),)
            a == b && (tH = (tH..., (CH[(0, 0, 0)], cmpGet(CH[(0, 0, 0)]), CA[(0, 0, 0)], cmpGet(CA[(0, 0, 0)]))))
            if a == b && QZ[1] !== nothing
                qz = QZ[1]; tH = (tH..., (a == qz[5] ? qz[3] : qz[1], nothing, a == qz[5] ? qz[4] : qz[2], nothing))
            end
            certSR!(SH, SA, tH, V)
        end
        xq = [[QI(xv) for xv in xs[i]] for i in 1:3]
        pd = [p for p in pats if p != (0, 0, 0)]
        # terms of the per-offset bound: (wH, wA, v1, v2, v3), vectors over the offsets of each axis
        function panTerms(lo, hi, sq)
            m_ = (lo + hi) / 2; h_ = (hi - lo) / 2
            best = nothing; bv = T(Inf)
            for ρ in RHOS
                e = ellT(m_, h_, T(ρ), sq); e === nothing && continue
                # (symAx rows are V(x) + V(-x): twice the majorant, which is even in x)
                vs = [[(i == symAx ? 2 : 1) .* axMajV(axk[i], xq[i], get(pers, i, nothing), k, e, tol) for k in 0:2] for i in 1:3]
                dH, pH, dA, pA = wHA(y, cr, ci, e.ihi, e.thi)
                g = (64 / T(15)) * h_ * T(ρ)^(-2nG) / (T(ρ)^2 - 1) * (sq ? 2 * e.uabs : one(T)) * exp(max(x, zero(T)) * e.thi) / V
                sep = g * (max(dH, dA) * prod(i -> maximum(vs[i][1]), 1:3) + max(pH, pA) * maximum(p -> prod(i -> maximum(vs[i][p[i] + 1]), 1:3), pd))
                isfinite(sep) || continue
                sep < bv && (bv = sep; best = (g, dH, pH, dA, pA, vs))
            end
            best === nothing && return nothing
            g, dH, pH, dA, pA, vs = best
            [(p == (0, 0, 0) ? g * dH : g * pH, p == (0, 0, 0) ? g * dA : g * pA, vs[1][p[1] + 1], vs[2][p[2] + 1], vs[3][p[3] + 1]) for p in pats]
        end
        # the first u-panel [0, hi] (rs0Bnd per offset): polynomial part of the kinks at distance 0 plus the Gaussian part ε,
        # Π(|P| + ε) - Π|P| <= Σ_i ε_i Π_{j != i}(|P_j| + ε_j)
        # the exact (Rational) part of rs0Terms' kink data per axis and offset, once per certification (it does not depend on the panel end hi):
        # T0, C0, D1 and the nonzero kink distances |y| in T; the bisection rounds then only redo the Gaussian part (the Rational
        # arithmetic per round and offset was most of the BigInt churn left in a fill)
        # in Rational{Int128} (rs0Dat): the same exact values without BigInt temporaries; QI where Int128 overflows
        rs0D = rs0Dat(T, axk, xq, isper, pers)
        function rs0Terms(hi)
            dat = map(rs0D) do di
                map(di) do (T0, C0, D1, ys)
                    local e0 = zero(T); local e1 = zero(T); local e2 = zero(T)
                    for ya in ys
                        local v = ya / (2 * hi); local ev = exp(-v^2) / sqrt(T(π))
                        e0 += hi * ev / (2 * v^2); e1 += ev / (2 * v); e2 += ev
                    end
                    (T0, C0, D1, (e0, e1, e2))
                end
            end
            symAx > 0 && (dat[symAx] = [(2 * d[1], 2 * d[2], 2 * d[3], 2 .* d[4]) for d in dat[symAx]])
            Pk(i, k, ua) = [k == 0 ? d[1] + d[2] * ua : k == 1 ? d[3] : d[2] for d in dat[i]]
            Ek(i, k) = [d[4][k + 1] for d in dat[i]]
            az = abs(complex(x, y)); h_ = hi / 2
            wd(fH, fA) = ((4 * T(π)^2 / V) * fH, (4 * T(π)^2 / V) * fA)
            best = nothing; bv = T(Inf)
            for ρ in RHOS
                ua = h_ + h_ * (T(ρ) + 1 / T(ρ)) / 2
                g = (64 / T(15)) * h_ * T(ρ)^(-2nG) / (T(ρ)^2 - 1)
                sy = sinh(abs(y) * ua^2); cy = cosh(abs(y) * ua^2)
                gH = g * exp(az * ua^2); gA = g * exp(abs(x) * ua^2)
                sep = gH * prod(i -> maximum(Pk(i, 0, ua)), 1:3)
                isfinite(sep) && sep < bv && (bv = sep; best = (ua, gH, gA, sy, cy))
            end
            ua, gH, gA, sy, cy = best
            terms = Tuple{T,T,Vector{T},Vector{T},Vector{T}}[]
            j2(p, u_) = any(==(2), p) ? one(T) : 2 * u_
            for p in pats
                if p == (0, 0, 0)
                    wH = gH * 4 * T(π)^2 / V; wA = gA * 4 * T(π)^2 * sy / V
                else
                    wH = gH / (abs2(f) * V); wA = gA * (sy * abs(cr) + cy * abs(ci)) / V
                end
                push!(terms, (wH * j2(p, ua), wA * j2(p, ua), Pk(1, p[1], ua), Pk(2, p[2], ua), Pk(3, p[3], ua)))
                # the Gaussian (ε) part on the real panel, |sin(yt)| <= min(1, |y| t)
                sr = min(one(T), abs(y) * hi^2); ex = 2 * hi * exp(max(x, zero(T)) * hi^2)
                eH = p == (0, 0, 0) ? 4 * T(π)^2 / V : 1 / (abs2(f) * V); eA = p == (0, 0, 0) ? 4 * T(π)^2 * sr / V : (sr * abs(cr) + abs(ci)) / V
                for i in 1:3
                    vv = [j == i ? Ek(j, p[j]) : Pk(j, p[j], hi) .+ Ek(j, p[j]) for j in 1:3]
                    push!(terms, (ex * eH * j2(p, hi), ex * eA * j2(p, hi), vv[1], vv[2], vv[3]))
                end
            end
            terms
        end
        # EV t-panel (route R, d < 3): the EV-only Poisson majorant of the periodic axes (independent of the periodic offset) times the
        # open-axis majorants per open offset (evBnd)
        function evTerms(lo, hi)
            m_ = (lo + hi) / 2; h_ = (hi - lo) / 2
            best = nothing; bv = T(Inf)
            local Pv = [pers[ax[i]] for i in 1:D]
            for ρ in RHOS
                e = ellT(m_, h_, T(ρ), false); e === nothing && continue
                rgs = [poiRngE(T, Pv[i], e.rlo, α + 40, νe2) for i in 1:D]
                Mp = zeros(T, 3, 3)
                for mm in Iterators.product(rgs...)
                    mm in SPset && continue
                    ν = ntuple(i -> nuP(T, mm[i], Pv[i]), D)
                    c = exp(-uOrd(ν, fa, fb) * e.rlo) * prod(i -> abs(poiPre(T, mm[i], Pv[i], QI0)[1][1]), 1:D)
                    for k1 in 0:2, k2 in 0:(D == 2 ? 2 - k1 : 0)
                        Mp[k1 + 1, k2 + 1] += c * (2 * T(π) * abs(ν[1]))^k1 * (D == 2 ? (2 * T(π) * abs(ν[2]))^k2 : one(T))
                    end
                end
                vo = Dict(o => [axMajV(axk[o], xq[o], nothing, k, e, tol) for k in 0:2] for o in op)
                dH, pH, dA, pA = wHA(y, cr, ci, e.ihi, e.thi)
                g = (64 / T(15)) * h_ * T(ρ)^(-2nG) / (T(ρ)^2 - 1) / V
                mpk(p) = Mp[p[ax[1]] + 1, D == 2 ? p[ax[2]] + 1 : 1]
                sep = g * maximum(p -> (p == (0, 0, 0) ? max(dH, dA) : max(pH, pA)) * mpk(p) * prod(o -> maximum(vo[o][p[o] + 1]), op), pats)
                isfinite(sep) || continue
                sep < bv && (bv = sep; best = (g, dH, pH, dA, pA, vo, mpk))
            end
            best === nothing && return nothing
            g, dH, pH, dA, pA, vo, mpk = best
            vec3(p, i) = i in op ? vo[i][p[i] + 1] : fill(T(i == symAx ? 2 : 1), nn[i])
            [((p == (0, 0, 0) ? dH : pH) * g * mpk(p), (p == (0, 0, 0) ? dA : pA) * g * mpk(p), vec3(p, 1), vec3(p, 2), vec3(p, 3)) for p in pats]
        end
        function accum(tl)
            EH = zeros(T, nn...); EA = zeros(T, nn...)
            isempty(tl) && return EH, EA
            # (the block-threaded kernel, 1024 terms at a time)
            for r in Iterators.partition(eachindex(tl), 1024)
                V1 = reduce(hcat, [tl[q][3] for q in r]); V2 = reduce(hcat, [tl[q][4] for q in r]); V3 = reduce(hcat, [tl[q][5] for q in r])
                krAddAB!(EH, V1, V2, V3, [tl[q][1] for q in r]); krAddAB!(EA, V1, V2, V3, [tl[q][2] for q in r])
            end
            EH, EA
        end
        for (nm, pl0, sq, addF) in ((:RS, rsP, true, rsAdd!), (:EV, evP, false, evAdd!))
            isempty(pl0) && continue
            pl = copy(pl0); nb = 0
            # a panel's terms do not change within the certification: memoized across the rounds (bmo)
            local bm_ = BMemo()
            local wPv = Ref((T(Inf), false))
            for round_ in 0:12
                tms = tMap(pn -> bmo(bm_, pn, () -> pn[1] == 0 ? rs0Terms(pn[2]) : sq ? panTerms(pn[1], pn[2], true) : evTerms(pn[1], pn[2])), pl)
                bad0 = findall(t_ -> t_ === nothing, tms)
                tl = reduce(vcat, [t_ for t_ in tms if t_ !== nothing]; init = Tuple{T,T,Vector{T},Vector{T},Vector{T}}[])
                EH, EA = accum(tl)
                # ratios, not EH/(tol·SH): tol·SH underflowed at Im f = 1e-300 (AH ~ 1e-301, NaN "not certified")
                rH = [SH[j] > 0 ? EH[j] / SH[j] / tol : zero(T) for j in eachindex(EH)]
                rA = [SA[j] > 0 ? EA[j] / SA[j] / tol : (EA[j] > 0 ? T(Inf) : zero(T)) for j in eachindex(EA)]
                rH[vec(zAll)] .= 0; rA[vec(zAll)] .= 0
                worst = max(maximum(rH), maximum(rA))
                isempty(bad0) && worst <= 1 && break
                crtStl(wPv, worst, round_, "real-axis ($nm)")
                round_ == 12 && error("blcFil: real-axis ($nm) quadrature not certified at the target: $(count(>(1), rH) + count(>(1), rA)) offsets, worst bound/target $(Float64(worst))")
                # the worst offset; panels whose bound there exceeds 1/(2·#panels) of its target (or whose majorant is infinite) are halved
                jw = rH[argmax(rH)] >= rA[argmax(rA)] ? argmax(rH) : argmax(rA)
                jc = CartesianIndices(EH)[jw]; useA = rA[jw] > rH[jw]; sc_ = useA ? SA[jw] : SH[jw]
                split = Set(bad0)
                for (q, t_) in enumerate(tms)
                    t_ === nothing && continue
                    bq = sum(tt[useA ? 2 : 1] * tt[3][jc[1]] * tt[4][jc[2]] * tt[5][jc[3]] for tt in t_)
                    (sc_ > 0 ? bq / sc_ : T(Inf)) > tol / (2 * length(pl)) && push!(split, q)
                end
                old = [pl[q] for q in split]; new = Tuple{T,T}[]
                for (lo, hi) in old
                    c = lo == 0 ? hi / 2 : (sq ? sqrt(lo * hi) : (lo + hi) / 2); push!(new, (lo, c)); push!(new, (c, hi))
                end
                nb += length(old)
                to, wo = sq ? rsNod(old) : tNod(old); tw, ww = sq ? rsNod(new) : tNod(new)
                addF(to, -wo); addF(tw, ww)
                pl = vcat([pl[q] for q in eachindex(pl) if !(q in split)], new)
            end
        end

        # propagating orders (route R, d < 3): per order and panel the blcTns bound (ray: ATAP 19.3 with the open-axis majorants per open
        # offset, ellC neg; pole form: poleMaj) as an array over the open offsets; the pole bound is increasing in X = |x_o| + cp, so it is
        # evaluated at <= 8 levels of X (d_o = 2: of X1 + X2) and each offset takes the level at or above its own X
        if !isempty(spRecF)
            xo = [[QI(xv) for xv in xs[o]] for o in op]
            Xof = [[abs(toT(T, xv)) + toT(T, axk[o].cp) for xv in xo[k]] for (k, o) in enumerate(op)]
            pdOrd = [p_ for p_ in pats]
            Smin = reshape([SH[j] > 0 && SA[j] > 0 ? min(SH[j], SA[j]) : max(SH[j], SA[j]) for j in eachindex(SH)], size(SH))
            nOp = Tuple(length.(xo))
            function ordBnd(r, lo, hi)
                out = Dict{NTuple{3,Int},Array{T}}()
                if r.pinf.kind == :ray
                    e = r.pinf.e; tm = τ + (lo + hi) / 2 * e; Δ = (hi - lo) / 2 * e
                    best = nothing; bv = T(Inf)
                    for ρ in RHOS
                        e_ = ellC(tm, Δ, T(ρ); neg = true); e_ === nothing && continue
                        vo = [[axMajV(axk[o], xo[k], nothing, kk, e_, tol) for kk in 0:2] for (k, o) in enumerate(op)]
                        g = (64 / T(15)) * abs(Δ) * T(ρ)^(-2nG) / (T(ρ)^2 - 1) * exp(e_.rmax(-r.p))
                        sep = g * maximum(p_ -> prod(k -> maximum(vo[k][p_[op[k]] + 1]), eachindex(op)), pdOrd)
                        isfinite(sep) || continue
                        sep < bv && (bv = sep; best = (g, vo))
                    end
                    best === nothing && return nothing
                    g, vo = best
                    for p_ in pdOrd
                        out[p_] = length(op) == 2 ? g .* vo[1][p_[op[1]] + 1] .* transpose(vo[2][p_[op[2]] + 1]) : g .* vo[1][p_[op[1]] + 1]
                    end
                    return out
                end
                # pole form: levels of X
                Xall = length(op) == 2 ? vec([a_ + b_ for a_ in Xof[1], b_ in Xof[2]]) : Xof[1]
                lv = sort(unique([maximum(Xall); [quantile_(Xall, q) for q in (0.5, 0.75, 0.875, 0.94, 0.97)]; minimum(Xall)]))
                w0s = [axk[o].w0 for o in op]
                for p_ in pdOrd
                    bl = map(lv) do X
                        local pd = PoleDat{T}(collect(op), w0s, length(op) == 2 ? [X, zero(T)] : [X], τ, ysp)
                        gBndR(lo, hi, nG, (δm, hA, hB) -> poleMaj(pd, p_, r.p, r.up, δm, hA, hB))
                    end
                    lev(X) = bl[findfirst(>=(X - 1e-12 * X), lv)]
                    out[p_] = length(op) == 2 ? [lev(Xof[1][a_] + Xof[2][b_]) for a_ in eachindex(Xof[1]), b_ in eachindex(Xof[2])] : [lev(Xv) for Xv in Xof[1]]
                end
                out
            end
            quantile_(v, q) = sort(v)[clamp(ceil(Int, q * length(v)), 1, length(v))]
            # |periodic prefactor| per pattern as an array over the periodic axes, shaped for broadcasting against the open arrays
            shpA(v, dims) = reshape(v, ntuple(k -> k in dims ? size(v, findfirst(==(k), dims)) : 1, 3))
            function perW(r, p_)
                pv = [abs.([r.PP[i][j][p_[ax[i]] + 1] for j in eachindex(r.PP[i])]) for i in 1:D]
                w = (p_ == (0, 0, 0) ? 4 * T(π)^2 : abs(complex(cr, ci))) / V
                D == 1 ? w .* shpA(pv[1], (ax[1],)) : w .* shpA(pv[1], (ax[1],)) .* shpA(pv[2], (ax[2],))
            end
            opA(b) = length(op) == 2 ? shpA(b, (op[1], op[2])) : shpA(b, (op[1],))
            pls = [copy(r.pinf.pl) for r in spRecF]; nb = 0
            local bm_ = BMemo()   # the panels' bounds memoized across the rounds
            local wPv = Ref((T(Inf), false))
            for round_ in 0:12
                bnds = tMap2(q -> bmo(bm_, q, () -> ordBnd(spRecF[q[1]], q[2], q[3])), [[(g, lo, hi) for (lo, hi) in pls[g]] for g in eachindex(spRecF)])
                E = zeros(T, nn...); bad0 = Tuple{Int,Int}[]
                # per order and pattern the panels' open-offset bounds summed first, then one outer product with the periodic factor
                for g in eachindex(spRecF)
                    Bs = Dict{NTuple{3,Int},Array{T}}()
                    for (q, b) in enumerate(bnds[g])
                        b === nothing && (push!(bad0, (g, q)); continue)
                        for p_ in pdOrd; haskey(Bs, p_) ? (Bs[p_] .+= b[p_]) : (Bs[p_] = copy(b[p_])); end
                    end
                    for (p_, B_) in Bs; tAdd!(*, E, perW(spRecF[g], p_), opA(B_)); end
                end
                rr = [Smin[j] > 0 ? E[j] / Smin[j] / tol : zero(T) for j in eachindex(E)]
                rr[vec(zAll)] .= 0
                worst = maximum(rr)
                if isempty(bad0) && worst <= 1
                    for g in eachindex(spRecF); spPlF[g] = pls[g]; end
                    break
                end
                crtStl(wPv, worst, round_, "propagating-order (SP)")
                round_ == 12 && error("blcFil: propagating-order (SP) quadrature not certified at the target: $(count(>(1), rr)) offsets, worst bound/target $(Float64(worst))")
                jc = CartesianIndices(E)[argmax(rr)]; sc_ = Smin[jc]; ntot = sum(length, pls)
                for g in eachindex(spRecF)
                    split = Set(q for (g_, q) in bad0 if g_ == g)
                    for (q, b) in enumerate(bnds[g])
                        b === nothing && continue
                        bq = sum(perW(spRecF[g], p_)[min(jc[1], end), min(jc[2], end), min(jc[3], end)] * opA(b[p_])[min(jc[1], size(opA(b[p_]), 1)), min(jc[2], size(opA(b[p_]), 2)), min(jc[3], size(opA(b[p_]), 3))] for p_ in pdOrd)
                        bq / sc_ > tol / (2 * ntot) && push!(split, q)
                    end
                    isempty(split) && continue
                    new = Tuple{T,T}[]; oldS = Tuple{T,T}[]; newS = Tuple{T,T}[]
                    for q in eachindex(pls[g])
                        lo, hi = pls[g][q]
                        if q in split
                            c = (lo + hi) / 2; push!(new, (lo, c)); push!(new, (c, hi)); push!(oldS, (lo, hi)); push!(newS, (lo, c)); push!(newS, (c, hi))
                        else
                            push!(new, (lo, hi))
                        end
                    end
                    nb += length(split)
                    # only the changed panels are recomputed: the order's value on the old and on the new sub-panels (the pole form's
                    # closed part is the same deterministic number in both and cancels in the difference)
                    r = spRecF[g]
                    # (an order formed elsewhere, PPa masked, needs only its new panels)
                    if r.PPa === r.PP
                        # (the remainders alone where spFil returns them: the closed parts cancel exactly)
                        rmOf(z) = haskey(z[4], :rm) ? z[4].rm : z[1]
                        Iold = rmOf(spFil(T, axk, op, xs, r.p, r.up, r.cont, ysp, τ, α, tol, nG, r.ap, s; pans = oldS))
                        Inew = rmOf(spFil(T, axk, op, xs, r.p, r.up, r.cont, ysp, τ, α, tol, nG, r.ap, s; pans = newS))
                        # (a ZZD order's patterns 0 and oo are recomputed on the final panels in Double64: no deltas for them here)
                        if r.zzD !== nothing
                            k0_ = (0, 0, 0); koo_ = ntuple(i -> i == op[1] ? 2 : 0, 3)
                            Iold = copy(Iold); Inew = copy(Inew); Iold[k0_] = zero.(Iold[k0_]); Iold[koo_] = zero.(Iold[koo_]); Inew[k0_] = zero.(Inew[k0_]); Inew[koo_] = zero.(Inew[koo_])
                        end
                        spAdd!(r.PPa, Iold, nothing, nothing, -1); spAdd!(r.PPa, Inew, nothing, nothing, 1)
                    end
                    pls[g] = new
                end
            end
        end
    end

    info = Dict{Symbol,Any}()
    if route == :C
        cFil!(CH, CA, s, f, ax, xs, ms, nn, axk, sTa, pers, pats, τ, T(12), T(4), rpx, α, tol, nG, D2, D2o; zAll, MH, MA,
              hotC = isempty(hot) ? nothing : (hot = hot, axkD = axkD, sTaD = sTaD, persD = persD, tolD = tolD, faD = faD, fbD = fbD, symAx = symAx, rowC = rowC, ppD = ppD))
    elseif D == 3
        VΛ = prod(i -> toT(T, L[i]), 1:3)
        kS = α + 12
        Pv = [pers[i] for i in 1:3]
        # cutoff relative to the largest term e^{-u_S τ} (blcTns,: with every order evanescent the absolute cutoff dropped all but
        # the orders within e^{-5} of the largest, an unseen absolute floor)
        uS0 = uOrd(ntuple(i -> νmn(Pv[i]), 3), fa, fb); uS = max(uS0, zero(T))
        function cqTab(τq)
            local rngs = [begin
                twv = Pv[i].tw isa QI ? toT(T, Pv[i].tw) : Pv[i].tw
                νm = sqrt(max(x, zero(T)) + uS + kS / τq) / (2 * T(π)) + 1 / toT(T, Pv[i].L)
                collect((floor(Int, twv - νm * toT(T, Pv[i].L)) - 1):(ceil(Int, twv + νm * toT(T, Pv[i].L)) + 1))
            end for i in 1:3]
            # coefficient tensors per pattern, then three mode products with the phase matrices E_i[m, q] = e^{2πiν_q x_m}
            local nq = length.(rngs)
            local Cq = Dict(p => (zeros(Complex{T}, nq...), zeros(Complex{T}, nq...)) for p in pats)
            # Σ|terms| per order and pattern (H, A): the products apart, not |C_q|
            local CqM = Dict(p => (zeros(T, nq...), zeros(T, nq...)) for p in pats)
            for (i1, m1) in enumerate(rngs[1]), (i2, m2) in enumerate(rngs[2]), (i3, m3) in enumerate(rngs[3])
                local mm, ν, uq, e, den, ReW, ImW, ReWf, ImWf, ReE, ImE, c
                mm = (m1, m2, m3)
                ν = ntuple(i -> nuP(T, mm[i], Pv[i]), 3)
                uq = uOrd(ν, fa, fb)
                (uq - uS) * τq > kS && continue
                abs(uq) <= 64 * eps(T) * abs(x) && abs(y) <= 64 * eps(T) * abs(x) && error("blcFil: empty-lattice mode (3D) at rounding distance, orders $mm")
                e = exp(-uq * τq); den = uq^2 + y^2
                ReW = e * (uq * cos(y * τq) - y * sin(y * τq)) / den; ImW = e * (uq * sin(y * τq) + y * cos(y * τq)) / den
                ReWf = ReW * cr - ImW * ci; ImWf = ReW * ci + ImW * cr
                ReE = e * cos(y * τq); ImE = e * sin(y * τq)
                c = V^2 / VΛ * prod(i -> sinc(ν[i] * toT(T, s[i]))^2, 1:3)
                for p in pats
                    p == (0, 0, 0) && continue
                    local a, Sa, νa, zr, zi, azr, azi, νν
                    if any(==(2), p)
                        # the diagonal entry a (pattern 2e_a) carries the δ part too: 4π²W - 4π²ν_a²W/f² = (-E + 4π² S_a W)/f², E = e^{-pτ},
                        # S_a = Σ_{d≠a} ν_d² (blcTns,: both forms carry the pole 1/p and cancel it near a mode along axis a); pattern 0 empty
                        a = findfirst(==(2), p); Sa = 4 * T(π)^2 * sum(d -> d == a ? zero(T) : ν[d]^2, 1:3); νa = 4 * T(π)^2 * ν[a]^2
                        if dgE(Sa, νa, hypot(ReW, ImW), e, hypot(cr, ci))
                            zr = Sa * ReW - ReE; zi = Sa * ImW - ImE; azr = Sa * abs(ReW) + abs(ReE); azi = Sa * abs(ImW) + abs(ImE)
                            Cq[p][1][i1, i2, i3] = c * (zr * cr - zi * ci); Cq[p][2][i1, i2, i3] = c * (zr * ci + zi * cr)
                            CqM[p][1][i1, i2, i3] = c * (azr * abs(cr) + azi * abs(ci)); CqM[p][2][i1, i2, i3] = c * (azr * abs(ci) + azi * abs(cr))
                        else
                            Cq[p][1][i1, i2, i3] = c * (4 * T(π)^2 * ReW - νa * ReWf); Cq[p][2][i1, i2, i3] = c * (4 * T(π)^2 * ImW - νa * ImWf)
                            CqM[p][1][i1, i2, i3] = c * (4 * T(π)^2 * abs(ReW) + νa * (abs(ReW * cr) + abs(ImW * ci))); CqM[p][2][i1, i2, i3] = c * (4 * T(π)^2 * abs(ImW) + νa * (abs(ReW * ci) + abs(ImW * cr)))
                        end
                    else
                        # (2πiν)^{p}: the spectral ∂∂ factor is -4π² ν_a ν_b
                        νν = prod(i -> (2 * T(π) * ν[i])^p[i], 1:3)
                        Cq[p][1][i1, i2, i3] = -c * νν * ReWf; Cq[p][2][i1, i2, i3] = -c * νν * ImWf
                        CqM[p][1][i1, i2, i3] = c * abs(νν) * (abs(ReW * cr) + abs(ImW * ci)); CqM[p][2][i1, i2, i3] = c * abs(νν) * (abs(ReW * ci) + abs(ImW * cr))
                    end
                end
            end
            rngs, Cq, CqM
        end
        # evanescent extension (blcTns): every order evanescent with u_S > 4|y|; the RS part continued to τE where the tail's
        # Σ|terms| (the same at every offset, |phase| = 1) is <= |X_RS,e(m)|/2 for every entry above entT of its offset's max
        if uS0 > 4 * abs(y) && max(x, zero(T)) * τ < 200
            for it in 1:12
                _, _, CqM_ = cqTab(τ)
                rr = zero(T)
                for (Cd, S_) in ((CH, (p -> sum(CqM_[p][1]))), (CA, (p -> sum(CqM_[p][2]))))
                    Xe = [begin
                        Xp = Cd[patOf(a, b)] .+ something(cmpGet(Cd[patOf(a, b)]), 0)
                        a == b ? Xp .+ Cd[(0, 0, 0)] .+ something(cmpGet(Cd[(0, 0, 0)]), 0) : Xp
                    end for (a, b) in ents]
                    Se = [S_(patOf(a, b)) for (a, b) in ents]
                    for I in CartesianIndices(Xe[1])
                        zAll[I] && continue
                        mI = maximum(q -> abs(Xe[q][I]), eachindex(ents)); mI == 0 && (maximum(Se) > 0 && (rr = T(Inf)); continue)
                        for (q, (a, b)) in enumerate(ents)
                            any(i -> zKl[i][I[i]] != 0 && zSig(a, b, i) == zKl[i][I[i]], 1:3) && continue
                            rr = max(rr, Se[q] / max(abs(Xe[q][I]), T(entT) * mI))
                        end
                    end
                end
                rr <= 1 / 2 && break
                τn = min(τ + (log(2 * rr) + 1) / uS0, 4 * τ)
                x > 0 && (τn = min(τn, 200 / x))
                τn <= τ && break
                ue_ = sqrt(τ); uN = sqrt(τn); eg = T[ue_]
                while ue_ < uN; ue_ = min(2 * ue_, uN); push!(eg, ue_); end
                newP = Tuple{T,T}[]
                for p in 1:length(eg)-1
                    lo0, hi0 = eg[p], eg[p + 1]
                    kph = max(1, ceil(Int, abs(y) * (hi0^2 - lo0^2) / T(3 // 2)))
                    sub = sqrt.(shpUnion(lo0^2, hi0^2, ueRS, D2, EMRS))
                    length(sub) - 1 < kph && (sub = [lo0 * (hi0 / lo0)^(T(j) / kph) for j in 0:kph])
                    for q in 1:length(sub)-1; push!(newP, (sub[q], sub[q + 1])); end
                end
                append!(rsP, newP); tnE, twE = rsNod(newP); rsAdd!(tnE, twE)
                τ = τn
            end
        end
        rngs, Cq, CqM = cqTab(τ)
        nq = length.(rngs)
        Ep = [[phsP(T, mq, Pv[i], xq) for xq in xs[i], mq in rngs[i]] for i in 1:3]
        # ±m pairs on the axes with an exact twist 0 or 1/2 (prMat): per pattern the even or odd phase matrix of each axis, and every offset's
        # Σ|terms| as the mode product of the term magnitudes with |E| (no longer the same at every offset)
        prA = any(i -> prTw(Pv[i]), 1:3)
        EpP = [prMat(T, Pv[i], rngs[i], [((xq, 1),) for xq in xs[i]], Ep[i]) for i in 1:3]
        Epp(p) = [isodd(p[i]) ? EpP[i][2] : EpP[i][1] for i in 1:3]
        EpA = [(abs.(EpP[i][1]), abs.(EpP[i][2])) for i in 1:3]
        mode(Cm, E) = begin
            X1 = reshape(E[1] * reshape(Cm, nq[1], :), nn[1], nq[2], nq[3])
            X2 = permutedims(reshape(E[2] * reshape(permutedims(X1, (2, 1, 3)), nq[2], :), nn[2], nn[1], nq[3]), (2, 1, 3))
            permutedims(reshape(E[3] * reshape(permutedims(X2, (3, 1, 2)), nq[3], :), nn[3], nn[1], nn[2]), (2, 3, 1))
        end
        # hot slabs (x_i ≡ 0 rows of a hot axis): coefficients and all three mode products in Double64 (the order pairs (ν, -ν) of the hot
        # axis cancel in the first product; where two axes are hot, in two)
        CqD = nothing
        if !isempty(hot)
            τD = TD(τ); yD = 8 * TD(π)^2 * faD * fbD; afD = abs2(complex(faD, fbD))^2
            crD = (faD - fbD) * (faD + fbD) / afD; ciD = -2 * faD * fbD / afD; VD = prod(i -> toT(TD, s[i]), 1:3); VΛD = prod(i -> toT(TD, L[i]), 1:3)
            PvD = [persD[i] for i in 1:3]
            CqD = Dict(p => (zeros(Complex{TD}, nq...), zeros(Complex{TD}, nq...)) for p in pats)
            for (i1, m1) in enumerate(rngs[1]), (i2, m2) in enumerate(rngs[2]), (i3, m3) in enumerate(rngs[3])
                mm = (m1, m2, m3)
                uq = uOrd(ntuple(i -> nuP(T, mm[i], Pv[i]), 3), fa, fb); (uq - uS) * τ > kS && continue
                ν = ntuple(i -> nuP(TD, mm[i], PvD[i]), 3); uqD = uOrd(ν, faD, fbD)
                e = exp(-uqD * τD); den = uqD^2 + yD^2
                ReW = e * (uqD * cos(yD * τD) - yD * sin(yD * τD)) / den; ImW = e * (uqD * sin(yD * τD) + yD * cos(yD * τD)) / den
                ReWf = ReW * crD - ImW * ciD; ImWf = ReW * ciD + ImW * crD
                ReE = e * cos(yD * τD); ImE = e * sin(yD * τD)
                c = VD^2 / VΛD * prod(i -> sinc(ν[i] * toT(TD, s[i]))^2, 1:3)
                for p in pats
                    p == (0, 0, 0) && continue
                    if any(==(2), p)
                        a = findfirst(==(2), p); Sa = 4 * TD(π)^2 * sum(d -> d == a ? zero(TD) : ν[d]^2, 1:3); νa = 4 * TD(π)^2 * ν[a]^2
                        if dgE(Sa, νa, hypot(ReW, ImW), e, hypot(crD, ciD))
                            zr = Sa * ReW - ReE; zi = Sa * ImW - ImE
                            CqD[p][1][i1, i2, i3] = c * (zr * crD - zi * ciD); CqD[p][2][i1, i2, i3] = c * (zr * ciD + zi * crD)
                        else
                            CqD[p][1][i1, i2, i3] = c * (4 * TD(π)^2 * ReW - νa * ReWf); CqD[p][2][i1, i2, i3] = c * (4 * TD(π)^2 * ImW - νa * ImWf)
                        end
                    else
                        νν = prod(i -> (2 * TD(π) * ν[i])^p[i], 1:3)
                        CqD[p][1][i1, i2, i3] = -c * νν * ReWf; CqD[p][2][i1, i2, i3] = -c * νν * ImWf
                    end
                end
            end
            EpD = [[sum(c_ * phsP(TD, mq, PvD[i], x_) for (_, x_, c_) in rowC(i, j)) for j in eachindex(xs[i]), mq in rngs[i]] for i in 1:3]
            EpDP = [prMat(TD, PvD[i], rngs[i], [[(x_, c_) for (_, x_, c_) in rowC(i, j)] for j in eachindex(xs[i])], EpD[i]) for i in 1:3]
        end
        modeE(Cm, E) = begin
            m1, m2, m3 = size(E[1], 1), size(E[2], 1), size(E[3], 1)
            X1 = reshape(E[1] * reshape(Cm, nq[1], :), m1, nq[2], nq[3])
            X2 = permutedims(reshape(E[2] * reshape(permutedims(X1, (2, 1, 3)), nq[2], :), m2, m1, nq[3]), (2, 1, 3))
            permutedims(reshape(E[3] * reshape(permutedims(X2, (3, 1, 2)), nq[3], :), m3, m1, m2), (2, 3, 1))
        end
        # the mode products of every pattern, and of every hot slab in Double64, on the threads; consumed in the original order below
        hl_ = collect(hot); pd_ = [p for p in pats if p != (0, 0, 0)]
        XHA_ = Dict(zip(pd_, tMap(pd_) do p
            local E_ = Epp(p); local Ea_ = [isodd(p[i]) ? EpA[i][2] : EpA[i][1] for i in 1:3]
            prA ? (mode(Cq[p][1], E_), mode(Cq[p][2], E_), mode(CqM[p][1], Ea_), mode(CqM[p][2], Ea_)) : (mode(Cq[p][1], E_), mode(Cq[p][2], E_))
        end))
        YD_ = CqD === nothing ? nothing : Dict(zip([(p, i) for p in pd_ for (i, _) in hl_], tMap([(p, i, hj) for p in pd_ for (i, hj) in hl_]) do (p, i, hj)
            local Es = [k == i ? EpDP[k][isodd(p[k]) ? 2 : 1][hj, :] : EpDP[k][isodd(p[k]) ? 2 : 1] for k in 1:3]
            (Complex{T}.(modeE(CqD[p][1], Es)), Complex{T}.(modeE(CqD[p][2], Es)))
        end))
        for p in pats
            # (pattern 0 of the spectral tail is empty: its δ part sits in the diagonal patterns)
            p == (0, 0, 0) && continue
            XH, XA = XHA_[p][1], XHA_[p][2]
            CH[p] .+= XH; CA[p] .+= XA
            # |E_i| = 1: every offset's Σ|terms| is Σ_q of the order's term magnitudes (paired axes: the mode product with |E|)
            sH = prA ? XHA_[p][3] : sum(CqM[p][1]); sA = prA ? XHA_[p][4] : sum(CqM[p][2])
            MH[p] .+= sH; MA[p] .+= sA
            CqD === nothing && continue
            rD = T(eps(TD) / eps(T)); sHD = T(sum(abs, CqD[p][1])) * rD; sAD = T(sum(abs, CqD[p][2])) * rD
            # the value and Σ|terms| applied so far per offset (two hot slabs meet on a line: each replaces what is there)
            aH = XH; aA = XA; bH = prA ? copy(sH) : fill(sH, nn...); bA = prA ? copy(sA) : fill(sA, nn...)
            for (i, hj) in hl_
                sl = ntuple(k -> k == i ? hj : Colon(), 3)
                YH, YA = YD_[(p, i)]
                CH[p][sl...] .+= YH .- aH[sl...]; CA[p][sl...] .+= YA .- aA[sl...]; aH[sl...] .= YH; aA[sl...] .= YA
                zH = abs.(YH) .+ sHD; zA = abs.(YA) .+ sAD
                MH[p][sl...] .+= zH .- bH[sl...]; MA[p][sl...] .+= zA .- bA[sl...]; bH[sl...] .= zH; bA[sl...] .= zA
            end
        end
    else
        Pv = [pers[ax[i]] for i in 1:D]
        ordR = [begin
            twv = Pv[i].tw isa QI ? toT(T, Pv[i].tw) : Pv[i].tw
            νm = max(sqrt(max(x + abs(y), zero(T))) / (2 * T(π)), abs(fa)) + 1 / toT(T, Pv[i].L)
            (floor(Int, twv - νm * toT(T, Pv[i].L)) - 1):(ceil(Int, twv + νm * toT(T, Pv[i].L)) + 1)
        end for i in 1:D]
        SP = NTuple{D,Int}[]
        for mm in Iterators.product(ordR...)
            νq = ntuple(i -> nuP(T, mm[i], Pv[i]), D)
            (uOrd(νq, fa, fb) <= abs(y) || (fb < 0 && abs(fa) >= sqrt(sum(abs2, νq)))) && push!(SP, mm)
        end
        box = [begin
            twv = Pv[i].tw isa QI ? toT(T, Pv[i].tw) : Pv[i].tw
            νm = sqrt(max(x, zero(T)) + 2 * abs(y)) / (2 * T(π)) + 3 / toT(T, Pv[i].L)
            (floor(Int, twv - νm * toT(T, Pv[i].L)) - 2):(ceil(Int, twv + νm * toT(T, Pv[i].L)) + 2)
        end for i in 1:D]
        umin = T(Inf)
        for mm in Iterators.product(box...)
            mm in SP && continue
            umin = min(umin, uOrd(ntuple(i -> nuP(T, mm[i], Pv[i]), D), fa, fb))
        end
        abs(umin) <= 64 * eps(T) * max(abs(x), 1) && error("blcFil: Rayleigh-Wood anomaly (evanescent side) at rounding distance")
        ρmx = sqrt(maximum(D2o))
        αp = α + 12
        tmax = max(τ + αp / umin, ((sqrt(αp) + sqrt(αp + 2 * sqrt(umin) * ρmx)) / (2 * sqrt(umin)))^2)
        spProj = [unique(g[i] for g in SP) for i in 1:D]; append!(SPset, SP); νe2 = (umin + x) / (4 * T(π)^2)
        # EV nodes
        tlo = τ
        while tlo < tmax
            thi = min(2 * tlo, tmax)
            nsub = max(1, ceil(Int, abs(y) * (thi - tlo) / 2))
            sub = shpUnion(tlo, thi, t -> umin, D2o, d2 -> -sqrt(umin * d2))
            length(sub) - 1 < nsub && (sub = [tlo * (thi / tlo)^(T(j) / nsub) for j in 0:nsub])
            for q in 1:length(sub)-1; push!(evP, (sub[q], sub[q + 1])); end
            tlo = thi
        end
        te, we = tNod(evP)
        uSPm = minimum((uOrd(ntuple(i -> nuP(T, g[i], Pv[i]), D), fa, fb) for g in SP); init = umin)
        # EV factor: EV-only Poisson sum also where (image - SP) would cancel by e^{(u_min - u_SP,min) t} > 4 (see blcTns)
        allP(t) = all(i -> !useReal(T, Pv[i], t), 1:D) || (umin - uSPm) * t > log(T(4))
        # per periodic axis and node: the EV periodic factor; rank-1 in d = 1, a matrix over (m1, m2) in d = 2
        #   all axes in Poisson form: Σ over the orders not in SP (no subtraction); else image form minus the SP Poisson terms
        αq = -log(tol) + 3
        phE = Dict(i => TMemo{Int,Vector{Complex{T}}}() for i in 1:3)
        pcol(i, m) = get!(() -> [phsP(T, m, pers[i], xq) for xq in xs[i]], phE[i], m)
        function pcoef(i, m, t; ex = false)
            Pq = pers[i]; ν = nuP(T, m, Pq)
            # ex = true: e^{xt} folded into the exponent, e^{-(4π²ν² - x)t} (d = 1 EV Poisson form; e^{xt} alone overflowed: NaN)
            c = Pq.A.w0 / toT(T, Pq.L) * sinc(ν * toT(T, Pq.A.a)) * sinc(ν * toT(T, Pq.A.b)) * (ex ? exp(-uOrd((ν,), fa, fb) * t) : exp(-4 * T(π)^2 * ν^2 * t))
            iq = complex(zero(T), 2 * T(π) * ν)
            (complex(c), c * iq, c * iq^2)
        end
        pcoef0(i, m) = (Pq = pers[i]; ν = nuP(T, m, Pq); c = Pq.A.w0 / toT(T, Pq.L) * sinc(ν * toT(T, Pq.A.a)) * sinc(ν * toT(T, Pq.A.b));
                        iq = complex(zero(T), 2 * T(π) * ν); ((complex(c), c * iq, c * iq^2), ν))
        # the 64-node E/EA buffers of the d = 2 EV block (set by evAdd!/evCAdd!)
        evBf = Ref{Tuple{Dict{NTuple{2,Int},Matrix{Complex{T}}},Dict{NTuple{2,Int},Matrix{T}}}}()
        evAdd1! = function (te, we)
        exE = [allP(t) ? one(T) : exp(x * t) for t in te]
        Vo = Dict(o => openV(o, te) for o in op)
        if D == 1
            i1 = ax[1]
            Vp = [zeros(Complex{T}, nn[i1], length(te)) for k in 1:3]; Vpa = [zeros(T, nn[i1], length(te)) for k in 1:3]
            Sfull, Sfa = perVec(i1, te)
            for (it, t) in enumerate(te)
                if allP(t)
                    for m in poiRngE(T, Pv[1], t, αq, νe2)
                        m in spProj[1] && continue
                        cf = pcoef(i1, m, t; ex = true); col = pcol(i1, m)
                        for k in 1:3; Vp[k][:, it] .+= cf[k] .* col; Vpa[k][:, it] .+= abs(cf[k]); end
                    end
                else
                    for k in 1:3; Vp[k][:, it] .= Sfull[k][:, it]; Vpa[k][:, it] .= Sfa[k][:, it]; end
                    for m in spProj[1]
                        cf = pcoef(i1, m, t); col = pcol(i1, m)
                        for k in 1:3; Vp[k][:, it] .-= cf[k] .* col; Vpa[k][:, it] .+= abs(cf[k]); end
                    end
                end
            end
            if haskey(hot, i1)
                # hot rows in Double64: the EV-only Poisson sum, or the full sum minus the SP Poisson terms
                for j in hot[i1]
                    local Sd, Sda = perRowD(i1, j, te)
                    local Vd = zeros(Complex{TD}, 3, length(te))
                    local Vda = zeros(TD, 3, length(te))
                    for (it, t) in enumerate(te)
                        local tD = TD(t)
                        if allP(t)
                            for mq in poiRngE(T, Pv[1], t, αq, νe2)
                                mq in spProj[1] && continue
                                local pr_ = ppD(mq, i1, j)
                                local ν_ = nuP(TD, mq, persD[i1])
                                local e_ = exp(-uOrd((ν_,), faD, fbD) * tD)
                                for q in 1:3; Vd[q, it] += pr_[q] * e_; Vda[q, it] += abs(pr_[q] * e_); end
                            end
                        else
                            for q in 1:3; Vd[q, it] = Sd[q, it]; Vda[q, it] = Sda[q, it]; end
                            for mq in spProj[1]
                                local pr_ = ppD(mq, i1, j)
                                local ν_ = nuP(TD, mq, persD[i1])
                                local e_ = exp(-4 * TD(π)^2 * ν_^2 * tD)
                                for q in 1:3; Vd[q, it] -= pr_[q] * e_; Vda[q, it] += abs(pr_[q] * e_); end
                            end
                        end
                    end
                    putRow!(Vp, Vpa, j, Vd, Vda)
                end
            end
            VsEV = Vector{Vector{Matrix{Complex{T}}}}(undef, 3); VaEV = Vector{Vector{Matrix{T}}}(undef, 3)
            VsEV[i1] = Vp; VaEV[i1] = Vpa; for o in op; VsEV[o] = Vo[o][1]; VaEV[o] = Vo[o][2]; end
            addPats!(VsEV, te, we, exE, VaEV)
        else
            i1, i2 = ax; o = op[1]
            n1, n2, n3 = nn[i1], nn[i2], nn[o]
            S1, S1a = perVec(i1, te); S2, S2a = perVec(i2, te)
            # E[k1,k2] : (n1*n2) × nt, and its Σ|terms| EA
            # views of 64-column buffers allocated once per evAdd! call (evBf); every column is overwritten below
            E = Dict(k => view(v, :, 1:length(te)) for (k, v) in evBf[][1]); EA = Dict(k => view(v, :, 1:length(te)) for (k, v) in evBf[][2])
            tFor(length(te)) do it
                local t = te[it]
                local r1, r2, Φ1, Φ2, eu, C, M, Ma, c1_, c2_
                if allP(t)
                    r1 = collect(poiRngE(T, Pv[1], t, αq, νe2)); r2 = collect(poiRngE(T, Pv[2], t, αq, νe2))
                    Φ1 = hcat([pcol(i1, m) for m in r1]...); Φ2 = hcat([pcol(i2, m) for m in r2]...)
                    local c1 = [pcoef0(i1, m) for m in r1]; local c2 = [pcoef0(i2, m) for m in r2]
                    eu = [exp(-uOrd((c1[a][2], c2[b][2]), fa, fb) * t) for a in eachindex(r1), b in eachindex(r2)]
                    for (k1, k2) in keys(E)
                        C = [(g1, g2) in SP ? zero(Complex{T}) : c1[a][1][k1 + 1] * c2[b][1][k2 + 1] * eu[a, b] for (a, g1) in enumerate(r1), (b, g2) in enumerate(r2)]
                        # (the product written into E's column, no temporaries of the block's size)
                        mul!(reshape(view(E[(k1, k2)], :, it), n1, n2), Φ1 * C, transpose(Φ2)); EA[(k1, k2)][:, it] .= sum(abs, C)
                    end
                else
                    # E = S1 S2ᵀ - Σ_g (c1_g φ1_g)(c2_g φ2_g)ᵀ as one product [S1 | -c1 φ1 ...] [S2 | c2 φ2 ...]ᵀ written into E's column
                    local pc1 = [pcoef(i1, g[1], t) for g in SP]; local pc2 = [pcoef(i2, g[2], t) for g in SP]
                    for (k1, k2) in keys(E)
                        local L1 = hcat(S1[k1 + 1][:, it], reduce(hcat, [-pc1[q][k1 + 1] .* pcol(i1, g[1]) for (q, g) in enumerate(SP)]; init = zeros(Complex{T}, n1, 0)))
                        local L2 = hcat(S2[k2 + 1][:, it], reduce(hcat, [pc2[q][k2 + 1] .* pcol(i2, g[2]) for (q, g) in enumerate(SP)]; init = zeros(Complex{T}, n2, 0)))
                        mul!(reshape(view(E[(k1, k2)], :, it), n1, n2), L1, transpose(L2))
                        local sa_ = sum(q -> abs(pc1[q][k1 + 1]) * abs(pc2[q][k2 + 1]), eachindex(SP); init = zero(T))
                        Ma = reshape(view(EA[(k1, k2)], :, it), n1, n2)
                        mul!(Ma, view(S1a[k1 + 1], :, it:it), transpose(view(S2a[k2 + 1], :, it:it))); Ma .+= sa_
                    end
                end
            end
            # hot rows (x_{i1} ≡ 0) and hot columns (x_{i2} ≡ 0) of E in Double64: the Poisson pair sums over the hot axis (and, where both are
            # hot, over both) formed before the one rounding; the other axis enters with its Float64 factors (no cancellation along it)
            if haskey(hot, i1) || haskey(hot, i2)
                local h1 = get(hot, i1, Int[])
                local h2 = get(hot, i2, Int[])
                local S1d = Dict(j => perRowD(i1, j, te) for j in h1)
                local S2d = Dict(j => perRowD(i2, j, te) for j in h2)
                local prs = unique(vcat([(j1, j2) for j1 in h1 for j2 in 1:n2], [(j1, j2) for j1 in 1:n1 for j2 in h2]))
                for (it, t) in enumerate(te)
                    local tD = TD(t)
                    if allP(t)
                        local r1 = collect(poiRngE(T, Pv[1], t, αq, νe2))
                        local r2 = collect(poiRngE(T, Pv[2], t, αq, νe2))
                        local q1c = [poiPre(TD, m, persD[i1], QI0) for m in r1]
                        local q2c = [poiPre(TD, m, persD[i2], QI0) for m in r2]
                        local Φ1 = reduce(hcat, [pcolD(i1, m) for m in r1])
                        local Φ2 = reduce(hcat, [pcolD(i2, m) for m in r2])
                        local eu = [(g1, g2) in SP ? zero(TD) : exp(-uOrd((q1c[a][2], q2c[b][2]), faD, fbD) * tD) for (a, g1) in enumerate(r1), (b, g2) in enumerate(r2)]
                        for (k1, k2) in keys(E)
                            local Cd = [q1c[a][1][k1 + 1] * q2c[b][1][k2 + 1] * eu[a, b] for a in eachindex(r1), b in eachindex(r2)]
                            local ca = T(sum(abs, Cd)) * T(eps(TD) / eps(T))
                            # hot rows: (Φ1[h1, :] C) Φ2ᵀ; hot columns: Φ1 (C Φ2[h2, :]ᵀ)
                            if !isempty(h1)
                                local Wr = (Φ1[h1, :] * Cd) * transpose(Φ2)
                                for (q, j1) in enumerate(h1), j2 in 1:n2
                                    local vr = Complex{T}(Wr[q, j2]); E[(k1, k2)][j1 + n1 * (j2 - 1), it] = vr; EA[(k1, k2)][j1 + n1 * (j2 - 1), it] = abs(vr) + ca
                                end
                            end
                            if !isempty(h2)
                                local Wc = Φ1 * (Cd * transpose(Φ2[h2, :]))
                                for (q, j2) in enumerate(h2), j1 in 1:n1
                                    local vc = Complex{T}(Wc[j1, q]); E[(k1, k2)][j1 + n1 * (j2 - 1), it] = vc; EA[(k1, k2)][j1 + n1 * (j2 - 1), it] = abs(vc) + ca
                                end
                            end
                        end
                    else
                        # per SP order the two per-axis columns (Double64): A1[j1, g] = c1_g φ1_g(j1), A2[j2, g] = c2_g φ2_g(j2) e^{-4π²(ν1² + ν2²)t}; the
                        # hot rows/columns of E are then S1 ⊗ S2 - A1 A2ᵀ restricted to them (small Double64 products; per-entry loops over the orders
                        # cost ~30 s per 64³ d2 hot fill). Typed: with every order evanescent (no SP order) the untyped n×0 comprehensions were Matrix{Any} and A1 A2ᵀ raised
                        local spc = [(poiPre(TD, g[1], persD[i1], QI0), poiPre(TD, g[2], persD[i2], QI0)) for g in SP]
                        local spe = [exp(-4 * TD(π)^2 * (q1[2]^2 + q2[2]^2) * tD) for (q1, q2) in spc]
                        for (k1, k2) in keys(E)
                            local A1 = Complex{TD}[spc[gq][1][1][k1 + 1] * pcolD(i1, g[1])[j] for j in 1:n1, (gq, g) in enumerate(SP)]
                            local A2 = Complex{TD}[spc[gq][2][1][k2 + 1] * pcolD(i2, g[2])[j] * spe[gq] for j in 1:n2, (gq, g) in enumerate(SP)]
                            local S1v = [haskey(S1d, j) ? S1d[j][1][k1 + 1, it] : Complex{TD}(S1[k1 + 1][j, it]) for j in 1:n1]
                            local S2v = [haskey(S2d, j) ? S2d[j][1][k2 + 1, it] : Complex{TD}(S2[k2 + 1][j, it]) for j in 1:n2]
                            local S1w = [haskey(S1d, j) ? S1d[j][2][k1 + 1, it] : TD(S1a[k1 + 1][j, it]) for j in 1:n1]
                            local S2w = [haskey(S2d, j) ? S2d[j][2][k2 + 1, it] : TD(S2a[k2 + 1][j, it]) for j in 1:n2]
                            local aA1 = abs.(A1)
                            local aA2 = abs.(A2)
                            if !isempty(h1)
                                local Vr = S1v[h1] * transpose(S2v) .- A1[h1, :] * transpose(A2)
                                local Va = S1w[h1] * transpose(S2w) .+ aA1[h1, :] * transpose(aA2)
                                for (q, j1) in enumerate(h1), j2 in 1:n2
                                    local v_ = Complex{T}(Vr[q, j2]); E[(k1, k2)][j1 + n1 * (j2 - 1), it] = v_; EA[(k1, k2)][j1 + n1 * (j2 - 1), it] = abs(v_) + T(Va[q, j2]) * T(eps(TD) / eps(T))
                                end
                            end
                            if !isempty(h2)
                                local Vc = S1v * transpose(S2v[h2]) .- A1 * transpose(A2[h2, :])
                                local Va = S1w * transpose(S2w[h2]) .+ aA1 * transpose(aA2[h2, :])
                                for (q, j2) in enumerate(h2), j1 in 1:n1
                                    local v_ = Complex{T}(Vc[j1, q]); E[(k1, k2)][j1 + n1 * (j2 - 1), it] = v_; EA[(k1, k2)][j1 + n1 * (j2 - 1), it] = abs(v_) + T(Va[j1, q]) * T(eps(TD) / eps(T))
                                end
                            end
                        end
                    end
                end
            end
            ex = exE
            wHd = 4 * T(π)^2 .* ex .* cos.(y .* te) .* we; wAd = 4 * T(π)^2 .* ex .* sin.(y .* te) .* we
            wHp = ex .* (cos.(y .* te) .* cr .- sin.(y .* te) .* ci) .* we; wAp = ex .* (sin.(y .* te) .* cr .+ cos.(y .* te) .* ci) .* we
            # per E block (k1, k2) one product over the node chunk for every pattern and kernel that uses it (the columns of evG2), added
            # into evAc (compensated across all chunks of the call, folded into CH/CA by evAdd!)
            for (kk, ps_) in evG2(i1, i2)
                local Y, Ya, ac_
                Y = reduce(vcat, [Vo[o][1][p[o] + 1] .* transpose(h_ == 1 ? (p == (0, 0, 0) ? wHd : wHp) : (p == (0, 0, 0) ? wAd : wAp)) for p in ps_ for h_ in 1:2])
                Ya = reduce(vcat, [Vo[o][2][p[o] + 1] .* transpose(abs.(h_ == 1 ? (p == (0, 0, 0) ? wHd : wHp) : (p == (0, 0, 0) ? wAd : wAp))) for p in ps_ for h_ in 1:2])
                ac_ = evAc[][kk]
                cMulAcc!(ac_[1], ac_[2], E[kk], Matrix(transpose(Y))); rMulAcc!(ac_[3], EA[kk], Matrix(transpose(Ya)))
            end
        end
        end
        # the patterns that read E block (k1, k2), in pats order
        evG2(i1, i2) = [(kk, [p for p in pats if (p[i1], p[i2]) == kk]) for kk in sort(unique([(p[i1], p[i2]) for p in pats]))]
        evAc = Ref{Any}(nothing)
        # in chunks of 64 nodes: the node-by-offset arrays (E: n1·n2 × nodes × 6 in d = 2) held GBs at 64³ (7 GB RSS)
        evAdd! = function (te, we)
            local i1, i2, o, n12, perm, n3
            if D == 2
                i1, i2 = ax; o = op[1]; n12 = nn[i1] * nn[i2]
                evAc[] = Dict(kk => (zeros(Complex{T}, n12, 2 * nn[o] * length(ps_)), zeros(Complex{T}, n12, 2 * nn[o] * length(ps_)), zeros(T, n12, 2 * nn[o] * length(ps_))) for (kk, ps_) in evG2(i1, i2))
                evBf[] = (Dict((k1, k2) => Matrix{Complex{T}}(undef, n12, 64) for k1 in 0:2, k2 in 0:2 if k1 + k2 <= 2), Dict((k1, k2) => Matrix{T}(undef, n12, 64) for k1 in 0:2, k2 in 0:2 if k1 + k2 <= 2))
            end
            for r in Iterators.partition(eachindex(te), 64); evAdd1!(te[r], we[r]); end
            if D == 2
                perm = invperm([i1, i2, o]); n3 = nn[o]
                for (kk, ps_) in evG2(i1, i2), (q, p) in enumerate(ps_), (h_, Cd, Md) in ((1, CH, MH), (2, CA, MA))
                    local cols = (2 * (q - 1) + h_ - 1) * n3 .+ (1:n3); local ac_ = evAc[][kk]
                    addPerm!(Cd[p], reshape(view(ac_[1], :, cols) .+ view(ac_[2], :, cols), nn[i1], nn[i2], n3), perm)
                    addPerm!(Md[p], reshape(ac_[3][:, cols], nn[i1], nn[i2], n3), perm)
                end
                evAc[] = nothing
            end
        end
        evAdd!(te, we)
        # SP orders: prefactor vectors on the periodic axes times the open-axis integrals over all open offsets
        MHp = Dict(p => zeros(T, nn...) for p in pats); MAp = Dict(p => zeros(T, nn...) for p in pats)
        # one order's contribution: periodic prefactor vectors times the open-axis integrals, pattern by pattern (sgn = -1: remove it;
        # mag: also its Σ|terms| into MHp/MAp)
        spAdd! = function (PP, Iop, IaR, IaI, sgn, ovr = nothing)
            for pt in pats
                ko = ntuple(i -> i in op ? pt[i] : 0, 3)
                # ovr: per-pattern replacements of the open-axis integral (patterns 0 and oo without their closed parts)
                Io = (ovr !== nothing && haskey(ovr, pt)) ? ovr[pt] : Iop[ko]
                # per-axis vectors: periodic prefactors (order pt[i]), open: the integral array (rank not 1 for d_o = 2)
                W0 = pt == (0, 0, 0) ? 4 * T(π)^2 .* Io : Io .* complex(cr, ci)
                # outer product: periodic prefactor vectors along their axes, the open-axis array along the open axes
                pv = [[PP[i][j][pt[ax[i]] + 1] for j in eachindex(PP[i])] for i in 1:D]
                shp(v, i) = reshape(v, ntuple(k -> k == i ? length(v) : 1, 3))
                cper = D == 1 ? shp(pv[1], ax[1]) : shp(pv[1], ax[1]) .* shp(pv[2], ax[2])
                Wr = real.(W0); Wi = imag.(W0)
                if IaR !== nothing
                    aR = IaR[ko]; aI = IaI[ko]
                    mR = pt == (0, 0, 0) ? 4 * T(π)^2 .* aR : aR .* abs(cr) .+ aI .* abs(ci)
                    mI = pt == (0, 0, 0) ? 4 * T(π)^2 .* aI : aR .* abs(ci) .+ aI .* abs(cr)
                end
                # the order's outer products are collected per pattern and added as one product over the orders (spFlush!), at the end
                # of this call or, in the SP loop, once after every order
                push!(spBuf[pt], (sgn .* vec(cper), vec(Wr), vec(Wi), IaR === nothing ? nothing : (abs.(vec(cper)), vec(mR), vec(mI))))
            end
            spBat[] || spFlush!()
        end
        spBuf = Dict(pt => Any[] for pt in pats); spBat = Ref(false); spXb = Ref{Tuple{Matrix{Complex{T}},Matrix{T}}}()
        spFlush! = function ()
            local pd_ = sort(collect(ax)); local dPO = (nn[pd_]..., nn[collect(op)]...); local pPO = invperm([pd_..., op...])
            for pt in pats
                local L = spBuf[pt]; isempty(L) && continue
                local Pm = reduce(hcat, [l[1] for l in L])
                # the products into two buffers kept across flushes (spXb), threaded over fixed row blocks (mMulB!)
                (isassigned(spXb) && size(spXb[][1]) == (size(Pm, 1), prod(nn[collect(op)]))) || (spXb[] = (Matrix{Complex{T}}(undef, size(Pm, 1), prod(nn[collect(op)])), Matrix{T}(undef, size(Pm, 1), prod(nn[collect(op)]))))
                local Xb, Xr = spXb[]
                for (dst, k) in ((CH, 2), (CA, 3))
                    addPerm!(dst[pt], reshape(mMulB!(Xb, Pm, transpose(reduce(hcat, [Complex{T}.(l[k]) for l in L]))), dPO), pPO)
                end
                local La = [l[4] for l in L if l[4] !== nothing]
                if !isempty(La)
                    local Pa = reduce(hcat, [l[1] for l in La])
                    for (dst, k) in ((MHp, 2), (MAp, 3))
                        addPerm!(dst[pt], reshape(mMulB!(Xr, Pa, transpose(reduce(hcat, [l[k] for l in La]))), dPO), pPO)
                    end
                end
                empty!(L)
            end
        end
        # propagating-order pairs of the hot axis (d = 1): orders m, m' = 2 tw0 - m (tw0 the nearer of 0, 1/2) both in SP. Their sum
        # on the hot rows is O(θ - θ_0) and is formed by spPair2 after the certification; spAdd! leaves those rows out (PPa)
        pairOf = Dict{NTuple{D,Int},NTuple{D,Int}}()
        # d = 2 with a hot axis: every SP order is recomputed in Double64 for the hot rows (spFil with d_o = 1 is cheap; pairs and the quadruples
        # of two hot axes are then formed exactly)
        hot2 = D == 2 && !isempty(hot)
        # d = 2: the groups of orders related by the reflections g_j -> m0_j - g_j along the hot axes (the whole product set in SP); only
        # grouped orders are left out of spAdd! and formed after the certification (singles have no reflected partner to cancel against)
        grp2 = Dict{NTuple{D,Int},Any}()
        if hot2
            m02 = Dict(i => (w_ = mod(Float64(θ[j]) / (2 * π), 1.0); abs(w_ - 0.5) < min(w_, 1 - w_) ? 1 : 0) for (j, i) in enumerate(ax) if haskey(hot, i))
            SPs2 = Set(SP)
            for g in SP
                hs = [j for j in 1:D if haskey(hot, ax[j]) && m02[ax[j]] - g[j] != g[j]]
                for Sc in (length(hs) == 2 ? [hs, [hs[1]], [hs[2]]] : length(hs) == 1 ? [hs] : Vector{Int}[])
                    memc = [ntuple(j -> j in Sc && b[findfirst(==(j), Sc)] ? m02[ax[j]] - g[j] : g[j], D) for b in Iterators.product(ntuple(_ -> (false, true), length(Sc))...)]
                    if all(in(SPs2), memc)
                        grp2[g] = (vals = [j in Sc ? sort(unique([g[j], m02[ax[j]] - g[j]])) : [g[j]] for j in 1:D], leader = minimum(memc)); break
                    end
                end
            end
        end
        if D == 1 && haskey(hot, ax[1])
            w_ = mod(Float64(θ[1]) / (2 * π), 1.0); m0 = abs(w_ - 0.5) < min(w_, 1 - w_) ? 1 : 0
            for g in SP
                g2 = (m0 - g[1],); (g2 != g && g2 in SP) && (pairOf[g] = g2)
            end
        end
        spBat[] = true
        # the open-axis integrals of every order on the threads (they depend on the order alone); the loop below consumes them in order
        spPre = tMap(SP) do g
            local ν = ntuple(i -> nuP(T, g[i], Pv[i]), D)
            local up = uOrdX(T, g, Pv, fa, fb)
            local p = complex(up, -ysp)
            abs(p) <= 64 * eps(T) * max(abs(x), one(T)) && error("blcFil: Rayleigh-Wood anomaly at rounding distance, order $g")
            local cont = fb < 0 && ysp != 0 && abs(fa) > sqrt(sum(abs2, ν))
            fb < 0 && abs(abs(fa) - sqrt(sum(abs2, ν))) <= 64 * eps(T) * abs(fa) && error("blcFil: Im f < 0 on the cut Re k = |q∥| of order $(g) (J10 continuation undefined there)")
            local ap = angle(p); ysp == 0 && up < 0 && (ap = -T(π)); cont && (ap -= 2 * T(π))
            local own = !(haskey(pairOf, g) || haskey(grp2, g))
            spFil(T, axk, op, xs, p, up, cont, ysp, τ, α, tol, nG, ap, s; panOnly = !own, s0z = (D == 2 && own) ? (fa, fb) : nothing)
        end
        for (gq, g) in enumerate(SP)
            ν = ntuple(i -> nuP(T, g[i], Pv[i]), D)
            up = uOrdX(T, g, Pv, fa, fb)
            p = complex(up, -ysp)
            abs(p) <= 64 * eps(T) * max(abs(x), one(T)) && error("blcFil: Rayleigh-Wood anomaly at rounding distance, order $g")
            cont = fb < 0 && ysp != 0 && abs(fa) > sqrt(sum(abs2, ν))
            fb < 0 && abs(abs(fa) - sqrt(sum(abs2, ν))) <= 64 * eps(T) * abs(fa) && error("blcFil: Im f < 0 on the cut Re k = |q∥| of order $(g) (J10 continuation undefined there)")
            ap = angle(p); ysp == 0 && up < 0 && (ap = -T(π)); cont && (ap -= 2 * T(π))
            # periodic prefactor vectors per derivative order
            PP = [ax[i] == symAx ? [Complex{T}.(ppD(g[i], ax[i], j)) for j in eachindex(xs[ax[i]])] : [poiPre(T, g[i], Pv[i], xq)[1] for xq in xs[ax[i]]] for i in 1:D]
            # every row, not only the hot ones: at any row the pair's two terms may cancel (e.g. an even pattern at x = L/4 for m = ±1, where
            # PP₊ + PP₋ ∝ cos(2πx/L)), and the two orders' u, formed separately near grazing (Σν² - f² with relative error κ_u eps each), then
            # leave 3.9e3 eps (10 eps(1 + κ_up)) on an entry 6.7e-3 of its max (d2 c8 L = 3λ/2 f = 1 θ = (1.26e-5, 0.7), q/c21, c22)
            PPa = (haskey(pairOf, g) || haskey(grp2, g)) ? [[zero.(PP[i][j]) for j in eachindex(PP[i])] for i in 1:D] : PP
            upD = isempty(hot) ? zero(TD) : uOrd(ntuple(i -> nuP(TD, g[i], persD[ax[i]]), D), faD, fbD)
            # open-axis integrals I[pt] over all open offsets (arrays over the open axes)
            Iop, IaR, IaI, pinf = spPre[gq]
            zzD = nothing
            # (up <= 0 only: the closed-part coefficient 4π²Σν² + i y is 4π²f² - ξc² with ξc² = -up; a J10 order with up > 0 (Im f < 0 near
            # grazing) has ξc = 0 and no near-normal cancellation: Double64 fill d2 L = λ f = 1-0.1i H_zz 73% off)
            if D == 2 && PPa === PP && haskey(pinf, :cl) && up <= 0
                # the closed parts of patterns 0 and oo (I_oo,cl = -ξc² I_0,cl exactly): 4π² I_0 + I_oo/f² = I_0 (4π² f² - ξc²)/f² with
                # 4π² f² - ξc² = 4π² Σν² + i y (ξc² = -u = 4π²(fa² - fb² - Σν²)): no cancellation (d2 L_z = 1/512 θ = 1.26e-5 f = 1, AH_yy
                # 1.1e-6 of its max, 8.6e5 eps at κ_up 12)
                o_ = op[1]; k0 = (0, 0, 0); koo = ntuple(i -> i == o_ ? 2 : 0, 3)
                qz = qzGet(o_)
                cl0 = pinf.cl[k0]
                shq_(v, i) = reshape(v, ntuple(k -> k == i ? length(v) : 1, 3))
                cp0 = shq_([PP[1][j][1] for j in eachindex(PP[1])], ax[1]) .* shq_([PP[2][j][1] for j in eachindex(PP[2])], ax[2])
                begin
                    # OODIAG2: the order's patterns 0 and oo leave CH altogether: the periodic-axis diagonals get the whole pattern-0 term
                    # (closed part + remainder) from qz[1:2], the open-axis diagonal the whole combined term (cl0 s0 + rmZ)/f² from qz[3:4]
                    rz, zR, zI, sz = pinf.rmZ; c_ = complex(cr, ci)
                    Wq = 4 * T(π)^2 .* (cl0 .+ pinf.rm[k0])
                    # (the closed part with s = 4π²Σν² + i y from ν directly: at real f the anti-Hermitian zz term is Im(cl_0) s alone, and
                    # 4π²f² - ξc² at the rounded ξc carries the eps|u| rounding of u against a small s (d2 L_z = 1/512 f = 1 θ = 1.26e-5: AH_yy
                    # 2.9e5 eps); the remainder keeps sz, its identity's own coefficient)
                    clz = cl0 .* complex(4 * T(π)^2 * sum(abs2, ν), ysp)
                    Wz = (clz .+ rz) .* c_
                    tAdd!(*, qz[1], cp0, shq_(real.(Wq), o_)); tAdd!(*, qz[2], cp0, shq_(imag.(Wq), o_))
                    tAdd!(*, qz[3], cp0, shq_(real.(Wz), o_)); tAdd!(*, qz[4], cp0, shq_(imag.(Wz), o_))
                    acp = abs.(cp0)
                    aRq = 4 * T(π)^2 .* IaR[k0]; aIq = 4 * T(π)^2 .* IaI[k0]
                    aRz = abs.(real.(clz)) .+ zR; aIz = abs.(imag.(clz)) .+ zI
                    tAdd!(*, qz[6], acp, shq_(aRq, o_)); tAdd!(*, qz[7], acp, shq_(aIq, o_))
                    tAdd!(*, qz[8], acp, shq_(aRz .* abs(cr) .+ aIz .* abs(ci), o_)); tAdd!(*, qz[9], acp, shq_(aRz .* abs(ci) .+ aIz .* abs(cr), o_))
                    zO = zeros(Complex{T}, size(cl0)); zS = zeros(T, size(cl0))
                    IaR2 = copy(IaR); IaI2 = copy(IaI); IaR2[k0] = zS; IaR2[koo] = zS; IaI2[k0] = zS; IaI2[koo] = zS
                    spAdd!(PPa, Iop, IaR2, IaI2, 1, Dict(k0 => zO, koo => zO))
                    # the zz term's own a posteriori λ (its Σ|terms| over its value, per open offset): above LAMZ its closed part and remainder
                    # cancel (far open offsets, the pole form's O(K(ξc)) closed part against a decayed value) and the order's patterns 0 and oo
                    # are recomputed in Double64 after the certification (d2 c32 L = λ/2 θ = 1.26e-5 1±0.1i, AH_zz at 15 cells:
                    # the SP piece lost 780-3450 eps of itself, 1.2-5 eps(1+κ_up) on the entry)
                    # (per part: the H entries take Re Wz, the anti-Hermitian ones Im Wz, each with its own Σ|terms|)
                    λo = maximum(j -> max((aRz[j] * abs(cr) + aIz[j] * abs(ci)) / max(abs(real(Wz[j])), floatmin(T)),
                                          (aRz[j] * abs(ci) + aIz[j] * abs(cr)) / max(abs(imag(Wz[j])), floatmin(T))), eachindex(aRz))
                    if T === Float64 && λo > LAMZ
                        zzD = (Wq = Wq, Wz = Wz, cp0 = cp0, o = o_, sz = (qzS8 = acp .* shq_(aRz .* abs(cr) .+ aIz .* abs(ci), o_), qzS9 = acp .* shq_(aRz .* abs(ci) .+ aIz .* abs(cr), o_)), λo = λo)
                    end
                end
            else
                spAdd!(PPa, Iop, IaR, IaI, 1)
            end
            push!(spRecF, (p = p, up = up, cont = cont, ap = ap, PP = PP, PPa = PPa, Iop = Iop, pinf = pinf, g = g, upD = upD, zzD = zzD))
        end
        spFlush!(); spBat[] = false
        # Σ|terms| of the propagating orders per offset (the e^{Im(k) r} of route R lives there), max over patterns, raw normalization
        info[:MH] = reduce((a, b) -> max.(a, b), values(MHp)) ./ V; info[:MA] = reduce((a, b) -> max.(a, b), values(MAp)) ./ V
        for p in pats; MH[p] .+= MHp[p]; MA[p] .+= MAp[p]; end
    end
    # certification of the real-axis stages per offset (in the fill): every RS u-panel and EV t-panel gets per-offset
    # ATAP 19.3 bounds (panBnd/evBnd/rs0Bnd majorants, per axis a vector over the offsets, so the bound of every offset is a sum of
    # separable products, accumulated by BLAS); every offset must meet tol·max|H(m)| and tol·max|AH(m)| (tol = eps·entT); failing
    # offsets bisect the panels that dominate their bound, the stage is recomputed on the new panels; after 12 rounds the fill raises
    certFil!()
    # ZZD: the flagged d = 2 orders' patterns 0 and oo in Double64 on their certified panels; the difference to the Float64 values
    # already in QZ formed in Double64 and added once (the pole centre is the Float64 one, u = up in both: the same p for closed part and
    # remainder, the consistency spPair2 needs)
    if D == 2 && any(r -> r.zzD !== nothing, spRecF)
        axkDz = [AxK(TD, s[i], s[i]) for i in 1:3]; rDz = T(eps(TD) / eps(T))
        # the order's u, y and 1/f² in Double64 from the exact inputs (the Float64 u carries eps|u| of rounding, which the zz term's cancellation
        # between closed part and remainder amplifies; hot2's upD likewise)
        f2D = complex(faD, fbD)^2; cD = inv(f2D); yD = ysp == 0 ? zero(TD) : 8 * TD(π)^2 * faD * fbD
        persZ = Dict(i => (j = findfirst(==(i), ax); PerAx(axkDz[i], L[j], θ[j] isa Rational ? twT(TD, θ[j]) : twT(TD, TD(θ[j])), θ[j])) for i in ax)
        # the flagged orders' Double64 integrals on the threads, consumed in order below
        zzQ = [q for (q, r) in enumerate(spRecF) if r.zzD !== nothing]
        zzV = tMap(zzQ) do q
            local r = spRecF[q]
            local pl = get(spPlF, q, r.pinf.pl)
            local upq = uOrd(ntuple(i -> nuP(TD, r.g[i], persZ[ax[i]]), D), faD, fbD)
            local scF = sqrt(max(-r.up, zero(T))); local scD = sqrt(max(-upq, zero(TD)))
            local plD = [(TD(scF) + TD(a) - scD, TD(scF) + TD(b) - scD) for (a, b) in pl]
            spFil(TD, axkDz, op, xs, complex(upq, -yD), upq, r.cont, yD, TD(τ), TD(α), TD(tol), nG, TD(r.ap), s; pans = plD)
        end
        zzID = Dict(zip(zzQ, zzV))
        for (q, r) in enumerate(spRecF)
            z = r.zzD; z === nothing && continue
            pl = get(spPlF, q, r.pinf.pl)
            upq = uOrd(ntuple(i -> nuP(TD, r.g[i], persZ[ax[i]]), D), faD, fbD)
            # (panels in δ = ξ - ξc: moved from the Float64 centre to the Double64 one, as hot2's IoD, else the Double64 integral starts at
            # ξc_D - ξc_F ≠ 0)
            scF = sqrt(max(-r.up, zero(T))); scD = sqrt(max(-upq, zero(TD)))
            plD = [(TD(scF) + TD(a) - scD, TD(scF) + TD(b) - scD) for (a, b) in pl]
            ID = zzID[q]
            o_ = z.o; k0 = (0, 0, 0); koo = ntuple(i -> i == o_ ? 2 : 0, 3)
            WqD = 4 * TD(π)^2 .* ID[1][k0]; WzD = WqD .+ ID[1][koo] .* cD
            dq = Complex{T}.(WqD .- Complex{TD}.(z.Wq)); dz = Complex{T}.(WzD .- Complex{TD}.(z.Wz))
            shz(v) = reshape(v, ntuple(k -> k == o_ ? length(v) : 1, 3))
            qz = QZ[1]
            qz[1] .+= z.cp0 .* shz(real.(dq)); qz[2] .+= z.cp0 .* shz(imag.(dq))
            qz[3] .+= z.cp0 .* shz(real.(dz)); qz[4] .+= z.cp0 .* shz(imag.(dz))
            # Σ|terms| of the zz term: the Float64 one replaced by |value| + (eps(D64)/eps(F64)) Σ
            sD = T.(abs.(WzD)) .+ rDz .* (4 * T(π)^2 .* (ID[2][k0] .+ ID[3][k0]) .+ abs(complex(cr, ci)) .* (ID[2][koo] .+ ID[3][koo]))
            qz[8] .+= abs.(z.cp0) .* shz(sD) .- z.sz.qzS8; qz[9] .+= abs.(z.cp0) .* shz(sD) .- z.sz.qzS9
        end
    end
    # the propagating-order pairs on the hot rows (d = 1): (PP + PP')(I + I')/2 + (PP - PP') D with PP ± PP' in Double64 and
    # D = (I - I')/2 from spPair2, on the union of both orders' certified panels
    if pairOf !== nothing && !isempty(pairOf)
        i1 = ax[1]; hj = hot[i1]; done_ = Set{NTuple{D,Int}}()
        gi = Dict(r.g => q for (q, r) in enumerate(spRecF))
        hj = collect(1:nn[i1])
        # every pair's open-axis integrals on the threads (the pairs in pairOf's order), consumed in that order below
        prL = Tuple{NTuple{D,Int},NTuple{D,Int}}[]; let dd = Set{NTuple{D,Int}}()
            for (g, g2) in pairOf; (g in dd || g2 in dd) && continue; push!(dd, g); push!(dd, g2); push!(prL, (g, g2)); end
        end
        prV = tMap(prL) do gg
            local rp = spRecF[gi[gg[1]]]; local rm = spRecF[gi[gg[2]]]
            local plp = get(spPlF, gi[gg[1]], rp.pinf.pl); local plm = get(spPlF, gi[gg[2]], rm.pinf.pl)
            pairInt(T, axk, op, xs, rp, rm, ysp, τ, α, tol, nG, s, plp, plm)
        end
        prD = Dict(zip(prL, prV))
        for (g, g2) in pairOf
            (g in done_ || g2 in done_) && continue
            push!(done_, g); push!(done_, g2)
            rp = spRecF[gi[g]]; rm = spRecF[gi[g2]]
            plp = get(spPlF, gi[g], rp.pinf.pl); plm = get(spPlF, gi[g2], rm.pinf.pl)
            Ip, Im_, Dv, DA = prD[(g, g2)]
            shpO = ntuple(k -> k == op[1] ? nn[op[1]] : k == op[2] ? nn[op[2]] : 1, 3)
            for j in hj
                xq = xs[i1][j]
                qp = ppD(g[1], i1, j); qm = ppD(g2[1], i1, j)
                for pt in pats
                    ko = ntuple(i -> i in op ? pt[i] : 0, 3); k1 = pt[i1] + 1
                    ps = Complex{T}(qp[k1] + qm[k1]); pd = Complex{T}(qp[k1] - qm[k1])
                    Is = (Ip[ko] .+ Im_[ko]) ./ 2; Dk = Dv[ko]
                    wf(Io) = pt == (0, 0, 0) ? 4 * T(π)^2 .* Io : Io .* complex(cr, ci)
                    Ws = wf(Is); Wd = wf(Dk)
                    sl = ntuple(k -> k == i1 ? (j:j) : Colon(), 3)
                    CH[pt][sl...] .+= reshape(ps .* real.(Ws) .+ pd .* real.(Wd), shpO)
                    CA[pt][sl...] .+= reshape(ps .* imag.(Ws) .+ pd .* imag.(Wd), shpO)
                    aw = pt == (0, 0, 0) ? 4 * T(π)^2 : abs(complex(cr, ci))
                    sA = aw .* (abs(ps) .* (abs.(Ip[ko]) .+ abs.(Im_[ko])) ./ 2 .+ abs(pd) .* DA[ko])
                    MH[pt][sl...] .+= reshape(sA, shpO); MA[pt][sl...] .+= reshape(sA, shpO)
                end
            end
        end
    end
    if D == 2 && route != :C && !isempty(hot) && !isempty(spRecF)
        # d = 2 with a hot axis: the SP orders grouped by the reflections g_j -> m0_j - g_j along the hot axes (the whole product set in SP, as
        # cFil!'s groups). Per group the members' open integrals I_g in Double64 (spFil, d_o = 1: cheap), and the sum/difference components
        # Σ_g PP1_g1 PP2_g2 I_g = Σ_c P1_c P2_c W_c: P_c the per-axis sums and differences of the reflected prefactors, W_c the matching
        # (±)-averages of the I_g, each formed in Double64 and rounded once; the components are rank-1 over the periodic axes and are added on
        # every row in Float64 (their products do not cancel)
        i1, i2 = ax; o = op[1]
        axkDo = [AxK(TD, s[i], s[i]) for i in 1:3]
        m0 = Dict(i => (w_ = mod(Float64(θ[j]) / (2 * π), 1.0); abs(w_ - 0.5) < min(w_, 1 - w_) ? 1 : 0) for (j, i) in enumerate(ax) if haskey(hot, i))
        SPs = Set(r.g for r in spRecF); gi = Dict(r.g => q for (q, r) in enumerate(spRecF))
        IoDc = Dict{Int,Any}()
        IoD(q) = get!(() -> IoD0(q), IoDc, q)
        function IoD0(q)
            begin
                local r = spRecF[q]
                local pl = get(spPlF, q, r.pinf.pl)
                # pole-form panels are δ around the order's own centre: moved from the Float64 centre to the Double64 one (δ_D = c_F + δ_F - c_D,
                # exact), else the integral would start at c_D - c_F ≠ 0 (a 1e-16 relative error per order, 1e6 eps on the hot entries); d_o = 1:
                # the δ of the pole form is ξ - ξc, ξc = √sc
                local scF = sqrt(max(-r.up, zero(T)))
                local scD = sqrt(max(-TD(r.upD), zero(TD)))
                local plD = r.pinf.kind == :pole ? [(TD(scF) + TD(a) - scD, TD(scF) + TD(b) - scD) for (a, b) in pl] : [(TD(a), TD(b)) for (a, b) in pl]
                spFil(TD, axkDo, op, xs, complex(TD(r.upD), -TD(ysp)), TD(r.upD), r.cont, TD(ysp), TD(τ), TD(α), TD(tol), nG, TD(r.ap), s; pans = plD)[1]
            end
        end
        rD = T(eps(TD) / eps(T)); aw(pt) = pt == (0, 0, 0) ? 4 * T(π)^2 : abs(complex(cr, ci))
        shq(v, i) = reshape(v, ntuple(k -> k == i ? length(v) : 1, 3))
        # the grouped orders' Double64 integrals on the threads first (IoD then reads them)
        let qs = unique([gi[(v1, v2)] for r in spRecF if haskey(grp2, r.g) for v1 in grp2[r.g].vals[1] for v2 in grp2[r.g].vals[2]])
            for (q, v) in zip(qs, tMap(IoD0, qs)); IoDc[q] = v; end
        end
        done2 = Set{NTuple{D,Int}}()
        for r in spRecF
            g = r.g; (g in done2 || !haskey(grp2, g)) && continue
            vals = grp2[g].vals
            for v1 in vals[1], v2 in vals[2]; push!(done2, (v1, v2)); end
            Pc = map(1:D) do j
                pv = [[ppD(v, ax[j], jr) for jr in eachindex(xs[ax[j]])] for v in vals[j]]
                length(vals[j]) == 1 ? [pv[1]] : [[pv[1][q] .+ pv[2][q] for q in eachindex(pv[1])], [pv[1][q] .- pv[2][q] for q in eachindex(pv[1])]]
            end
            IDs = Dict((v1, v2) => IoD(gi[(v1, v2)]) for v1 in vals[1], v2 in vals[2])
            k1s = collect(keys(IDs[(vals[1][1], vals[2][1])]))
            sumAbs = Dict(k => sum(abs.(Complex{T}.(IDs[mm][k])) for mm in keys(IDs)) for k in k1s)
            nrm = length(vals[1]) * length(vals[2])
            for c1 in eachindex(Pc[1]), c2 in eachindex(Pc[2])
                W = Dict(k => zeros(Complex{TD}, nn[o]) for k in k1s)
                for (q1, v1) in enumerate(vals[1]), (q2, v2) in enumerate(vals[2])
                    sg = (c1 == 2 && q1 == 2 ? -1 : 1) * (c2 == 2 && q2 == 2 ? -1 : 1)
                    for k in k1s; W[k] .+= sg .* IDs[(v1, v2)][k]; end
                end
                koo2 = ntuple(i -> i == o ? 2 : 0, 3)
                for pt in pats
                    ko = ntuple(i -> i in op ? pt[i] : 0, 3)
                    # OODIAG2: the open-axis diagonal of the group formed in Double64 (4π² W_0 + W_oo/f², rounded once) into QZ, pattern 0
                    # into QZ's periodic-axis diagonals; neither enters CH (their separate roundings are O(max) and cancel on the zz entry)
                    oz = pt == (0, 0, 0) || pt == koo2
                    oz && pt == koo2 && continue
                    Wc = W[ko] ./ nrm
                    W0 = Complex{T}.(pt == (0, 0, 0) ? 4 * TD(π)^2 .* Wc : Wc .* complex(TD(cr), TD(ci)))
                    p1 = [Complex{T}(v[pt[i1] + 1]) for v in Pc[1][c1]]; p2 = [Complex{T}(v[pt[i2] + 1]) for v in Pc[2][c2]]
                    cper = shq(p1, i1) .* shq(p2, i2)
                    sa = abs.(W0) .+ (aw(pt) * rD / nrm) .* sumAbs[ko]
                    if oz
                        qz = qzGet(o)
                        Wz = Complex{T}.(4 * TD(π)^2 .* Wc .+ (W[koo2] ./ nrm) .* complex(TD(cr), TD(ci)))
                        saz = abs.(Wz) .+ (rD / nrm) .* (4 * T(π)^2 .* sumAbs[ko] .+ abs(complex(cr, ci)) .* sumAbs[koo2])
                        tAdd!(*, qz[1], cper, shq(real.(W0), o)); tAdd!(*, qz[2], cper, shq(imag.(W0), o))
                        tAdd!(*, qz[3], cper, shq(real.(Wz), o)); tAdd!(*, qz[4], cper, shq(imag.(Wz), o))
                        tAdd!((c_, w_) -> abs(c_) * w_, qz[6], cper, shq(sa, o)); tAdd!((c_, w_) -> abs(c_) * w_, qz[7], cper, shq(sa, o))
                        tAdd!((c_, w_) -> abs(c_) * w_, qz[8], cper, shq(saz, o)); tAdd!((c_, w_) -> abs(c_) * w_, qz[9], cper, shq(saz, o))
                        continue
                    end
                    tAdd!(*, CH[pt], cper, shq(real.(W0), o)); tAdd!(*, CA[pt], cper, shq(imag.(W0), o))
                    tAdd!((c_, w_) -> abs(c_) * w_, MH[pt], cper, shq(sa, o)); tAdd!((c_, w_) -> abs(c_) * w_, MA[pt], cper, shq(sa, o))
                end
            end
        end
    end
    for p in pats; fold!(CH[p]); fold!(CA[p]); end
    # reflection zeros (symZ) per axis and offset
    H = zeros(Complex{T}, 3, 3, nn...); A = zeros(Complex{T}, 3, 3, nn...)
    # λ_e per unique entry (ents order) and offset, Float32 (0 at reflection zeros and where the entry is 0)
    λH = zeros(Float32, 6, nn...); λA = zeros(Float32, 6, nn...)
    # the entry's arrays formed in place in four buffers (same operations in the same order)
    p0_ = patOf(ents[1]...); Hab = similar(CH[p0_]); Aab = similar(CA[p0_]); SHab = similar(MH[p0_]); SAab = similar(MA[p0_])
    # every twist exact 0 or 1/2: -θ ≡ θ, so H = [G + conj G]/2 and AH are real entry by entry (as the reflection zeros, an exact symmetry); their
    # imaginary parts are rounding of the route (route C's complex path: 1e-14 relative at c8 θ = 0 1+0.1i)
    rlP = all(i -> prTw(pers[i]), ax)
    for (q, (a, b)) in enumerate(ents)
        p = patOf(a, b)
        Hab .= CH[p] ./ V; Aab .= CA[p] ./ V; SHab .= MH[p] ./ V; SAab .= MA[p] ./ V
        if a == b
            Hab .= Hab .+ CH[(0, 0, 0)] ./ V; Aab .= Aab .+ CA[(0, 0, 0)] ./ V; SHab .= SHab .+ MH[(0, 0, 0)] ./ V; SAab .= SAab .+ MA[(0, 0, 0)] ./ V
            if QZ[1] !== nothing
                qz = QZ[1]; Hab .= Hab .+ (a == qz[5] ? qz[3] : qz[1]) ./ V; Aab .= Aab .+ (a == qz[5] ? qz[4] : qz[2]) ./ V
                SHab .= SHab .+ (a == qz[5] ? qz[8] : qz[6]) ./ V; SAab .= SAab .+ (a == qz[5] ? qz[9] : qz[7]) ./ V
            end
        end
        rlP && (Hab .= real.(Hab); Aab .= real.(Aab))
        finEnt!(λH, λA, Hab, Aab, SHab, SAab, zKl, a, b, q)
        H[a, b, :, :, :] .= Hab; A[a, b, :, :, :] .= Aab
        H[b, a, :, :, :] .= Hab; A[b, a, :, :, :] .= Aab
    end
    zRw === nothing || (for I in CartesianIndices(zAll); zAll[I] && (H[:, :, I] .= 0; A[:, :, I] .= 0; λH[:, I] .= 0; λA[:, I] .= 0); end)
    info[:ms] = ms; info[:λe] = (λH, λA)
    return H, A, info
end

# per-order flops of the d_o = 2 pole form beyond which the fill refuses, as filBlk's a priori estimate counts them (M(√S) on every panel, an
# upper bound of the per-panel rule): the largest d = 1 block of the coverage, c4 (16,64,64) at a 4λ period, f = 1, estimates 1.32e12 for its
# costliest order; a cap of 1e11 refused far smaller blocks; open extents of 32λ estimate ~2e13
const FILMAX = 4.0e12
# pole-form flops of one whole fill (every spFil call: first passes, certification re-evaluations, Double64 reruns) beyond which it refuses:
# an out-of-coverage d = 1 block passes every per-order FILMAX check and then spends its time in the re-evaluations;
# the largest d = 1 blocks of the coverage need <= 5.1e13 (c4 (16,64,64), 4λ period, 1+1e-3i: 976 s at 8 threads)
const FILTOT = 1.0e14
const FILACC = ScopedValue{Union{Nothing,Threads.Atomic{Float64}}}(nothing)
# route C of the mixed route on its own task concurrently with route R (the threaded build sets it; the sources run sequentially)
const PARRC = true
const LAMZ = 16
# one ray node of an SP order into the open-offset arrays (function barrier, as remAcc!)
function rayAcc!(O, OR, ep, ks, pt, nO)
    if length(nO) == 2
        @inbounds for j2 in 1:nO[2], j1 in 1:nO[1]
            v = ep * ks[1][j1][pt[1] + 1] * ks[2][j2][pt[2] + 1]; O[j1, j2] += v; OR[j1, j2] += abs(v)
        end
    else
        @inbounds for j1 in 1:nO[1]
            v = ep * ks[1][j1][pt[1] + 1]; O[j1] += v; OR[j1] += abs(v)
        end
    end
    nothing
end
# one node of the pole-form remainder into the compensated sums (a function barrier: the Dict values are untyped in spFil)
function remAcc!(S_, C_, R_, I_, Ak, Ck, wq, rw, iw)
    @inbounds for i in eachindex(S_)
        d = (Ak[i] - Ck[i]) * wq; dr = d * rw; di = d * iw
        R_[i] += abs(dr); I_[i] += abs(di)
        S_[i], C_[i] = kaddC(S_[i], C_[i], complex(dr, di))
    end
    nothing
end
# open-axis integrals of one SP order for every open offset: Dict(pattern over the open axes => array over the open offsets),
# the same formulas as blcTns's spcSP (spectral pole route) or its evanescent steepest-descent ray
function spFil(::Type{T}, axk, op, xs, p, up, cont, y, τ, α, tol, nGt, ap, s; pans = nothing, Mx::Int = 0, panOnly::Bool = false, s0z = nothing) where {T}
    xt, wt = glq(T, nGt)
    xsO = [toT.(T, xs[o]) for o in op]; nO = length.(xsO)
    # the open offsets in cells and the edges once, not one Rational{BigInt} division per node and offset
    ksI_ = [[Int(xq / s[o]) for xq in xs[o]] for o in op]; sTo_ = [toT(T, s[o]) for o in op]
    pats = length(op) == 2 ? [(0, 0), (1, 0), (0, 1), (2, 0), (0, 2), (1, 1)] : [(0,), (1,), (2,)]
    key(pt) = length(op) == 2 ? (ntuple(i -> i == op[1] ? pt[1] : i == op[2] ? pt[2] : 0, 3)) : ntuple(i -> i == op[1] ? pt[1] : 0, 3)
    out = Dict(key(pt) => zeros(Complex{T}, nO...) for pt in pats)
    # Σ|terms| of the Hermitian / anti-Hermitian parts per pattern and open offset (a posteriori route selection, as blcTns's Ia)
    oR = Dict(key(pt) => zeros(T, nO...) for pt in pats); oI = Dict(key(pt) => zeros(T, nO...) for pt in pats)
    ρo = sqrt(sum(maximum(xq -> max(abs(xq) - toT(T, axk[o].cp), zero(T))^2, xsO[k]) for (k, o) in enumerate(op)))
    X2 = sum(k -> (maximum(abs, xsO[k]) + toT(T, axk[op[k]].cp))^2, eachindex(op))
    useSpc = cont || (up < 0 && exp(real(sqrt(p)) * ρo) <= 4)
    if !useSpc
        φ = -ap
        if abs(φ) > T(π) / 2
            cψ = 1 / (1 + 8 * τ * T(1 // 2) / max(X2, eps(T)))
            φ = sign(φ) * min(abs(φ), T(π) / 2 + min(acos(cψ), T(13) / 10))
        end
        rate = abs(p) * cos(ap + φ); e = cis(φ)
        rate > 0 || error("blcFil: order with no decaying ray, p = $p")
        smax = (α + 12 + ρo^2 / (4 * τ)) / rate
        pl = Tuple{T,T}[]
        s0 = zero(T); s1 = τ
        while s0 < smax
            s1 = min(s1, smax)
            subS = shpSub(τ + s0, τ + s1, t -> rate * t + ρo^2 / (4 * t), T(4)) .- τ
            for sb in 1:length(subS)-1; push!(pl, (subS[sb], subS[sb + 1])); end
            s0 = s1; s1 = 2 * s1
        end
        pans === nothing || (pl = pans)
        # panOnly: the panels alone (an order whose sum is formed elsewhere, pairs/groups; only its certification needs them)
        panOnly && return out, oR, oI, (kind = :ray, pl = pl, e = e)
        begin
            # the nodes' open factors on the threads, 256 nodes at a time; accumulated in node order
            ndR = [(lo, hi, j) for (lo, hi) in pl for j in 1:nGt]
            for c0 in 1:256:length(ndR)
                nv = tMap(ndR[c0:min(c0 + 255, end)]) do (lo, hi, j)
                    local sn = (hi - lo) / 2 * xt[j] + (hi + lo) / 2; local wq = (hi - lo) / 2 * wt[j]
                    local t = τ + sn * e
                    local ks = [[kapL(axk[o], sTo_[k_], ksI_[k_][j_], t, tol) for j_ in eachindex(xs[o])] for (k_, o) in enumerate(op)]
                    (ks, exp(-p * t) * e * wq)
                end
                for (ks, ep) in nv, pt in pats
                    rayAcc!(out[key(pt)], oR[key(pt)], ep, ks, pt, nO)
                end
            end
        end
        for k in keys(oR); oI[k] .= oR[k]; end
        return out, oR, oI, (kind = :ray, pl = pl, e = e)
    end
    sr = -up; sc = max(sr, zero(T)); gap = sc - sr
    S = sc + (α + 12) / τ
    cy = cos(y * τ); sy = sin(y * τ)
    wRI(v) = (local e = exp(-v * τ) / (v^2 + y^2); (e * (v * cy - y * sy), e * (v * sy + y * cy)))
    if length(op) == 2
        # angular nodes per panel (spcSP): circM at the panel's largest radius, one rule M_in = M(√(2 s_c)) for s <= 2 s_c and
        # the closed part, M(2√S) for the continuation only; Mx (spPair2's shared rule) forces one M everywhere
        Mr(r_) = max(Mx, maximum(circM([axk[o] for o in op], [maximum(abs, xsO[k]) for k in 1:2], [pt[1], pt[2]], r_, tol) for pt in pats))
        M2 = Mr(sqrt(S) * 2); Min = Mx > 0 ? M2 : Mr(sqrt(2 * sc))
        csT = Dict{Int,Tuple{Vector{T},Vector{T}}}()
        cs_(M) = get!(() -> ([cospi(2 * T(j) / M) for j in 0:M-1], [sinpi(2 * T(j) / M) for j in 0:M-1]), csT, M)
        # (the wide tables: BigFloat ones at the precision of the scope they are first built in, wideDo's, the same for every node of the call)
        csTW = Dict{Int,Tuple{Vector{wideT(T)},Vector{wideT(T)}}}()
        csW_(M) = get!(() -> csWide(wideT(T), M), csTW, M)
        abT = [(toT(T, axk[op[k]].a), toT(T, axk[op[k]].b)) for k in 1:2]
        # the angular factors of axis k at radius r for every offset and node, the trapezoid mass and trig once per (node, offset): pFac3's numbers
        # (every name assigned in these closures is local: spFil assigns r0, cM, sM, P1, P2 itself)
        # (the wide phase of blcTns's spcSP per offset where r_g|x_q| > PHX, r_g = √(2 s_c) on s <= 2 s_c; rW the node's radius in wideT(T))
        function qAx(k, r, cv, M, rW, cvW, rg)
            local A_ = axk[op[k]]; local xk = xsO[k]; local Q = [zeros(typeof(r), length(xk), M) for _ in 1:3]
            local wq = [rW !== nothing && rg * abs(xq) > PHX for xq in xk]
            for j in 1:M
                local ξ = r * cv[j]; local kh = A_.w0 * sncH(ξ, abT[k][1]) * sncH(ξ, abT[k][2])
                for (q, xq) in enumerate(xk)
                    if wq[q]
                        local P = rW isa Tuple ? pFac3W(A_, mul2(rW[1], rW[2], HI(cvW[j]), LO(cvW[j]))..., xq, abT[k][1], abT[k][2]) : pFac3W(A_, rW * cvW[j], xq, abT[k][1], abT[k][2])
                        Q[1][q, j] = P[1]; Q[2][q, j] = P[2]; Q[3][q, j] = P[3]
                    else
                        local c, sn = csn(ξ * xq)
                        Q[1][q, j] = kh * c; Q[2][q, j] = -kh * ξ * sn; Q[3][q, j] = -kh * ξ^2 * c
                    end
                end
            end
            Q
        end
        # the node s0 + δ0 as two numbers (the wide radius from both)
        function aV(s0, δ0, M)
            local sv = s0 + δ0; local r = sqrt(sv)
            local cM, sM = cs_(M)
            local rg = sv isa Real ? (sv <= 2 * sc ? sqrt(2 * sc) : r) : abs(r)
            local wd = WPH[] && rg * max(maximum(abs, xsO[1]), maximum(abs, xsO[2])) > PHX
            # the continuation a(-p) at complex radius: factors and products in W (blcTns's spcSP)
            if wd && !(sv isa Real)
                return wideDo(T, s0) do
                    local rW = sqrt(toW(T, s0) + toW(T, δ0)); local cW, sW = csW_(M)
                    local Q1 = [pFac3(axk[op[1]], rW * cW[j], xq, abT[1][1], abT[1][2]) for xq in xsO[1], j in 1:M]
                    local Q2 = [pFac3(axk[op[2]], rW * sW[j], xq, abT[2][1], abT[2][2]) for j in 1:M, xq in xsO[2]]
                    Dict(key(pt) => Complex{T}.((getindex.(Q1, pt[1] + 1) * getindex.(Q2, pt[2] + 1)) ./ (4 * real(typeof(rW))(π) * M)) for pt in pats)
                end
            end
            local P1, P2 = if wd
                wideDo(T, s0) do
                    local rW = T === Float64 && sv isa Real ? sqrt2(s0, δ0) : sqrt(toW(T, s0) + toW(T, δ0)); local cW, sW = csW_(M)
                    (qAx(1, r, cM, M, rW, cW, rg), qAx(2, r, sM, M, rW, sW, rg))
                end
            else
                (qAx(1, r, cM, M, nothing, nothing, rg), qAx(2, r, sM, M, nothing, nothing, rg))
            end
            P2 = [permutedims(Q) for Q in P2]
            Dict(key(pt) => (P1[pt[1] + 1] * P2[pt[2] + 1]) ./ (4 * T(π) * M) for pt in pats)
        end
        # the closed part's angular average summed in Double64 (the M products per offset pair summed in Float64 by BLAS cost 10-27 eps on
        # anti-Hermitian entries at real f, whose only term this is); the remainder keeps aV
        ac = if T === Float64
            r0 = sqrt(sc); cM, sM = cs_(Min); rg0 = sqrt(2 * sc)
            wd0 = WPH[] && rg0 * max(maximum(abs, xsO[1]), maximum(abs, xsO[2])) > PHX
            rW0 = wd0 ? sqrt2(sc, zero(T)) : nothing; cW0, sW0 = wd0 ? csW_(Min) : (nothing, nothing)
            P1 = [Double64.(Q) for Q in qAx(1, r0, cM, Min, rW0, cW0, rg0)]; P2 = [Double64.(permutedims(Q)) for Q in qAx(2, r0, sM, Min, rW0, sW0, rg0)]
            Dict(key(pt) => T.((P1[pt[1] + 1] * P2[pt[2] + 1]) ./ (4 * Double64(π) * Min)) for pt in pats)
        else
            aV(sc, zero(T), Min)
        end
        win2 = anmWin(up, p, τ)
        E1 = win2 ? e1Win(p, y, τ) : (y == 0 && up < 0) ? complex(real(expintG(1, complex(up * τ, zero(T)))), T(π)) : expintG(1, p * τ)
        for k in keys(out); out[k] .= ac[k] .* E1; end
        e1R = win2 ? T(0.58) + abs(log(abs(p) * τ)) + 2 * abs(p) * τ : abs(E1); e1I = win2 ? T(π) + 2 * abs(p) * τ : abs(E1)
        for k in keys(out); oR[k] .= abs.(ac[k]) .* e1R; oI[k] .= abs.(ac[k]) .* e1I; end
        w = y != 0 ? abs(y) / 4 : max(sc, 1 / τ) / 8
        pts = oscSplit(grdPts(-sc, S - sc, zero(T), w), sum(k -> maximum(abs, xsO[k]) + toT(T, axk[op[k]].cp), eachindex(op)), true, sc)
        pl = pans === nothing ? [(pts[q], pts[q + 1]) for q in 1:length(pts)-1] : pans
        Mpl = [T(hi_) <= sc ? Min : max(Min, Mr(sqrt(sc + T(hi_)))) for (lo_, hi_) in pl]
        # (the angular tables of every M filled here, before any node: aV then only reads csT, also from several threads)
        foreach(cs_, unique(Mpl)); cs_(M2)
        wideDo(T, sc) do; foreach(csW_, unique([Mpl; Min; M2])); end
        # a priori flops of this order's remainder (angular factors and their products over the open-offset grid per node); the fill has no
        # other budget, and at open extents of 32λ one order needed ~1e11 (> 25 min); 64³ in coverage needs <= ~2e10
        fl_ = nGt * sum(Mpl; init = 0) * (3 * (nO[1] + nO[2]) + 12 * nO[1] * nO[2])
        fl_ > FILMAX && error("blcFil: a priori cost of a propagating order's pole form, $(round(Float64(fl_); sigdigits = 3)) flops (M up to $(maximum(Mpl; init = Min)) angular nodes, $(length(pl)) panels, open extent $(maximum(abs, xsO[1]))λ x $(maximum(abs, xsO[2]))λ), exceeds $(FILMAX); refused")
        panOnly && return out, oR, oI, (kind = :pole, pl = pl, e = one(Complex{T}))
        (fa_ = FILACC[]) === nothing || (ft_ = Threads.atomic_add!(fa_, Float64(fl_)) + Float64(fl_)) <= FILTOT ||
            error("blcFil: the propagating orders' pole forms (first passes and certification re-evaluations) need more than $(FILTOT) flops ($(round(ft_; sigdigits = 3)) so far, open extent $(maximum(abs, xsO[1]))λ x $(maximum(abs, xsO[2]))λ); refused")
        # the remainder summed on its own with Neumaier compensation, then added to the closed-form part (as blcTns)
        rs_ = Dict(k => zeros(Complex{T}, size(out[k])) for k in keys(out)); rc_ = Dict(k => zeros(Complex{T}, size(out[k])) for k in keys(out))
        # the nodes' angular averages on the threads, 64 nodes at a time; accumulated in node order (pl untyped: typed panel ends)
        ndP = [(T(lo_)::T, T(hi_)::T, j, Mpl[q_]) for (q_, (lo_, hi_)) in enumerate(pl) for j in 1:nGt]
        for c0 in 1:64:length(ndP)
            nv = tMap(ndP[c0:min(c0 + 63, end)]) do (lo, hi, j, Mq)
                local δ = (hi - lo) / 2 * xt[j] + (hi + lo) / 2; local wq = (hi - lo) / 2 * wt[j]
                local rw, iw = wRI(δ + gap)
                (aV(sc, δ, Mq), wq, T(rw)::T, (y == 0 ? zero(T) : T(iw))::T)
            end
            for (av, wq, rw, iw) in nv, k in keys(out)
                remAcc!(rs_[k], rc_[k], oR[k], oI[k], av[k], ac[k], wq, rw, iw)
            end
        end
        for k in keys(out); out[k] .+= rs_[k] .+ rc_[k]; end
        if cont
            a0 = aV(-p, zero(T), M2)
            for k in keys(out); out[k] .+= 2 * T(π) * im .* a0[k]; oR[k] .+= 2 * T(π) .* abs.(a0[k]); oI[k] .+= 2 * T(π) .* abs.(a0[k]); end
        end
    else
        o = op[1]; A = axk[o]
        ξc = sqrt(sc); Ξ = sqrt(S)
        win = anmWin(up, p, τ)
        if win
            C1, bw = c1Win(p, τ)
        else
            sp = (y == 0 && up < 0) ? complex(zero(T), -sqrt(sr)) : sqrt(p)
            C1 = T(π) / (2 * sp) * erfcG(sp * sqrt(τ))
        end
        Pc = Dict(key(pt) => [pFac(A, ξc, xq, pt[1]) for xq in xsO[1]] for pt in pats)
        for k in keys(out); out[k] .= Pc[k] .* C1; end
        if win
            Ew = erfiQ(-p * τ); ab2 = abs2(bw)
            c1R = T(π) * abs(imag(bw)) / (2 * ab2) + T(π) * sqrt(τ) / 2 * abs(real(Ew)); c1I = T(π) * abs(real(bw)) / (2 * ab2) + T(π) * sqrt(τ) / 2 * abs(imag(Ew))
        else
            c1R = abs(C1); c1I = abs(C1)
        end
        for k in keys(out); oR[k] .= abs.(Pc[k]) .* c1R; oI[k] .= abs.(Pc[k]) .* c1I; end
        w = y != 0 ? abs(y) / (4 * max(ξc, sqrt(abs(y)))) : max(ξc, 1 / sqrt(τ)) / 8
        pts = oscSplit(grdPts(-ξc, Ξ - ξc, zero(T), w), maximum(abs, xsO[1]) + toT(T, axk[op[1]].cp), false, zero(T))
        pl = pans === nothing ? [(pts[q], pts[q + 1]) for q in 1:length(pts)-1] : pans
        panOnly && return out, oR, oI, (kind = :pole, pl = pl, e = one(Complex{T}))
        rs_ = Dict(k => zeros(Complex{T}, size(out[k])) for k in keys(out)); rc_ = Dict(k => zeros(Complex{T}, size(out[k])) for k in keys(out))
        z0_ = zeros(T, length(xsO[1])); aT_ = toT(T, A.a); bT_ = toT(T, A.b)
        # s0z = (fa, fb): the open-axis diagonal's remainder 4π² rm_0 + rm_oo/f² formed as one integral, ∫ [s D_0(δ) - δ(2ξc + δ) pF_0(ξc + δ)]
        # w dδ / f² with s = 4π² f² - ξc² at the remainder's own (rounded) subtraction point ξc (pF_2 = -ξ² pF_0 exactly; Re s in extended
        # precision, it is 4π²Σν² up to rounding and cancels), whose terms do not cancel; the separate remainders are O(max) and cancel to it
        # (d2 c32 L = λ/2 θ = 1.26e-5 1+0.1i, AH_zz at 15 cells, 3.6e-6 of its max, 5.25e5 eps = 4.6 eps(1+κ_up))
        # (Im f < 0, the J10 continuation term too: its zz combination (4π² f² - b²) pF_0(b)/f², b² = -p, is formed the same way below)
        zq = (s0z === nothing || gap != 0) ? nothing : (zeros(Complex{T}, nO[1]), zeros(Complex{T}, nO[1]), zeros(T, nO[1]), zeros(T, nO[1]))
        TE = T === Float64 ? Double64 : T
        F2E = zq === nothing ? nothing : 4 * TE(π)^2 * complex(TE(s0z[1]), TE(s0z[2]))^2
        zq === nothing || (s0z = complex(T(real(F2E) - TE(ξc)^2), y))
        Pc0_ = Pc[key((0,))]
        # the nodes' differences on the threads, 256 nodes at a time; accumulated in node order (typed panel ends; the three derivative
        # orders of the difference at once)
        ndP = [(T(lo_)::T, T(hi_)::T, j) for (lo_, hi_) in pl for j in 1:nGt]
        for c0 in 1:256:length(ndP)
          nv = tMap(ndP[c0:min(c0 + 255, end)]) do (lo, hi, j)
              local δ = (hi - lo) / 2 * xt[j] + (hi + lo) / 2; local wq = (hi - lo) / 2 * wt[j]
              local rw, iw = wRI(δ * (2 * ξc + δ) + gap)
              y == 0 && (iw = zero(T))
              (δ, wq, rw, iw, [pDif3(A, ξc, δ, xq, aT_, bT_) for xq in xsO[1]])
          end
          for (δ, wq, rw, iw, dq) in nv
            for pt in pats
                k = key(pt)
                remAcc!(rs_[k], rc_[k], oR[k], oI[k], getindex.(dq, pt[1] + 1), z0_, wq, rw, iw)
            end
            if zq !== nothing
                vz = δ * (2 * ξc + δ); wz = complex(rw, iw) * wq
                @inbounds for i in eachindex(dq)
                    tz = (s0z * dq[i][1] - vz * (Pc0_[i] + dq[i][1])) * wz
                    zq[3][i] += abs(real(tz)); zq[4][i] += abs(imag(tz))
                    zq[1][i], zq[2][i] = kaddC(zq[1][i], zq[2][i], tz)
                end
            end
          end
        end
        # closed part and remainder (+ continuation term) of each pattern kept apart (the open-axis diagonal f²·I_0 + I_oo of a
        # near-normal order cancels between the two closed parts, whose sum is formed exactly by the caller)
        clq = Dict(k => out[k] ./ T(π) for k in keys(out)); rmq = Dict(k => (rs_[k] .+ rc_[k]) ./ T(π) for k in keys(out))
        for k in keys(out); out[k] .+= rs_[k] .+ rc_[k]; out[k] ./= T(π); oR[k] ./= T(π); oI[k] ./= T(π); end
        rZ = zq === nothing ? nothing : (zq[1] .+ zq[2]) ./ T(π); zR = zq === nothing ? nothing : zq[3] ./ T(π); zI = zq === nothing ? nothing : zq[4] ./ T(π)
        if cont && win
            δb = (-p - sc) / (bw + ξc)
            for pt in pats
                v = im .* [pDif(A, complex(ξc), δb, xq, pt[1]) for xq in xsO[1]] ./ bw; out[key(pt)] .+= v; oR[key(pt)] .+= abs.(v); oI[key(pt)] .+= abs.(v)
                rmq[key(pt)] .+= v
            end
            if zq !== nothing
                # 4π² [pF_0(b) - pF_0(ξc)] + [pF_2(b) - pF_2(ξc)]/f² = [(4π²f² - b²) D_0b - δb(2ξc + δb) pF_0(ξc)]/f², b = ξc + δb
                sb = Complex{T}(F2E - Complex{TE}(bw)^2)
                vz = im .* [sb * pDif(A, complex(ξc), δb, xq, 0) - δb * (2 * ξc + δb) * pFac(A, ξc, xq, 0) for xq in xsO[1]] ./ bw
                rZ .+= vz; zR .+= abs.(vz); zI .+= abs.(vz)
            end
        elseif cont
            s0 = sqrt(-p)
            for pt in pats
                v = im .* [pFac(A, s0, xq, pt[1]) for xq in xsO[1]] ./ s0; out[key(pt)] .+= v; oR[key(pt)] .+= abs.(v); oI[key(pt)] .+= abs.(v)
                rmq[key(pt)] .+= v
            end
            if zq !== nothing
                # 4π² pF_0(s0) + pF_2(s0)/f² = (4π²f² - s0²) pF_0(s0)/f², 4π²f² - s0² = 4π²f² + p up to rounding (≈ 4π²Σν²)
                sb = Complex{T}(F2E - Complex{TE}(s0)^2)
                vz = im .* [sb * pFac(A, s0, xq, 0) for xq in xsO[1]] ./ s0
                rZ .+= vz; zR .+= abs.(vz); zI .+= abs.(vz)
            end
        end
        zq === nothing || return out, oR, oI, (kind = :pole, pl = pl, e = one(Complex{T}), cl = clq, rm = rmq, rmZ = (rZ, zR, zI, s0z))
        return out, oR, oI, (kind = :pole, pl = pl, e = one(Complex{T}), cl = clq, rm = rmq)
    end
    out, oR, oI, (kind = :pole, pl = pl, e = one(Complex{T}))
end

# a propagating-order pair on the steepest-descent ray of the + order (both evanescent-side or far-open orders take the ray form): I(p₊), I(p₋)
# and D = [I(p₊) - I(p₋)]/2 on the same nodes, the weight (e^{-p₊t} - e^{-p₋t})/2 formed in Double64 from the Double64 u± (the ray
# analogue of spPair2: d1 c32 L = λ, 1+0.1i, θ = 1.26e-5, the pair (1, -1) at far open offsets, 2e6 eps on the hot entries before, q/c28)
function spPairRay(::Type{T}, axk, op, xs, rp, rm, y::T, τ::T, tol::T, nGt::Int, s, e::Complex{T}, pl) where {T}
    TD = T === Float64 ? Double64 : T
    xt, wt = glq(T, nGt)
    xsO = [toT.(T, xs[o]) for o in op]; nO = length.(xsO)
    # the open offsets in cells and the edges once, not one Rational{BigInt} division per node and offset
    ksI_ = [[Int(xq / s[o]) for xq in xs[o]] for o in op]; sTo_ = [toT(T, s[o]) for o in op]
    pats = length(op) == 2 ? [(0, 0), (1, 0), (0, 1), (2, 0), (0, 2), (1, 1)] : [(0,), (1,), (2,)]
    key(pt) = length(op) == 2 ? (ntuple(i -> i == op[1] ? pt[1] : i == op[2] ? pt[2] : 0, 3)) : ntuple(i -> i == op[1] ? pt[1] : 0, 3)
    Ip = Dict(key(pt) => zeros(Complex{T}, nO...) for pt in pats); Im_ = Dict(k => zeros(Complex{T}, nO...) for k in keys(Ip)); Dv = Dict(k => zeros(Complex{T}, nO...) for k in keys(Ip))
    Ap = Dict(k => zeros(T, nO...) for k in keys(Ip)); DA = Dict(k => zeros(T, nO...) for k in keys(Ip))
    pD = (complex(TD(rp.upD), -TD(y)), complex(TD(rm.upD), -TD(y)))
    for (lo, hi) in pl, j in 1:nGt
        sn = (hi - lo) / 2 * xt[j] + (hi + lo) / 2; wq = (hi - lo) / 2 * wt[j]
        t = τ + sn * e; tD = Complex{TD}(t)
        ks = [[kapL(axk[o], sTo_[k_], ksI_[k_][j_], t, tol) for j_ in eachindex(xs[o])] for (k_, o) in enumerate(op)]
        e1 = exp(-pD[1] * tD); e2 = exp(-pD[2] * tD)
        ep = Complex{T}(e1) * e * wq; em = Complex{T}(e2) * e * wq; ed = Complex{T}((e1 - e2) / 2) * e * wq
        for pt in pats
            k = key(pt)
            rayAcc!(Ip[k], Ap[k], ep, ks, pt, nO); rayAcc!(Im_[k], Ap[k], em, ks, pt, nO); rayAcc!(Dv[k], DA[k], ed, ks, pt, nO)
        end
    end
    Ip, Im_, Dv, DA
end

# one node of spPair2's three compensated sums (function barrier)
function pairAcc!(Rp, Cp, Rm, Cm, Rd, Cd, Ad, A_, Δa, fp, fm, fd)
    @inbounds for i in eachindex(A_)
        tp = A_[i] * fp; tm = (A_[i] + Δa[i]) * fm; td = A_[i] * fd - Δa[i] * fm
        Rp[i], Cp[i] = kaddC(Rp[i], Cp[i], tp)
        Rm[i], Cm[i] = kaddC(Rm[i], Cm[i], tm)
        Rd[i], Cd[i] = kaddC(Rd[i], Cd[i], td)
        Ad[i] += abs(A_[i] * fd) + abs(Δa[i] * fm)
    end
    nothing
end
# pFac3(A, ξc + δ, x) - pFac3(A, ξc, x) for the three derivative orders, absolute error O(eps |δ|) (pDif, edges in T)
function pDif3(A::AxK{T}, ξc::S, δ::S, x::T, aT::T, bT::T) where {T,S}
    ξ = ξc + δ; ca = aT / (2 * T(π)); cb = bT / (2 * T(π))
    f1c = sinc(ξc * ca); f1 = sinc(ξ * ca); d1 = sincD(ξc * ca, δ * ca)
    f2c = sinc(ξc * cb); f2 = sinc(ξ * cb); d2 = sincD(ξc * cb, δ * cb)
    sh = sin(δ * x / 2); mA = (2 * ξc + δ) * x / 2; cx = cos(ξ * x); sx = sin(ξ * x)
    g0 = cx; e0 = -2 * sin(mA) * sh
    g1 = -ξ * sx; e1 = -(δ * sx + ξc * 2 * cos(mA) * sh)
    g2 = -ξ^2 * cx; e2 = -(δ * (2 * ξc + δ) * cx - ξc^2 * 2 * sin(mA) * sh)
    w = d1 * f2 + f1c * d2; c = f1c * f2c
    (A.w0 * (w * g0 + c * e0), A.w0 * (w * g1 + c * e1), A.w0 * (w * g2 + c * e2))
end

# A propagating-order pair (ν, ν') of the d_o = 2 pole form (d = 1), ν + ν' ∝ θ - θ_0 small: I(p₊), I(p₋) and D = [I(p₊) - I(p₋)]/2
# for every open offset and pattern, on one set of radial nodes s (the union of both orders' certified panels). With the pole subtraction
# points σ± (the Float64 poles), a± = a(σ±) (angular average of the open factors), I(p) = a(σ)E(p) + ∫ (a(s) - a(σ)) Φ(s + p) ds,
# Φ(w) = e^{-wτ}/w, E the closed part (E_1(pτ), its anomaly window, the J10 continuation term):
#   D = [a₊E₊ - a₋E₋]/2 + (1/2) ∫ [A₊(s) ΔΦ(s) - Δa Φ₋(s)] ds,  A₊ = a(s) - a₊ (pDif3 products), ΔΦ = Φ₊ - Φ₋, Δa = a₊ - a₋,
# an exact rearrangement of (I₊ - I₋)/2 in which every term is O(ν + ν') and formed without cancellation: ΔΦ from
# Φ(w + d) - Φ(w - d) = -2 e^{-wτ} (w sinh dτ + d cosh dτ)/(w² - d²) and the closed parts and a± in Double64, the same Double64 p± in both
# (the rearrangement is exact only if the closed part and the remainder see the same p and σ). Near the poles A₊ΔΦ and ΔaΦ₋ are each
# O(Δa/(s - σ)) and cancel: their rounding eps·Δa/h enters with the node weight ~h, i.e. eps·Δa per node, O(eps) of D.
# Returns (I₊, I₋, D, Σ|terms| of D) as Dicts pattern key => array over the open offsets.
function spPair2(::Type{T}, axk, op, xs, rp, rm, y::T, τ::T, α, tol::T, nGt::Int, s, plp, plm; Mx::Int = 0) where {T}
    TD = T === Float64 ? Double64 : T
    xt, wt = glq(T, nGt)
    xsO = [toT.(T, xs[o]) for o in op]; nO = length.(xsO)
    pats = [(0, 0), (1, 0), (0, 1), (2, 0), (0, 2), (1, 1)]
    key(pt) = ntuple(i -> i == op[1] ? pt[1] : i == op[2] ? pt[2] : 0, 3)
    abT = [(toT(T, axk[op[k]].a), toT(T, axk[op[k]].b)) for k in 1:2]
    axD = [AxK(TD, s[o], s[o]) for o in op]; abD = [(toT(TD, axk[op[k]].a), toT(TD, axk[op[k]].b)) for k in 1:2]
    xsD = [toT.(TD, xs[o]) for o in op]
    # the Double64 poles s_p = -u± (the caller's Double64 u), the Float64 subtraction radius rB = fl(√σ₊) shared by both orders
    upD = (TD(rp.upD), TD(rm.upD)); pD = (complex(upD[1], -TD(y)), complex(upD[2], -TD(y)))
    σ = (max(-T(upD[1]), zero(T)), max(-T(upD[2]), zero(T)))
    Smax = max(σ[1], σ[2]) + (α + 12) / τ
    M = max(Mx, maximum(circM([axk[o] for o in op], [maximum(abs, xsO[k]) for k in 1:2], [pt[1], pt[2]], sqrt(Smax) * 2, tol) for pt in pats))
    cM = [cospi(2 * T(j) / M) for j in 0:M-1]; sM = [sinpi(2 * T(j) / M) for j in 0:M-1]
    cMD = TD.(cM); sMD = TD.(sM)
    rB = sqrt(σ[1]); b1 = rB .* cM; b2 = rB .* sM; b1D = TD.(b1); b2D = TD.(b2); rBD = TD(rB)
    # a_C(ρ): the angular average on the point set b_j + (ρ - rB) e_j (Double64), i.e. exactly the set the remainder's pDif3 differences see
    # (their base points b_j = fl(rB e_j) are Float64 roundings); a real pole's residue a_C(s) - a_C(s_p) then vanishes at s_p on the remainder's
    # own points. The two orders share rB, so the remaining O(eps) deformation of the circle is common to both and cancels in D
    function aC(ρ)
        Q1 = [pFac3(axD[1], b1D[j] + (ρ - rBD) * cMD[j], xq, abD[1][1], abD[1][2]) for xq in xsD[1], j in 1:M]
        Q2 = [pFac3(axD[2], b2D[j] + (ρ - rBD) * sMD[j], xq, abD[2][1], abD[2][2]) for j in 1:M, xq in xsD[2]]
        P1 = [getindex.(Q1, k) for k in 1:3]; P2 = [getindex.(Q2, k) for k in 1:3]
        Dict(key(pt) => (P1[pt[1] + 1] * P2[pt[2] + 1]) ./ (4 * TD(π) * M) for pt in pats)
    end
    # the exact circle at a (complex) radius: the J10 continuation term only (no subtraction pairs with it)
    function aVD(r)
        Q1 = [pFac3(axD[1], r * cMD[j], xq, abD[1][1], abD[1][2]) for xq in xsD[1], j in 1:M]
        Q2 = [pFac3(axD[2], r * sMD[j], xq, abD[2][1], abD[2][2]) for j in 1:M, xq in xsD[2]]
        P1 = [getindex.(Q1, k) for k in 1:3]; P2 = [getindex.(Q2, k) for k in 1:3]
        Dict(key(pt) => (P1[pt[1] + 1] * P2[pt[2] + 1]) ./ (4 * TD(π) * M) for pt in pats)
    end
    τD = TD(τ); yD = TD(y)
    a0 = aC(rBD)
    closed = map(1:2) do q
        r = q == 1 ? rp : rm
        ac = aC(sqrt(max(-upD[q], zero(TD))))
        E1 = anmWin(T(upD[q]), complex(T(upD[q]), -y), τ) ? e1Win(pD[q], yD, τD) : (y == 0 && upD[q] < 0) ? complex(real(expintG(1, complex(upD[q] * τD, zero(TD)))), TD(π)) : expintG(1, pD[q] * τD)
        out = Dict(k => ac[k] .* E1 for k in keys(ac))
        if r.cont
            ak = aVD(sqrt(-pD[q])); for k in keys(out); out[k] .+= 2 * TD(π) * im .* ak[k]; end
        end
        ac, out
    end
    aP = closed[1][1]; aM = closed[2][1]
    Δa = Dict(k => Complex{T}.(aP[k] .- aM[k]) for k in keys(aP))
    Rq = [Dict(k => Complex{T}.(c[1][k] .- a0[k]) for k in keys(a0)) for c in closed]
    # the common radial panels: every breakpoint of both orders' certified panels (δ-panels around their σ)
    bp = sort(unique(vcat([σ[1] + T(e) for (lo, hi) in plp for e in (lo, hi)], [σ[2] + T(e) for (lo, hi) in plm for e in (lo, hi)])))
    bp = unique(clamp.(bp, zero(T), Smax)); length(bp) < 2 && (bp = [zero(T), Smax])
    P1c = [pFac3(axk[op[1]], b1[j], xq, abT[1]...) for xq in xsO[1], j in 1:M]
    P1ck = [getindex.(P1c, q) for q in 1:3]
    a0F = Dict(k => T.(real.(a0[k])) for k in keys(a0))
    Rp = Dict(key(pt) => zeros(Complex{T}, nO...) for pt in pats); Rm = Dict(k => zeros(Complex{T}, nO...) for k in keys(Rp)); Rd = Dict(k => zeros(Complex{T}, nO...) for k in keys(Rp))
    Cp = Dict(k => zeros(Complex{T}, nO...) for k in keys(Rp)); Cm = Dict(k => zeros(Complex{T}, nO...) for k in keys(Rp)); Cd = Dict(k => zeros(Complex{T}, nO...) for k in keys(Rp))
    Ad = Dict(k => zeros(T, nO...) for k in keys(Rp))
    ΦD(w) = exp(-w * τD) / w
    # the nodes' angular averages A (per pattern) and weights on the threads, 32 nodes at a time; accumulated in node order (pairAcc!)
    ndQ = [(q, j) for q in 1:length(bp)-1 for j in 1:nGt if bp[q + 1] > bp[q]]
    for c0 in 1:32:length(ndQ)
        nv = tMap(ndQ[c0:min(c0 + 31, end)]) do (q, j)
            local lo = bp[q]; local hi = bp[q + 1]
            local sv = (hi - lo) / 2 * xt[j] + (hi + lo) / 2; local wq = (hi - lo) / 2 * wt[j]
            # the node's radial offset from rB (Float64) and its exact square: Φ is evaluated at the radius the angular points see
            local δr = (sv - rB^2) / (sqrt(sv) + rB)
            local d1 = δr .* cM; local d2 = δr .* sM
            # near the poles (|δr| <= rB/2) A = a_C(s) - a_C(rB) by pDif3 products (absolute error O(eps |δr|)); farther the plain difference of
            # the node's angular average and a_C(rB) (error eps|a|, multiplied there by ΔΦ = O(Δu) and by Φ₋ ~ 1/|s - σ| <= 2/σ: O(eps) of D;
            # pDif3 everywhere cost 66 s for one pair of a 64³ d1 fill)
            local near = abs(δr) <= rB / 2
            local Q2 = [pFac3(axk[op[2]], b2[jj] + d2[jj], xq, abT[2]...) for jj in 1:M, xq in xsO[2]]
            local Q2k = [getindex.(Q2, q_) for q_ in 1:3]
            local svD = (rBD + TD(δr))^2; local wb = svD + (upD[1] + upD[2]) / 2 - im * yD; local dd = (upD[1] - upD[2]) / 2
            local Φp = ΦD(svD + upD[1] - im * yD); local Φm = ΦD(svD + upD[2] - im * yD)
            local ΔΦ = -2 * exp(-wb * τD) * (wb * sinh(dd * τD) + dd * cosh(dd * τD)) / (wb^2 - dd^2)
            if y == 0
                Φp = complex(real(Φp)); Φm = complex(real(Φm)); ΔΦ = complex(real(ΔΦ))
            end
            local As
            if near
                local dQ1 = [pDif3(axk[op[1]], b1[jj], d1[jj], xq, abT[1]...) for xq in xsO[1], jj in 1:M]
                local dQ2 = [pDif3(axk[op[2]], b2[jj], d2[jj], xq, abT[2]...) for jj in 1:M, xq in xsO[2]]
                local dQ1k = [getindex.(dQ1, q_) for q_ in 1:3]; local dQ2k = [getindex.(dQ2, q_) for q_ in 1:3]
                # A = a_C(s) - a_C(rB) (pDif3 products); N₊ = A - (a_C(s_p₊) - a_C(rB)), N₋ = N₊ + Δa
                As = [(dQ1k[pt[1] + 1] * Q2k[pt[2] + 1] .+ P1ck[pt[1] + 1] * dQ2k[pt[2] + 1]) ./ (4 * T(π) * M) .- Rq[1][key(pt)] for pt in pats]
            else
                local Q1n = [pFac3(axk[op[1]], b1[jj] + d1[jj], xq, abT[1]...) for xq in xsO[1], jj in 1:M]
                local Q1k = [getindex.(Q1n, q_) for q_ in 1:3]
                As = [(Q1k[pt[1] + 1] * Q2k[pt[2] + 1]) ./ (4 * T(π) * M) .- a0F[key(pt)] .- Rq[1][key(pt)] for pt in pats]
            end
            (As, Complex{T}(Φp) * wq, Complex{T}(Φm) * wq, Complex{T}(ΔΦ) * wq)
        end
        for (As, fp, fm, fd) in nv, (ip, pt) in enumerate(pats)
            k = key(pt)
            pairAcc!(Rp[k], Cp[k], Rm[k], Cm[k], Rd[k], Cd[k], Ad[k], As[ip], Δa[k], fp, fm, fd)
        end
    end
    Ip = Dict(k => Complex{T}.(closed[1][2][k]) .+ (Rp[k] .+ Cp[k]) for k in keys(Rp))
    Im_ = Dict(k => Complex{T}.(closed[2][2][k]) .+ (Rm[k] .+ Cm[k]) for k in keys(Rp))
    Dv = Dict(k => Complex{T}.((closed[1][2][k] .- closed[2][2][k]) ./ 2) .+ (Rd[k] .+ Cd[k]) ./ 2 for k in keys(Rp))
    DA = Dict(k => abs.(Complex{T}.((closed[1][2][k] .- closed[2][2][k]) ./ 2)) .+ Ad[k] ./ 2 for k in keys(Rp))
    Ip, Im_, Dv, DA
end

# per-axis vectors over the offsets xs at complex nodes: periodic (image form while 1/Re(1/t) <= L²/4π, else Poisson; lattice
# κ(k s, t) cached per node), open (κ at the offsets)
# fold = (Re f, Im f): the Poisson-form nodes carry e^{xt} folded into each term, e^{-u_m t} (the caller's weight is then e^{iyt})
function axVecC(::Type{T}, i, tv::Vector{Complex{T}}, xs, ms, axk, sTa, pers, P_, tol; excl = Int[], forceP = t -> false, fold = nothing, ab::Bool = false) where {T}
    n = length(xs[i]); M = [zeros(Complex{T}, n, length(tv)) for k in 1:3]
    # Σ|terms| of every entry (per-entry λ): open axes |κ|, periodic axes the chosen form's own Σ|terms|
    Ma = [zeros(T, n, length(tv)) for k in 1:3]
    αq = -log(tol) + 3
    if !haskey(pers, i)
        # erf regime: the kink terms depend on |k ± 1|, |k| only, so each lattice kink is evaluated once per node (3x fewer erfc)
        A = axk[i]; sT = sTa[i]; kmx = maximum(abs, ms[i]) + 1
        kq = [k * A.a for k in ms[i]]   # (the offsets once, not one Rational{BigInt} product per node)
        tFor(length(tv)) do it
            local t = tv[it]
            local v0 = Vector{Complex{T}}(undef, kmx + 1); local v1 = similar(v0); local v2 = similar(v0)
            local u = sqrt(t); local done = false
            for (j, k) in enumerate(ms[i])
                if momReg(sT, abs(k) * sT, t)
                    v = kap(A, kq[j], t, tol)
                else
                    if !done
                        for q in 0:kmx
                            w = q * sT / (2 * u)
                            v0[q + 1] = u * ierfcC(w); v1[q + 1] = erfcC(w) / 2; v2[q + 1] = exp(-w^2) / (2 * sqrt(T(π)) * u)
                        end
                        done = true
                    end
                    tri = sT * (abs(k + 1) - 2 * abs(k) + abs(k - 1)) / 2
                    dtr = T(sign(k + 1) - 2 * sign(k) + sign(k - 1)) / 2
                    k0 = complex(tri); k1 = complex(dtr); k2 = zero(Complex{T})
                    for (kk, sg) in ((k + 1, 1), (k, -2), (k - 1, 1))
                        q = abs(kk) + 1
                        k0 += sg * v0[q]; kk != 0 && (k1 -= (kk > 0 ? sg : -sg) * v1[q]); k2 += sg * v2[q]
                    end
                    v = (k0, k1, k2)
                end
                M[1][j, it] = v[1]; M[2][j, it] = v[2]; M[3][j, it] = v[3]
            end
        end
        return ab ? (M, [abs.(Mk) for Mk in M]) : M
    end
    Pq = pers[i]
    tFor(length(tv)) do it
        local t = tv[it]
        if useRealC(T, Pq, t) && isempty(excl) && !forceP(t)
            cache = Dict{Int,NTuple{3,Complex{T}}}()
            kv(k) = get!(() -> kapL(axk[i], sTa[i], k, t, tol), cache, k)
            ri = real(1 / t)
            for (j, m) in enumerate(ms[i])
                dn(nn) = max(abs(m + nn * P_) - 1, 0) * sTa[i]
                n0 = round(Int, -m / P_); dmin = min(dn(n0), dn(n0 - 1), dn(n0 + 1))
                D2 = dmin^2 + 4 * αq / ri
                nlo = n0; while dn(nlo - 1)^2 <= D2; nlo -= 1; end
                nhi = n0; while dn(nhi + 1)^2 <= D2; nhi += 1; end
                a0 = zero(Complex{T}); a1 = zero(Complex{T}); a2 = zero(Complex{T}); s0 = zero(T); s1 = zero(T); s2 = zero(T)
                for nn in nlo:nhi
                    k = kv(m + nn * P_); ph = phsN(T, nn, Pq)
                    a0 += ph * k[1]; a1 += ph * k[2]; a2 += ph * k[3]; s0 += abs(k[1]); s1 += abs(k[2]); s2 += abs(k[3])
                end
                av = (a0, a1, a2); sv = (s0, s1, s2)
                # as perSk: where the image form's Σ|terms|/|S^{(k)}| > 4 (odd derivative at a symmetry point, the value set by
                # images at distance L) the Poisson form is evaluated too and the one of smaller Σ|terms| kept per k
                if s0 > 4 * abs(a0) || s1 > 4 * abs(a1) || s2 > 4 * abs(a2)
                    pv = [zero(Complex{T}), zero(Complex{T}), zero(Complex{T})]; ps = zeros(T, 3)
                    for mq in poiRng(T, Pq, real(t), αq)
                        pt = poiTrm(T, mq, Pq, xs[i][j], t)
                        for q in 1:3; pv[q] += pt[q]; ps[q] += abs(pt[q]); end
                    end
                    av = ntuple(q -> ps[q] < sv[q] ? pv[q] : av[q], 3); sv = ntuple(q -> min(ps[q], sv[q]), 3)
                end
                M[1][j, it] = av[1]; M[2][j, it] = av[2]; M[3][j, it] = av[3]
                Ma[1][j, it] = sv[1]; Ma[2][j, it] = sv[2]; Ma[3][j, it] = sv[3]
            end
        elseif useRealC(T, Pq, t) && !forceP(t)
            # image form minus the excluded Poisson terms
            Mf, Mfa = axVecC(T, i, [t], xs, ms, axk, sTa, pers, P_, tol; ab = true)
            for k in 1:3; M[k][:, it] .= Mf[k][:, 1]; Ma[k][:, it] .= Mfa[k][:, 1]; end
            for m in excl
                for (j, xq) in enumerate(xs[i])
                    pt = poiTrm(T, m, Pq, xq, t)
                    for k in 1:3; M[k][j, it] -= pt[k]; Ma[k][j, it] += abs(pt[k]); end
                end
            end
        else
            for m in poiRngE(T, Pq, real(t), αq, nu2Ex(T, Pq, excl))
                m in excl && continue
                ef = fold === nothing ? nothing : exp(-uOrd((nuP(T, m, Pq),), fold[1], fold[2]) * t)
                for (j, xq) in enumerate(xs[i])
                    pt = ef === nothing ? poiTrm(T, m, Pq, xq, t) : (pr = poiPre(T, m, Pq, xq)[1]; (pr[1] * ef, pr[2] * ef, pr[3] * ef))
                    for k in 1:3; M[k][j, it] += pt[k]; Ma[k][j, it] += abs(pt[k]); end
                end
            end
        end
    end
    ab ? (M, Ma) : M
end

"""
    cFil!(CH, CA, ...)

Route C of the fill (cPart of blcTns for all offsets at once): complex segment τ → τc along φ* (lattice sum), then the tail from τc
(d = 3: closed form per q by three mode products; d < 3: orders with u <= U on their own rays, the others on t = τc + s). The complex
part C(θ; m) enters as H += [C(m) + conj C(-m)]/2, AH += [C(m) - conj C(-m)]/2i, C(-θ; R) = C(θ; -R) (K even), with C(-m) the reflected
array (periodic axes reversed, open axes (-1)^{p_o}).
"""
function cFil!(CH, CA, s, f::Complex{T}, ax::NTuple{D,Int}, xs, ms, nn, axk, sTa, pers, pats, τ1::T, c2::T, cU::T, rpx::T, α, tol, nG, D2s, D2o; zAll = nothing, MH = nothing, MA = nothing, hotC = nothing) where {T,D}
    fa = real(f); fb = imag(f)
    x = 4 * T(π)^2 * (fa - fb) * (fa + fb); y = 8 * T(π)^2 * fa * fb; z = complex(x, y)
    af4 = abs2(f)^2; f2i = complex((fa - fb) * (fa + fb) / af4, -2 * fa * fb / af4)
    V = prod(sTa); αp = α + 12
    op = Tuple(setdiff(1:3, ax)); isper(i) = haskey(pers, i)
    Pc = Dict(ax[i] => Int(pers[ax[i]].L / s[ax[i]]) for i in 1:D)
    xg, wg = glq(T, nG)
    φ = -angle(complex(-x, -y)) / 2; eφ = cis(φ)
    kk = 2 * T(π) * abs(f); ik = 2 * T(π) * fb
    sc = max(zero(T), (ik * rpx + c2 + x * τ1) / (kk * ik))
    D < 3 && (sc = min(sc, max(zero(T), minimum(i -> toT(T, pers[ax[i]].L)^2, 1:D) / (4 * T(π)) - τ1)))
    U = cU * abs(y)
    CC = Dict(p => zeros(Complex{T}, nn...) for p in pats); CCa = Dict(p => zeros(T, nn...) for p in pats)
    # hot rows (as in filBlk): the periodic per-axis vectors of those rows in Double64, rounded once
    hot = hotC === nothing ? Dict{Int,Vector{Int}}() : hotC.hot
    symC = hotC === nothing ? 0 : hotC.symAx
    TD = T === Float64 ? Double64 : T; rD = T(eps(TD) / eps(T))
    # Double64 rows (hot[i] or given) of axis i at the nodes: sums over hotC.rowC (symAx: the ±x pair) of axVecC's per-axis vectors
    function axRowsD(i, tvD, rows; excl = Int[], forceP = t -> false, fold = nothing)
        local MD = [zeros(Complex{TD}, length(rows), length(tvD)) for q in 1:3]
        local MaD = [zeros(TD, length(rows), length(tvD)) for q in 1:3]
        for c in 1:2
            local sub = [(r_, hotC.rowC(i, j)[c]) for (r_, j) in enumerate(rows) if length(hotC.rowC(i, j)) >= c]
            isempty(sub) && continue
            local xsS = [k == i ? [e[2][2] for e in sub] : xs[k] for k in 1:3]
            local msS = [k == i ? [e[2][1] for e in sub] : ms[k] for k in 1:3]
            local M1, M1a = axVecC(TD, i, tvD, xsS, msS, hotC.axkD, hotC.sTaD, hotC.persD, get(Pc, i, 0), hotC.tolD; excl, forceP = t -> forceP(Complex{T}(t)), fold = fold === nothing ? nothing : (hotC.faD, hotC.fbD), ab = true)
            for q in 1:3, (u_, e) in enumerate(sub), it in eachindex(tvD)
                MD[q][e[1], it] += e[2][3] * M1[q][u_, it]; MaD[q][e[1], it] += e[2][3] * M1a[q][u_, it]
            end
        end
        MD, MaD
    end
    function axVecH(i, tv; excl = Int[], forceP = t -> false, fold = nothing)
        local M_, Ma_ = axVecC(T, i, tv, xs, ms, axk, sTa, pers, get(Pc, i, 0), tol; excl, forceP, fold, ab = true)
        if haskey(hot, i)
            local rows = hot[i]
            local MD, MaD = axRowsD(i, Complex{TD}.(tv), rows; excl, forceP, fold)
            for q in 1:3, (r_, j) in enumerate(rows), it in eachindex(tv)
                M_[q][j, it] = Complex{T}(MD[q][r_, it]); Ma_[q][j, it] = abs(M_[q][j, it]) + T(MaD[q][r_, it]) * rD
            end
        end
        M_, Ma_
    end
    function addC!(Vs, tv, wv, fl, Va)
        ew = [(fl[j] ? exp(complex(zero(T), y) * tv[j]) : exp(z * tv[j])) * wv[j] for j in eachindex(tv)]
        wD = 4 * T(π)^2 .* ew; wP = ew .* f2i
        for p in pats
            krAdd!(CC[p], Vs[1][p[1] + 1], Vs[2][p[2] + 1], Vs[3][p[3] + 1], p == (0, 0, 0) ? wD : wP)
            krAddA!(CCa[p], Va[1][p[1] + 1], Va[2][p[2] + 1], Va[3][p[3] + 1], abs.(p == (0, 0, 0) ? wD : wP))
        end
    end
    # segment
    az = abs(z)
    segP = Tuple{T,T}[]
    if sc > 0
        np = max(1, ceil(Int, az * sc / T(3 // 2)))
        np > 10^5 && error("cFil!: segment of $np panels")
        d2 = minimum(D2s[D2s .> 0]; init = T(Inf))
        for q in 1:np
            lo0 = sc * T(q - 1) / np; hi0 = sc * T(q) / np
            sub = isfinite(d2) ? shpSub(τ1 + lo0, τ1 + hi0, tt -> az * tt + d2 / (4 * tt), T(4)) .- τ1 : [lo0, hi0]
            for r in 1:length(sub)-1; push!(segP, (sub[r], sub[r + 1])); end
        end
    end
    segNod(pl) = (tv = Complex{T}[]; wv = Complex{T}[]; for (lo, hi) in pl, j in 1:nG; push!(tv, τ1 + ((hi - lo) / 2 * xg[j] + (hi + lo) / 2) * eφ); push!(wv, (hi - lo) / 2 * wg[j] * eφ); end; (tv, wv))
    segAdd!(tv, wv) = isempty(tv) || (VV = [isper(i) ? axVecH(i, tv) : axVecC(T, i, tv, xs, ms, axk, sTa, pers, get(Pc, i, 0), tol; ab = true) for i in 1:3]; addC!([v[1] for v in VV], tv, wv, falses(length(tv)), [v[2] for v in VV]))
    tn, tw = segNod(segP); segAdd!(tn, tw)
    ordRec = Any[]; evCP = Tuple{T,T}[]; evCAdd! = nothing; SLc = Any[]; νE2c = T(Inf)
    ordPlF = Dict{Int,Any}(); grpOf = Dict{Any,Any}(); rayOf = Dict{Any,Any}(); ordConF = nothing
    τc = τ1 + sc * eφ
    if D == 3
        VΛ = prod(i -> toT(T, pers[i].L), 1:3)
        Pv = [pers[i] for i in 1:3]; rτ = real(τc)
        rngs = [begin
            twv = Pv[i].tw isa QI ? toT(T, Pv[i].tw) : Pv[i].tw
            νm = sqrt(max(x, zero(T)) + αp / rτ) / (2 * T(π)) + 1 / toT(T, Pv[i].L)
            collect((floor(Int, twv - νm * toT(T, Pv[i].L)) - 1):(ceil(Int, twv + νm * toT(T, Pv[i].L)) + 1))
        end for i in 1:3]
        Ep = [[phsP(T, mq, Pv[i], xq) for xq in xs[i], mq in rngs[i]] for i in 1:3]
        # ±m pairs at exact twists 0, 1/2 (prMat, as filBlk's route R d = 3 tail)
        prA = any(i -> prTw(Pv[i]), 1:3)
        EpP = [prMat(T, Pv[i], rngs[i], [((xq, 1),) for xq in xs[i]], Ep[i]) for i in 1:3]
        Epp(p) = [isodd(p[i]) ? EpP[i][2] : EpP[i][1] for i in 1:3]
        EpA = [(abs.(EpP[i][1]), abs.(EpP[i][2])) for i in 1:3]
        nq = length.(rngs)
        uMin = minimum(uOrd(ntuple(i -> nuP(T, mm[i], Pv[i]), 3), fa, fb) for mm in Iterators.product(rngs...))
        Cq = Dict(p => zeros(Complex{T}, nq...) for p in pats); CqM = Dict(p => zeros(T, nq...) for p in pats)
        for (i1, m1) in enumerate(rngs[1]), (i2, m2) in enumerate(rngs[2]), (i3, m3) in enumerate(rngs[3])
            mm = (m1, m2, m3); ν = ntuple(i -> nuP(T, mm[i], Pv[i]), 3)
            uq = uOrd(ν, fa, fb)
            (uq - uMin) * rτ > αp && continue
            p_ = complex(uq, -y); E_ = exp(-p_ * τc); W = E_ / p_
            c = V^2 / VΛ * prod(i -> sinc(ν[i] * sTa[i])^2, 1:3)
            for p in pats
                p == (0, 0, 0) && continue
                if any(==(2), p)
                    # the diagonal with its δ part, (-E + 4π² S_a W)/f² (filBlk route R d = 3)
                    a = findfirst(==(2), p); Sa = 4 * T(π)^2 * sum(d -> d == a ? zero(T) : ν[d]^2, 1:3); νa = 4 * T(π)^2 * ν[a]^2
                    if dgE(Sa, νa, abs(W), abs(E_), abs(f2i))
                        Cq[p][i1, i2, i3] = c * (Sa * W - E_) * f2i; CqM[p][i1, i2, i3] = c * (Sa * abs(W) + abs(E_)) * abs(f2i)
                    else
                        Cq[p][i1, i2, i3] = c * (4 * T(π)^2 * W - νa * W * f2i); CqM[p][i1, i2, i3] = c * abs(W) * (4 * T(π)^2 + νa * abs(f2i))
                    end
                else
                    Cq[p][i1, i2, i3] = -c * prod(i -> (2 * T(π) * ν[i])^p[i], 1:3) * W * f2i; CqM[p][i1, i2, i3] = abs(Cq[p][i1, i2, i3])
                end
            end
        end
        mode(Cm, E) = begin
            X1 = reshape(E[1] * reshape(Cm, nq[1], :), nn[1], nq[2], nq[3])
            X2 = permutedims(reshape(E[2] * reshape(permutedims(X1, (2, 1, 3)), nq[2], :), nn[2], nn[1], nq[3]), (2, 1, 3))
            permutedims(reshape(E[3] * reshape(permutedims(X2, (3, 1, 2)), nq[3], :), nn[3], nn[1], nn[2]), (2, 3, 1))
        end
        # hot slabs in Double64 (as filBlk's route R d = 3)
        CqD = nothing
        if !isempty(hot)
            PvD = [hotC.persD[i] for i in 1:3]; τcD = Complex{TD}(τc); yD = 8 * TD(π)^2 * hotC.faD * hotC.fbD
            afD = abs2(complex(hotC.faD, hotC.fbD))^2; f2iD = complex((hotC.faD - hotC.fbD) * (hotC.faD + hotC.fbD) / afD, -2 * hotC.faD * hotC.fbD / afD)
            VD = prod(i -> toT(TD, s[i]), 1:3); VΛD = prod(i -> toT(TD, pers[i].L), 1:3)
            CqD = Dict(p => zeros(Complex{TD}, nq...) for p in pats)
            for (i1, m1) in enumerate(rngs[1]), (i2, m2) in enumerate(rngs[2]), (i3, m3) in enumerate(rngs[3])
                mm = (m1, m2, m3)
                (uOrd(ntuple(i -> nuP(T, mm[i], Pv[i]), 3), fa, fb) - uMin) * rτ > αp && continue
                ν = ntuple(i -> nuP(TD, mm[i], PvD[i]), 3); pq = complex(uOrd(ν, hotC.faD, hotC.fbD), -yD)
                ED = exp(-pq * τcD); W = ED / pq; c = VD^2 / VΛD * prod(i -> sinc(ν[i] * toT(TD, s[i]))^2, 1:3)
                for p in pats
                    p == (0, 0, 0) && continue
                    if any(==(2), p)
                        a = findfirst(==(2), p); Sa = 4 * TD(π)^2 * sum(d -> d == a ? zero(TD) : ν[d]^2, 1:3); νa = 4 * TD(π)^2 * ν[a]^2
                        CqD[p][i1, i2, i3] = dgE(Sa, νa, abs(W), abs(ED), abs(f2iD)) ? c * (Sa * W - ED) * f2iD : c * (4 * TD(π)^2 * W - νa * W * f2iD)
                    else
                        CqD[p][i1, i2, i3] = -c * prod(i -> (2 * TD(π) * ν[i])^p[i], 1:3) * W * f2iD
                    end
                end
            end
            EpD = [[sum(c_ * phsP(TD, mq, PvD[i], x_) for (_, x_, c_) in hotC.rowC(i, j)) for j in eachindex(xs[i]), mq in rngs[i]] for i in 1:3]
            EpDP = [prMat(TD, PvD[i], rngs[i], [[(x_, c_) for (_, x_, c_) in hotC.rowC(i, j)] for j in eachindex(xs[i])], EpD[i]) for i in 1:3]
        end
        modeE(Cm, E) = begin
            m1, m2, m3 = size(E[1], 1), size(E[2], 1), size(E[3], 1)
            X1 = reshape(E[1] * reshape(Cm, nq[1], :), m1, nq[2], nq[3])
            X2 = permutedims(reshape(E[2] * reshape(permutedims(X1, (2, 1, 3)), nq[2], :), m2, m1, nq[3]), (2, 1, 3))
            permutedims(reshape(E[3] * reshape(permutedims(X2, (3, 1, 2)), nq[3], :), m3, m1, m2), (2, 3, 1))
        end
        # mode products of every pattern and hot slab on the threads, consumed in order below
        hl_ = collect(hot); pd_ = [p for p in pats if p != (0, 0, 0)]
        XC_ = Dict(zip(pd_, tMap(pd_) do p
            prA ? (mode(Cq[p], Epp(p)), mode(CqM[p], [isodd(p[i]) ? EpA[i][2] : EpA[i][1] for i in 1:3])) : (mode(Cq[p], Ep), nothing)
        end))
        YC_ = CqD === nothing ? nothing : Dict(zip([(p, i) for p in pd_ for (i, _) in hl_], tMap([(p, i, hj) for p in pd_ for (i, hj) in hl_]) do (p, i, hj)
            Complex{T}.(modeE(CqD[p], [k == i ? EpDP[k][isodd(p[k]) ? 2 : 1][hj, :] : EpDP[k][isodd(p[k]) ? 2 : 1] for k in 1:3]))
        end))
        for p in pats
            # (pattern 0 of the spectral tail is empty: its δ part sits in the diagonal patterns)
            p == (0, 0, 0) && continue
            XC = XC_[p][1]; sC = prA ? XC_[p][2] : sum(CqM[p])
            CC[p] .+= XC; CCa[p] .+= sC
            CqD === nothing && continue
            sCD = T(sum(abs, CqD[p])) * rD
            aC_ = XC; bC_ = prA ? copy(sC) : fill(sC, nn...)
            for (i, hj) in hl_
                sl = ntuple(k -> k == i ? hj : Colon(), 3)
                Y = YC_[(p, i)]
                CC[p][sl...] .+= Y .- aC_[sl...]; aC_[sl...] .= Y
                zz = abs.(Y) .+ sCD; CCa[p][sl...] .+= zz .- bC_[sl...]; bC_[sl...] .= zz
            end
        end
    else
        Pv = [pers[ax[i]] for i in 1:D]
        ordR = [begin
            twv = Pv[i].tw isa QI ? toT(T, Pv[i].tw) : Pv[i].tw
            νm = sqrt(max(x + U, zero(T))) / (2 * T(π)) + 1 / toT(T, Pv[i].L)
            (floor(Int, twv - νm * toT(T, Pv[i].L)) - 1):(ceil(Int, twv + νm * toT(T, Pv[i].L)) + 1)
        end for i in 1:D]
        SL = NTuple{D,Int}[]
        for mm in Iterators.product(ordR...)
            uOrd(ntuple(i -> nuP(T, mm[i], Pv[i]), D), fa, fb) <= U && push!(SL, mm)
        end
        ρo2 = maximum(D2o); ρo = sqrt(ρo2); ri = real(1 / τc)
        # node count per order-ray panel from ATAP 19.3: the smallest n with (32/15) Rρ ρ^{-2n}/(ρ² - 1) <= tol/10 for some ρ, where
        # Rρ = max over the open offsets and derivative orders of M(E_ρ)/M(segment) (the panel's majorant on the ellipse over the
        # same majorant on the panel itself: e^{max Re(-p t)} Π_o κ_o majorants, axMaj): each panel's Gauss error is then
        # below tol/10 of its own a priori scale 2h·M(segment), offset by offset (the vacuum library's farTol convention)
        xaO = Dict(o => unique([abs(toT(T, xq)) for xq in xs[o]]) for o in op)
        function omaj(o, k, e)
            cp = sTa[o]; w0 = axk[o].w0
            [w0 * gk(k, xa + cp, e.tlo) / sqrt(4 * T(π) * e.tlo) * exp(-max(xa - cp, zero(T))^2 * e.rinv / 4) for xa in xaO[o]]
        end
        cands = filter(<=(nG), [4, 6, 8, 10, 12, 14, 16, 20, 24, nG])
        function nOrdSel(tm, Δ, pq)
            e1 = ellC(tm, Δ, one(T)); e1 === nothing && return nG
            m1 = Dict((o, k) => omaj(o, k, e1) for o in op, k in 0:2)
            best = nG
            for ρ in RHOS
                e = ellC(tm, Δ, T(ρ)); e === nothing && continue
                Rρ = exp(e.rmax(-pq) - e1.rmax(-pq))
                for o in op
                    Rρ *= maximum(maximum(omaj(o, k, e) ./ m1[(o, k)]) for k in 0:2)
                end
                isfinite(Rρ) || continue
                for n in cands
                    n >= best && break
                    (32 / T(15)) * Rρ * T(ρ)^(-2n) / (T(ρ)^2 - 1) <= tol / 10 && (best = n; break)
                end
            end
            best
        end
        shp(v, i) = reshape(v, ntuple(k -> k == i ? length(v) : 1, 3))
        ordA = Dict{Any,Vector{Vector{Complex{T}}}}(); ordB = Dict{Any,Vector{Vector{Complex{T}}}}(); npanC = Ref(0)
        ordAa = Dict{Any,Vector{Vector{T}}}(); ordBa = Dict{Any,Vector{Vector{T}}}()
        # one order on given (lo, hi, n) panels of its ray t = τc + s ep: per pattern the periodic column and the open row
        function ordCon(r, pl)
            tr = Complex{T}[]; wr = Complex{T}[]
            for (lo, hi, n) in pl
                tm = τc + (hi + lo) / 2 * r.ep; Δ = (hi - lo) / 2 * r.ep
                xq, wq = glq(T, n)
                for j in 1:n; push!(tr, tm + Δ * xq[j]); push!(wr, Δ * wq[j]); end
            end
            ew = exp.(-r.p .* tr) .* wr
            # in chunks of 2048 nodes: the open-offset × node arrays of one ray held GBs at open offsets of 8λ
            out = nothing; outA = nothing
            for rg in Iterators.partition(eachindex(tr), 2048)
                trc = tr[rg]; ewc = ew[rg]; ewa = abs.(ewc)
                KK = Dict(o => axVecC(T, o, trc, xs, ms, axk, sTa, pers, 0, tol; ab = true) for o in op)
                Ko = Dict(o => KK[o][1] for o in op); Ka = Dict(o => KK[o][2] for o in op)
                part = Dict(pt => (D == 1 ? vec(cMul(Ko[op[1]][pt[op[1]] + 1] .* transpose(ewc), transpose(Ko[op[2]][pt[op[2]] + 1]))) : vec(cMul(Ko[op[1]][pt[op[1]] + 1], reshape(ewc, :, 1)))) for pt in pats)
                partA = Dict(pt => (D == 1 ? vec((Ka[op[1]][pt[op[1]] + 1] .* transpose(ewa)) * transpose(Ka[op[2]][pt[op[2]] + 1])) : vec(Ka[op[1]][pt[op[1]] + 1] * reshape(ewa, :, 1))) for pt in pats)
                if out === nothing
                    out = part; outA = partA
                else
                    for pt in pats; out[pt] .+= part[pt]; outA[pt] .+= partA[pt]; end
                end
            end
            out, length(tr), outA
        end
        cperOf(PP, pt) = D == 1 ? [PP[1][j][pt[ax[1]] + 1] for j in eachindex(PP[1])] : vec([PP[1][j1][pt[ax[1]] + 1] * PP[2][j2][pt[ax[2]] + 1] for j1 in eachindex(PP[1]), j2 in eachindex(PP[2])])
        # hot-axis groups: the orders related by the reflections g_j -> m0_j - g_j along hot axes whose whole product set lies in SL
        # share the leader's ray; their sum Σ_g PP_g e^{-p_g t} = e^{-p_L t} Π_j [Σ_v PP_j(v) e^{-4π²(ν_v² - ν_L,j²) t}] is formed per axis in
        # Double64 after the certification (the reflected orders' separate terms cancel on every row: the route R pairs)
        m0 = Dict(i => (w_ = mod(Float64(pers[i].tw isa QI ? toT(T, pers[i].tw) : pers[i].tw), 1.0); abs(w_ - 0.5) < min(w_, 1 - w_) ? 1 : 0) for i in keys(hot))
        grpOf = Dict{NTuple{D,Int},Any}(); SLs = Set(SL)
        for g in SL
            hs = [j for j in 1:D if haskey(hot, ax[j]) && m0[ax[j]] - g[j] != g[j]]
            cands_ = length(hs) == 2 ? [hs, [hs[1]], [hs[2]]] : length(hs) == 1 ? [hs] : Vector{Int}[]
            for S in cands_
                mem = vec([ntuple(j -> j in S && b[findfirst(==(j), S)] ? m0[ax[j]] - g[j] : g[j], D) for b in Iterators.product(ntuple(_ -> (false, true), length(S))...)])
                if all(in(SLs), mem)
                    grpOf[g] = (leader = minimum(mem), members = sort(unique(mem)), vals = [j in S ? sort(unique([g[j], m0[ax[j]] - g[j]])) : [g[j]] for j in 1:D]); break
                end
            end
        end
        rayOf = Dict{NTuple{D,Int},Any}()
        SL = sort(SL; by = g -> (haskey(grpOf, g) ? grpOf[g].leader : g, g))
        for g in SL
            ν = ntuple(i -> nuP(T, g[i], Pv[i]), D)
            p_ = complex(uOrdX(T, g, Pv, fa, fb), -y)
            abs(p_) <= 64 * eps(T) * max(abs(x), one(T)) && error("cFil!: Rayleigh-Wood anomaly at rounding distance, order $g")
            φp = -angle(p_) / 2; ep = cis(φp); rate = abs(p_) * cos(angle(p_) / 2)
            ld = haskey(grpOf, g) ? grpOf[g].leader : g
            if ld != g && haskey(rayOf, ld)
                # a group member on its leader's ray and panels
                ep = rayOf[ld].ep
                PP = [ax[i] == symC ? [Complex{T}.(hotC.ppD(g[i], ax[i], j)) for j in eachindex(xs[ax[i]])] : [poiPre(T, g[i], Pv[i], xq)[1] for xq in xs[ax[i]]] for i in 1:D]
                PPa = [[zero.(PP[i][j]) for j in eachindex(PP[i])] for i in 1:D]
                pl = copy(rayOf[ld].pl)
                r = (p = p_, ep = ep, PP = PP, PPa = PPa, g = g); push!(ordRec, (r, pl))
                # (the group sum is formed after the certification: no ray integral of its own)
                nq_ = D == 1 ? nn[op[1]] * nn[op[2]] : nn[op[1]]
                Ior = Dict(pt => zeros(Complex{T}, nq_) for pt in pats); IorA = Dict(pt => zeros(T, nq_) for pt in pats)
                for pt in pats
                    push!(get!(() -> Vector{Complex{T}}[], ordA, pt), cperOf(PPa, pt)); push!(get!(() -> Vector{Complex{T}}[], ordB, pt), Ior[pt])
                    push!(get!(() -> Vector{T}[], ordAa, pt), abs.(cperOf(PPa, pt))); push!(get!(() -> Vector{T}[], ordBa, pt), IorA[pt])
                end
                continue
            end
            smax = max((αp + ρo2 * ri / 4) / rate, ((sqrt(αp) + sqrt(αp + 2 * sqrt(rate) * ρo)) / (2 * sqrt(rate)))^2)
            pl = Tuple{T,T,Int}[]
            s0 = zero(T); s1 = min(smax, max(abs(τc), 1 / rate) / 4)
            while s0 < smax
                s1 = min(s1, smax)
                osc = max(1, ceil(Int, abs(p_) * (s1 - s0) / T(3 // 2)))
                npanC[] += osc
                npanC[] > CPMAX && error("cFil!: panel budget: more than $CPMAX complex-path panels (order rays at open distance ρ = $ρo)")
                for q in 1:osc
                    lo = s0 + (s1 - s0) * T(q - 1) / osc; hi = s0 + (s1 - s0) * T(q) / osc
                    n = nOrdSel(τc + (hi + lo) / 2 * ep, (hi - lo) / 2 * ep, p_)
                    push!(pl, (lo, hi, n))
                end
                s0 = s1; s1 = 2 * s1
            end
            PP = [ax[i] == symC ? [Complex{T}.(hotC.ppD(g[i], ax[i], j)) for j in eachindex(xs[ax[i]])] : [poiPre(T, g[i], Pv[i], xq)[1] for xq in xs[ax[i]]] for i in 1:D]
            PPa = haskey(grpOf, g) ? [[zero.(PP[i][j]) for j in eachindex(PP[i])] for i in 1:D] : PP
            haskey(grpOf, g) && (rayOf[g] = (ep = ep, pl = copy(pl), p = p_))
            r = (p = p_, ep = ep, PP = PP, PPa = PPa, g = g); push!(ordRec, (r, pl))
            if PPa === PP
                Ior, _, IorA = ordCon(r, pl)
            else
                nq_ = D == 1 ? nn[op[1]] * nn[op[2]] : nn[op[1]]
                Ior = Dict(pt => zeros(Complex{T}, nq_) for pt in pats); IorA = Dict(pt => zeros(T, nq_) for pt in pats)
            end
            # per order and pattern: the periodic column (over the periodic offsets) and the open row (over the open offsets); the sum
            # over orders is one chunk-compensated product per pattern below (a full-array broadcast per order and pattern cost 139 s
            # of the 160 s 64³ d = 2 1+i fill)
            for pt in pats
                push!(get!(() -> Vector{Complex{T}}[], ordA, pt), cperOf(PPa, pt) .* (pt == (0, 0, 0) ? 4 * T(π)^2 : f2i))
                push!(get!(() -> Vector{Complex{T}}[], ordB, pt), Ior[pt])
                push!(get!(() -> Vector{T}[], ordAa, pt), abs.(cperOf(PPa, pt)) .* (pt == (0, 0, 0) ? 4 * T(π)^2 : abs(f2i)))
                push!(get!(() -> Vector{T}[], ordBa, pt), IorA[pt])
            end
        end
        # add one order's contribution difference (certification refinement) into CC
        function ordDelta!(r, dI)
            nper = D == 1 ? (nn[ax[1]],) : (nn[ax[1]], nn[ax[2]]); nop = D == 1 ? (nn[op[1]], nn[op[2]]) : (nn[op[1]],)
            for pt in pats
                X = (cperOf(r.PPa, pt) .* (pt == (0, 0, 0) ? 4 * T(π)^2 : f2i)) * transpose(dI[pt])
                addPerm!(CC[pt], reshape(X, nper..., nop...), invperm([ax..., op...]))
            end
        end
        append!(SLc, SL)
        for pt in keys(ordA)
            X = cMul(reduce(hcat, ordA[pt]), transpose(reduce(hcat, ordB[pt])))
            nper = D == 1 ? (nn[ax[1]],) : (nn[ax[1]], nn[ax[2]]); nop = D == 1 ? (nn[op[1]], nn[op[2]]) : (nn[op[1]],)
            Cc_ = get!(() -> zeros(Complex{T}, size(CC[pt])), CMPS[], CC[pt])
            kaddPerm!(CC[pt], Cc_, reshape(X, nper..., nop...), invperm([ax..., op...]))
            Xa = rMulB(reduce(hcat, ordAa[pt]), transpose(reduce(hcat, ordBa[pt])))
            addPerm!(CCa[pt], reshape(Xa, nper..., nop...), invperm([ax..., op...]))
        end
        # the other orders: t = τc + s
        spProj = [unique(g[i] for g in SL) for i in 1:D]
        boxE = [begin
            twv = Pv[i].tw isa QI ? toT(T, Pv[i].tw) : Pv[i].tw
            νm = sqrt(max(x + U, zero(T))) / (2 * T(π)) + 3 / toT(T, Pv[i].L)
            (floor(Int, twv - νm * toT(T, Pv[i].L)) - 2):(ceil(Int, twv + νm * toT(T, Pv[i].L)) + 2)
        end for i in 1:D]
        uEV = minimum((uOrd(ntuple(i -> nuP(T, mm[i], Pv[i]), D), fa, fb) for mm in Iterators.product(boxE...) if !(mm in SL)); init = U)
        uSL = minimum((uOrd(ntuple(i -> nuP(T, g[i], Pv[i]), D), fa, fb) for g in SL); init = U)
        cancE(t) = (uEV - uSL) * real(t) > log(T(4))
        νE2 = (uEV + x) / (4 * T(π)^2); νE2c = min(νE2c, νE2)
        smaxE = max((αp + ρo2 * ri / 4) / U, ((sqrt(αp) + sqrt(αp + 2 * sqrt(U) * ρo)) / (2 * sqrt(U)))^2)
        s0 = zero(T); s1 = min(smaxE, max(abs(τc), 1 / U) / 4)
        while s0 < smaxE
            s1 = min(s1, smaxE)
            osc = max(1, ceil(Int, abs(y) * (s1 - s0) / T(3 // 2)))
            for q in 1:osc
                push!(evCP, (s0 + (s1 - s0) * T(q - 1) / osc, s0 + (s1 - s0) * T(q) / osc))
            end
            s0 = s1; s1 = 2 * s1
        end
        evNod(pl) = (tv = Complex{T}[]; wv = Complex{T}[]; for (lo, hi) in pl, j in 1:nG; push!(tv, τc + ((hi - lo) / 2 * xg[j] + (hi + lo) / 2)); push!(wv, complex((hi - lo) / 2 * wg[j])); end; (tv, wv))
        te, we = evNod(evCP)
        # the 64-node E/EA buffers of the d = 2 EV block (set by evAdd!/evCAdd!)
        evBf = Ref{Tuple{Dict{NTuple{2,Int},Matrix{Complex{T}}},Dict{NTuple{2,Int},Matrix{T}}}}()
        evCAdd1! = function (te, we)
        VoA = Dict(o => axVecC(T, o, te, xs, ms, axk, sTa, pers, 0, tol; ab = true) for o in op)
        Vo = Dict(o => VoA[o][1] for o in op)
        if D == 1
            i1 = ax[1]
            Vp, Vpa = axVecH(i1, te; excl = spProj[1], forceP = cancE, fold = (fa, fb))
            Vs = Vector{Vector{Matrix{Complex{T}}}}(undef, 3); Vs[i1] = Vp; for o in op; Vs[o] = Vo[o]; end
            Va = Vector{Vector{Matrix{T}}}(undef, 3); Va[i1] = Vpa; for o in op; Va[o] = VoA[o][2]; end
            addC!(Vs, te, we, [!useRealC(T, Pv[1], t) || cancE(t) for t in te], Va)
        else
            i1, i2 = ax; o = op[1]
            n1, n2, n3 = nn[i1], nn[i2], nn[o]
            αq = -log(tol) + 3
            # views of 64-column buffers allocated once per evCAdd! call (evBf); every column is overwritten below
            E = Dict(k => view(v, :, 1:length(te)) for (k, v) in evBf[][1]); EA = Dict(k => view(v, :, 1:length(te)) for (k, v) in evBf[][2])
            S1, S1a = axVecH(i1, te); S2, S2a = axVecH(i2, te)
            pcol(i, m) = [phsP(T, m, pers[i], xq) for xq in xs[i]]
            pco(i, m, t) = poiTrm(T, m, pers[i], QI0, t)
            aP = [all(i -> !useRealC(T, Pv[i], t), 1:D) || cancE(t) for t in te]
            pcS1 = [pcol(i1, g[1]) for g in SL]; pcS2 = [pcol(i2, g[2]) for g in SL]
            tFor(length(te)) do it
                local t = te[it]
                local r1, r2, Φ1, Φ2, c1v, c2v, C, M, Ma, c1_, c2_
                if aP[it]
                    # EV-only Poisson pairs with e^{xt} folded, e^{-u_G t} (the weight below is then e^{iyt}; evC!)
                    r1 = collect(poiRngE(T, Pv[1], real(t), αq, νE2)); r2 = collect(poiRngE(T, Pv[2], real(t), αq, νE2))
                    Φ1 = hcat([pcol(i1, m) for m in r1]...); Φ2 = hcat([pcol(i2, m) for m in r2]...)
                    c1v = [poiPre(T, m, pers[i1], QI0) for m in r1]; c2v = [poiPre(T, m, pers[i2], QI0) for m in r2]
                    for (k1, k2) in keys(E)
                        C = [(g1, g2) in SL ? zero(Complex{T}) : c1v[a][1][k1 + 1] * c2v[b][1][k2 + 1] * exp(-uOrd((c1v[a][2], c2v[b][2]), fa, fb) * t) for (a, g1) in enumerate(r1), (b, g2) in enumerate(r2)]
                        # (the product written into E's column)
                        mul!(reshape(view(E[(k1, k2)], :, it), n1, n2), Φ1 * C, transpose(Φ2)); EA[(k1, k2)][:, it] .= sum(abs, C)
                    end
                else
                    # E = S1 S2ᵀ - Σ_g (c1_g φ1_g)(c2_g φ2_g)ᵀ as one product written into E's column (the phase columns once per chunk)
                    local pc1 = [pco(i1, g[1], t) for g in SL]; local pc2 = [pco(i2, g[2], t) for g in SL]
                    for (k1, k2) in keys(E)
                        local L1 = hcat(S1[k1 + 1][:, it], reduce(hcat, [-pc1[q][k1 + 1] .* pcS1[q] for q in eachindex(SL)]; init = zeros(Complex{T}, n1, 0)))
                        local L2 = hcat(S2[k2 + 1][:, it], reduce(hcat, [pc2[q][k2 + 1] .* pcS2[q] for q in eachindex(SL)]; init = zeros(Complex{T}, n2, 0)))
                        mul!(reshape(view(E[(k1, k2)], :, it), n1, n2), L1, transpose(L2))
                        local sa_ = sum(q -> abs(pc1[q][k1 + 1]) * abs(pc2[q][k2 + 1]), eachindex(SL); init = zero(T))
                        Ma = reshape(view(EA[(k1, k2)], :, it), n1, n2)
                        mul!(Ma, view(S1a[k1 + 1], :, it:it), transpose(view(S2a[k2 + 1], :, it:it))); Ma .+= sa_
                    end
                end
            end
            # hot rows/columns of E in Double64 (as filBlk's route R block, complex t)
            if haskey(hot, i1) || haskey(hot, i2)
                local h1 = get(hot, i1, Int[])
                local h2 = get(hot, i2, Int[])
                local tD = Complex{TD}.(te)
                rowD(i, rows) = axRowsD(i, tD, rows)
                local S1D = isempty(h1) ? nothing : rowD(i1, h1)
                local S2D = isempty(h2) ? nothing : rowD(i2, h2)
                local pcD = Dict{Tuple{Int,Int},Vector{Complex{TD}}}()
                pcolD(i, m) = get!(() -> [sum(c_ * phsP(TD, m, hotC.persD[i], x_) for (_, x_, c_) in hotC.rowC(i, j)) for j in eachindex(xs[i])], pcD, (i, m))
                local prs = unique(vcat([(j1, j2) for j1 in h1 for j2 in 1:n2], [(j1, j2) for j1 in 1:n1 for j2 in h2]))
                for (it, t) in enumerate(te)
                    if aP[it]
                        local r1 = collect(poiRngE(T, Pv[1], real(t), αq, νE2))
                        local r2 = collect(poiRngE(T, Pv[2], real(t), αq, νE2))
                        local q1c = [poiPre(TD, m, hotC.persD[i1], QI0) for m in r1]
                        local q2c = [poiPre(TD, m, hotC.persD[i2], QI0) for m in r2]
                        local Φ1 = reduce(hcat, [pcolD(i1, m) for m in r1])
                        local Φ2 = reduce(hcat, [pcolD(i2, m) for m in r2])
                        local eu = [(g1, g2) in SL ? zero(Complex{TD}) : exp(-uOrd((q1c[a][2], q2c[b][2]), hotC.faD, hotC.fbD) * tD[it]) for (a, g1) in enumerate(r1), (b, g2) in enumerate(r2)]
                        for (k1, k2) in keys(E)
                            local Cd = [q1c[a][1][k1 + 1] * q2c[b][1][k2 + 1] * eu[a, b] for a in eachindex(r1), b in eachindex(r2)]
                            local ca = T(sum(abs, Cd)) * rD
                            if !isempty(h1)
                                local Wr = (Φ1[h1, :] * Cd) * transpose(Φ2)
                                for (q, j1) in enumerate(h1), j2 in 1:n2
                                    local vr = Complex{T}(Wr[q, j2]); E[(k1, k2)][j1 + n1 * (j2 - 1), it] = vr; EA[(k1, k2)][j1 + n1 * (j2 - 1), it] = abs(vr) + ca
                                end
                            end
                            if !isempty(h2)
                                local Wc = Φ1 * (Cd * transpose(Φ2[h2, :]))
                                for (q, j2) in enumerate(h2), j1 in 1:n1
                                    local vc = Complex{T}(Wc[j1, q]); E[(k1, k2)][j1 + n1 * (j2 - 1), it] = vc; EA[(k1, k2)][j1 + n1 * (j2 - 1), it] = abs(vc) + ca
                                end
                            end
                        end
                    else
                        # per SP order the two per-axis columns (Double64): A1[j1, g] = c1_g φ1_g(j1), A2[j2, g] = c2_g φ2_g(j2) e^{-4π²(ν1² + ν2²)t}; the
                        # hot rows/columns of E are then S1 ⊗ S2 - A1 A2ᵀ restricted to them (small Double64 products; per-entry loops over the orders
                        # cost ~30 s per 64³ d2 hot fill). Typed: with every order evanescent (no SP order) the untyped n×0 comprehensions were Matrix{Any} and A1 A2ᵀ raised
                        local spc = [(poiPre(TD, g[1], hotC.persD[i1], QI0), poiPre(TD, g[2], hotC.persD[i2], QI0)) for g in SL]
                        local spe = [exp(-4 * TD(π)^2 * (q1[2]^2 + q2[2]^2) * tD[it]) for (q1, q2) in spc]
                        for (k1, k2) in keys(E)
                            local A1 = Complex{TD}[spc[gq][1][1][k1 + 1] * pcolD(i1, g[1])[j] for j in 1:n1, (gq, g) in enumerate(SL)]
                            local A2 = Complex{TD}[spc[gq][2][1][k2 + 1] * pcolD(i2, g[2])[j] * spe[gq] for j in 1:n2, (gq, g) in enumerate(SL)]
                            local S1v = [(r_ = findfirst(==(j), h1); r_ !== nothing ? S1D[1][k1 + 1][r_, it] : Complex{TD}(S1[k1 + 1][j, it])) for j in 1:n1]
                            local S2v = [(r_ = findfirst(==(j), h2); r_ !== nothing ? S2D[1][k2 + 1][r_, it] : Complex{TD}(S2[k2 + 1][j, it])) for j in 1:n2]
                            local S1w = [(r_ = findfirst(==(j), h1); r_ !== nothing ? S1D[2][k1 + 1][r_, it] : TD(S1a[k1 + 1][j, it])) for j in 1:n1]
                            local S2w = [(r_ = findfirst(==(j), h2); r_ !== nothing ? S2D[2][k2 + 1][r_, it] : TD(S2a[k2 + 1][j, it])) for j in 1:n2]
                            local aA1 = abs.(A1)
                            local aA2 = abs.(A2)
                            if !isempty(h1)
                                local Vr = S1v[h1] * transpose(S2v) .- A1[h1, :] * transpose(A2)
                                local Va = S1w[h1] * transpose(S2w) .+ aA1[h1, :] * transpose(aA2)
                                for (q, j1) in enumerate(h1), j2 in 1:n2
                                    local v_ = Complex{T}(Vr[q, j2]); E[(k1, k2)][j1 + n1 * (j2 - 1), it] = v_; EA[(k1, k2)][j1 + n1 * (j2 - 1), it] = abs(v_) + T(Va[q, j2]) * rD
                                end
                            end
                            if !isempty(h2)
                                local Vc = S1v * transpose(S2v[h2]) .- A1 * transpose(A2[h2, :])
                                local Va = S1w * transpose(S2w[h2]) .+ aA1 * transpose(aA2[h2, :])
                                for (q, j2) in enumerate(h2), j1 in 1:n1
                                    local v_ = Complex{T}(Vc[j1, q]); E[(k1, k2)][j1 + n1 * (j2 - 1), it] = v_; EA[(k1, k2)][j1 + n1 * (j2 - 1), it] = abs(v_) + T(Va[j1, q]) * rD
                                end
                            end
                        end
                    end
                end
            end
            ew = [(aP[j] ? exp(complex(zero(T), y) * te[j]) : exp(z * te[j])) * we[j] for j in eachindex(te)]
            # per E block one product over the chunk for every pattern that reads it, into evCAc (compensated across the call's chunks)
            for (kk, ps_) in evG2C(i1, i2)
                local Y = reduce(vcat, [Vo[o][p[o] + 1] .* transpose(p == (0, 0, 0) ? 4 * T(π)^2 .* ew : ew .* f2i) for p in ps_])
                local Ya = reduce(vcat, [VoA[o][2][p[o] + 1] .* transpose(abs.(p == (0, 0, 0) ? 4 * T(π)^2 .* ew : ew .* f2i)) for p in ps_])
                local ac_ = evCAc[][kk]
                cMulAcc!(ac_[1], ac_[2], E[kk], Matrix(transpose(Y))); rMulAcc!(ac_[3], EA[kk], Matrix(transpose(Ya)))
            end
        end
        end
        evG2C(i1, i2) = [(kk, [p for p in pats if (p[i1], p[i2]) == kk]) for kk in sort(unique([(p[i1], p[i2]) for p in pats]))]
        evCAc = Ref{Any}(nothing)
        # in chunks of 64 nodes: the node-by-offset arrays (E: n1·n2 × nodes × 6 in d = 2) held GBs at 64³ (7 GB RSS)
        evCAdd! = function (te, we)
            local i1, i2, o, n12, perm, n3
            if D == 2
                i1, i2 = ax; o = op[1]; n12 = nn[i1] * nn[i2]
                evCAc[] = Dict(kk => (zeros(Complex{T}, n12, nn[o] * length(ps_)), zeros(Complex{T}, n12, nn[o] * length(ps_)), zeros(T, n12, nn[o] * length(ps_))) for (kk, ps_) in evG2C(i1, i2))
                evBf[] = (Dict((k1, k2) => Matrix{Complex{T}}(undef, n12, 64) for k1 in 0:2, k2 in 0:2 if k1 + k2 <= 2), Dict((k1, k2) => Matrix{T}(undef, n12, 64) for k1 in 0:2, k2 in 0:2 if k1 + k2 <= 2))
            end
            for r in Iterators.partition(eachindex(te), 64); evCAdd1!(te[r], we[r]); end
            if D == 2
                perm = invperm([i1, i2, o]); n3 = nn[o]
                for (kk, ps_) in evG2C(i1, i2), (q, p) in enumerate(ps_)
                    local cols = (q - 1) * n3 .+ (1:n3); local ac_ = evCAc[][kk]
                    addPerm!(CC[p], reshape(view(ac_[1], :, cols) .+ view(ac_[2], :, cols), nn[i1], nn[i2], n3), perm)
                    addPerm!(CCa[p], reshape(ac_[3][:, cols], nn[i1], nn[i2], n3), perm)
                end
                evCAc[] = nothing
            end
        end
        evCAdd!(te, we)
    end
    # certification of the complex path per offset (as the real-axis stages in blcFil): every segment panel (lattice-sum majorants of
    # majLat per axis and offset), order-ray panel (its own node count; e^{max Re(-pt)}, |periodic prefactors| and the open-axis
    # majorants, ellC neg) and EV-ray panel (majEVf: EV-only Poisson majorant times the open-axis majorants) gets per-offset
    # ATAP 19.3 bounds as sums of separable products; C's error enters H and AH as (E(m) + E(-m))/2 against tol·min(max|H(m)|, max|AH(m)|)
    function cCert!()
        SH = zeros(T, nn...); SA = zeros(T, nn...)
        ents_ = [(1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)]
        pOf(a, b) = ntuple(i -> dOrd(a, b, i), 3)
        # one threaded pass per entry (certS!), the same operations per element as the broadcasts it replaces
        rv_ = ntuple(i -> i in ax && i != symC, 3)
        for (a, b) in ents_
            certS!(SH, SA, map(p -> (CH[p], cmpGet(CH[p]), CA[p], cmpGet(CA[p]), CC[p], cmpGet(CC[p]), iseven(sum(o -> p[o], op; init = 0)) ? one(T) : -one(T)),
                                (a == b ? (pOf(a, b), (0, 0, 0)) : (pOf(a, b),))), rv_, V)
        end
        Smin = reshape([SH[j] > 0 && SA[j] > 0 ? min(SH[j], SA[j]) : max(SH[j], SA[j]) for j in eachindex(SH)], size(SH))
        xq = [[QI(xv) for xv in xs[i]] for i in 1:3]
        pdn = [p for p in pats if p != (0, 0, 0)]
        wP(p) = p == (0, 0, 0) ? 4 * T(π)^2 : abs(f2i)
        function segT(lo, hi)
            tm = τ1 + (lo + hi) / 2 * eφ; Δ = (hi - lo) / 2 * eφ; best = nothing; bv = T(Inf)
            for ρ in RHOS
                e = ellC(tm, Δ, T(ρ)); e === nothing && continue
                vs = [[(hotC !== nothing && i == hotC.symAx ? 2 : 1) .* axMajV(axk[i], xq[i], get(pers, i, nothing), k, e, tol) for k in 0:2] for i in 1:3]
                g = (64 / T(15)) * abs(Δ) * T(ρ)^(-2nG) / (T(ρ)^2 - 1) * exp(e.rmax(z)) / V
                sep = g * maximum(p -> wP(p) * prod(i -> maximum(vs[i][p[i] + 1]), 1:3), pats)
                isfinite(sep) && sep < bv && (bv = sep; best = (g, vs))
            end
            best === nothing && return nothing
            g, vs = best
            [(g * wP(p), vs[1][p[1] + 1], vs[2][p[2] + 1], vs[3][p[3] + 1]) for p in pats]
        end
        function ordT(r, lo, hi, n)
            tm = τc + (lo + hi) / 2 * r.ep; Δ = (hi - lo) / 2 * r.ep; best = nothing; bv = T(Inf)
            pv = [[[abs(r.PP[i][j][k + 1]) for j in eachindex(r.PP[i])] for k in 0:2] for i in 1:D]
            for ρ in RHOS
                e = ellC(tm, Δ, T(ρ); neg = true); e === nothing && continue
                vo = Dict(o => [axMajV(axk[o], xq[o], nothing, k, e, tol) for k in 0:2] for o in op)
                g = (64 / T(15)) * abs(Δ) * T(ρ)^(-2n) / (T(ρ)^2 - 1) * exp(e.rmax(-r.p)) / V
                vec3(p, i) = i in op ? vo[i][p[i] + 1] : pv[findfirst(==(i), ax)][p[i] + 1]
                sep = g * maximum(p -> wP(p) * prod(i -> maximum(vec3(p, i)), 1:3), pats)
                isfinite(sep) && sep < bv && (bv = sep; best = [(g * wP(p), vec3(p, 1), vec3(p, 2), vec3(p, 3)) for p in pats])
            end
            best
        end
        function evT(lo, hi)
            tm = τc + complex((lo + hi) / 2); Δ = complex((hi - lo) / 2); best = nothing; bv = T(Inf)
            Pv_ = [pers[ax[i]] for i in 1:D]
            for ρ in RHOS
                e = ellC(tm, Δ, T(ρ)); e === nothing && continue
                rgs = [poiRngE(T, Pv_[i], e.rlo, α + 40, νE2c) for i in 1:D]
                Mp = zeros(T, 3, 3)
                for mm in Iterators.product(rgs...)
                    (D == 2 ? mm in SLc : mm[1] in [g[1] for g in SLc]) && continue
                    ν = ntuple(i -> nuP(T, mm[i], Pv_[i]), D)
                    c = exp(-uOrd(ν, fa, fb) * e.rlo) * prod(i -> abs(poiPre(T, mm[i], Pv_[i], QI0)[1][1]), 1:D)
                    for k1 in 0:2, k2 in 0:(D == 2 ? 2 - k1 : 0)
                        Mp[k1 + 1, k2 + 1] += c * (2 * T(π) * abs(ν[1]))^k1 * (D == 2 ? (2 * T(π) * abs(ν[2]))^k2 : one(T))
                    end
                end
                vo = Dict(o => [axMajV(axk[o], xq[o], nothing, k, e, tol) for k in 0:2] for o in op)
                g = (64 / T(15)) * abs(Δ) * T(ρ)^(-2nG) / (T(ρ)^2 - 1) * exp(e.rmax(complex(zero(T), y))) / V
                mpk(p) = Mp[p[ax[1]] + 1, D == 2 ? p[ax[2]] + 1 : 1]
                vec3(p, i) = i in op ? vo[i][p[i] + 1] : fill(T(hotC !== nothing && i == hotC.symAx ? 2 : 1), nn[i])
                sep = g * maximum(p -> wP(p) * mpk(p) * prod(o -> maximum(vo[o][p[o] + 1]), op), pats)
                isfinite(sep) && sep < bv && (bv = sep; best = [(g * wP(p) * mpk(p), vec3(p, 1), vec3(p, 2), vec3(p, 3)) for p in pats])
            end
            best
        end
        # in chunks of 1024 terms: the (n2·n3) × #terms matrix B of all terms at once was GBs at 64³ (d2 1+i > 7.6 GB)
        function accum(tl)
            E = zeros(T, nn...)
            isempty(tl) && return E
            # (the block-threaded kernel, 1024 terms at a time)
            for r in Iterators.partition(eachindex(tl), 1024)
                V1 = reduce(hcat, [tl[q][2] for q in r]); V2 = reduce(hcat, [tl[q][3] for q in r]); V3 = reduce(hcat, [tl[q][4] for q in r])
                krAddAB!(E, V1, V2, V3, [tl[q][1] for q in r])
            end
            E
        end
        # stages: (name, panel list, bound function, recompute on changed panels)
        stg = Any[]
        isempty(segP) || push!(stg, (:segment, [(lo, hi, nG) for (lo, hi) in segP], (lo, hi, n) -> segT(lo, hi),
                                     (old, new) -> begin to, wo = segNod([(a, b) for (a, b, _) in old]); segAdd!(to, -wo); tn_, tw2 = segNod([(a, b) for (a, b, _) in new]); segAdd!(tn_, tw2) end))
        for (g, (r, pl)) in enumerate(ordRec)
            push!(stg, (Symbol("order", g), copy(pl), (lo, hi, n) -> ordT(r, lo, hi, n),
                        (old, new) -> begin r.PPa === r.PP || return; Io, _, _ = ordCon(r, old); In, _, _ = ordCon(r, new); ordDelta!(r, Dict(pt => In[pt] .- Io[pt] for pt in pats)) end))
        end
        isempty(evCP) || push!(stg, (:evRay, [(lo, hi, nG) for (lo, hi) in evCP], (lo, hi, n) -> evT(lo, hi),
                                     (old, new) -> begin to, wo = evNod([(a, b) for (a, b, _) in old]); evCAdd!(to, -wo); tn_, tw2 = evNod([(a, b) for (a, b, _) in new]); evCAdd!(tn_, tw2) end))
        pls = [st[2] for st in stg]; nb = 0; worst = zero(T)
        bm_ = BMemo()   # the panels' bounds memoized across the rounds
        wPv = Ref((T(Inf), false))
        for round_ in 0:12
            bnds = tMap2(q -> bmo(bm_, q, () -> stg[q[1]][3](q[2]...)), [[(k, pn) for pn in pls[k]] for k in eachindex(stg)])
            tl = Tuple{T,Vector{T},Vector{T},Vector{T}}[]; bad0 = Tuple{Int,Int}[]
            for k in eachindex(stg), (q, b) in enumerate(bnds[k])
                b === nothing ? push!(bad0, (k, q)) : append!(tl, b)
            end
            E = accum(tl); Er = E; for i in ax; symC == i && continue; Er = reverse(Er; dims = i); end
            rr = certRR(E, Er, Smin, tol)   # (function barrier)
            zAll === nothing || (rr[vec(zAll)] .= 0)
            worst = maximum(rr)
            if isempty(bad0) && worst <= 1
                for (k, st) in enumerate(stg)
                    nm_ = string(st[1]); startswith(nm_, "order") && (ordPlF[parse(Int, nm_[6:end])] = pls[k])
                end
                return
            end
            crtStl(wPv, worst, round_, "complex-path (route C)")
            round_ == 12 && error("blcFil: complex-path (route C) quadrature not certified at the target: $(count(>(1), rr)) offsets, worst bound/target $(Float64(worst))")
            jc = CartesianIndices(E)[argmax(rr)]; jr = CartesianIndex(ntuple(i -> i in ax && i != symC ? nn[i] + 1 - jc[i] : jc[i], 3))
            sc_ = Smin[jc]; ntot = sum(length, pls)
            for (k, st) in enumerate(stg)
                split = Set(q for (k_, q) in bad0 if k_ == k)
                for (q, b) in enumerate(bnds[k])
                    b === nothing && continue
                    bq = sum(t_[1] * (t_[2][jc[1]] * t_[3][jc[2]] * t_[4][jc[3]] + t_[2][jr[1]] * t_[3][jr[2]] * t_[4][jr[3]]) / 2 for t_ in b)
                    bq / sc_ > tol / (2 * ntot) && push!(split, q)
                end
                isempty(split) && continue
                old = [pls[k][q] for q in split]; newP = Tuple{T,T,Int}[]
                # an order-ray panel below nG nodes first gets nG nodes: with n fixed, halving stops paying once the best ellipse is the
                # largest of RHOS (ρ = 8: the bound tends to (64/15)·length·M·8^{-2n}/63, a floor independent of the panel count; e.g.
                # n = 4, 1.03e-30 against a target 3.6e-33 at the nearly vanishing m = P/2 offset of θ = π - 1e-5, 30948 panels after 12 rounds)
                for (lo, hi, n) in old
                    if n < nG
                        push!(newP, (lo, hi, nG))
                    else
                        c = (lo + hi) / 2; push!(newP, (lo, c, n)); push!(newP, (c, hi, n))
                    end
                end
                st[4](old, newP); nb += length(old)
                pls[k] = vcat([pls[k][q] for q in eachindex(pls[k]) if !(q in split)], newP)
            end
        end
    end
    cCert!()
    # the hot-axis groups (d < 3): on the union of the members' certified panels of the common ray
    if !isempty(grpOf)
        recIdx = Dict(r.g => q for (q, (r, _)) in enumerate(ordRec))
        xgn, wgn = glq(T, nG)
        for ld in unique(v.leader for v in values(grpOf))
            grp = grpOf[ld]
            rL = ordRec[recIdx[ld]][1]
            bps = sort(unique(vcat([[e for (lo, hi, _) in get(ordPlF, recIdx[g], ordRec[recIdx[g]][2]) for e in (lo, hi)] for g in grp.members]...)))
            tr = Complex{T}[]; wr = Complex{T}[]
            for q in 1:length(bps)-1
                tm = τc + (bps[q + 1] + bps[q]) / 2 * rL.ep; Δ = (bps[q + 1] - bps[q]) / 2 * rL.ep
                for j in 1:nG; push!(tr, tm + Δ * xgn[j]); push!(wr, Δ * wgn[j]); end
            end
            ew = exp.(-rL.p .* tr) .* wr
            # per periodic axis j the sum/difference components: Σ_v PP_v(x) E_v(t) = Σ_c P_c(x) E_c(t) with (P_s, E_s) = (PP₁ + PP₂, (E₁ + E₂)/2),
            # (P_d, E_d) = (PP₁ - PP₂, (E₁ - E₂)/2), E_v = e^{-4π²(ν_v² - ν_L²) t}: each factor formed in Double64 and rounded once, so the group is
            # Σ_{c1,c2} [P_c1 ⊗ P_c2] ⊗ I_c1c2 with I = Σ_t e^{-p_L t} w E_c1 E_c2 K_open (the ordA/ordB structure; per-node Double64 rows over
            # every periodic offset cost 100 s at 64³ d2)
            comps = map(1:D) do j
                i = ax[j]; νL = nuP(TD, ld[j], hotC.persD[i]); vs = grp.vals[j]
                Pv = [[hotC.ppD(v, i, jr) for jr in eachindex(xs[i])] for v in vs]
                Ev = [exp.(-4 * TD(π)^2 * (nuP(TD, v, hotC.persD[i]) - νL) * (nuP(TD, v, hotC.persD[i]) + νL) .* Complex{TD}.(tr)) for v in vs]
                if length(vs) == 1
                    [(P = [Complex{T}.(q_) for q_ in Pv[1]], E = Complex{T}.(Ev[1]), Pa = [abs.(Complex{T}.(q_)) for q_ in Pv[1]])]
                else
                    [(P = [Complex{T}.(Pv[1][r_] .+ Pv[2][r_]) for r_ in eachindex(Pv[1])], E = Complex{T}.((Ev[1] .+ Ev[2]) ./ 2), Pa = [abs.(Complex{T}.(Pv[1][r_])) .+ abs.(Complex{T}.(Pv[2][r_])) for r_ in eachindex(Pv[1])]),
                     (P = [Complex{T}.(Pv[1][r_] .- Pv[2][r_]) for r_ in eachindex(Pv[1])], E = Complex{T}.((Ev[1] .- Ev[2]) ./ 2), Pa = [abs.(Complex{T}.(Pv[1][r_])) .+ abs.(Complex{T}.(Pv[2][r_])) for r_ in eachindex(Pv[1])])]
                end
            end
            combos = D == 1 ? [(c,) for c in comps[1]] : [(c1, c2) for c1 in comps[1], c2 in comps[2]]
            nper = D == 1 ? (nn[ax[1]],) : (nn[ax[1]], nn[ax[2]]); nop = D == 1 ? (nn[op[1]], nn[op[2]]) : (nn[op[1]],)
            Ios = Vector{Any}(nothing, length(combos)); IoAs = Vector{Any}(nothing, length(combos))
            for rg in Iterators.partition(eachindex(tr), 2048)
                trc = tr[rg]
                KK = Dict(o => axVecC(T, o, trc, xs, ms, axk, sTa, pers, 0, tol; ab = true) for o in op)
                for (ic, cb) in enumerate(combos)
                    ewc = ew[rg] .* (D == 1 ? cb[1].E[rg] : cb[1].E[rg] .* cb[2].E[rg]); ewa = abs.(ewc) .* (1 + rD)
                    part = Dict(pt => (D == 1 ? vec(cMul(KK[op[1]][1][pt[op[1]] + 1] .* transpose(ewc), transpose(KK[op[2]][1][pt[op[2]] + 1]))) : vec(cMul(KK[op[1]][1][pt[op[1]] + 1], reshape(ewc, :, 1)))) for pt in pats)
                    partA = Dict(pt => (D == 1 ? vec((KK[op[1]][2][pt[op[1]] + 1] .* transpose(ewa)) * transpose(KK[op[2]][2][pt[op[2]] + 1])) : vec(KK[op[1]][2][pt[op[1]] + 1] * reshape(ewa, :, 1))) for pt in pats)
                    if Ios[ic] === nothing
                        Ios[ic] = part; IoAs[ic] = partA
                    else
                        for pt in pats; Ios[ic][pt] .+= part[pt]; IoAs[ic][pt] .+= partA[pt]; end
                    end
                end
            end
            for (ic, cb) in enumerate(combos)
                Io = Ios[ic]; IoA = IoAs[ic]
                for pt in pats
                    wv = pt == (0, 0, 0) ? 4 * T(π)^2 : f2i
                    cp_ = D == 1 ? [q_[pt[ax[1]] + 1] for q_ in cb[1].P] : vec([a_[pt[ax[1]] + 1] * b_[pt[ax[2]] + 1] for a_ in cb[1].P, b_ in cb[2].P])
                    ca_ = D == 1 ? [q_[pt[ax[1]] + 1] for q_ in cb[1].Pa] : vec([a_[pt[ax[1]] + 1] * b_[pt[ax[2]] + 1] for a_ in cb[1].Pa, b_ in cb[2].Pa])
                    X = (cp_ .* wv) * transpose(Io[pt])
                    addPerm!(CC[pt], reshape(X, nper..., nop...), invperm([ax..., op...]))
                    Xa = (ca_ .* abs(wv)) * transpose(IoA[pt])
                    addPerm!(CCa[pt], reshape(Xa, nper..., nop...), invperm([ax..., op...]))
                end
            end
        end
    end
    for p in pats; fold!(CC[p]); end
    # C(-θ; m) = C(θ; -m): periodic axes reversed (ms symmetric), open axes by parity (-1)^{p_o}
    for p in pats
        # (the symAx rows Σ(k) = C(k) + C(-k) are their own reflection)
        # the reflection as a reversed view, not one reversed copy per axis
        rv_ = ntuple(i -> (i in ax && i != symC) ? (size(CC[p], i):-1:1) : (1:size(CC[p], i)), 3)
        Cr = view(CC[p], rv_...)
        sg = iseven(sum(o -> p[o], op; init = 0)) ? one(T) : -one(T)
        CH[p] .+= (CC[p] .+ sg .* conj.(Cr)) ./ 2
        CA[p] .+= (CC[p] .- sg .* conj.(Cr)) ./ (2im)
        if MH !== nothing
            Cra = view(CCa[p], rv_...)
            MH[p] .+= (CCa[p] .+ Cra) ./ 2; MA[p] .+= (CCa[p] .+ Cra) ./ 2
        end
    end
    nothing
end

# the mixed route's per-entry choice (a function barrier over typed arrays, threaded over the last offset index; the same rule): every
# entry of the route C sub-block (offsets beyond lo_) takes route C where its λ_e is smaller
function selEnt!(HR::Array{S,5}, AR::Array{S,5}, HC::Array{S,5}, AC::Array{S,5}, eRH::Array{Float32,4}, eRA::Array{Float32,4}, eCH::Array{Float32,4}, eCA::Array{Float32,4}, lo_) where {S}
    nn = size(HR)[3:5]
    tFor(nn[3]) do k
        k > lo_[3] || return
        for j2 in (lo_[2] + 1):nn[2], j1 in (lo_[1] + 1):nn[1]
            c1 = j1 - lo_[1]; c2 = j2 - lo_[2]; c3 = k - lo_[3]
            for (q, (a, b)) in enumerate(((1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)))
                @inbounds if eCH[q, c1, c2, c3] < eRH[q, j1, j2, k]
                    HR[a, b, j1, j2, k] = HC[a, b, c1, c2, c3]; HR[b, a, j1, j2, k] = HC[b, a, c1, c2, c3]; eRH[q, j1, j2, k] = eCH[q, c1, c2, c3]
                end
                @inbounds if eCA[q, c1, c2, c3] < eRA[q, j1, j2, k]
                    AR[a, b, j1, j2, k] = AC[a, b, c1, c2, c3]; AR[b, a, j1, j2, k] = AC[b, a, c1, c2, c3]; eRA[q, j1, j2, k] = eCA[q, c1, c2, c3]
                end
            end
        end
    end
end

# filD64's scan for flagged entries (a function barrier over typed arrays; the same rule and order)
function flgScan(H::Array{S,5}, A::Array{S,5}, λH::Array{Float32,4}, λA::Array{Float32,4}, ms, s, ax, P, θ) where {S}
    ents = ((1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3))
    flg = falses(size(H)[3:5]...); rmn = 1.0; nfl = 0
    for I in CartesianIndices(flg)
        any(q -> λH[q, I] > LAMF || λA[q, I] > LAMF, 1:6) || continue
        for (X, λ) in ((H, λH), (A, λA))
            mx = maximum(abs, view(X, :, :, I)); mx == 0 && continue
            for (q, (a, b)) in enumerate(ents)
                λ[q, I] > LAMF || continue
                nearSym(ms, s, ax, P, θ, I, a, b) || continue
                ra = abs(X[a, b, I]) / mx
                (ra >= RMINF || !isfinite(λ[q, I])) || continue
                flg[I] = true; nfl += 1
                ra > 0 && (rmn = min(rmn, ra))
            end
        end
    end
    flg, rmn, nfl
end

# the reflection zeros and the per-entry rounding factors λ_e of one entry (a, b) over every offset (a function barrier, on the threads)
function finEnt!(λH, λA, Hab::Array{Complex{T},3}, Aab, SHab, SAab, zKl, a, b, q) where {T}
    cI = CartesianIndices(Hab)
    tFor(cld(length(cI), 4096)) do c
        for l in ((c - 1) * 4096 + 1):min(c * 4096, length(cI))
            I = cI[l]
            if any(i -> zKl[i][I[i]] != 0 && zSig(a, b, i) == zKl[i][I[i]], 1:3)
                Hab[I] = 0; Aab[I] = 0
            else
                λH[q, I] = Hab[I] != 0 ? Float32(min(SHab[I] / abs(Hab[I]), T(floatmax(Float32)))) : SHab[I] > 0 ? Inf32 : 0f0
                λA[q, I] = Aab[I] != 0 ? Float32(min(SAab[I] / abs(Aab[I]), T(floatmax(Float32)))) : SAab[I] > 0 ? Inf32 : 0f0
            end
        end
    end
    nothing
end

# one propagating-order pair's open-axis integrals (I₊, I₋, D = (I₊ - I₋)/2, Σ|terms| of D): the pole pair form, the common-ray form, or the two
# orders as they are when their kinds differ
function pairInt(::Type{T}, axk, op, xs, rp, rm, ysp, τ, α, tol, nG, s, plp, plm) where {T}
    if rp.pinf.kind == :pole && rm.pinf.kind == :pole
        return spPair2(T, axk, op, xs, rp, rm, ysp, τ, α, tol, nG, s, plp, plm)
    elseif rp.pinf.kind == :ray && rm.pinf.kind == :ray
        # the common ray: the + order's direction; its panels merged with the - order's (both certified on their own rays)
        bpr = sort(unique(vcat([e_ for (lo, hi) in plp for e_ in (lo, hi)], [e_ for (lo, hi) in plm for e_ in (lo, hi)])))
        return spPairRay(T, axk, op, xs, rp, rm, ysp, τ, tol, nG, s, rp.pinf.e, [(bpr[q], bpr[q + 1]) for q in 1:length(bpr)-1])
    end
    Ip = spFil(T, axk, op, xs, rp.p, rp.up, rp.cont, ysp, τ, α, tol, nG, rp.ap, s; pans = plp)[1]
    Im_ = spFil(T, axk, op, xs, rm.p, rm.up, rm.cont, ysp, τ, α, tol, nG, rm.ap, s; pans = plm)[1]
    Ip, Im_, Dict(k => (Ip[k] .- Im_[k]) ./ 2 for k in keys(Ip)), Dict(k => abs.(Ip[k]) ./ 2 .+ abs.(Im_[k]) ./ 2 for k in keys(Ip))
end
