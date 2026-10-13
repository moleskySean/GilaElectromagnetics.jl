# Closed-form moments I_m = int_A int_B |x - y|^m dS' dS of the face pairs of
# touching cuboid cells, m = -1, 0, 1, ... The 4D integral collapses onto the
# separation w = x - y with a per-axis box-box convolution weight, so every pair
# is a positively weighted sum of axis-aligned boxes (parBxs): 2D with a frozen
# offset (box2), 3D with none (box3). Generic in the number type.
using Base.Threads
using DoubleFloats
using StaticArrays

# the asinh difference log((t2 + sqrt(t2^2 + dsq))/(t1 + sqrt(t1^2 + dsq))), avoiding
# catastrophic cancellation
function asnD(t1::T, t2::T, dsq::T) where {T}
    iszero(dsq) && return log(t2 / t1)
    A1 = sqrt(t1^2 + dsq); A2 = sqrt(t2^2 + dsq)
    x = ((t2 - t1) + (t2 - t1) * (t2 + t1) / (A2 + A1)) / (t1 + A1)
    return x < 1//2 ? log1p(x) : log((t2 + A2) / (t1 + A1))
end

# "cosh minus": d cosh(d) - sinh(d), avoiding catastrophic cancellation
function cshMns(d::T) where {T}
    d > 3 && return d * cosh(d) - sinh(d)
    s = zero(T); trm = d^3 / T(3); k = 1
    while k < 500
        s += trm
        k += 1
        trm *= d^2 * T(k) / (T(k - 1) * T(2k) * T(2k + 1))
        abs(trm) <= eps(T) * abs(s) && (s += trm; break)
    end
    return s
end

# the centred rung int_t1^t2 ((t1 + t2)/2 - t)/sqrt(t^2 + dsq) dt, zt its asnD
function ctrM1(t1::T, t2::T, dsq::T, zt::T) where {T}
    A1 = sqrt(t1^2 + dsq)
    dl = zt / 2
    return 2 * (t1 * cosh(dl) + A1 * sinh(dl)) * cshMns(dl)
end

# the falling-ladder rung int_t1^t2 (t2 - t)/sqrt(t^2 + dsq) dt, S its asnD and H the
# difference sqrt(t2^2 + dsq) - sqrt(t1^2 + dsq); avoiding catastrophic cancellation
function ladFal(t1::T, t2::T, dsq::T, S::T, H::T) where {T}
    iszero(dsq) && iszero(t1) && return T(Inf)
    S <= 2 && return (t2 - t1) / 2 * S + ctrM1(t1, t2, dsq, S)
    return t2 * S - H
end

# the ladder in m: for m = m0, m0 + 2, ... mMax the three rungs int_t1^t2 w(t)
# (t^2 + dsq)^(m/2) dt with w = 1, t - t1 and t2 - t
function lad1(m0::Int, mMax::Int, t1::T, t2::T, dsq::T) where {T}
    n = (mMax - m0) ÷ 2 + 1
    out = Vector{NTuple{3,T}}(undef, n)
    L = t2 - t1; h = L / 2
    q1 = t1^2 + dsq; q2 = t2^2 + dsq
    A1 = sqrt(q1); A2 = sqrt(q2)
    if m0 == -1
        S = asnD(t1, t2, dsq)
        H = L * (t2 + t1) / (A2 + A1)                    # A2 - A1, rationalized
        rs = iszero(t1) ? H : H - t1 * S
        out[1] = (S, rs, ladFal(t1, t2, dsq, S, H))
    else
        out[1] = (L, L^2 / 2, L^2 / 2)
    end
    for k in 2:n
        m = m0 + 2 * (k - 1)
        p2 = isodd(m) ? A2 * q2^((m - 1) ÷ 2) : q2^(m ÷ 2)
        p1 = isodd(m) ? A1 * q1^((m - 1) ÷ 2) : q1^(m ÷ 2)
        po, pr, pf = out[k - 1]
        cd = iszero(dsq) ? zero(T) : T(m) * dsq
        jo = (t2 * p2 - t1 * p1 + cd * po) / T(m + 1)
        jr = (L * t2 * p2 - t1 * jo + cd * pr) / T(m + 2)
        jf = (t2 * jo - L * t1 * p1 + cd * pf) / T(m + 2)
        out[k] = (jo, jr, jf)
    end
    return out
end

# lad1's rungs at offset dh minus the same rungs at offset dl, formed without either:
# dm1 the w = 1 rung at m = -1, the other three the w = 1, t - t1, t2 - t rungs at m = 1
struct OffDif{T}
    dm1::T                                # asinh difference at m = -1, negative
    dOne::T
    dRis::T
    dFal::T
end

# "pick combination": dh^2 Jh - dl^2 Jl, by whichever of its two exact forms cancels less
@inline function pikCmb(dhsq::T, dlsq::T, ddsq::T, D::T, Jh::T, Jl::T) where {T}
    aA = dhsq * abs(Jh) + dlsq * abs(Jl)
    aB = dhsq * abs(D) + ddsq * abs(Jl)
    return aA <= aB ? dhsq * Jh - dlsq * Jl : dhsq * D + ddsq * Jl
end

function offDif(t1::T, t2::T, dlsq::T, dhsq::T, ddsq::T) where {T}
    A1l = sqrt(t1^2 + dlsq); A2l = sqrt(t2^2 + dlsq)
    A1h = sqrt(t1^2 + dhsq); A2h = sqrt(t2^2 + dhsq)
    L = t2 - t1; h = L / 2; sq = L * (t2 + t1)
    h2 = ddsq / (A2h + A2l)                                          # A2h - A2l
    h1 = iszero(A1h + A1l) ? zero(T) : ddsq / (A1h + A1l)            # A1h - A1l
    Sh = asnD(t1, t2, dhsq); Hh = sq / (A2h + A1h)
    hyp = Sh <= 2 && !(iszero(dlsq) && iszero(t1))
    Ch = hyp ? ctrM1(t1, t2, dhsq, Sh) : zero(T)
    jhO = Sh
    jhR = iszero(t1) ? Hh : Hh - t1 * Sh
    jhF = ladFal(t1, t2, dhsq, Sh, Hh)
    if iszero(dlsq)
        dmO = T(-Inf); dmR = T(-Inf); dmF = T(-Inf)
        cO = dhsq * jhO; cR = dhsq * jhR; cF = dhsq * jhF
    else
        Sl = asnD(t1, t2, dlsq); Hl = sq / (A2l + A1l)
        jlO = Sl
        jlR = iszero(t1) ? Hl : Hl - t1 * Sl
        jlF = ladFal(t1, t2, dlsq, Sl, Hl)
        num = ddsq * (t2 / (A1h + A1l) - t1 / (A2h + A2l) + sq / (A1h * A2l + A2h * A1l))
        X = num / ((t1 + A1h) * (t2 + A2l))
        dmO = X < 1//2 ? log1p(-X) :
              log(((t2 + A2h) * (t1 + A1l)) / ((t1 + A1h) * (t2 + A2l)))
        gap = sq * (h2 + h1) / ((A2h + A1h) * (A2l + A1l))            # positive
        dmR = -gap - t1 * dmO
        dmFa = t2 * dmO + gap
        Cl = hyp && Sl <= 2 ? ctrM1(t1, t2, dlsq, Sl) : zero(T)
        useB = hyp && Sl <= 2 &&
               abs(h * dmO) + abs(Ch) + abs(Cl) < abs(t2 * dmO) + gap
        dmF = useB ? h * dmO + (Ch - Cl) : dmFa
        cO = pikCmb(dhsq, dlsq, ddsq, dmO, jhO, jlO)
        cR = pikCmb(dhsq, dlsq, ddsq, dmR, jhR, jlR)
        cF = pikCmb(dhsq, dlsq, ddsq, dmF, jhF, jlF)
    end
    dO = (t2 * h2 - t1 * h1 + cO) / T(2)
    dR = (L * t2 * h2 - t1 * dO + cR) / T(3)
    dF = (t2 * dO - L * t1 * h1 + cF) / T(3)
    return OffDif{T}(dmO, dO, dR, dF)
end

# asinh(z) - z, avoiding catastrophic cancellation
function asnMns(z::T) where {T}
    z > 1//2 && return asinh(z) - z
    s = zero(T); t = -z^3 / 6; k = 1
    while k < 400
        s += t; k += 1
        t *= -z^2 * T(2k - 1)^2 / (T(2k) * T(2k + 1))
        abs(t) <= eps(T) * abs(s) && (s += t; break)
    end
    return s
end
# x - atan(x), avoiding catastrophic cancellation
function atnMns(x::T) where {T}
    x > 1//2 && return x - atan(x)
    s = zero(T); t = x^3 / 3; k = 1
    while k < 400
        s += t; k += 1
        t *= -x^2 * T(2k - 1) / T(2k + 1)
        abs(t) <= eps(T) * abs(s) && (s += t; break)
    end
    return s
end

# int_0^p int_0^q du dv/sqrt(u^2 + v^2 + c^2), avoiding catastrophic cancellation
function potCrn(p::T, q::T, c::T) where {T}
    csq = c^2
    al = sqrt(p^2 + csq); be = sqrt(q^2 + csq); R = sqrt(p^2 + q^2 + csq)
    if !iszero(c) && 2 * p <= al && 2 * q <= be
        return p * q * (1 / be + q^2 / (al * R * (R + al))) +
               p * asnMns(q / al) + q * asnMns(p / be) + c * atnMns(p * q / (c * R))
    end
    at = iszero(c) ? zero(T) : c * atan(p * q / (c * R))
    return p * asinh(q / al) + q * asinh(p / be) - at
end

# the Y with atan(Y) = atan(t2 s/(c R2)) - atan(t1 s/(c R1)), R = sqrt(t^2 + s^2 + c^2)
@inline function atnY(t1::T, t2::T, s::T, c::T) where {T}
    bs = s^2 + c^2
    R1 = sqrt(t1^2 + bs); R2 = sqrt(t2^2 + bs)
    return c * s * bs * (t2 - t1) * (t2 + t1) /
           ((t2 * R1 + t1 * R2) * (c^2 * R1 * R2 + t1 * t2 * s^2))
end

# the four-corner difference of atan(u v/(c sqrt(u^2 + v^2 + c^2))) over
# [u1,u2] x [v1,v2], avoiding catastrophic cancellation
function atnDD(u1::T, u2::T, v1::T, v2::T, c::T) where {T}
    at(u, v) = atan(u * v / (c * sqrt(u^2 + v^2 + c^2)))
    a22 = at(u2, v2); a12 = at(u1, v2); a21 = at(u2, v1); a11 = at(u1, v1)
    nv = (a22 - a12) - (a21 - a11)
    !iszero(nv) && (a22 + a12 + a21 + a11) <= 4 * abs(nv) && return nv
    Y1 = atnY(u1, u2, v1, c); Y2 = atnY(u1, u2, v2, c)
    Z1 = atnY(v1, v2, u1, c); Z2 = atnY(v1, v2, u2, c)
    gY = abs(Y2 - Y1) * (Z1 + Z2); gZ = abs(Z2 - Z1) * (Y1 + Y2)
    return gY >= gZ ? atan((Y2 - Y1) / (1 + Y1 * Y2)) :
                      atan((Z2 - Z1) / (1 + Z1 * Z2))
end

# int_u1^u2 int_v1^v2 du dv/sqrt(u^2 + v^2 + c^2), avoiding catastrophic cancellation
function potBox(u1::T, u2::T, v1::T, v2::T, c::T) where {T}
    iszero(u1) && iszero(v1) && return potCrn(u2, v2, c)
    csq = c^2
    a1 = u1^2 + csq; a2 = u2^2 + csq
    b1 = v1^2 + csq; b2 = v2^2 + csq
    s = zero(T)
    if iszero(u1)
        s += u2 * asnD(v1, v2, a2)
    else
        hu = (u2 - u1) / 2; um = (u2 + u1) / 2
        z2 = asnD(v1, v2, a2); z1 = asnD(v1, v2, a1)
        s += hu * (z2 + z1) + um * offDif(v1, v2, a1, a2, (u2 - u1) * (u2 + u1)).dm1
    end
    if iszero(v1)
        s += v2 * asnD(u1, u2, b2)
    else
        hv = (v2 - v1) / 2; vm = (v2 + v1) / 2
        z2 = asnD(u1, u2, b2); z1 = asnD(u1, u2, b1)
        s += hv * (z2 + z1) + vm * offDif(u1, u2, b1, b2, (v2 - v1) * (v2 + v1)).dm1
    end
    iszero(c) || (s -= c * atnDD(u1, u2, v1, v2, c))
    return s
end

# box2Lad's W at a single m, from a finitely truncated Taylor series about the box centre:
# exact and elementary, not a quadrature. Nothing where the series does not converge
function box2Tay(m::Int, u1::T, u2::T, v1::T, v2::T, c::T;
                 nMax::Int = 140, tol::T = eps(T)/16, cndMax::T = T(4)) where {T}
    hu0 = (u2 - u1)/2; um0 = (u2 + u1)/2
    hv0 = (v2 - v1)/2; vm0 = (v2 + v1)/2
    scl = max(um0, vm0, c)
    (scl > 0 && hu0 > 0 && hv0 > 0) || return nothing
    aa = hu0^2 + hv0^2; bb = 2*(um0*hu0 + vm0*hv0); rr = um0^2 + vm0^2 + c^2
    lam = (-bb + sqrt(bb^2 + 4*aa*rr))/(2*aa)
    lam > 1//2 || return nothing           # certainly hopeless, do not even try
    e = exponent(scl)
    hu = ldexp(hu0, -e); um = ldexp(um0, -e)
    hv = ldexp(hv0, -e); vm = ldexp(vm0, -e); cs = ldexp(c, -e)
    rsq = um^2 + vm^2 + cs^2

    mus = Vector{NTuple{3,T}}();  nus = Vector{NTuple{3,T}}()
    function pushMom!(v, h, i)
        if iseven(i)
            ev = 2*h^(i+1)/T(i+1)
            push!(v, (ev, h*ev, h*ev))
        else
            od = 2*h^(i+2)/T(i+2)
            push!(v, (zero(T), od, -od))
        end
    end
    cf = Vector{Vector{T}}()                         # cf[n+1][i+1] = c_{i, n-i}
    lvs = Vector{Matrix{T}}()
    runA = zeros(T, 3, 3); relPr = one(T); nUse = -1
    for n in 0:nMax
        pushMom!(mus, hu, n); pushMom!(nus, hv, n)
        cn = Vector{T}(undef, n + 1)
        if n == 0
            cn[1] = rsq^(T(m)/2)
        else
            s = T(m - 2*(n-1))*vm*cf[n][1]
            n >= 2 && (s += T(m - n + 2)*cf[n-1][1])
            cn[1] = s/(rsq*T(n))
            for i in 1:n
                ip = i - 1; j = n - i
                s = T(m - 2*ip)*um*cf[n][ip+1]
                ip >= 1 && (s += T(m - ip + 1)*cf[n-1][ip])
                j >= 1 && (s -= 2*vm*T(i)*cf[n][i+1])
                j >= 2 && (s -= T(i)*cf[n-1][i+1])
                cn[i+1] = s/(rsq*T(i))
            end
        end
        push!(cf, cn)
        L = zeros(T, 3, 3); A = zeros(T, 3, 3)
        for i in 0:n
            j = n - i
            cij = cn[i+1]
            iszero(cij) && continue
            mu = mus[i+1]; nu = nus[j+1]
            for a in 1:3, b in 1:3
                t = cij*mu[a]*nu[b]
                L[a,b] += t; A[a,b] += abs(t)
            end
        end
        push!(lvs, L); runA .+= A
        all(isfinite, runA) || return nothing
        rel = zero(T)
        for a in 1:3, b in 1:3
            iszero(runA[a,b]) && return nothing
            rel = max(rel, A[a,b]/runA[a,b])
        end
        if n >= 3 && rel <= tol && relPr <= tol   # two consecutive small levels
            nUse = n; break
        end
        # single-level ratios oscillate, so project on the geometric mean
        if n >= 10 && n < nMax
            r = max(rel, eps(T)^2)^(one(T)/T(n))
            (r >= 1 || log(tol)/log(r) > nMax) && return nothing
        end
        relPr = rel
    end
    nUse < 0 && return nothing
    W = zeros(T, 3, 3)
    for n in nUse:-1:0, a in 1:3, b in 1:3               # smallest levels first
        W[a,b] += lvs[n+1][a,b]
    end
    dg = (0, 1, 1); cnd = zero(T)
    for a in 1:3, b in 1:3
        (W[a,b] > 0 && runA[a,b] <= cndMax*W[a,b]) || return nothing
        cnd = max(cnd, runA[a,b]/W[a,b])
        W[a,b] = ldexp(W[a,b], e*(m + 2 + dg[a] + dg[b]))
    end
    return (W, nUse, cnd)
end

# the 2D ladder: W[k][a,b] = int_u1^u2 int_v1^v2 w_a(u) w_b(v) (u^2 + v^2 + c^2)^(m/2),
# m = m0 + 2(k-1) up to mMax and w = (1, u - u1, u2 - u), i.e. flat, rising and falling
function box2Lad(m0::Int, mMax::Int, u1::T, u2::T, v1::T, v2::T, c::T;
                 tayOn::Bool = true) where {T}
    @assert m0 == -1 || m0 == 0
    n = (mMax - m0) ÷ 2 + 1
    csq = c^2
    a1 = u1^2 + csq; a2 = u2^2 + csq
    b1 = v1^2 + csq; b2 = v2^2 + csq
    Lu = u2 - u1; Lv = v2 - v1
    hu = Lu / 2; hv = Lv / 2
    Jv2 = lad1(m0, mMax, v1, v2, a2)
    Jv1 = iszero(u1) ? Jv2 : lad1(m0, mMax, v1, v2, a1)
    Ju2 = lad1(m0, mMax, u1, u2, b2)
    Ju1 = iszero(v1) ? Ju2 : lad1(m0, mMax, u1, u2, b1)

    W = [zeros(T, 3, 3) for _ in 1:n]
    tayOK = false
    if m0 == -1
        WT = tayOn ? box2Tay(-1, u1, u2, v1, v2, c) : nothing
        if WT !== nothing
            W[1] .= WT[1]
            tayOK = true
            @goto seeded
        end
        P = potBox(u1, u2, v1, v2, c)
        Bv = offDif(v1, v2, a1, a2, Lu * (u2 + u1))
        Bu = offDif(u1, u2, b1, b2, Lv * (v2 + v1))
        Bvv = (Bv.dOne, Bv.dRis, Bv.dFal); Buu = (Bu.dOne, Bu.dRis, Bu.dFal)
        um = (u2 + u1) / 2; vm = (v2 + v1) / 2
        # both routes are exact; averaging restores the u<->v symmetry of the box
        A1r = (P, iszero(u1) ? Bvv[1] : Bvv[1] - u1 * P, hu * P + (um * P - Bvv[1]))
        A2r = (P, iszero(v1) ? Buu[1] : Buu[1] - v1 * P, hv * P + (vm * P - Buu[1]))
        for a in 1:3, b in 1:3
            r1 = b == 1 ? A1r[a] :
                 b == 2 ? (iszero(v1) ? Buu[a] : Buu[a] - v1 * A1r[a]) :
                          hv * A1r[a] + (vm * A1r[a] - Buu[a])
            r2 = a == 1 ? A2r[b] :
                 a == 2 ? (iszero(u1) ? Bvv[b] : Bvv[b] - u1 * A2r[b]) :
                          hu * A2r[b] + (um * A2r[b] - Bvv[b])
            W[1][a, b] = (r1 + r2) / 2
        end
        @label seeded
    else
        Iu = (Lu, Lu^2 / 2, Lu^2 / 2); Iv = (Lv, Lv^2 / 2, Lv^2 / 2)
        for a in 1:3, b in 1:3
            W[1][a, b] = Iu[a] * Iv[b]
        end
        tayOK = tayOn && box2Tay(0, u1, u2, v1, v2, c) !== nothing
    end

    wu2 = (one(T), Lu, zero(T)); wu1 = (one(T), zero(T), Lu)
    wv2 = (one(T), Lv, zero(T)); wv1 = (one(T), zero(T), Lv)
    eps2 = (0, 1, 1)
    ku = (zero(T), u1, -u2); kv = (zero(T), v1, -v2)

    for k in 2:n
        m = m0 + 2 * (k - 1)
        Wm = W[k]; Wp = W[k - 1]
        c2 = iszero(c) ? zero(T) : T(m) * csq
        bt = function (a, b)
            s = u2 * wu2[a] * Jv2[k][b] + v2 * wv2[b] * Ju2[k][a]
            iszero(u1) || (s -= u1 * wu1[a] * Jv1[k][b])
            iszero(v1) || (s -= v1 * wv1[b] * Ju1[k][a])
            return s
        end
        for (a, b) in ((1, 1), (2, 1), (3, 1), (1, 2), (1, 3),
                       (2, 2), (2, 3), (3, 2), (3, 3))
            s = bt(a, b) + c2 * Wp[a, b]
            a == 1 || (s -= ku[a] * Wm[1, b])
            b == 1 || (s -= kv[b] * Wm[a, 1])
            Wm[a, b] = s / T(m + 2 + eps2[a] + eps2[b])
        end
        # convergence does not depend on m, so a seeded series serves every rung
        if tayOK
            WT = box2Tay(m, u1, u2, v1, v2, c)
            WT === nothing || (Wm .= WT[1])
        end
    end
    return W
end

function wgtSpl(cw::T, dw::T, t1::T, t2::T) where {T}
    r = cw + dw * t1; s = cw + dw * t2
    iszero(r) && return (zero(T), dw, zero(T))
    iszero(s) && return (zero(T), zero(T), -dw)
    dw > 0 && return (r, dw, zero(T))
    return (s, zero(T), -dw)
end

function box2(m0::Int, mMax::Int, lo::Vector{T}, hi::Vector{T},
              cw::Vector{T}, dw::Vector{T}, off2::T; tayOn::Bool = true) where {T}
    u1, v1 = lo[1], lo[2]; u2, v2 = hi[1], hi[2]
    W = box2Lad(m0, mMax, u1, u2, v1, v2, sqrt(off2); tayOn = tayOn)
    pu = wgtSpl(cw[1], dw[1], u1, u2)
    pv = wgtSpl(cw[2], dw[2], v1, v2)
    out = Vector{T}(undef, length(W))
    for k in eachindex(W)
        s = zero(T)
        for a in 1:3, b in 1:3
            (iszero(pu[a]) || iszero(pv[b])) && continue
            s += pu[a] * pv[b] * W[k][a, b]
        end
        out[k] = s
    end
    return out
end

# "log minus": x - log(1 + x), avoiding catastrophic cancellation
function logMns(x::T) where {T}
    x > T(1) / 2 && return x - log1p(x)
    s = zero(T)
    t = x * x / 2
    k = 2
    while true
        s += t
        t = -t * x * k / (k + 1)
        k += 1
        (abs(t) <= eps(T) * abs(s) || k > 500) && break
    end
    return s
end

# sum_{i=0}^{n-1} A^i B^{n-1-i}, so A^n - B^n = (A^2-B^2) powSum/(A+B)
function powSum(A::T, B::T, n::Integer) where {T}
    s = zero(T)
    for i in 0:(n - 1)
        s += A^i * B^(n - 1 - i)
    end
    return s
end

# J_n = int_lo^hi (t^2 + d^2)^(n/2) dt, avoiding catastrophic cancellation
function powSeg(n::Integer, lo::T, hi::T, d::T) where {T}
    Sl = sqrt(lo^2 + d^2)
    Sh = sqrt(hi^2 + d^2)
    dl = hi - lo
    if n < -1                   # J_{-3} exact and positive; below it, downwards
        v = (hi^2 - lo^2) / (Sh * Sl * (hi * Sl + lo * Sh))
        k = -3
        while k > n
            k -= 2
            v = ((k + 3) * v - hi * Sh^(k + 2) + lo * Sl^(k + 2)) / ((k + 2) * d^2)
        end
        return v
    end
    if isodd(n)
        v = log1p((dl + (hi^2 - lo^2) / (Sh + Sl)) / (lo + Sl))
        k = -1
    else
        v = dl
        k = 0
    end
    while k < n
        k += 2
        tp = dl * Sh^k
        if !iszero(lo)
            tp += lo * dl * (hi + lo) * powSum(Sh, Sl, k) / (Sh + Sl)
        end
        v = (tp + k * d^2 * v) / (k + 1)
    end
    return v
end

# J_n(d1) - J_n(d0), e2 = d1^2 - d0^2, avoiding catastrophic cancellation
function powSegD(n::Integer, lo::T, hi::T, d0::T, d1::T, e2::T) where {T}
    S0l = sqrt(lo^2 + d0^2); S1l = sqrt(lo^2 + d1^2)
    S0h = sqrt(hi^2 + d0^2); S1h = sqrt(hi^2 + d1^2)
    pdf(A, B, k) = k > 0 ? e2 * powSum(A, B, k) / (A + B) :
                   k == 0 ? zero(T) :
                   -e2 * powSum(A, B, -k) / ((A + B) * (A * B)^(-k))
    if n < -1                      # downwards from the seed, low-order use only
        iszero(d0) && return powSeg(n, lo, hi, d1) - (hi^(n + 1) - lo^(n + 1)) / (n + 1)
        N = e2 * (hi / (S1l + S0l) - lo / (S0h + S1h) +
                  (hi^2 - lo^2) / (S1l * S0h + S1h * S0l))
        v = -log1p(N / ((hi + S1h) * (lo + S0l)))
        k = -1
        while k > n
            k -= 2
            v = (((k + 3) * v - hi * pdf(S1h, S0h, k + 2) + lo * pdf(S1l, S0l, k + 2)) /
                 (k + 2) - e2 * powSeg(k, lo, hi, d1)) / d0^2
        end
        return v
    end
    if isodd(n)
        if iszero(d0) && iszero(lo)
            v = T(NaN)                     # true value -Inf; killed by d0^2 = 0
        else
            N = e2 * (hi / (S1l + S0l) - lo / (S0h + S1h) +
                      (hi^2 - lo^2) / (S1l * S0h + S1h * S0l))
            v = -log1p(N / ((hi + S1h) * (lo + S0l)))
        end
        k = -1
    else
        v = zero(T)
        k = 0
    end
    while k < n
        k += 2
        tp = hi * e2 * powSum(S1h, S0h, k) / (S1h + S0h)
        if !iszero(lo)
            tp -= lo * e2 * powSum(S1l, S0l, k) / (S1l + S0l)
        end
        lw = iszero(d0) ? zero(T) : d0^2 * v
        v = (tp + k * (lw + e2 * powSeg(k - 2, lo, hi, d1))) / (k + 1)
    end
    return v
end

# wmnSeg's n = -1 seed, avoiding catastrophic cancellation
function wmnSed(lo::T, hi::T, d::T, q::T) where {T}
    Sl = sqrt(lo^2 + d^2)
    Sh = sqrt(hi^2 + d^2)
    dl = hi - lo
    B0 = lo + Sl
    kap = 1 + (lo + hi) / (Sl + Sh)
    x = dl * kap / B0
    hs = ((hi^2 - lo^2) * (hi^2 + lo^2) + d^2 * (hi^2 - lo^2)) / (hi * Sh + lo * Sl)
    Pp = dl * ((lo + hi) * dl + hs) / (B0 * (Sl + Sh))
    v = Pp - hi * logMns(x)
    if q != hi
        v += (q - hi) * powSeg(-1, lo, hi, d)
    end
    return v
end

# "weight minus": int_lo^hi (q - t)(t^2 + d^2)^(n/2) dt, with q >= hi
function wmnSeg(n::Integer, lo::T, hi::T, d::T, q::T) where {T}
    Sl = sqrt(lo^2 + d^2)
    Sh = sqrt(hi^2 + d^2)
    dl = hi - lo
    if isodd(n)
        v = wmnSed(lo, hi, d, q)
        k = -1
    else
        v = dl * (q - (hi + lo) / 2)
        k = 0
    end
    while k < n
        k += 2
        tp = q * powSeg(k, lo, hi, d)
        if q != hi
            tp += hi * (q - hi) * Sh^k
        end
        if !iszero(lo)
            tp -= lo * (q - lo) * Sl^k
        end
        v = (tp + k * d^2 * v) / (k + 2)
    end
    return v
end

# the solid angle the rectangle t x [a1,a2] x [b1,b2] subtends at the origin. Van
# Oosterom-Strackee per triangle: the four-corner arctangent difference cancels
function solAng(t::T, a1::T, a2::T, b1::T, b2::T) where {T}
    iszero(t) && return zero(T)
    num = t * (a2 - a1) * (b2 - b1)
    nrm(a, b) = sqrt(t^2 + a^2 + b^2)
    dt(a, b, c, d) = t^2 + a * c + b * d
    function tri(a1_, b1_, a2_, b2_, a3_, b3_)
        n1 = nrm(a1_, b1_); n2 = nrm(a2_, b2_); n3 = nrm(a3_, b3_)
        den = n1 * n2 * n3 + dt(a1_, b1_, a2_, b2_) * n3 +
              dt(a1_, b1_, a3_, b3_) * n2 + dt(a2_, b2_, a3_, b3_) * n1
        return 2 * atan(num, den)
    end
    return tri(a1, b1, a2, b1, a2, b2) + tri(a1, b1, a2, b2, a1, b2)
end

# Psi_n = int_a1^a2 int_b1^b2 (t^2 + a^2 + b^2)^(n/2) da db, the rectangle moment
function psiBox(n::Integer, t::T, a1::T, a2::T, b1::T, b2::T) where {T}
    r1 = sqrt(t^2 + a1^2); r2 = sqrt(t^2 + a2^2)
    s1 = sqrt(t^2 + b1^2); s2 = sqrt(t^2 + b2^2)
    function bdy(k)
        z = (a2 - a1) * powSeg(k, b1, b2, r2) + (b2 - b1) * powSeg(k, a1, a2, s2)
        iszero(a1) || (z += a1 * powSegD(k, b1, b2, r1, r2, a2^2 - a1^2))
        iszero(b1) || (z += b1 * powSegD(k, a1, a2, s1, s2, b2^2 - b1^2))
        return z
    end
    if n < -1
        # only the thin-axis expansion needs this, and there t = 0 kills the Psi_{n-2}
        iszero(t) || error("psiBox: n < -1 requires t = 0")
        return bdy(n) / (n + 2)
    end
    if isodd(n)
        v = bdy(-1) - t * solAng(t, a1, a2, b1, b2)
        k = -1
    else
        v = (a2 - a1) * (b2 - b1)
        k = 0
    end
    while k < n
        k += 2
        v = (bdy(k) + k * t^2 * v) / (k + 2)
    end
    return v
end

# psiBox with the weight q - a, q >= a2
function psiWmn(n::Integer, t::T, a1::T, a2::T, b1::T, b2::T, q::T) where {T}
    r1 = sqrt(t^2 + a1^2); r2 = sqrt(t^2 + a2^2)
    s1 = sqrt(t^2 + b1^2); s2 = sqrt(t^2 + b2^2)
    function bdy(k)
        wh = wmnSeg(k, a1, a2, s2, q)
        z = (b2 - b1) * wh
        iszero(b1) || (z += b1 * (wh - wmnSeg(k, a1, a2, s1, q)))
        q == a2 || (z += a2 * (q - a2) * powSeg(k, b1, b2, r2))
        return z
    end
    function led(k)
        p = psiBox(k, t, a1, a2, b1, b2)
        z = (q - a1) * p
        iszero(a1) || (z += a1 * (p - (q - a1) * powSeg(k, b1, b2, r1)))
        return z
    end
    if isodd(n)
        p = psiBox(-1, t, a1, a2, b1, b2)
        v = (q - a1) * p
        v += (iszero(a1) ? zero(T) : a1 * p) - powSegD(1, b1, b2, r1, r2, a2^2 - a1^2)
        k = -1
    else
        v = (b2 - b1) * (a2 - a1) * (q - (a1 + a2) / 2)
        k = 0
    end
    while k < n
        k += 2
        v = (led(k) + k * t^2 * v + bdy(k)) / (k + 3)
    end
    return v
end

# psiBox with the weight a - p, p <= a1
function psiWpl(n::Integer, t::T, a1::T, a2::T, b1::T, b2::T, p::T) where {T}
    r1 = sqrt(t^2 + a1^2); r2 = sqrt(t^2 + a2^2)
    v = powSegD(n + 2, b1, b2, r1, r2, a2^2 - a1^2) / (n + 2)
    iszero(p) || (v -= p * psiBox(n, t, a1, a2, b1, b2))
    return v
end

# K_m = int over the box [lo, hi] of (x^2 + y^2 + z^2)^(m/2) dV, by the divergence
# identity (m+3) K_m = sum_axes [hi_i Psi_m(hi_i) - lo_i Psi_m(lo_i)]
function box3Mom(m::Integer, lo, hi)
    T = eltype(lo)
    v = zero(T)
    for i in 1:3
        j = i % 3 + 1; k = j % 3 + 1
        ph = psiBox(m, hi[i], lo[j], hi[j], lo[k], hi[k])
        if iszero(lo[i])
            v += hi[i] * ph
        else
            pl = psiBox(m, lo[i], lo[j], hi[j], lo[k], hi[k])
            v += (hi[i] - lo[i]) * ph + lo[i] * (ph - pl)
        end
    end
    return v / (m + 3)
end

# box3Mom with the weight q - u, u the first coordinate and q >= hi[1]
function box3Wmn(m::Integer, lo, hi, q)
    T = eltype(lo)
    u1, u2 = lo[1], hi[1]; v1, v2 = lo[2], hi[2]; w1, w2 = lo[3], hi[3]
    K = box3Mom(m, lo, hi)
    if iszero(u1)
        s = q * K
    else
        pu = psiBox(m, u1, v1, v2, w1, w2)
        s = q * (K - (q - u1) * pu) + (q - u1)^2 * pu
    end
    q == u2 || (s += u2 * (q - u2) * psiBox(m, u2, v1, v2, w1, w2))
    bv = psiWmn(m, v2, u1, u2, w1, w2, q)
    s += (v2 - v1) * bv
    iszero(v1) || (s += v1 * (bv - psiWmn(m, v1, u1, u2, w1, w2, q)))
    bw = psiWmn(m, w2, u1, u2, v1, v2, q)
    s += (w2 - w1) * bw
    iszero(w1) || (s += w1 * (bw - psiWmn(m, w1, u1, u2, v1, v2, q)))
    return s / (m + 4)
end

# box3Mom with the weight u - p, u the first coordinate and p <= lo[1]
function box3Wpl(m::Integer, lo, hi, p)
    T = eltype(lo)
    u1, u2 = lo[1], hi[1]; v1, v2 = lo[2], hi[2]; w1, w2 = lo[3], hi[3]
    pu2 = psiBox(m, u2, v1, v2, w1, w2)
    if iszero(p)
        s = u2^2 * pu2
    else
        s = p * ((u2 - p) * pu2 - box3Mom(m, lo, hi)) + (u2 - p)^2 * pu2
    end
    (iszero(u1) || u1 == p) ||
        (s -= u1 * (u1 - p) * psiBox(m, u1, v1, v2, w1, w2))
    cv = psiWpl(m, v2, u1, u2, w1, w2, p)
    s += (v2 - v1) * cv
    iszero(v1) || (s += v1 * (cv - psiWpl(m, v1, u1, u2, w1, w2, p)))
    cw = psiWpl(m, w2, u1, u2, v1, v2, p)
    s += (w2 - w1) * cw
    iszero(w1) || (s += w1 * (cw - psiWpl(m, w1, u1, u2, v1, v2, p)))
    return s / (m + 4)
end

# the polynomial moment int_lo^hi u^p (c + d u) du, avoiding catastrophic cancellation
function polMom(p::Integer, lo::T, hi::T, c::T, d::T) where {T}
    dl = hi - lo
    if p >= 0
        wl = c + d * lo; wh = c + d * hi
        v = zero(T); bc = one(T)
        for j in 0:p
            br = d >= 0 ? wl * dl^(j + 1) / (j + 1) + d * dl^(j + 2) / (j + 2) :
                 wh * dl^(j + 1) / (j + 1) - d * dl^(j + 2) / ((j + 1) * (j + 2))
            v += bc * lo^(p - j) * br
            bc = bc * (p - j) / (j + 1)
        end
        return v
    end
    pw(q) = q == 0 ? log(hi / lo) : (hi^q - lo^q) / q      # int u^{q-1}, lo > 0
    v = iszero(c) ? zero(T) : c * pw(p + 1)
    iszero(d) || (v += d * pw(p + 2))
    return v
end

# the axis that dominates the box throughout, 4(x_j^2 + x_k^2) <= x_d^2, else 0; box3Ser
# expands in (x_j^2 + x_k^2)/x_d^2 there, box3Mom's face terms exceeding the integral
function domAxs(lo, hi)
    for d in 1:3
        j = d % 3 + 1; k = j % 3 + 1
        4 * (hi[j]^2 + hi[k]^2) <= lo[d]^2 && return d
    end
    return 0
end

function box3Ser(m::Integer, lo, hi, cw, dw, d::Integer)
    T = eltype(lo)
    j = d % 3 + 1; k = j % 3 + 1
    kmx = 600
    X = T[]; Y = T[]
    v = zero(T); bin = one(T)
    for n in 0:kmx
        iszero(bin) && break
        push!(X, polMom(2n, lo[j], hi[j], cw[j], dw[j]))
        push!(Y, polMom(2n, lo[k], hi[k], cw[k], dw[k]))
        s = zero(T)
        c = one(T)
        for i in 0:n
            s += c * X[i + 1] * Y[n - i + 1]
            c = c * (n - i) / (i + 1)
        end
        t = bin * s * polMom(m - 2n, lo[d], hi[d], cw[d], dw[d])
        v += t
        bin = bin * (T(m) / 2 - n) / (n + 1)
        n > 2 && abs(t) <= eps(T) * abs(v) / 8 && break
    end
    return v
end

# the axis negligible against the other two, hi_i^2 <= (lo_j^2 + lo_k^2)/10000, else 0
function thnAxs(lo, hi)
    T = eltype(lo)
    for i in 1:3
        j = i % 3 + 1; k = j % 3 + 1
        hi[i]^2 <= (lo[j]^2 + lo[k]^2) / 10000 && return i
    end
    return 0
end

function wgt2(p::Integer, t::T, a1::T, a2::T, b1::T, b2::T, cw::T, dw::T) where {T}
    if p >= -1
        wl = cw + dw * a1; wh = cw + dw * a2; dl = a2 - a1
        v = zero(T)
        iszero(wl) || (v += (wl / dl) * psiWmn(p, t, a1, a2, b1, b2, a2))
        iszero(wh) || (v += (wh / dl) * psiWpl(p, t, a1, a2, b1, b2, a1))
        return v
    end
    return cw * psiBox(p, t, a1, a2, b1, b2) +
           dw * powSegD(p + 2, b1, b2, sqrt(t^2 + a1^2), sqrt(t^2 + a2^2),
                      a2^2 - a1^2) / (p + 2)
end

function box3Thn(m::Integer, lo, hi, cw, dw, i::Integer)
    T = eltype(lo)
    j = i % 3 + 1; k = j % 3 + 1
    v = zero(T); bin = one(T)
    for n in 0:60
        iszero(bin) && break       # even m: the expansion terminates at n = m/2
        p = m - 2n
        if !iszero(dw[j])
            th = cw[k] * wgt2(p, zero(T), lo[j], hi[j], lo[k], hi[k], cw[j], dw[j])
        elseif !iszero(dw[k])
            th = cw[j] * wgt2(p, zero(T), lo[k], hi[k], lo[j], hi[j], cw[k], dw[k])
        else
            th = cw[j] * cw[k] * psiBox(p, zero(T), lo[j], hi[j], lo[k], hi[k])
        end
        t = bin * polMom(2n, lo[i], hi[i], cw[i], dw[i]) * th
        v += t
        bin = bin * (T(m) / 2 - n) / (n + 1)
        n > 0 && abs(t) <= eps(T) * abs(v) / 8 && break
    end
    return v
end

# the up to two slender axes, 2(hi - lo) < hi, thinnest relative to hi first, else 0. The
# gate is strict so hi = 2 lo -- all parBxs makes away from the origin -- stays analytic
function slvAxs(lo, hi)
    a = 0; b = 0
    rt(i) = (hi[i] - lo[i]) / hi[i]
    for i in 1:3
        (2 * (hi[i] - lo[i]) < hi[i] && !iszero(hi[i])) || continue
        if a == 0 || rt(i) < rt(a)
            b = a; a = i
        elseif b == 0 || rt(i) < rt(b)
            b = i
        end
    end
    return (a, b)
end

# the Gauss-Legendre order that reaches eps(T) on [lo, hi]: slvAxs' gate bounds hf/md
# by 1/3, i.e. Bernstein rho >= 3 + sqrt(8), and an n-point rule converges like rho^{-2n}
function slvOrd(lo::T, hi::T) where {T}
    hf = (hi - lo) / 2; md = (lo + hi) / 2
    rho = (md + sqrt(md^2 - hf^2)) / hf
    return clamp(ceil(Int, -log(eps(T)) / (2 * log(rho))) + 3, 3, 240)
end

# the n-point Gauss-Legendre nodes and weights on [-1, 1], generic in T at any precision
function gauLeg(n::Int, ::Type{T}) where {T}
    xs = zeros(T, n); ws = zeros(T, n)
    for i in 1:((n + 1) ÷ 2)
        x = cos(T(pi) * T(4i - 1) / T(4n + 2))
        pn = zero(T); dp = zero(T)
        for _ in 1:200
            p0 = one(T); pn = x
            for k in 2:n
                p0, pn = pn, ((2k - 1) * x * pn - (k - 1) * p0) / T(k)
            end
            dp = n * (x * pn - p0) / (x^2 - 1)
            dx = pn / dp
            x -= dx
            abs(dx) <= eps(T) && break
        end
        w = 2 / ((1 - x^2) * dp^2)
        xs[i] = -x; ws[i] = w
        xs[n + 1 - i] = x; ws[n + 1 - i] = w
    end
    return xs, ws
end

# "weight plus": int_lo^hi (t - lo)(t^2 + d^2)^(n/2) dt, by the exact form that cancels less
function wplSeg(n::Integer, lo::T, hi::T, d::T) where {T}
    Sl = sqrt(lo^2 + d^2); Sh = sqrt(hi^2 + d^2)
    dl = hi - lo
    J = powSeg(n, lo, hi, d)
    P = dl * (hi + lo) * powSum(Sh, Sl, n + 2) / ((Sh + Sl) * T(n + 2))
    W = wmnSeg(n, lo, hi, d, hi)
    return P + lo * J <= dl * J + W ? P - lo * J : dl * J - W
end

# int_lo^hi (cw + dw t)(t^2 + d^2)^(m/2) dt
function seg1(m::Integer, lo::T, hi::T, d::T, cw::T, dw::T) where {T}
    iszero(dw) && return cw * powSeg(m, lo, hi, d)
    dl = hi - lo
    wl = cw + dw * lo; wh = cw + dw * hi
    v = zero(T)
    iszero(wl) || (v += (wl / dl) * wmnSeg(m, lo, hi, d, hi))
    iszero(wh) || (v += (wh / dl) * wplSeg(m, lo, hi, d))
    return v
end

# box3's integral with the slender axes i and i2 taken by Gauss-Legendre instead of the
# closed forms, which amplify by hi/(hi - lo) per slender axis
function box3Slv(m::Integer, lo, hi, cw, dw, i::Integer, i2::Integer)
    T = eltype(lo)
    dl = hi[i] - lo[i]; md = (lo[i] + hi[i]) / 2; hf = dl / 2
    xs, gws = gauLeg(slvOrd(lo[i], hi[i]), T)
    v = zero(T)
    if i2 == 0
        j = i % 3 + 1; k = j % 3 + 1
        for (x, gw) in zip(xs, gws)
            t = md + hf * x
            w = cw[i] + dw[i] * t
            if !iszero(dw[j])
                th = cw[k] * wgt2(m, t, lo[j], hi[j], lo[k], hi[k], cw[j], dw[j])
            elseif !iszero(dw[k])
                th = cw[j] * wgt2(m, t, lo[k], hi[k], lo[j], hi[j], cw[k], dw[k])
            else
                th = cw[j] * cw[k] * psiBox(m, t, lo[j], hi[j], lo[k], hi[k])
            end
            v += hf * gw * w * th
        end
        return v
    end
    j = 6 - i - i2                                  # the one axis left analytic
    dl2 = hi[i2] - lo[i2]; md2 = (lo[i2] + hi[i2]) / 2; hf2 = dl2 / 2
    xs2, gws2 = gauLeg(slvOrd(lo[i2], hi[i2]), T)
    for (x, gw) in zip(xs, gws), (x2, gw2) in zip(xs2, gws2)
        t = md + hf * x; t2 = md2 + hf2 * x2
        w = (cw[i] + dw[i] * t) * (cw[i2] + dw[i2] * t2)
        v += hf * gw * hf2 * gw2 * w *
             seg1(m, lo[j], hi[j], sqrt(t^2 + t2^2), cw[j], dw[j])
    end
    return v
end

# int over the box [lo, hi] of prod_i (cw_i + dw_i x_i) (x^2 + y^2 + z^2)^(m/2) dV, by
# whichever route the box shape can carry. At most one axis may have dw != 0, and its
# weight splits as alp (hi - u) + bet (u - lo), both parts nonnegative
function box3(m::Integer, lo, hi, cw, dw)
    T = eltype(lo)
    sa, sb = slvAxs(lo, hi)
    sa == 0 || return box3Slv(m, lo, hi, cw, dw, sa, sb)
    d = domAxs(lo, hi)
    d == 0 || return box3Ser(m, lo, hi, cw, dw, d)
    d = thnAxs(lo, hi)
    d == 0 || return box3Thn(m, lo, hi, cw, dw, d)
    k = 0
    for i in 1:3
        if !iszero(dw[i])
            k == 0 || error("box3: more than one linearly weighted axis")
            k = i
        end
    end
    if k == 0
        return prod(cw) * box3Mom(m, lo, hi)
    end
    prm = (k, k % 3 + 1, (k % 3 + 1) % 3 + 1)
    lo2 = T[lo[i] for i in prm]; hi2 = T[hi[i] for i in prm]
    cst = cw[prm[2]] * cw[prm[3]]
    wlo = cw[k] + dw[k] * lo[k]
    whi = cw[k] + dw[k] * hi[k]
    dl = hi[k] - lo[k]
    v = zero(T)
    iszero(wlo) || (v += (wlo / dl) * box3Wmn(m, lo2, hi2, hi2[1]))
    iszero(whi) || (v += (whi / dl) * box3Wpl(m, lo2, hi2, lo2[1]))
    return cst * v
end

# per-axis box-box convolution weight; (:frozen, offset) or
# (:pieces, [(lo, hi, c, d)]) with weight c + d w on [lo, hi]. tol, for coordinates with
# no exact rational form, merges breakpoints closer than that fraction of the span
function axsCnv(A, B, tol = nothing)
    p, q = A
    r, s = B
    if p == q && r == s
        return (:frozen, p - r)
    elseif p == q
        return (:pieces, [(p - s, p - r, one(p), zero(p))])
    elseif r == s
        return (:pieces, [(p - r, q - r, one(p), zero(p))])
    end
    dns(w) = max(zero(p), min(q, s + w) - max(p, r + w))
    lo = p - s; hi = q - r
    bps = sort(unique(filter(x -> lo <= x <= hi, [p - s, q - s, p - r, q - r])))
    if tol !== nothing                       # merge near-coincident breakpoints
        scl = max(abs(lo), abs(hi))
        kep = [bps[1]]
        for x in bps[2:end]
            x - kep[end] > tol * scl && push!(kep, x)
        end
        bps = kep
    end
    pcs = Tuple{typeof(p),typeof(p),typeof(p),typeof(p)}[]
    for i in 1:(length(bps) - 1)
        al, be = bps[i], bps[i + 1]
        be <= al && continue
        da, db = dns(al), dns(be)
        d = (db - da) / (be - al)
        c = da - d * al
        push!(pcs, (al, be, c, d))
    end
    return (:pieces, pcs)
end

# exact conversion to Rational, so parBxs can form its breakpoints without two that
# coincide mathematically rounding an ulp apart into a sliver box. Rational{BigInt} of a
# Double64 routes through BigFloat and throws, so the two Float64 halves are summed
cnvQ(::Type{Q}, x::Double64) where {Q<:Rational} =
    Q(Rational{BigInt}(DoubleFloats.HI(x)) + Rational{BigInt}(DoubleFloats.LO(x)))
cnvQ(::Type{Q}, x) where {Q} = Q(x)

function parBxs(pA, pB, ::Type{T}, tol = nothing) where {T}
    Q = tol === nothing ? Rational{BigInt} : typeof(pA[1][1])
    qA = ((cnvQ(Q, pA[1][1]), cnvQ(Q, pA[1][2])), (cnvQ(Q, pA[2][1]), cnvQ(Q, pA[2][2])),
          (cnvQ(Q, pA[3][1]), cnvQ(Q, pA[3][2])))
    qB = ((cnvQ(Q, pB[1][1]), cnvQ(Q, pB[1][2])), (cnvQ(Q, pB[2][1]), cnvQ(Q, pB[2][2])),
          (cnvQ(Q, pB[3][1]), cnvQ(Q, pB[3][2])))
    of2 = zero(Q)
    axs = Vector{Vector{NTuple{4,Q}}}()
    for i in 1:3
        kind, val = axsCnv(qA[i], qB[i], tol)
        if kind == :frozen
            of2 += val^2
        else
            push!(axs, val)
        end
    end
    d = length(axs)
    bxs = Tuple{Vector{T},Vector{T},Vector{T},Vector{T}}[]
    for cmb in Iterators.product(axs...)
        splt = Vector{Vector{NTuple{4,Q}}}()
        for (lo, hi, c, dd) in cmb                    # split at 0, then reflect
            sub = NTuple{4,Q}[]
            if lo < 0 < hi
                push!(sub, (lo, zero(Q), c, dd))
                push!(sub, (zero(Q), hi, c, dd))
            else
                push!(sub, (lo, hi, c, dd))
            end
            push!(splt, sub)
        end
        for sc in Iterators.product(splt...)
            lov = Vector{T}(undef, d); hiv = similar(lov)
            cv = similar(lov); dv = similar(lov)
            bad = false
            for (i, (lo, hi, c, dd)) in enumerate(sc)
                l, h, cc, ddd = hi <= 0 ? (-hi, -lo, c, -dd) : (lo, hi, c, dd)
                h <= l && (bad = true; break)
                lov[i] = T(l); hiv[i] = T(h); cv[i] = T(cc); dv[i] = T(ddd)
            end
            bad && continue
            push!(bxs, (lov, hiv, cv, dv))
        end
    end
    return bxs, T(of2)
end

# faces in the order yzL yzU xzL xzU xyL xyU: (degenerate axis, side)
const FACES = [(1, -1), (1, 1), (2, -1), (2, 1), (3, -1), (3, 1)]

function boxFace(c::Int, sig::Int, ctr::NTuple{3,T}, s::NTuple{3,T}) where {T}
    iv = Vector{NTuple{2,T}}(undef, 3)
    for d in 1:3
        if d == c
            v = ctr[d] + T(sig) * s[d] / 2
            iv[d] = (v, v)
        else
            iv[d] = (ctr[d] - s[d] / 2, ctr[d] + s[d] / 2)
        end
    end
    return (iv[1], iv[2], iv[3])
end

# the panel pair (face F of the cell centred at R, face Fp of the cell at the origin);
# R is a displacement, not a cell index, so the cells need not sit on their own grid
function parFac(R::NTuple{3,T}, F::Int, Fp::Int, s::NTuple{3,T}) where {T}
    c, sig = FACES[F]
    cp, sigp = FACES[Fp]
    return (boxFace(c, sig, R, s), boxFace(cp, sigp, ntuple(_ -> zero(T), 3), s))
end

parFac(D::NTuple{3,Int}, F::Int, Fp::Int, s::NTuple{3,T}) where {T} =
    parFac(T.(D) .* s, F, Fp, s)

# exact multiplication by 2^k, whatever the panels are given as
scl2(x::AbstractFloat, k::Integer) = ldexp(x, k)
scl2(x::Union{Integer,Rational}, k::Integer) = x * Rational{BigInt}(2)^k
sclPan(p, k::Integer) = ntuple(d -> (scl2(p[d][1], k), scl2(p[d][2], k)), 3)

function panExp(pA, pB)
    mx = zero(float(typeof(pA[1][1])))
    for p in (pA, pB), d in 1:3, x in p[d]
        a = abs(float(x)); a > mx && (mx = a)
    end
    return iszero(mx) ? 0 : -exponent(mx)
end

# I_m for m = -1 .. mMax, out[m + 2] = I_m. I_m(lam s) = lam^{m+4} I_m(s), so the pair is
# prescaled to [1, 2) and each moment scaled back, both steps only moving exponents
parMom(pA, pB, mMax::Integer; tol = nothing) =
    parMom(pA, pB, mMax, float(typeof(pA[1][1])); tol = tol)

function parMom(pA, pB, mMax::Integer, ::Type{T}; tol = nothing) where {T}
    mMax >= -1 || error("parMom: mMax < -1")
    kx = panExp(pA, pB)
    bxs, of2 = parBxs(sclPan(pA, kx), sclPan(pB, kx), T, tol)
    out = zeros(T, mMax + 2)
    isempty(bxs) && return out
    d = length(bxs[1][1])
    if d == 2
        mo = isodd(mMax) ? mMax : mMax - 1                    # top odd m, >= -1
        me = iseven(mMax) ? mMax : mMax - 1           # top even m, or -2 (none)
        for (lo, hi, cw, dw) in bxs
            vo = box2(-1, mo, lo, hi, cw, dw, of2)
            for k in eachindex(vo)
                out[2 * k - 2 + 1] += vo[k]                    # m = -1 + 2(k-1)
            end
            me < 0 && continue
            ve = box2(0, me, lo, hi, cw, dw, of2)
            for k in eachindex(ve)
                out[2 * k] += ve[k]                                 # m = 2(k-1)
            end
        end
    elseif d == 3
        iszero(of2) || error("parMom: 3D domain with a frozen offset")
        for (lo, hi, cw, dw) in bxs, m in -1:mMax
            out[m + 2] += box3(m, lo, hi, cw, dw)
        end
    else
        error("parMom: $d-dimensional domain")
    end
    iszero(kx) && return out
    for m in -1:mMax
        out[m + 2] = ldexp(out[m + 2], -kx * (m + 4))
    end
    return out
end

facMom(D::NTuple{3,Int}, F::Int, Fp::Int, s::NTuple{3,T}, mMax::Integer;
            tol = nothing) where {T} =
    parMom(parFac(D, F, Fp, s)..., mMax; tol = tol)

# int_A int_B exp(2 pi i f r)/(4 pi r f^2) = (1/(4 pi f^2)) sum_n (2 pi i f)^n
# I_{n-1}/n!. rel = |last term|/|sum| is the term-count stopping rule, NOT an
# error bound: it certifies term decay, not the conditioning of the partial sum.
momSer(pA, pB, frq, mMax::Integer) = momSer(parMom(pA, pB, mMax), frq)

function momSer(mom::AbstractVector{T}, frq) where {T}
    F = Complex{T}
    f = F(frq)
    z = 2 * T(pi) * im * f
    acc = zero(F); cof = one(F); lst = zero(F)
    for n in eachindex(mom)
        lst = cof * mom[n]
        acc += lst
        cof *= z / n
    end
    val = acc / (4 * T(pi) * f^2)
    return (val, abs(lst) / abs(acc))
end

#=
Returns the scalar (Helmholtz) Green function. The separation dstMag is assumed 
to be scaled by wavelength. 
=#
@inline function sclEgo_(dstMag::Number, frqPhz::Number)
    # cispi reduces the argument exactly, as in sclEgoN_ below
    return cispi(2 * dstMag * frqPhz) / (4 * π * dstMag * frqPhz^2)
end
function sclEgo(dstMag::Number, frqPhz::ComplexF64)
    if imag(frqPhz) == zero(real(typeof(frqPhz)))
        return sclEgo_(dstMag, real(frqPhz))
    end
    return sclEgo_(dstMag, frqPhz)
end
sclEgo(dstMag::Number, frqPhz::Number) = sclEgo_(dstMag, frqPhz)
#=
Returns the scalar (Helmholtz) Green function with the singularity removed. The 
separation distance dstMag is assumed to be scaled by the wavelength.
=#
@inline function sclEgoN_(dstMag::Number, frqPhz::Number)
    # Computes g = [exp(im z) - 1] / (4π dstMag frqPhz^2) with z = 2π dstMag frqPhz.
    # Note that expressing g in terms of a sinc improves numerical stability for
    # small z and for complex frqPhz. For very large imaginary parts, the sinc
    # stays within a few ulp of a 256 bit BigFloat reference.
    xPrd = dstMag * frqPhz
    return im * cispi(xPrd) * sinc(xPrd) / (2 * frqPhz)
end
function sclEgoN(dstMag::Number, frqPhz::ComplexF64)
    if imag(frqPhz) == zero(real(typeof(frqPhz)))
        return sclEgoN_(dstMag, real(frqPhz))
    end
    return sclEgoN_(dstMag, frqPhz)
end
sclEgoN(dstMag::Number, frqPhz::Number) = sclEgoN_(dstMag, frqPhz)

#=
All unique pairs of cube faces. Row 1 is the target face and row 2 the source
face, the convention srfSum! assembles by: the tensor component (a, b) sums the
face pairs whose row 1 normal is a and row 2 normal is b.
=#
const facPar = SMatrix{2,36,Int}(vcat(repeat(1:6, inner=6)', repeat(1:6, outer=6)'))

#=
Update egoCrc to hold Green function interactions. The storage format of egoCrc 
is [[ii, ji, ki]^{T}; [ij, jj, kj]^{T}; [ik, jk, kk]^{T}]. 
See documentation for explanation.
=#
function srfSum!(egoCrc::AbstractMatrix{<:Complex}, srfMat::AbstractVector{<:Complex})
    # ii
    egoCrc[1,1] = srfMat[15] - srfMat[16] - srfMat[21] + 
    srfMat[22] + srfMat[29] - srfMat[30] - srfMat[35] + srfMat[36]
    # ji
    egoCrc[2,1] = - srfMat[13] + srfMat[14] + srfMat[19] - srfMat[20] 
    # ki
    egoCrc[3,1] = - srfMat[25] + srfMat[26] + srfMat[31] - srfMat[32] 
    # ij
    egoCrc[1,2] = - srfMat[3] + srfMat[4] + srfMat[9] - srfMat[10]
    # jj
    egoCrc[2,2] = srfMat[1] - srfMat[2] - srfMat[7] + srfMat[8] + 
    srfMat[29] - srfMat[30] - srfMat[35] + srfMat[36]
    # kj
    egoCrc[3,2] = - srfMat[27] + srfMat[28] + srfMat[33] - srfMat[34]
    # ik
    egoCrc[1,3] = - srfMat[5] + srfMat[6] + srfMat[11] - srfMat[12]
    # jk
    egoCrc[2,3] = - srfMat[17] + srfMat[18] + srfMat[23] - srfMat[24]
    # kk
    egoCrc[3,3] = srfMat[1] - srfMat[2] - srfMat[7] + srfMat[8] + 
    srfMat[15] - srfMat[16] - srfMat[21] + srfMat[22]
    return nothing
end

const cntInd = CartesianIndices((2, 2, 2))
# frequency-free, so one table serves every frequency at a given cell geometry
const momMem = Dict{Tuple{NTuple{3,Rational},NTuple{3,Rational},Int},
    Array{Vector{Double64},4}}()
const momLck = ReentrantLock()

#=
Smallest N with (k rMax)^(N + 1) / (N + 1)! exp(k rMax) below eps(Double64),
rMax being the corner-to-corner reach of the (1,1,1) offset. The bound governs
truncation only; momSer's own indicator does not detect the cancellation the
Double64 working type is there to absorb.
=#
function serOrd(scl::NTuple{3,Rational}, stp::NTuple{3,Rational}, frq::Number)
    rMax = sqrt(sum(abs2, Float64.(scl .+ stp)))
    arg = 2 * π * abs(frq) * rMax
    lim = log(eps(Double64)) - arg
    lgt = 0.0
    for n ∈ 1:400
        lgt += log(arg) - log(n)
        lgt < lim && return max(4, n - 1)
    end
    error("serOrd: contact series does not truncate at scl $scl, frq $frq.")
end

function cntMom(scl::NTuple{3,Rational}, stp::NTuple{3,Rational}, mMax::Int)
    key = (scl, stp, mMax)
    hit = lock(momLck) do
        get(momMem, key, nothing)
    end
    isnothing(hit) || return hit
    # 288 face pairs of Double64 closed forms, so the table is built outside the lock
    sclDbl = Double64.(scl)
    stpDbl = Double64.(stp)
    mom = Array{Vector{Double64}}(undef, 2, 2, 2, 36)
    @threads for lin ∈ 1:288
        celItr, fp = fldmod1(lin, 36)
        sep = (Tuple(cntInd[celItr]) .- 1) .* stpDbl
        mom[cntInd[celItr], fp] = parMom(parFac(sep, facPar[1, fp], facPar[2, fp],
            sclDbl)..., mMax)
    end
    return lock(momLck) do
        get!(momMem, key, mom)
    end
end

# the contact block of egoToe, from the face-pair moments, which are exact touching or
# not. The signed 36-pair assembly runs at Double64 whatever genPrc is: its amplification
# and the series' digit loss together overrun Float64.
function cntBlk!(egoToe::AbstractArray{<:Complex,5}, vol::GlaVol,
    cmpInf::GlaKerOpt)

    stp = ntuple(itr -> Rational(step(vol.grd[itr])), 3)
    frq = ComplexDF64(frqPhz(cmpInf))
    mom = cntMom(vol.scl, stp, serOrd(vol.scl, stp, frqPhz(cmpInf)))
    celInv = inv(prod(Double64.(vol.scl)))
    srfMat = zeros(MVector{36,ComplexDF64})
    egoCel = zeros(MMatrix{3,3,ComplexDF64})
    for posInd ∈ CartesianIndices(ntuple(itr -> min(vol.cel[itr], 2), 3))
        for fp ∈ 1:36
            srfMat[fp] = celInv * first(momSer(mom[posInd, fp], frq))
        end
        srfSum!(egoCel, srfMat)
        view(egoToe, :, :, posInd) .= egoCel
    end
    return nothing
end

# The quasistatic block at the cell offsets Ds: the n = 0 term of momSer's series,
# I_{-1} / (4 pi f^2), less the 1/f^2 that genEgoSlf! applies once at the end.  What lands in
# egoToe is therefore S itself, which carries no frequency.  genEgoSlf! calls this for the contact
# offsets, then again for the near band farBlkQss! refuses.  The signed 36-pair assembly runs at
# Double64 whatever genPrc is; its amplification alone overruns Float64.
function momBlkQss!(egoToe::AbstractArray{<:Complex,5}, vol::GlaVol, Ds::Vector{NTuple{3,Int}})

    stp = ntuple(dir -> Int(step(vol.grd[dir]) // vol.scl[dir]), 3)
    sclDbl = Double64.(vol.scl)
    celInv = inv(prod(sclDbl) * 4 * Double64(pi))
    srfMat = Matrix{ComplexDF64}(undef, 36, length(Ds))
    @threads for lin ∈ 1:(36 * length(Ds))
        off, fp = fldmod1(lin, 36)
        srfMat[fp, off] = celInv * only(parMom(parFac(Ds[off], facPar[1, fp],
            facPar[2, fp], sclDbl)..., -1))
    end
    egoCel = zeros(MMatrix{3,3,ComplexDF64})
    for off ∈ eachindex(Ds)
        srfSum!(egoCel, view(srfMat, :, off))
        view(egoToe, :, :, CartesianIndex(div.(Ds[off], stp) .+ 1)) .= egoCel
    end
    return nothing
end
