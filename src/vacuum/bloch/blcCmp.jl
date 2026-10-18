# Composite periodic unit cells: the H/AH blocks of every ordered region pair (t, s) in genEgoExt!'s circulant layout
# (trgCel + srcCel per axis, the slot trgCel + 1 zero), (t, t) included with its own images, from ONE equal-cell fill on the
# gcd lattice g (blcFil) and the gcd identity: for cells made of r_t and r_s sub-cells of g per axis,
#   G_ts(c) = (1/Π r_t) Σ_δ w(δ) G_gg(c + δ),  w_i(δ) = #{(j, l): j - l = δ, 0 <= j < r_t,i, 0 <= l < r_s,i},
# c = (target lower corner - source lower corner)/g (integer on the gcd lattice), applied per axis (three 1D convolutions with
# positive integer weights, so no cancellation of its own). Periodic offsets reduce by the twist G(m + P e_i) = e^{-iθ_i} G(m),
# open negative offsets by the reflection G_ab(-m_o) = σ_a σ_b G_ab(m_o), σ = -1 on axis o. H and AH are linear in G: same sums.
import ..GilaElectromagnetics as GVa
const LAMBOX = 8

# region: a GlaVol; periods L (λ) on the axes ax
function cmpChk(regs, ax, L)
    g = ntuple(i -> reduce(gcd, [QI(r.scl[i]) for r in regs]), 3)
    for (j, i) in enumerate(ax), (k, r) in enumerate(regs)
        isinteger(QI(L[j]) / QI(r.scl[i])) || error("blcCmp: period $(L[j]) along axis $i is not a whole multiple of the cell edge $(r.scl[i]) of region $k")
    end
    for (k, r) in enumerate(regs), i in 1:3
        isinteger((QI(r.grd[i].start) - QI(r.scl[i]) / 2) / g[i]) || error("blcCmp: region $k is off the gcd lattice $(g) along axis $i")
    end
    # no region longer than a period, no two regions overlapping, directly or modulo the periods
    lo(r, i) = QI(r.grd[i].start) - QI(r.scl[i]) / 2; hi(r, i) = lo(r, i) + r.cel[i] * QI(r.scl[i])
    for (j, i) in enumerate(ax), (k, r) in enumerate(regs)
        hi(r, i) - lo(r, i) <= QI(L[j]) || error("blcCmp: region $k spans $(hi(r, i) - lo(r, i)) along axis $i, longer than the period $(L[j])")
    end
    for a in eachindex(regs), b in a+1:length(regs)
        ra, rb = regs[a], regs[b]
        ovl = all(1:3) do i
            j = findfirst(==(i), ax)
            j === nothing && return lo(ra, i) < hi(rb, i) && lo(rb, i) < hi(ra, i)
            Lq = QI(L[j]); n0 = floor(Int, (lo(ra, i) - lo(rb, i)) / Lq)
            any(n -> lo(ra, i) < hi(rb, i) + n * Lq && lo(rb, i) + n * Lq < hi(ra, i), n0-1:n0+2)
        end
        ovl && error("blcCmp: regions $a and $b overlap" * (isempty(ax) ? "" : " (directly or modulo the periods $(Tuple(L)))"))
    end
    g
end

"""
    blcCmp(regs, f, ax, L, θ; tol, cache) -> B

B[(t, s)] = (H, A), arrays (3, 3, n1, n2, n3): t != s in genEgoExt!'s layout (egoCrcExt of trgVol = regs[t], srcVol = regs[s]);
t == s the Toeplitz form of blcFil (periodic m ∈ -(N-1):(N-1), open 0:(N-1)). Periods L must be whole multiples of every edge.
"""
function blcCmp(regs, f::Complex{T}, ax::NTuple{D,Int}, L::NTuple{D,<:Rational}, θ::NTuple{D,Any}; tol::T = eps(T), cache = nothing) where {T,D}
    blasMt() && return blas1(() -> blcCmp(regs, f, ax, L, θ; tol, cache))
    try
        cmpBlk(regs, f, ax, L, θ; tol, cache)
    catch e
        e isa ErrorException && startswith(e.msg, r"blc(Fil|Tns): ") ? error("blcCmp: " * e.msg[9:end]) : rethrow()
    end
end
function cmpBlk(regs, f::Complex{T}, ax::NTuple{D,Int}, L::NTuple{D,<:Rational}, θ::NTuple{D,Any}; tol::T, cache) where {T,D}
    inpChk("blcCmp", f, ax, L, θ)
    g = cmpChk(regs, ax, L)
    isper(i) = i in ax
    Pg = Dict(ax[j] => Int(QI(L[j]) / g[ax[j]]) for j in 1:D)
    lo(r, i) = Int((QI(r.grd[i].start) - QI(r.scl[i]) / 2) / g[i])      # lower corner, in g
    rat(r, i) = Int(QI(r.scl[i]) / g[i])
    # per ordered pair and axis: the corner offsets c of the circulant slots (nothing at the zero slot) and the δ range
    function slots(t, s, i)
        rt, rs = regs[t], regs[s]
        if t == s
            N = rt.cel[i]
            return (isper(i) ? collect(-(N - 1):(N - 1)) : collect(0:(N - 1))) .* rat(rt, i)
        end
        gT = GVa.sepGrd(rt, rs, 0); gS = GVa.sepGrd(rt, rs, 1); n = rt.cel[i] + rs.cel[i]
        [ind == rt.cel[i] + 1 ? typemin(Int) : Int((QI(GVa.grdSep(ind, rt.cel[i], i, gT, gS)) - (QI(rt.scl[i]) - QI(rs.scl[i])) / 2) / g[i]) for ind in 1:n]
    end
    prs = [(t, s) for t in eachindex(regs), s in eachindex(regs)]
    sl = Dict((t, s, i) => slots(t, s, i) for (t, s) in prs, i in 1:3)
    # open extent of the fine fill
    No = ntuple(i -> isper(i) ? Pg[i] : maximum(maximum(abs(c) + max(rat(regs[t], i), rat(regs[s], i)) for c in sl[(t, s, i)] if c != typemin(Int)) for (t, s) in prs) + 1, 3)
    H, A, inf = blcFil(g, f, ax, No, ntuple(j -> Pg[ax[j]], D), θ; tol, cache)
    # hot periodic axes (twist within DHOT of 0 or 1/2, not exact): a block between cells whose centres are aligned on such an axis
    # (modulo the period) is a positive box sum of the pairs G(X + d) + G(X - d) whose x-odd entries cancel to O(θ - θ_0); those slots are
    # summed from the symmetric-sum fill Σ(k) = G(k) + G(-k) of the same fine grid (blcFil symAx), formed without that cancellation
    hotAx = [i for (j, i) in enumerate(ax) if !(θ[j] isa Rational) && (w_ = mod(Float64(θ[j]) / (2 * π), 1.0); d0 = min(w_, 1 - w_, abs(w_ - 0.5)); 0 < d0 < DHOT)]
    Sy = Dict(i => blcFil(g, f, ax, No, ntuple(j -> Pg[ax[j]], D), θ; tol, cache, symAx = i) for i in hotAx)
    ms = inf[:ms]; off = [findfirst(==(0), ms[i]) for i in 1:3]
    θv = [T(θ[j] isa Rational ? 2 * T(π) * T(θ[j]) : θ[j]) for j in 1:D]
    tw(j, n) = θ[j] isa Rational ? cispi(-2 * n * T(θ[j])) : cis(-n * θv[j])
    # fine entry at integer offset m (any sign, any image): (index tuple, 3x3 sign/phase)
    function fineIdx(i, m)
        if isper(i)
            j = findfirst(==(i), ax); P = Pg[i]; mr = mod(m, P); n = div(m - mr, P)
            return off[i] + mr, tw(j, n), 1
        end
        return off[i] + abs(m), one(Complex{T}), m < 0 ? -1 : 1
    end
    # the box sums of every pair; aligned slots of the axes in Sy from the symmetric-sum fills. needS: periodic axes not in Sy whose aligned
    # slots have an entry (> 1e-8 of its block max) with box rounding factor Σ w|G_fine| / |X| > LAMBOX (the ±d pairs cancel also at a generic
    # twist when the entry is set by far images: cell A d1 θ = 0.7, H_xy 31 eps = 2.7 eps(1+κ_up))
    function boxAll(Sy)
    needS = Set{Int}()
    B = Dict{Tuple{Int,Int},Any}()
    for (t, s) in prs
        rt = ntuple(i -> rat(regs[t], i), 3); rs = ntuple(i -> rat(regs[s], i), 3)
        cs = [sl[(t, s, i)] for i in 1:3]
        # per axis: the slot x fine-index matrices W± (weights w(δ), twist phase, reflection sign for the entries odd on that axis),
        # then three mode products per entry (dense BLAS; each row has <= r_t + r_s - 1 nonzeros, zeros add exactly)
        Wm = map(1:3) do i
            n = length(cs[i]); Nf = size(H, 2 + i)
            Wp = zeros(Complex{T}, n, Nf); Wn = zeros(Complex{T}, n, Nf)
            for (k, c) in enumerate(cs[i])
                c == typemin(Int) && continue
                for δ in -(rs[i] - 1):(rt[i] - 1)
                    w = count(l -> 0 <= l + δ < rt[i], 0:rs[i]-1)
                    ix, ph, rf = fineIdx(i, c + δ)
                    Wp[k, ix] += w * ph; Wn[k, ix] += (rf < 0 ? -w : w) * ph
                end
            end
            (Wp, Wn)
        end
        mode3(C, W1, W2, W3) = begin
            n1, n2, n3 = size(C); m1, m2, m3 = size(W1, 1), size(W2, 1), size(W3, 1)
            X1 = reshape(W1 * reshape(C, n1, :), m1, n2, n3)
            X2 = permutedims(reshape(W2 * reshape(permutedims(X1, (2, 1, 3)), n2, :), m2, m1, n3), (2, 1, 3))
            permutedims(reshape(W3 * reshape(permutedims(X2, (3, 1, 2)), n3, :), m3, m1, m2), (2, 3, 1))
        end
        # per hot axis: the slots whose centre separation X (units g) is ≡ 0 mod the period, and their weights on Σ(k): e^{-inθ} w(k)
        # (w(0)/2 at k = 0), from Σ_δ w(δ) G(X + d_δ) = Σ_{d>0} w_d Σ(d) + w_0 Σ(0)/2 with w_d = w_{-d} (the box is symmetric about X)
        Wsy = Dict{Int,Any}()
        alig = Dict{Int,Vector{Int}}()
        for (j_, i) in enumerate(ax)
            alig[i] = [k for (k, c) in enumerate(cs[i]) if c != typemin(Int) && (X2 = 2 * c + (rt[i] - rs[i]); iseven(X2) && mod(X2 ÷ 2, Pg[i]) == 0)]
        end
        for i in keys(Sy)
            j_ = findfirst(==(i), ax); P_ = Pg[i]; Nf = length(Sy[i][3][:ms][i])
            rows = Int[]; Wq = zeros(Complex{T}, 0, Nf)
            for (k, c) in enumerate(cs[i])
                c == typemin(Int) && continue
                X2 = 2 * c + (rt[i] - rs[i]); (iseven(X2) && mod(X2 ÷ 2, P_) == 0) || continue
                Xc = X2 ÷ 2; n_ = div(Xc, P_); wrow = zeros(Complex{T}, 1, Nf)
                for δ in -(rs[i] - 1):(rt[i] - 1)
                    w = count(l -> 0 <= l + δ < rt[i], 0:rs[i]-1); d = c + δ - Xc
                    abs(d) < Nf || error("blcCmp: symmetric slot beyond the fine fill along axis $i")
                    wrow[1, abs(d) + 1] += w / 2 * tw(j_, n_)   # each δ carries half of its pair (Σ(0) = 2G(0) too)
                end
                push!(rows, k); Wq = vcat(Wq, wrow)
            end
            isempty(rows) || (Wsy[i] = (rows, Wq))
        end
        out = map(((H, 1), (A, 2))) do (M, q)
            X = zeros(Complex{T}, 3, 3, length.(cs)...)
            for b in 1:3, a in 1:3
                Ws = [((a == i) + (b == i)) == 1 ? Wm[i][2] : Wm[i][1] for i in 1:3]
                X[a, b, :, :, :] .= mode3(M[a, b, :, :, :], Ws...) ./ prod(rt)
                for (i, rows) in alig
                    (isempty(rows) || haskey(Sy, i) || i in needS) && continue
                    slq = ntuple(k -> k == i ? rows : Colon(), 3)
                    Xa = mode3(abs.(M[a, b, :, :, :]), [abs.(k == i ? Ws[k][rows, :] : Ws[k]) for k in 1:3]...) ./ prod(rt)
                    Xv = X[a, b, slq...]; mxv = maximum(abs, X)
                    any(e -> abs(Xv[e]) > 1e-8 * mxv && Xa[e] > LAMBOX * abs(Xv[e]), eachindex(Xv)) && push!(needS, i)
                end
                for (i, (rows, Wq)) in Wsy
                    MS = Sy[i][q]; W2 = [k == i ? Wq : Ws[k] for k in 1:3]
                    Y = mode3(MS[a, b, :, :, :], W2...) ./ prod(rt)
                    slq = ntuple(k -> k == i ? rows : Colon(), 3)
                    X[a, b, slq...] .= Y
                end
            end
            X
        end
        B[(t, s)] = out
    end
    B, needS
    end
    B, needS = boxAll(Sy)
    if !isempty(needS)
        for i in needS; Sy[i] = blcFil(g, f, ax, No, ntuple(j -> Pg[ax[j]], D), θ; tol, cache, symAx = i); end
        B, _ = boxAll(Sy)
    end
    B
end
