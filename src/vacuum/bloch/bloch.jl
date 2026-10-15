# Bloch-periodic cell-averaged Green tensor of Gila's pulse basis, periodic along one, two or three axes, at real and complex frequency,
#   G^per_ab(R; θ) = Σ_{n ∈ Z^d} e^{iθ·n} G_ab(R + Λn; sT, sS),
# with its Hermitian and anti-Hermitian kernels H = Σ_n e^{iθ·n} Re G_n, AH = Σ_n e^{iθ·n} Im G_n at real f and [G(θ) ± conj G(-θ)]/2(i) at
# complex f (so G = H + i AH), Gila's full entry (the -δ_ab/f² of the self image included), raw normalization 1/V_t.
# One heat-kernel representation: G = ∫_0^∞ [4π² e^{zt} K_t δ + (e^{zt}/f²) ∂∂K_t] dt, z = k² = x + iy, K_t a product of per-axis
# trapezoid heat kernels, each periodic axis image-summed or Poisson-summed per node; the real segment [0, τ] in real space, the tail per
# order in closed form (d = 3) or per order on the real axis (evanescent), on its steepest-descent ray or in the spectral pole form
# (propagating); route C (Im f > Re f/20) puts the segment on the saddle direction of every image. Every truncation and Gauss rule is
# certified per offset against a proven bound; the a posteriori rounding factor λ_e = Σ|terms|/|X| of every entry is tracked.
# Lengths in wavelengths, f complex (Float64, Double64 or BigFloat parts), exact Rational geometry (edges, offsets, periods), θ in radians
# (T) or as an exact twist θ/2π (Rational: θ = π is 1//2). At Im f < 0 the operator is the continuation at fixed Re f (J10).
#
# Entry points
#   blcTns(R, sT, sS, f, ax, L, θ)          one tensor pair (H, AH, info), any cells, per-entry rerun in the wider type
#   blcSet(s, f, ax, N, P)                  the θ-independent state of one unit cell at one frequency
#   blcFil(st, θ; form)                     both storage forms of the unit cell's Toeplitz/twisted-circulant block (threaded)
#   blcFil(s, f, ax, N, P, θ)               the same without the per-frequency cache
#   blcCmp(regs, f, ax, L, θ)               the blocks of every ordered region pair of a periodic composite unit cell
#   blcTwc(M, ms, ax, θ)                    twisted-circulant rows of a Toeplitz fill (P = N) and its twist residual
# Threads: the fill runs its producers (per-axis node factors, certification bounds, propagating orders' open-axis integrals, block
# accumulation) on Threads.nthreads() threads and consumes them in a fixed order: the result is bitwise the same for every thread count.
# Packages: DoubleFloats (hot rows, Double64 reruns), SpecialFunctions (Float64 erfc/erfcx/expint), QuadGK (Gauss nodes at +64 bits).
module Bloch

using LinearAlgebra, SpecialFunctions, DoubleFloats, QuadGK
using Base.ScopedValues
using ..GilaElectromagnetics: GlaVol, coordinates, regions, sepGrd
using SpecialFunctions: gamma

export blcTns, blcSet, blcFil, blcCmp, blcTwc, blcCrt, BlcSet

# one Dict per thread for memo tables whose values are pure functions of the key: every thread builds the same values, no lock, and the
# table never holds a Dict across the computation of a value (which may yield and move the task)
struct Miss end
const MISS = Miss()
struct TMemo{K,V}
    d::Vector{Dict{K,V}}
end
TMemo{K,V}() where {K,V} = TMemo{K,V}([Dict{K,V}() for _ in 1:Threads.maxthreadid()])
function Base.get!(f::Base.Callable, m::TMemo{K,V}, k) where {K,V}
    v = get(m.d[Threads.threadid()], k, MISS)
    v isa Miss || return v
    w = f()
    get!(m.d[Threads.threadid()], k, w)
end

# f(i), i = 1..n, on the threads (each i writes only its own slots); the error of the smallest failing i is raised, as a loop would
function tFor(f, n::Int)
    if n <= 1 || Threads.nthreads() == 1
        for i in 1:n; f(i); end
        return nothing
    end
    err = Vector{Any}(nothing, n)
    Threads.@threads :dynamic for i in 1:n
        try
            f(i)
        catch e
            err[i] = e
        end
    end
    for e in err; e === nothing || throw(e); end
    nothing
end
function tMap(f, xs)
    (length(xs) <= 1 || Threads.nthreads() == 1) && return map(f, xs)
    r = Vector{Any}(undef, length(xs)); v = collect(xs)
    tFor(i -> (r[i] = f(v[i]); nothing), length(v))
    map(identity, r)
end
# f over every element of a vector of vectors, the nesting kept
function tMap2(f, vv)
    fl = tMap(f, [x for v in vv for x in v]); k = Ref(0)
    [[fl[k[] += 1] for _ in v] for v in vv]
end

# J27: OpenBLAS threads called from several Julia threads crashed a fill once (signal 11 at 8 x 8 threads): with more than one
# Julia thread the entry points run BLAS on one thread and restore the caller's setting
blasMt() = Threads.nthreads() > 1 && BLAS.get_num_threads() > 1
function blas1(f)
    n = BLAS.get_num_threads(); BLAS.set_num_threads(1)
    try f() finally BLAS.set_num_threads(n) end
end

# the per-frequency cache of the fill: lattice-point trapezoid heat kernels κ(k s, t) at the exact nodes t met, keyed (edge, k, t), in
# NSH lock-sharded Dicts per (edge type, node type); at most nMax entries (about 120 bytes each), then values are computed, not stored.
# Every entry carries the phase (generation) that last used it; a phase that finds the cache above 3/4 full first drops the entries
# not used by the previous two phases (the nodes of θ-dependent panels: SP rays, refined panels), so the θ-independent nodes (RS, EV, route C
# segment and EV ray, the Double64 hot rows) stay cached through a whole sweep instead of the cache filling up with the first phases' keys
const NSH = 64
struct KapCache
    sh::Vector{Dict{Any,Any}}; lk::Vector{ReentrantLock}; n::Threads.Atomic{Int}; nMax::Int; gen::Threads.Atomic{Int}
end
KapCache(nMax::Int = 2_000_000) = KapCache([Dict{Any,Any}() for _ in 1:NSH], [ReentrantLock() for _ in 1:NSH], Threads.Atomic{Int}(0), nMax, Threads.Atomic{Int}(0))
function kcGet!(f, c::KapCache, k::Tuple{A,Int,S}) where {A,S}
    h = (hash(k) % NSH) + 1; g = c.gen[]
    d = @lock c.lk[h] get!(() -> Dict{Tuple{A,Int,S},Tuple{NTuple{3,S},Int}}(), c.sh[h], (A, S))::Dict{Tuple{A,Int,S},Tuple{NTuple{3,S},Int}}
    v = @lock c.lk[h] get(d, k, MISS)
    if !(v isa Miss)
        v[2] == g || @lock c.lk[h] (d[k] = (v[1], g))
        return v[1]
    end
    w = f()::NTuple{3,S}
    c.n[] >= c.nMax && return w
    @lock c.lk[h] begin
        haskey(d, k) && return d[k][1]
        d[k] = (w, g); Threads.atomic_add!(c.n, 1)
    end
    w
end
# a new phase: the generation advances; above 3/4 of nMax the entries last used before the previous phase are dropped
function kcAge!(c::KapCache)
    g = Threads.atomic_add!(c.gen, 1) + 1
    c.n[] > 3 * c.nMax ÷ 4 || return c
    m = 0
    for h in 1:NSH
        @lock c.lk[h] for (ks, d) in collect(c.sh[h])
            filter!(kv -> kv[2][2] >= g - 2, d); m += length(d)
            # (the evicted slots released: a copy sized for the entries left; called between phases, no fill holds the Dict)
            d2 = typeof(d)(); sizehint!(d2, length(d)); for kv in d; d2[kv[1]] = kv[2]; end; c.sh[h][ks] = d2
        end
    end
    c.n[] = m
    c
end
Base.length(c::KapCache) = c.n[]
Base.empty!(c::KapCache) = (for h in 1:NSH; @lock c.lk[h] empty!(c.sh[h]); end; c.n[] = 0; c)

include("blcSfn.jl")
include("blcTns.jl")
include("blcFil.jl")
include("blcCmp.jl")

# MOMMC is built while the package precompiles, with one table per thread of that process
__init__() = append!(MOMMC.d, [empty(MOMMC.d[1]) for _ in length(MOMMC.d)+1:Threads.maxthreadid()])

"""
    blcSet(s, f, ax, N, P; nMax = 2_000_000) -> BlcSet

The θ-independent state of the fill of a unit cell of `N` equal cells of edges `s` (exact rationals), periodic along the axes `ax` with
periods `P` (cells, P_i >= N_i), at frequency `f`: the cache of the lattice-point trapezoid heat kernels κ(k s, t) at the quadrature nodes
already met (at most `nMax` entries, ~120 B each). The geometry tables (trapezoid moments, Gauss rules) are kept globally, bounded.
Everything that depends on θ (panels, orders, phases) is recomputed per phase.
"""
struct BlcSet{T,D}
    s::NTuple{3,QI}; f::Complex{T}; ax::NTuple{D,Int}; N::NTuple{3,Int}; P::NTuple{D,Int}; kc::KapCache
end
blcSet(s, f::Complex{T}, ax::NTuple{D,Int}, N::NTuple{3,Int}, P::NTuple{D,Int}; nMax::Int = 2_000_000) where {T,D} =
    BlcSet{T,D}(q3(s), f, ax, N, P, KapCache(nMax))

"""
    blcFil(st::BlcSet, θ; form = :toe, kw...) -> (H, A, info)

The Hermitian and anti-Hermitian kernels of the unit cell at Bloch phases `θ` (one per periodic axis), G = H + i A:
- `form = :toe`: Toeplitz form, arrays (3, 3, n1, n2, n3) over m_i ∈ -(N_i-1):(N_i-1) on periodic axes, 0:(N_i-1) on open axes
  (info[:ms] lists them); drops into Gila's 2N circulant with the half storage turned off.
- `form = :twc` (P = N): twisted-circulant form, m_i ∈ 0:(N_i-1) on every axis, G(m + N_i e_i) = e^{-iθ_i} G(m); info[:twc] holds the
  twist residual max|M(m) - e^{-iθ} M(m - N e_i)|/max|M| of H and A over the Toeplitz block.
Keywords (defaults are the audited ones): `sheet = :J10 | :principal` (Im f < 0), `peRun = true` (J31: the entries flagged in info[:pe]
rerun in the wider type, Double64 for Float64 and BigFloat for Double64 or BigFloat; false: only reported), `wmax = Inf` (work budget),
`tol`, `entT = 1e-8` (entries below entT of their offset's max carry their κ, not the eps target), `qcert = true` (per-offset
certification; false only for probes), `route = :auto`.
"""
function blcFil(st::BlcSet, θ::Tuple; form::Symbol = :toe, kw...)
    form in (:toe, :twc) || error("blcFil: form = $form is neither :toe nor :twc")
    form == :twc && st.P != ntuple(j -> st.N[st.ax[j]], length(st.ax)) && error("blcFil: the twisted-circulant form needs P = N (P = $(st.P), N = $(st.N))")
    kcAge!(st.kc)
    H, A, inf = blcFil(st.s, st.f, st.ax, st.N, st.P, θ; cache = st.kc, kw...)
    inf[:cache] = length(st.kc)
    form == :toe && return H, A, inf
    Hw, eH = blcTwc(H, inf[:ms], st.ax, θ); Aw, eA = blcTwc(A, inf[:ms], st.ax, θ)
    inf[:twc] = (H = eH, A = eA)
    Hw, Aw, inf
end


"""
    blcCrt(R, sT, sS, f, ax, L, θ; kw...) -> (H, A, info)

blcTns in Float64 with a rigorous per-entry error certificate, at the cost of a second evaluation in Double64 at the same exact inputs with
tol = eps(Double64). info[:cert] = (H, A), each a 3×3 matrix of relative error bounds |X - X_exact| / |X_exact| <= cert, from
|X - X_D| + (truncation of the Double64 run, proven: 2 eps(Double64) of the offset's largest entry) + (its rounding: eps(Double64) (1 + 16 λ_D,e) |X_D|),
over |X_D| minus that same Double64 error. The only measured ingredient, the per-term rounding (16 ulp here against at most 4 measured), enters at
the eps(Double64) ≈ 5e-32 level, below the Float64 result's own rounding by ten orders. An entry exactly zero in Double64 gets cert = Inf
unless the Float64 value is zero too.
"""
function blcCrt(R, sT, sS, f::ComplexF64, ax::NTuple{D,Int}, L::NTuple{D,<:Rational}, θ::NTuple{D,Any}; kw...) where {D}
    H, A, inf = blcTns(R, sT, sS, f, ax, L, θ; kw...)
    W = Double64; θW = map(t -> t isa Rational ? t : W(t), θ)
    HW, AW, iW = blcTns(R, sT, sS, Complex{W}(f), ax, L, θW; kw..., tol = eps(W))
    λW = get(iW, :λe, nothing)
    m = max(maximum(abs, HW), maximum(abs, AW))
    function crt(X, XW, l)
        c = zeros(3, 3)
        for i in 1:9
            eW = 2 * eps(W) * m + eps(W) * (1 + 16 * (l === nothing ? 1 : l[i])) * abs(XW[i])
            den = abs(XW[i]) - eW
            c[i] = X[i] == 0 && XW[i] == 0 ? 0.0 : den <= 0 ? Inf : Float64((abs(Complex{W}(X[i]) - XW[i]) + eW) / den)
        end
        c
    end
    inf[:cert] = (H = crt(H, HW, λW === nothing ? nothing : λW[1]), A = crt(A, AW, λW === nothing ? nothing : λW[2]))
    H, A, inf
end

end
