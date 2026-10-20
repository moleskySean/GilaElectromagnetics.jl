# erfc, erfcx and E_p for T = Float64, Double64 (the part the heat-kernel route uses). Principal branches; on a
# cut the sign of a zero imaginary part picks the side, also for Double64 (whose BigFloat conversion drops the sign of -0.0: bg).
# Series that cancel are summed in BigFloat widened by their a priori amplification, continued fractions (modified Lentz) run in T.
const BG = BigFloat
wid(::Type{T}, xtr) where {T} = precision(T) + ceil(Int, xtr) + 24
bg(x::Real) = (b = BG(x); iszero(x) && signbit(x) ? -abs(b) : b)
cB(z::Complex) = Complex{BG}(bg(real(z)), bg(imag(z)))
# round a working-precision BigFloat result to T
rT(::Type{T}, z::Complex) where {T} = Complex{T}(T(real(z)), T(imag(z)))
# Sum_k t_k in BigFloat at precision p, t_{k+1} = t_k * r(k); stops when |t_k| < 2^-p |S|
function serB(t0, r, p; kMax = 1_000_000)
    S = t0; t = t0
    for k in 0:kMax
        t *= r(k); S += t
        abs(t) <= ldexp(abs(S), -p) && k > 2 && return S
    end
    error("serB: no convergence")
end
# continued fractions run in T
# Laplace continued fraction erfc(z) = e^{-z^2}/sqrt(pi) * cfErfc(z), Re z > 0; nothing if not converged in nMax steps
function cfErfc(z::Complex{T}; nMax = 4000) where {T}
    tiny = floatmin(T) * 2^20; e = eps(T)
    f = z; C = z; D = zero(z)
    for n in 1:nMax
        a = T(n) / 2
        D = z + a * D; D = iszero(D) ? tiny : inv(D)
        C = z + a / C; iszero(C) && (C = tiny)
        dl = C * D; f *= dl
        abs(dl - 1) < e && return inv(f)
    end
    nothing
end
# exp(-z^2) with z^2 formed exactly (fma for Float64, widened BigFloat otherwise): the rounding of z^2 would cost |z|^2 eps
function expM2(z::Complex{Float64})
    x, y = reim(z); a = x * x; b = y * y; c = 2x * y
    s = b - a; bb = s - b; es = (b - (s - bb)) + (-a - bb)          # TwoSum: s + es = b - a exactly
    lr = es + fma(y, y, -b) - fma(x, x, -a); li = fma(2x, y, -c)
    exp(complex(s, -c)) * (1 + complex(lr, -li))
end
expM2(z::Complex{T}) where {T} = rT(T, setprecision(() -> exp(-cB(z)^2), BG, precision(T) + 2 * max(0, exponent(abs(z))) + 16))
cfOK(z) = abs(z) >= 3 && real(z) > 0 && (real(z) >= abs(imag(z)) / 4 || abs(z) >= 8)
function erfcSer(z::Complex{T}) where {T}
    p = wid(T, 1.45 * (abs2(z) + max(real(z^2), 0)) + log2(1 + abs(z)) + 8)
    r = setprecision(BG, p) do
        zb = cB(z)
        1 - 2 * serB(zb, k -> -zb^2 * (2k + 1) / ((k + 1) * (2k + 3)), p) / sqrt(BG(pi))
    end
    rT(T, r)
end
function erfcG(z::Complex{T}) where {T}
    real(z) < 0 && return 2 - erfcG(-z)
    if cfOK(z)
        c = cfErfc(z); c === nothing || return expM2(z) * c / sqrt(T(pi))
    end
    erfcSer(z)
end
# E_p(z) = int_1^inf e^{-zt} t^{-p} dt, p integer >= 1
function expSer(p::Integer, z::Complex{T}) where {T}
    q = wid(T, 1.45 * (abs(z) + max(real(z), 0)) + 2 * abs(p) + log2(1 + abs(z)))
    r = setprecision(BG, q) do
        zb = cB(z)
        j = Int(p)
        psi = -BG(Base.MathConstants.eulergamma) + sum(BG(1) / k for k in 1:(j - 1); init = BG(0))
        S = (-zb)^(j - 1) / factorial(big(j - 1)) * (psi - log(zb))
        t = one(Complex{BG}); n = 0
        while true
            n != j - 1 && (S -= t / (n - j + 1))
            n += 1; t *= -zb / n
            n > j && abs(t) <= ldexp(abs(S), -q) && break
        end
        S
    end
    rT(T, r)
end
# Legendre continued fraction for e^z E_p(z), |arg z| < 3pi/4
function cfExp(p, z::Complex{T}) where {T}
    tiny = floatmin(T) * 2^20; e = eps(T); pT = T(p)
    b = z + pT; C = inv(tiny); D = inv(b); h = D
    for i in 1:100_000
        an = -i * (pT - 1 + i); b += 2
        D = inv(an * D + b); C = b + an / C; dl = C * D; h *= dl
        abs(dl - 1) < e && return h
    end
    error("cfExp: no convergence at z = $z")
end
# the CF's rounding grows with its step count (~ precision^2/|z|): series below |z| = max(2, precision(T)/12)
cfExpOK(::Type{T}, z) where {T} = abs(z) >= max(2, precision(T) / 12) && abs(angle(z)) < 3pi / 4
function expintG(p::Integer, z::Complex{T}) where {T}
    p < 1 && error("expintG: integer order must be >= 1")
    cfExpOK(T, z) && return cfExp(p, z) * exp(-z)
    expSer(p, z)
end
