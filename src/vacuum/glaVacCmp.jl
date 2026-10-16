using KernelAbstractions
using KernelAbstractions: supports_float64

"""
    GlaKerOpt{T<:AbstractFloat, B<:KernelAbstractions.Backend}

Options that determine how the Green function operator is computed. `T` is the
real storage precision of the operator data (`Complex{T}`), `genPrc` the
generation precision and `dev` the KernelAbstractions backend the operator
computes on.

# Fields
- `frqPhz::ComplexF64`: Multiplicative scaling factor allowing for complex frequencies
- `genPrc::Type{<:AbstractFloat}`: Generation precision (default `Float64`; `Float32` throws, it would return NaN)
- `qssApx::Bool`: Quasistatic approximation flag (true to generate the quasistatic kernel)
- `adjMod::Bool`: Adjoint mode flag (true if adjoint mode is enabled)
- `dev::B`: Backend the operator computes on
- `blc::Union{Nothing,NamedTuple}`: Bloch lattice `(ax, P, θ)` of a periodic unit cell: periodic axes, periods in cells, one Bloch phase per axis (radians, or an exact twist θ/2π as a `Rational`), a composite block's with the periods `L` in λ for `P`; `nothing` is vacuum
- `blcSet`: The `BlcSet` of the current frequency, kept for `setBlc!`

`Float64` storage on a backend without double precision throws an `ArgumentError`.
"""
mutable struct GlaKerOpt{T<:AbstractFloat, B<:KernelAbstractions.Backend}
    frqPhz::ComplexF64
    genPrc::Type{<:AbstractFloat}
    qssApx::Bool
    adjMod::Bool
    dev::B
    blc::Union{Nothing,NamedTuple}
    blcSet::Union{Nothing,BlcSet}
    function GlaKerOpt{T}(dev::B = CPU(); frqPhz::Number = 1.0+0.0im, genPrc::Type{<:AbstractFloat} = Float64,
        qssApx::Bool = false, adjMod::Bool = false, blc = nothing, blcSet = nothing) where {T<:AbstractFloat, B<:KernelAbstractions.Backend}
        chkGenPrc(genPrc)
        T === Float64 && !supports_float64(dev) && throw(ArgumentError("$(nameof(B)) has no Float64. Build the operator in Float32."))
        qssApx && blc !== nothing && throw(ArgumentError("The quasistatic kernel has no Bloch lattice."))
        return new{T, B}(frqPhz, genPrc, qssApx, adjMod, dev, blc, blcSet)
    end
end

# the options are mutable, so the constructor's genPrc guard has to hold here too
function Base.setproperty!(opt::GlaKerOpt, fld::Symbol, val)
    fld === :genPrc && chkGenPrc(val)
    return setfield!(opt, fld, convert(fieldtype(typeof(opt), fld), val))
end

# genPrc = Float32 overflows the h_l recursion near l ≈ 27 and returns NaN
chkGenPrc(genPrc::Type{<:AbstractFloat}) = genPrc === Float32 &&
    throw(ArgumentError("genPrc = Float32 would return NaN: the contact h_l recursion overflows near l ≈ 27. Use Float64 (default) or Double64."))

"""
    GlaKerOpt{T}(dev::KernelAbstractions.Backend=CPU(); frqPhz=1.0+0.0im, genPrc=Float64, qssApx=false, adjMod=false, blc=nothing, blcSet=nothing)
    GlaKerOpt(dev::KernelAbstractions.Backend=CPU(); kwargs...)

Options on `dev`, by default a phase factor of 1, `Float64` generation, the full
kernel, not adjoint and no Bloch lattice. The storage precision `T` defaults to `dflPrc`.
"""
GlaKerOpt(dev::KernelAbstractions.Backend=CPU(); kw...) = GlaKerOpt{dflPrc}(dev; kw...)

"""
    GlaKerOpt{T}(opt::GlaKerOpt; dev::KernelAbstractions.Backend=opt.dev)

Copy of `opt` with storage precision `T`, on `dev`.
"""
GlaKerOpt{T}(opt::GlaKerOpt; dev::KernelAbstractions.Backend=opt.dev) where T<:AbstractFloat =
    GlaKerOpt{T}(dev; opt.frqPhz, opt.genPrc, opt.qssApx, opt.adjMod, opt.blc, opt.blcSet)

"""
    devMem(dev::KernelAbstractions.Backend)

Bytes free on `dev` for the Fourier stage of operator creation, `typemax(Int)`
where the backend does not say. `GilaCUDAExt` answers for CUDA.
"""
devMem(::KernelAbstractions.Backend) = typemax(Int)
