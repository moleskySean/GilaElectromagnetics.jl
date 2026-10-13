_strKnd(opr::GlaOprVac) = "G₀"
_strKnd(opr::AsyGlaOprVac) = "Asym(G₀)"
_strKnd(opr::SymGlaOprVac) = "Sym(G₀)"
_strKnd(opr::SusOpr) = ndims(opr.sus) == 3 ? "anisotropic X" : "X"
_strKnd(opr::InvSctOpr) = "(I - XG₀)"
_strKnd(opr::SctOpr) = "(I - XG₀)⁻¹"
_strKnd(opr::GlaOpr) = "G₀(I - XG₀)⁻¹"
_strKnd(opr::GlaSndOprVac) = "fine mesh G₀"
_strKnd(opr::GlaCmpOprVac) = "composite G₀"
_strKnd(opr::AsyGlaCmpOprVac) = "composite Asym(G₀)"
_strKnd(opr::SymGlaCmpOprVac) = "composite Sym(G₀)"

# A scattering operator over a composite volume is marked composite too
_strKnd(::InvSctOpr{T, <:GlaCmpOprVac}) where T<:AbstractFloat = "composite (I - XG₀)"
_strKnd(::SctOpr{T, <:InvSctOpr{T, <:GlaCmpOprVac}}) where T<:AbstractFloat = "composite (I - XG₀)⁻¹"
_strKnd(::GlaOpr{T, <:SctOpr{T, <:InvSctOpr{T, <:GlaCmpOprVac}}}) where T<:AbstractFloat = "composite G₀(I - XG₀)⁻¹"

_srcVol(opr::Union{GlaOprVac, AsyGlaOprVac, SymGlaOprVac}) = opr.mem.srcVol
_srcVol(opr::InvSctOpr) = _srcVol(opr.oprVac)
_srcVol(opr::SctOpr) = _srcVol(opr.invSctOpr)
_srcVol(opr::GlaOpr) = _srcVol(opr.sctOpr)
_trgVol(opr::Union{GlaOprVac, AsyGlaOprVac, SymGlaOprVac}) = opr.mem.trgVol
_trgVol(opr::InvSctOpr) = _trgVol(opr.oprVac)
_trgVol(opr::SctOpr) = _trgVol(opr.invSctOpr)
_trgVol(opr::GlaOpr) = _trgVol(opr.sctOpr)

# self, external or overlapping, lowercase to match the compact and block forms
_relKnd(opr::AbstractGlaOpr) = isselfoperator(opr) ? "self" :
    isexternaloperator(opr) ? "external" : "overlapping"

# A byte count as MiB or GiB, one decimal place
_szStr(bytes::Integer) = bytes >= 2^30 ? "$(round(bytes / 2^30, digits=1)) GiB" :
    bytes >= 2^20 ? "$(round(bytes / 2^20, digits=1)) MiB" :
    "$(round(bytes / 2^10, digits=1)) KiB"

# Bytes actually stored in every egoFur array of an operator's Fourier data
_szBytes(opr::Union{GlaOprVac, AsyGlaOprVac, SymGlaOprVac}) = sum(sizeof, opr.mem.egoFur)
_szBytes(opr::GlaSndOprVac) = _szBytes(opr.opr)
_szBytes(opr::GlaCmpOprVac) = sum(_szBytes, opr.blkMat)
_szBytes(opr::Union{AsyGlaCmpOprVac, SymGlaCmpOprVac}) = _szBytes(opr.opr)

# The volume row: one volume for a self operator, source and target otherwise
_volRow(opr::Union{GlaOprVac, AsyGlaOprVac, SymGlaOprVac}) = isselfoperator(opr) ?
    _volDesc(_srcVol(opr), "  →  ") : string(join(_srcVol(opr).cel, "×"), " cells of ",
    _sclStr(_srcVol(opr).scl), "  →  ", join(_trgVol(opr).cel, "×"), " cells of ", _sclStr(_trgVol(opr).scl))

# The center separation of an external operator's two volumes, as already computed
_sepRow(opr::Union{GlaOprVac, AsyGlaOprVac, SymGlaOprVac}) =
    "(" * join(_ratStr.(_trgVol(opr).org .- _srcVol(opr).org), ", ") * ")λ"

_kerRow(opr::Union{GlaOprVac, AsyGlaOprVac, SymGlaOprVac}) =
    "$(isquasistatic(opr) ? "quasistatic" : "full"), frequency $(frqPhz(opr.mem.cmpInf))"

function _dofStr(opr::AbstractGlaOpr)
    m, n = size(opr)
    return m == n ? "$n dof" : "$m × $n dof"
end
_stoRow(opr::AbstractGlaOpr) = "$(_dofStr(opr)), egoFur $(_szStr(_szBytes(opr)))"

#= The rows of the vacuum family's block show. InvSctOpr, SctOpr and GlaOpr reach
this through GlaOprVac(opr), which resolves to the plain or composite method
depending on what they were built over, so no isa check is needed here. =#
function _oprRow(opr::AbstractGlaOpr)
    rows = [("volume", _volRow(opr))]
    isexternaloperator(opr) && push!(rows, ("separation", _sepRow(opr)))
    push!(rows, ("kernel", _kerRow(opr)))
    push!(rows, ("storage", _stoRow(opr)))
    return rows
end

#= The direction-1 component of a per-cell susceptibility vector, one entry per
cell in coordinates(cvol) order. Exact for the uniform and per-cell isotropic
cases; for a per-component diagonal vector it is a representative, not a mean. =#
function _susCel(sus::AbstractVector, cvol::GlaCmpVol)
    off = _dofOff(cvol)
    cel = similar(sus, off[end] ÷ 3)
    celOff = 0
    for (i, reg) in enumerate(regions(cvol))
        n = prod(reg.cel)
        cel[celOff+1:celOff+n] .= view(sus, off[i]+1:off[i]+n)
        celOff += n
    end
    return cel
end

# A one-line summary of a susceptibility: uniform, ranged, or anisotropic
function _susSummary(x::SusOpr)
    ndims(x.sus) == 3 && return "anisotropic, $(size(x.sus, 1)) cells"
    cel = _susCel(Array(x.sus), x.cvol)
    allequal(cel) && return "uniform χ = $(first(cel))"
    lo, hi = extrema(abs, cel)
    return "χ ∈ [$lo, $hi], $(count(!iszero, cel)) of $(length(cel)) cells nonzero"
end

_oprRow(opr::SusOpr) = [("volume", sprint(show, opr.cvol)),
    ("susceptibility", _susSummary(opr)),
    ("storage", "$(_dofStr(opr)), sus $(_szStr(sizeof(opr.sus)))")]

_oprRow(opr::InvSctOpr) = push!(_oprRow(GlaOprVac(opr)), ("susceptibility", _susSummary(opr.sus)))
_oprRow(opr::Union{SctOpr, GlaOpr}) =
    push!(_oprRow(GlaOprVac(opr)), ("solver", string(nameof(typeof(slv(opr))))))

# The regions row replaces the bespoke composite show with a short table
function _oprRow(opr::GlaCmpOprVac)
    numTrg, numSrc = size(opr.blkMat)
    numSnd = count(blk -> blk isa GlaSndOprVac, opr.blkMat)
    rows = [("regions", "$numTrg target × $numSrc source, $numSnd fine mesh block$(numSnd == 1 ? "" : "s")"),
        ("targets", sprint(show, opr.trgCvl))]
    isselfoperator(opr) || push!(rows, ("sources", sprint(show, opr.srcCvl)))
    push!(rows, ("storage", _stoRow(opr)))
    return rows
end

function _oprRow(opr::Union{AsyGlaCmpOprVac, SymGlaCmpOprVac})
    numReg = nregions(opr.opr.srcCvl)
    numSnd = count(blk -> blk isa GlaSndOprVac, opr.opr.blkMat)
    return [("regions", "$numReg region$(numReg == 1 ? "" : "s"), $numSnd fine mesh block$(numSnd == 1 ? "" : "s")"),
        ("volume", sprint(show, opr.opr.srcCvl)),
        ("storage", _stoRow(opr))]
end

# The compact line's trailing size descriptor
function _dimStr(opr::Union{GlaOprVac, AsyGlaOprVac, SymGlaOprVac})
    isselfoperator(opr) && return join(_srcVol(opr).cel, "×")
    return join(_srcVol(opr).cel, "×") * " → " * join(_trgVol(opr).cel, "×")
end
_dimStr(opr::Union{InvSctOpr, SctOpr, GlaOpr}) = _dimStr(GlaOprVac(opr))
_dimStr(opr::SusOpr) = "$(size(opr, 2)) dof"

function _dimStr(opr::GlaCmpOprVac)
    numTrg, numSrc = size(opr.blkMat)
    return isselfoperator(opr) ? "$numSrc region$(numSrc == 1 ? "" : "s")" : "$numTrg × $numSrc regions"
end
_dimStr(opr::Union{AsyGlaCmpOprVac, SymGlaCmpOprVac}) =
    "$(nregions(opr.opr.srcCvl)) region$(nregions(opr.opr.srcCvl) == 1 ? "" : "s")"

function Base.show(io::IO, opr::AbstractGlaOpr)
    isadjoint(opr) && print(io, "Adjoint ")
    print(io, nameof(typeof(opr)), "{", real(eltype(opr)), "} ", _strKnd(opr), " ",
        _relKnd(opr), " ", nameof(typeof(get_backend(opr))), " ", _dimStr(opr))
end

function Base.show(io::IO, ::MIME"text/plain", opr::AbstractGlaOpr)
    isadjoint(opr) && print(io, "Adjoint ")
    print(io, nameof(typeof(opr)), "{", real(eltype(opr)), "} — ", _strKnd(opr), ", ",
        _relKnd(opr), ", ", nameof(typeof(get_backend(opr))))
    for (lbl, val) in _oprRow(opr)
        _shwRow(io, lbl, val)
    end
end

function Base.show(io::IO, opr::GlaSndOprVac)
    isadjoint(opr) && print(io, "Adjoint ")
    print(io, nameof(typeof(get_backend(opr))), " ")
    print(io, "fine mesh G₀ for $(eltype(opr)) (" *
        join(glaSze(opr, 2)[1:3], "×") * ") -> (" *
        join(glaSze(opr, 1)[1:3], "×") * ") volumes ")
    print(io, "on a (" * join(_srcVol(opr.opr).scl, "×") * ")λ³ mesh")
end
Base.show(io::IO, ::MIME"text/plain", opr::GlaSndOprVac) = show(io, opr)
