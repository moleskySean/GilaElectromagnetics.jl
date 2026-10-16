# One labelled row of a multi-line show; a multi-line val re-indents to align.
function _shwRow(io::IO, lbl::AbstractString, val::AbstractString)
    lines = split(val, '\n')
    print(io, "\n  ", rpad(lbl, 14), "  ", lines[1])
    for line in lines[2:end]
        print(io, "\n", " "^18, line)
    end
end

# The kernel a vacuum operator holds, whole or one Hermitian part
_kerStr(hrmPrt) = hrmPrt === :asy ? "Asym(G₀)" : hrmPrt === :sym ? "Sym(G₀)" : "G₀"

_strKnd(opr::GlaOprVac) = _kerStr(opr.mem.cmpInf.hrmPrt)
_strKnd(opr::SusOpr) = ndims(opr.sus) == 3 ? "anisotropic X" : "X"
_strKnd(opr::InvSctOpr) = "(I - XG₀)"
_strKnd(opr::SctOpr) = "(I - XG₀)⁻¹"
_strKnd(opr::GlaOpr) = "G₀(I - XG₀)⁻¹"
_strKnd(opr::GlaSndOprVac) = "fine mesh G₀"
_strKnd(opr::GlaCmpOprVac) = "composite " * _kerStr(_blkOpt(first(opr.blkMat)).hrmPrt)

# A scattering operator over a composite volume is marked composite too
_strKnd(::InvSctOpr{T, <:GlaCmpOprVac}) where T<:AbstractFloat = "composite (I - XG₀)"
_strKnd(::SctOpr{T, <:InvSctOpr{T, <:GlaCmpOprVac}}) where T<:AbstractFloat = "composite (I - XG₀)⁻¹"
_strKnd(::GlaOpr{T, <:SctOpr{T, <:InvSctOpr{T, <:GlaCmpOprVac}}}) where T<:AbstractFloat = "composite G₀(I - XG₀)⁻¹"

_srcVol(opr::GlaOprVac) = opr.mem.srcVol
_srcVol(opr::AnySctOpr) = _srcVol(vac(opr))
_trgVol(opr::GlaOprVac) = opr.mem.trgVol
_trgVol(opr::AnySctOpr) = _trgVol(vac(opr))

# A byte count as MiB or GiB, one decimal place
_szStr(bytes::Integer) = bytes >= 2^30 ? "$(round(bytes / 2^30, digits=1)) GiB" :
    bytes >= 2^20 ? "$(round(bytes / 2^20, digits=1)) MiB" :
    "$(round(bytes / 2^10, digits=1)) KiB"

# Bytes actually stored in every egoFur array of an operator's Fourier data
_szBytes(opr::GlaOprVac) = sum(sizeof, opr.mem.egoFur)
_szBytes(opr::GlaSndOprVac) = _szBytes(opr.opr)
_szBytes(opr::GlaCmpOprVac) = sum(_szBytes, opr.blkMat)

# The volume row: one volume for a self operator, source and target otherwise
_volRow(opr::GlaOprVac) = isselfoperator(opr) ?
    _volDesc(_srcVol(opr), "  →  ") : string(join(_srcVol(opr).cel, "×"), " cells of ",
    _sclStr(_srcVol(opr).scl), "  →  ", join(_trgVol(opr).cel, "×"), " cells of ", _sclStr(_trgVol(opr).scl))

function _dofStr(opr::AbstractGlaOpr)
    m, n = size(opr)
    return m == n ? "$n dof" : "$m × $n dof"
end
_stoRow(opr::AbstractGlaOpr) = "$(_dofStr(opr)), egoFur $(_szStr(_szBytes(opr)))"

#= The rows under the header of the multi-line show. A scattering operator shows
the vacuum operator it was built over, plain or composite. =#
_oprRow(opr::AbstractGlaOpr) = [("storage", _stoRow(opr))]
_oprRow(opr::GlaOprVac) = [("volume", _volRow(opr)),
    ("kernel", "$(isquasistatic(opr) ? "quasistatic" : "full"), frequency $(opr.mem.cmpInf.frqPhz)"),
    ("storage", _stoRow(opr))]
_oprRow(opr::AnySctOpr) = _oprRow(vac(opr))
_oprRow(opr::SusOpr) = [("volume", sprint(show, opr.cvol)), ("storage", "$(_dofStr(opr)), sus $(_szStr(sizeof(opr.sus)))")]

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

# The compact line's trailing size descriptor
function _dimStr(opr::GlaOprVac)
    isselfoperator(opr) && return join(_srcVol(opr).cel, "×")
    return join(_srcVol(opr).cel, "×") * " → " * join(_trgVol(opr).cel, "×")
end
_dimStr(opr::AnySctOpr) = _dimStr(vac(opr))
_dimStr(opr::GlaSndOprVac) = join(glaSze(opr, 2)[1:3], "×") * " → " * join(glaSze(opr, 1)[1:3], "×")
_dimStr(opr::SusOpr) = "$(size(opr, 2)) dof"

function _dimStr(opr::GlaCmpOprVac)
    numTrg, numSrc = size(opr.blkMat)
    return isselfoperator(opr) ? "$numSrc region$(numSrc == 1 ? "" : "s")" : "$numTrg × $numSrc regions"
end

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
