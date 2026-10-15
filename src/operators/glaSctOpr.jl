"""
    InvSctOpr{T, O}

The inverse scattering operator (I - XG₀), with X over the source tiling of G₀.
"""
mutable struct InvSctOpr{T<:AbstractFloat, O<:AbstractGlaVacOpr{T}} <: AbstractGlaOpr{T}
    oprVac::O
    # setSus! may store either storage shape
    sus::SusOpr{T}

    #= The susceptibility comes in any of the forms SusOpr accepts, and is the
    one place a precision conversion is allowed. =#
    function InvSctOpr{T}(oprVac::AbstractGlaVacOpr{T}, sus) where T<:AbstractFloat
        if oprVac isa GlaCmpOprVac && !isselfoperator(oprVac)
            throw(ArgumentError("An inverse scattering operator needs a self operator, and this composite operator maps between two different tilings."))
        end
        return new{T, typeof(oprVac)}(oprVac, sus isa SusOpr ? sus :
            SusOpr{T}(_srcCvl(oprVac), sus; dev=get_backend(oprVac)))
    end
end

InvSctOpr(oprVac::AbstractGlaVacOpr{T}, sus) where T<:AbstractFloat = InvSctOpr{T}(oprVac, sus)

"""
    SctOpr{T, I, S}

The scattering operator (I - XG₀)⁻¹, applied by solving with `slv`.
"""
mutable struct SctOpr{T<:AbstractFloat, I<:InvSctOpr{T}, S<:GlaSlv} <: AbstractGlaOpr{T}
    invSctOpr::I
    slv::S
end
SctOpr{T}(invSctOpr::InvSctOpr{T}, slv::GlaSlv) where T<:AbstractFloat = SctOpr(invSctOpr, slv)

"""
    GlaOpr{T, S}

The full Green operator G₀(I - XG₀)⁻¹.
"""
mutable struct GlaOpr{T<:AbstractFloat, S<:SctOpr{T}} <: AbstractGlaOpr{T}
    sctOpr::S
end
GlaOpr{T}(sctOpr::SctOpr{T}) where T<:AbstractFloat = GlaOpr(sctOpr)

const InverseScatteringOperator = InvSctOpr
const ScatteringOperator = SctOpr
const GreenOperator = GlaOpr

# The scattering operators, which answer for the vacuum operator inside them
const AnySctOpr = Union{InvSctOpr, SctOpr, GlaOpr}

"""
    GlaOprVac(opr::Union{InvSctOpr, SctOpr, GlaOpr})

The vacuum operator inside a scattering operator, shared rather than copied.
"""
GlaOprVac(opr::InvSctOpr) = opr.oprVac
GlaOprVac(opr::SctOpr) = GlaOprVac(opr.invSctOpr)
GlaOprVac(opr::GlaOpr) = GlaOprVac(opr.sctOpr)

# A vacuum operator is already the vacuum operator the unwrapping looks for
GlaOprVac(opr::AbstractGlaVacOpr) = opr

"""
    InvSctOpr{T}(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray; dev=get_backend(sus), kwargs...)
    InvSctOpr{T}(vol::GlaVol, sus::AbstractArray; dev=get_backend(sus), kwargs...)

The inverse scattering operator (I - XG₀), with `sus` in any array form `SusOpr`
takes, on the source volume. Other keywords go to `GlaOprVac`.
"""
InvSctOpr{T}(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{<:Number}; dev::Backend=get_backend(sus), kw...) where T<:AbstractFloat =
    InvSctOpr{T}(GlaOprVac{T}(trgVol, srcVol; dev, kw...), sus)
InvSctOpr{T}(vol::GlaVol, sus::AbstractArray{<:Number}; kw...) where T<:AbstractFloat = InvSctOpr{T}(vol, vol, sus; kw...)
InvSctOpr(args...; kw...) = InvSctOpr{dflPrc}(args...; kw...)

"""
    InvSctOpr{T}(opr::InvSctOpr)

Convert the storage precision of the operator, and of its susceptibility, to `T`.
"""
InvSctOpr{T}(opr::InvSctOpr{T}) where T<:AbstractFloat = opr
InvSctOpr{T}(opr::InvSctOpr) where T<:AbstractFloat =
    InvSctOpr{T}(Base.typename(typeof(opr.oprVac)).wrapper{T}(opr.oprVac), SusOpr{T}(opr.sus))

"""
    InvSctOpr(opr::Union{SctOpr, GlaOpr})

The inverse scattering operator inside `opr`, shared rather than copied.
"""
InvSctOpr(opr::SctOpr) = opr.invSctOpr
InvSctOpr(opr::GlaOpr) = InvSctOpr(opr.sctOpr)

"""
    SctOpr{T}(vol::GlaVol, sus::AbstractArray; dev=get_backend(sus), slv=GCRODRSolver(), kwargs...)
    SctOpr{T}(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray; dev=get_backend(sus), slv=GCRODRSolver(), kwargs...)

The scattering operator (I - XG₀)⁻¹, applied by solving with `slv`. `sus` as for
`InvSctOpr`; other keywords go to `GlaOprVac`.
"""
SctOpr{T}(vol::GlaVol, sus::AbstractArray{<:Number}; kw...) where T<:AbstractFloat = SctOpr{T}(vol, vol, sus; kw...)
SctOpr{T}(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{<:Number}; slv::GlaSlv=GCRODRSolver(), kw...) where T<:AbstractFloat =
    SctOpr{T}(InvSctOpr{T}(trgVol, srcVol, sus; kw...), slv)
SctOpr(args...; kw...) = SctOpr{dflPrc}(args...; kw...)

"""
    SctOpr(opr::GlaOprVac, sus::AbstractArray{<:Number}; slv::GlaSlv=GCRODRSolver())

The scattering operator over an existing vacuum operator, which it shares.
"""
SctOpr(opr::GlaOprVac{T}, sus::AbstractArray{<:Number}; slv::GlaSlv=GCRODRSolver()) where T<:AbstractFloat =
    SctOpr{T}(InvSctOpr{T}(opr, sus), slv)

"""
    SctOpr{T}(opr::SctOpr)

Convert the storage precision of the operator, and of its susceptibility, to `T`. The solver is kept.
"""
SctOpr{T}(opr::SctOpr{T}) where T<:AbstractFloat = opr
SctOpr{T}(opr::SctOpr) where T<:AbstractFloat = SctOpr{T}(InvSctOpr{T}(opr.invSctOpr), opr.slv)

"""
    SctOpr(opr::GlaOpr)

The scattering operator inside `opr`, shared rather than copied.
"""
SctOpr(opr::GlaOpr) = opr.sctOpr

"""
    GlaOpr{T}(vol::GlaVol, sus::AbstractArray; dev=get_backend(sus), slv=GCRODRSolver(), kwargs...)
    GlaOpr{T}(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray; dev=get_backend(sus), slv=GCRODRSolver(), kwargs...)

The full Green operator G₀(I - XG₀)⁻¹. Arguments as for `SctOpr`.
"""
GlaOpr{T}(vol::GlaVol, sus::AbstractArray{<:Number}; kw...) where T<:AbstractFloat = GlaOpr{T}(SctOpr{T}(vol, sus; kw...))
GlaOpr{T}(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{<:Number}; kw...) where T<:AbstractFloat =
    GlaOpr{T}(SctOpr{T}(trgVol, srcVol, sus; kw...))
GlaOpr(args...; kw...) = GlaOpr{dflPrc}(args...; kw...)

"""
    GlaOpr(opr::Union{GlaOprVac, GlaCmpOprVac}, sus; slv::GlaSlv=GCRODRSolver())

The full Green operator over an existing vacuum operator, which it shares. `sus` as
for `SctOpr` over the same operator.
"""
GlaOpr(opr::Union{GlaOprVac, GlaCmpOprVac}, sus; kw...) = GlaOpr(SctOpr(opr, sus; kw...))

"""
    GlaOpr{T}(opr::GlaOpr)

Convert the storage precision of the operator, and of its susceptibility, to `T`. The solver is kept.
"""
GlaOpr{T}(opr::GlaOpr{T}) where T<:AbstractFloat = opr
GlaOpr{T}(opr::GlaOpr) where T<:AbstractFloat = GlaOpr{T}(SctOpr{T}(opr.sctOpr))

"""
    InvSctOpr{T}(cvol::GlaCmpVol, sus; dev, kwargs...)

The inverse scattering operator (I - XG₀) over a composite volume, with `sus` in
any form `SusOpr` takes and `dev` defaulting as there. X and the √ΔV normalization
are both diagonal, so they commute. Other keywords go to `GlaCmpOprVac`.
"""
InvSctOpr{T}(cvol::GlaCmpVol, sus; dev::Backend=susDev(sus), kw...) where T<:AbstractFloat =
    InvSctOpr{T}(GlaCmpOprVac{T}(cvol; dev, kw...), sus)

"""
    SctOpr{T}(cvol::GlaCmpVol, sus; dev, slv=GCRODRSolver(), kwargs...)

The scattering operator (I - XG₀)⁻¹ over a composite volume. Arguments as for
`InvSctOpr(cvol, sus)`.
"""
SctOpr{T}(cvol::GlaCmpVol, sus; slv::GlaSlv=GCRODRSolver(), kw...) where T<:AbstractFloat =
    SctOpr{T}(InvSctOpr{T}(cvol, sus; kw...), slv)

"""
    SctOpr(opr::GlaCmpOprVac, sus; slv::GlaSlv=GCRODRSolver())

The scattering operator over an existing self composite operator, which it shares.
"""
SctOpr(opr::GlaCmpOprVac{T}, sus; slv::GlaSlv=GCRODRSolver()) where T<:AbstractFloat =
    SctOpr{T}(InvSctOpr{T}(opr, sus), slv)

"""
    GlaOpr{T}(cvol::GlaCmpVol, sus; dev, slv=GCRODRSolver(), kwargs...)

The full Green operator G₀(I - XG₀)⁻¹ over a composite volume. Arguments as for
`SctOpr(cvol, sus)`.
"""
GlaOpr{T}(cvol::GlaCmpVol, sus; kw...) where T<:AbstractFloat = GlaOpr{T}(SctOpr{T}(cvol, sus; kw...))

Adapt.@adapt_structure InvSctOpr
Adapt.@adapt_structure SctOpr
Adapt.@adapt_structure GlaOpr

isadjoint(opr::AnySctOpr) = isadjoint(GlaOprVac(opr))
isquasistatic(opr::AnySctOpr) = isquasistatic(GlaOprVac(opr))
_relKnd(opr::AnySctOpr) = _relKnd(GlaOprVac(opr))
KernelAbstractions.get_backend(opr::AnySctOpr) = get_backend(GlaOprVac(opr))
glaSze(opr::AnySctOpr) = glaSze(GlaOprVac(opr))
glaSze(opr::AnySctOpr, dim::Int) = glaSze(GlaOprVac(opr), dim)
Base.size(opr::AnySctOpr) = size(GlaOprVac(opr))
Base.size(opr::AnySctOpr, dim::Int) = size(GlaOprVac(opr), dim)
_srcCvl(opr::AnySctOpr) = _srcCvl(GlaOprVac(opr))
# X lives on the source tiling, so (I - XG₀) and its inverse stay there
_trgCvl(opr::Union{InvSctOpr, SctOpr}) = _srcCvl(opr)

"""
    setSus!(opr::Union{InvSctOpr, SctOpr, GlaOpr}, sus)

Replace the susceptibility of `opr`, given as a `SusOpr` or in any form `SusOpr`
takes, converted to the precision of `opr`. Returns `opr`.
"""
function setSus!(opr::InvSctOpr{T}, sus) where T<:AbstractFloat
    opr.sus = sus isa SusOpr ? sus :
        SusOpr{T}(opr.sus.cvol, sus; dev=get_backend(opr.oprVac))
    return opr
end

setSus!(opr::Union{SctOpr, GlaOpr}, sus) = (setSus!(InvSctOpr(opr), sus); opr)

slv(opr::InvSctOpr) = slv(opr.oprVac)
slv(opr::SctOpr) = opr.slv
slv(opr::GlaOpr) = opr.sctOpr.slv

"""
    sus(opr::AbstractGlaOpr)

The susceptibility operator X of a scattering operator, shared rather than
copied, so an `adjoint!` on either is seen by both. `setSus!` replaces it, and an
earlier handle keeps the old one.
"""
sus(opr::InvSctOpr) = opr.sus
sus(opr::Union{SctOpr, GlaOpr}) = sus(InvSctOpr(opr))

#= (I - XG₀), in place on the input. In adjoint mode the vacuum operator is
already the adjoint and the susceptibility already the conjugate transpose,
which leaves I - G₀' X'. =#
function mulAct!(opr::InvSctOpr{T}, act::AbstractVector{Complex{T}}) where T<:AbstractFloat
    isadjoint(opr) && return act .-= mulAct!(opr.oprVac, mulAct!(opr.sus, copy(act)))
    return act .-= mulAct!(opr.sus, mulAct!(opr.oprVac, copy(act)))
end

mulAct!(opr::SctOpr{T}, act::AbstractVector{Complex{T}}) where T<:AbstractFloat = solve(opr.invSctOpr, act, opr.slv)

# G₀ after the solve, the two the other way around in adjoint mode
function mulAct!(opr::GlaOpr{T}, act::AbstractVector{Complex{T}}) where T<:AbstractFloat
    isadjoint(opr) && return mulAct!(opr.sctOpr, mulAct!(opr.sctOpr.invSctOpr.oprVac, act))
    return mulAct!(opr.sctOpr.invSctOpr.oprVac, mulAct!(opr.sctOpr, act))
end

function adjoint!(opr::InvSctOpr)
    opr.oprVac = adjoint!(opr.oprVac)
    adjoint!(opr.sus)
    return opr
end
function adjoint!(opr::SctOpr)
    opr.invSctOpr = adjoint!(opr.invSctOpr)
    return opr
end
function adjoint!(opr::GlaOpr)
    opr.sctOpr = adjoint!(opr.sctOpr)
    return opr
end

_invAct(opr::SctOpr, v) = opr.invSctOpr * v
# The inverse action is a matvec, so it is an admissible preconditioner
isvarying(::SctOpr) = false
#= The inverse of G₀(I - XG₀)⁻¹ is (I - XG₀)G₀⁻¹, so only the vacuum half is
solved for. The two halves swap in adjoint mode, as they do in mulAct!. =#
function _invAct(opr::GlaOpr, v)
    invSct = opr.sctOpr.invSctOpr
    return isadjoint(opr) ? solve(invSct.oprVac, invSct * v, slv(invSct)) :
        invSct * solve(invSct.oprVac, v, slv(invSct))
end

#= The full operator on a field is the vacuum operator after the solve, the two
the other way around in adjoint mode. Both halves take a field, so the tiling
rides along and the checks happen inside them. =#
Base.:*(opr::GlaOpr{T}, fld::GlaFld{T}) where T<:AbstractFloat = isadjoint(opr) ?
    opr.sctOpr * (opr.sctOpr.invSctOpr.oprVac * fld) :
    opr.sctOpr.invSctOpr.oprVac * (opr.sctOpr * fld)
