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
        isselfoperator(oprVac) || throw(ArgumentError("An inverse scattering operator needs a self operator, and this one is $(_relKnd(oprVac))."))
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
    vac(opr::AbstractGlaOpr)

The vacuum operator of `opr`, shared rather than copied. A vacuum operator is its own.
"""
vac(opr::AbstractGlaVacOpr) = opr
vac(opr::InvSctOpr) = opr.oprVac
vac(opr::SctOpr) = vac(opr.invSctOpr)
vac(opr::GlaOpr) = vac(opr.sctOpr)

"""
    InvSctOpr{T}(vol::Union{GlaVol, GlaCmpVol}, sus; dev, kwargs...)

The inverse scattering operator (I - XG₀) of `vol` on itself, with `sus` in any
form `SusOpr` takes and `dev` defaulting as there. Other keywords go to
`GlaOprVac`, or to `GlaCmpOprVac` for a composite volume. X and the √ΔV
normalization of a composite volume are both diagonal, so they commute.
"""
InvSctOpr{T}(vol::Union{GlaVol, GlaCmpVol}, sus; dev::Backend=susDev(sus), kw...) where T<:AbstractFloat =
    InvSctOpr{T}(GlaOprVac{T}(vol; dev, kw...), sus)
InvSctOpr(args...; kw...) = InvSctOpr{dflPrc}(args...; kw...)

"""
    InvSctOpr{T}(opr::InvSctOpr)

Convert the storage precision of the operator, and of its susceptibility, to `T`.
"""
InvSctOpr{T}(opr::InvSctOpr{T}) where T<:AbstractFloat = opr
InvSctOpr{T}(opr::InvSctOpr) where T<:AbstractFloat =
    InvSctOpr{T}(Base.typename(typeof(opr.oprVac)).wrapper{T}(opr.oprVac), SusOpr{T}(opr.sus))

"""
    SctOpr{T}(vol::Union{GlaVol, GlaCmpVol}, sus; dev, slv=GCRODRSolver(), kwargs...)

The scattering operator (I - XG₀)⁻¹, applied by solving with `slv`. Other
arguments as for `InvSctOpr`.
"""
SctOpr{T}(vol::Union{GlaVol, GlaCmpVol}, sus; slv::GlaSlv=GCRODRSolver(), kw...) where T<:AbstractFloat =
    SctOpr{T}(InvSctOpr{T}(vol, sus; kw...), slv)
SctOpr(args...; kw...) = SctOpr{dflPrc}(args...; kw...)

"""
    SctOpr(opr::Union{GlaOprVac, GlaCmpOprVac}, sus; slv::GlaSlv=GCRODRSolver())

The scattering operator over an existing vacuum operator, which it shares. A
composite operator has to be a self operator.
"""
SctOpr(opr::Union{GlaOprVac{T}, GlaCmpOprVac{T}}, sus; slv::GlaSlv=GCRODRSolver()) where T<:AbstractFloat =
    SctOpr{T}(InvSctOpr{T}(opr, sus), slv)

"""
    SctOpr{T}(opr::SctOpr)

Convert the storage precision of the operator, and of its susceptibility, to `T`. The solver is kept.
"""
SctOpr{T}(opr::SctOpr{T}) where T<:AbstractFloat = opr
SctOpr{T}(opr::SctOpr) where T<:AbstractFloat = SctOpr{T}(InvSctOpr{T}(opr.invSctOpr), opr.slv)

"""
    GlaOpr{T}(vol::Union{GlaVol, GlaCmpVol}, sus; dev, slv=GCRODRSolver(), kwargs...)

The full Green operator G₀(I - XG₀)⁻¹. Arguments as for `SctOpr`.
"""
GlaOpr{T}(vol::Union{GlaVol, GlaCmpVol}, sus; kw...) where T<:AbstractFloat = GlaOpr{T}(SctOpr{T}(vol, sus; kw...))
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

Adapt.@adapt_structure InvSctOpr
Adapt.@adapt_structure SctOpr
Adapt.@adapt_structure GlaOpr

isadjoint(opr::AnySctOpr) = isadjoint(vac(opr))
isquasistatic(opr::AnySctOpr) = isquasistatic(vac(opr))
_relKnd(opr::AnySctOpr) = _relKnd(vac(opr))
KernelAbstractions.get_backend(opr::AnySctOpr) = get_backend(vac(opr))
glaSze(opr::AnySctOpr) = glaSze(vac(opr))
glaSze(opr::AnySctOpr, dim::Int) = glaSze(vac(opr), dim)
Base.size(opr::AnySctOpr) = size(vac(opr))
Base.size(opr::AnySctOpr, dim::Int) = size(vac(opr), dim)
_srcCvl(opr::AnySctOpr) = _srcCvl(vac(opr))
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

setSus!(opr::SctOpr, sus) = (setSus!(opr.invSctOpr, sus); opr)
setSus!(opr::GlaOpr, sus) = (setSus!(opr.sctOpr, sus); opr)

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
sus(opr::SctOpr) = sus(opr.invSctOpr)
sus(opr::GlaOpr) = sus(opr.sctOpr)

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
