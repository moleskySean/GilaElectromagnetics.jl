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

# The tiling the currents of a vacuum operator live on
_srcCvl(oprVac::AbstractGlaVacOpr) = GlaCmpVol(oprVac.mem.srcVol)

_srcCvl(opr::GlaCmpOprVac) = opr.srcCvl

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
InvSctOpr{T}(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{<:Number}; dev::Backend=get_backend(sus), kwargs...) where T<:AbstractFloat =
    InvSctOpr{T}(GlaOprVac{T}(trgVol, srcVol; dev, kwargs...), sus)
InvSctOpr(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{<:Number}; dev::Backend=get_backend(sus), kwargs...) =
    InvSctOpr{dflPrc}(trgVol, srcVol, sus; dev, kwargs...)

InvSctOpr{T}(vol::GlaVol, sus::AbstractArray{<:Number}; dev::Backend=get_backend(sus), kwargs...) where T<:AbstractFloat =
    InvSctOpr{T}(vol, vol, sus; dev, kwargs...)
InvSctOpr(vol::GlaVol, sus::AbstractArray{<:Number}; dev::Backend=get_backend(sus), kwargs...) =
    InvSctOpr{dflPrc}(vol, vol, sus; dev, kwargs...)

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
SctOpr{T}(vol::GlaVol, sus::AbstractArray{<:Number}; dev::Backend=get_backend(sus), slv::GlaSlv=GCRODRSolver(), kwargs...) where T<:AbstractFloat =
    SctOpr{T}(vol, vol, sus; dev, slv, kwargs...)
SctOpr(vol::GlaVol, sus::AbstractArray{<:Number}; dev::Backend=get_backend(sus), slv::GlaSlv=GCRODRSolver(), kwargs...) =
    SctOpr{dflPrc}(vol, vol, sus; dev, slv, kwargs...)

function SctOpr{T}(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{<:Number}; dev::Backend=get_backend(sus), slv::GlaSlv=GCRODRSolver(), kwargs...) where T<:AbstractFloat
    invSctOpr = InvSctOpr{T}(trgVol, srcVol, sus; dev, kwargs...)
    return SctOpr{T}(invSctOpr, slv)
end
SctOpr(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{<:Number}; dev::Backend=get_backend(sus), slv::GlaSlv=GCRODRSolver(), kwargs...) =
    SctOpr{dflPrc}(trgVol, srcVol, sus; dev, slv, kwargs...)

"""
    SctOpr(opr::GlaOprVac, sus::AbstractArray{<:Number}; slv::GlaSlv=GCRODRSolver())

The scattering operator over an existing vacuum operator, which it shares.
"""
function SctOpr(opr::GlaOprVac{T}, sus::AbstractArray{<:Number}; slv::GlaSlv=GCRODRSolver()) where T<:AbstractFloat
    invSctOpr = InvSctOpr{T}(opr, sus)
    return SctOpr{T}(invSctOpr, slv)
end

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
GlaOpr{T}(vol::GlaVol, sus::AbstractArray{<:Number}; dev::Backend=get_backend(sus), slv::GlaSlv=GCRODRSolver(), kwargs...) where T<:AbstractFloat =
    GlaOpr{T}(vol, vol, sus; dev, slv, kwargs...)
GlaOpr(vol::GlaVol, sus::AbstractArray{<:Number}; dev::Backend=get_backend(sus), slv::GlaSlv=GCRODRSolver(), kwargs...) =
    GlaOpr{dflPrc}(vol, vol, sus; dev, slv, kwargs...)

function GlaOpr{T}(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{<:Number}; dev::Backend=get_backend(sus), slv::GlaSlv=GCRODRSolver(), kwargs...) where T<:AbstractFloat
    sctOpr = SctOpr{T}(trgVol, srcVol, sus; dev, slv, kwargs...)
    return GlaOpr{T}(sctOpr)
end
GlaOpr(trgVol::GlaVol, srcVol::GlaVol, sus::AbstractArray{<:Number}; dev::Backend=get_backend(sus), slv::GlaSlv=GCRODRSolver(), kwargs...) =
    GlaOpr{dflPrc}(trgVol, srcVol, sus; dev, slv, kwargs...)

"""
    GlaOpr(opr::GlaOprVac, sus::AbstractArray{<:Number}; slv::GlaSlv=GCRODRSolver())

The full Green operator over an existing vacuum operator, which it shares.
"""
GlaOpr(opr::GlaOprVac, sus::AbstractArray{<:Number}; slv::GlaSlv=GCRODRSolver()) = GlaOpr(SctOpr(opr, sus; slv=slv))

"""
    GlaOpr{T}(opr::GlaOpr)

Convert the storage precision of the operator, and of its susceptibility, to `T`. The solver is kept.
"""
GlaOpr{T}(opr::GlaOpr{T}) where T<:AbstractFloat = opr
GlaOpr{T}(opr::GlaOpr) where T<:AbstractFloat = GlaOpr{T}(SctOpr{T}(opr.sctOpr))

GlaOpr(opr::InvSctOpr) = GlaOpr(SctOpr(opr))

"""
    InvSctOpr{T}(cvol::GlaCmpVol, sus; dev, kwargs...)

The inverse scattering operator (I - XG₀) over a composite volume, with `sus` in
any form `SusOpr` takes and `dev` defaulting as there. X and the √ΔV normalization
are both diagonal, so they commute. Other keywords go to `GlaCmpOprVac`.
"""
InvSctOpr{T}(cvol::GlaCmpVol, sus; dev::Backend=susDev(sus), kwargs...) where T<:AbstractFloat =
    InvSctOpr{T}(GlaCmpOprVac{T}(cvol; dev, kwargs...), sus)
InvSctOpr(cvol::GlaCmpVol, sus; dev::Backend=susDev(sus), kwargs...) =
    InvSctOpr{dflPrc}(cvol, sus; dev, kwargs...)

"""
    SctOpr{T}(cvol::GlaCmpVol, sus; dev, slv=GCRODRSolver(), kwargs...)

The scattering operator (I - XG₀)⁻¹ over a composite volume. Arguments as for
`InvSctOpr(cvol, sus)`.
"""
SctOpr{T}(cvol::GlaCmpVol, sus; dev::Backend=susDev(sus), slv::GlaSlv=GCRODRSolver(), kwargs...) where T<:AbstractFloat =
    SctOpr{T}(InvSctOpr{T}(cvol, sus; dev, kwargs...), slv)
SctOpr(cvol::GlaCmpVol, sus; dev::Backend=susDev(sus), slv::GlaSlv=GCRODRSolver(), kwargs...) =
    SctOpr{dflPrc}(cvol, sus; dev, slv, kwargs...)

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
GlaOpr{T}(cvol::GlaCmpVol, sus; dev::Backend=susDev(sus), slv::GlaSlv=GCRODRSolver(), kwargs...) where T<:AbstractFloat =
    GlaOpr{T}(SctOpr{T}(cvol, sus; dev, slv, kwargs...))
GlaOpr(cvol::GlaCmpVol, sus; dev::Backend=susDev(sus), slv::GlaSlv=GCRODRSolver(), kwargs...) =
    GlaOpr{dflPrc}(cvol, sus; dev, slv, kwargs...)

"""
    GlaOpr(opr::GlaCmpOprVac, sus; slv::GlaSlv=GCRODRSolver())

The full Green operator over an existing self composite operator, which it shares.
"""
GlaOpr(opr::GlaCmpOprVac{T}, sus; slv::GlaSlv=GCRODRSolver()) where T<:AbstractFloat =
    GlaOpr{T}(SctOpr(opr, sus; slv=slv))

Adapt.@adapt_structure InvSctOpr
Adapt.@adapt_structure SctOpr
Adapt.@adapt_structure GlaOpr

isadjoint(opr::InvSctOpr) = isadjoint(opr.oprVac)
isadjoint(opr::SctOpr) = isadjoint(opr.invSctOpr)
isadjoint(opr::GlaOpr) = isadjoint(opr.sctOpr)

isquasistatic(opr::InvSctOpr) = isquasistatic(opr.oprVac)
isquasistatic(opr::SctOpr) = isquasistatic(opr.invSctOpr)
isquasistatic(opr::GlaOpr) = isquasistatic(opr.sctOpr)

isselfoperator(opr::InvSctOpr) = isselfoperator(opr.oprVac)
isselfoperator(opr::SctOpr) = isselfoperator(opr.invSctOpr)
isselfoperator(opr::GlaOpr) = isselfoperator(opr.sctOpr)

isexternaloperator(opr::InvSctOpr) = isexternaloperator(opr.oprVac)
isexternaloperator(opr::SctOpr) = isexternaloperator(opr.invSctOpr)
isexternaloperator(opr::GlaOpr) = isexternaloperator(opr.sctOpr)

isoverlappingoperator(opr::InvSctOpr) = isoverlappingoperator(opr.oprVac)
isoverlappingoperator(opr::SctOpr) = isoverlappingoperator(opr.invSctOpr)
isoverlappingoperator(opr::GlaOpr) = isoverlappingoperator(opr.sctOpr)

KernelAbstractions.get_backend(opr::InvSctOpr) = get_backend(opr.oprVac)
KernelAbstractions.get_backend(opr::SctOpr) = get_backend(opr.invSctOpr)
KernelAbstractions.get_backend(opr::GlaOpr) = get_backend(opr.sctOpr)

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

function setSus!(opr::SctOpr, sus)
    setSus!(opr.invSctOpr, sus)
    return opr
end

function setSus!(opr::GlaOpr, sus)
    setSus!(opr.sctOpr, sus)
    return opr
end

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

glaSze(opr::InvSctOpr) = glaSze(opr.oprVac)
glaSze(opr::SctOpr) = glaSze(opr.invSctOpr)
glaSze(opr::GlaOpr) = glaSze(opr.sctOpr)

glaSze(opr::InvSctOpr, dim::Int) = glaSze(opr.oprVac, dim)
glaSze(opr::SctOpr, dim::Int) = glaSze(opr.invSctOpr, dim)
glaSze(opr::GlaOpr, dim::Int) = glaSze(opr.sctOpr, dim)

Base.size(opr::InvSctOpr) = size(opr.oprVac)
Base.size(opr::SctOpr) = size(opr.invSctOpr)
Base.size(opr::GlaOpr) = size(opr.sctOpr)
Base.size(opr::Union{InvSctOpr, SctOpr, GlaOpr}, i::Int) = size(opr)[i]

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

function invMul!(w, opr::SctOpr, v, α, β)
    iszero(β) && return w .= α .* (opr.invSctOpr * v)
    return axpby!(α, opr.invSctOpr * v, β, w)
end
# The inverse action is a matvec, so it is an admissible preconditioner
isvarying(::SctOpr) = false
#= The inverse of G₀(I - XG₀)⁻¹ is (I - XG₀)G₀⁻¹, so only the vacuum half is
solved for. The two halves swap in adjoint mode, as they do in mulAct!. =#
function invMul!(w, opr::GlaOpr, v, α, β)
    invSct = opr.sctOpr.invSctOpr
    out = isadjoint(opr) ? solve(invSct.oprVac, invSct * v, slv(invSct)) :
        invSct * solve(invSct.oprVac, v, slv(invSct))
    iszero(β) && return w .= α .* out
    return axpby!(α, out, β, w)
end

#= The tiling a scattering operator reads its input on, checked against the
field. A plain vacuum operator carries a single volume, which is a tiling of one
region. =#
function _chkSctFld(oprVac::GlaCmpOprVac, fld::GlaFld)
    fld.cvol == oprVac.srcCvl || throw(ArgumentError("The field lives on a different composite volume than the operator. An operator only applies to fields on the tiling it was built for."))
    return oprVac.srcCvl
end

function _chkSctFld(oprVac::AbstractGlaVacOpr, fld::GlaFld)
    srcVol = oprVac.mem.srcVol
    if nregions(fld.cvol) != 1 || regions(fld.cvol)[1] != srcVol
        throw(ArgumentError("The field does not live on the source volume of the operator, which is a ($(join(srcVol.cel, "×"))) cell volume of ($(join(srcVol.scl, "×")))λ³ cells."))
    end
    return fld.cvol
end

"""
    *(opr::InvSctOpr, fld::GlaFld)

Apply an inverse scattering operator to a field on its tiling.
"""
function Base.:*(opr::InvSctOpr{T}, fld::GlaFld{T}) where T<:AbstractFloat
    cvol = _chkSctFld(opr.oprVac, fld)
    return GlaFld(opr * fld.dat, cvol)
end

"""
    *(opr::SctOpr, fld::GlaFld)

Apply a scattering operator to a field on its tiling, by an iterative solve.
"""
function Base.:*(opr::SctOpr{T}, fld::GlaFld{T}) where T<:AbstractFloat
    cvol = _chkSctFld(opr.invSctOpr.oprVac, fld)
    return GlaFld(solve(opr.invSctOpr, fld.dat, opr.slv), cvol)
end

#= The full operator on a field is the vacuum operator after the solve, the two
the other way around in adjoint mode. Both halves take a field, so the tiling
rides along and the checks happen inside them. =#
Base.:*(opr::GlaOpr{T}, fld::GlaFld{T}) where T<:AbstractFloat = isadjoint(opr) ?
    opr.sctOpr * (opr.sctOpr.invSctOpr.oprVac * fld) :
    opr.sctOpr.invSctOpr.oprVac * (opr.sctOpr * fld)
