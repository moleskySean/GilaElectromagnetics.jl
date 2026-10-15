using LinearAlgebra
using Adapt
using KernelAbstractions

"""
    GlaFld{T}

A field (in the physics sense, i.e., a vector at each point in space) over a volume, stored as a flat buffer.

The volume is always a `GlaCmpVol`. A plain `GlaVol` is wrapped into a tiling of
one region, so a field over an ordinary volume is the same type as a field over a
refined one.

The buffer holds √ΔV times the current density coefficient of each degree of
freedom, with ΔV the cell volume of the region the degree of freedom belongs to.
In this basis the Euclidean inner product of two fields is their physical L²
pairing, so `dot` and `norm` can be used as-is. `discretize!` and `regrid` convert
between densities and stored coefficients; from a flat density vector `c`, the
buffer is `sqrt.(cellvolumes(cvol)) .* c`. On a uniform mesh the √ΔV factor is
only one global scalar and the basis is the usual one (up to that scalar).

The layout inside the buffer is the flat degree of freedom layout of
`GlaCmpVol`: region blocks in `regions` order, and inside a block the `vec` of
an array of size `(reg.cel..., 3)`.

# Fields
- `dat::AbstractVector{Complex{T}}`: The flat buffer, on any KernelAbstractions
  backend, whose eltype fixes the storage precision `T`
- `cvol::GlaCmpVol`: The composite volume the field lives on
- `off::Vector{Int}`: Buffer offset of each region block, with the total length
  as a last entry
"""
struct GlaFld{T<:AbstractFloat, V<:AbstractVector{Complex{T}}} <: AbstractVector{Complex{T}}
    dat::V
    cvol::GlaCmpVol
    off::Vector{Int}

    function GlaFld(dat::AbstractVector{Complex{T}}, cvol::GlaCmpVol) where T<:AbstractFloat
        off = _dofOff(cvol)
        if length(dat) != off[end]
            throw(ArgumentError("A buffer of length $(length(dat)) does not fit a composite volume with $(off[end]) degrees of freedom."))
        end
        return new{T, typeof(dat)}(dat, cvol, off)
    end
end

const MultiScaleField = GlaFld

"""
    GlaFld(cvol::GlaCmpVol; dev::Backend=CPU())

Construct the zero field over a composite volume. Same as `zerofield`.

# Arguments
- `cvol::GlaCmpVol`: The composite volume
- `dev::Backend=CPU()`: The KernelAbstractions backend holding the buffer

# Returns
- `GlaFld`: A field of zeros with one entry per degree of freedom
"""
GlaFld(cvol::GlaCmpVol; dev::Backend=CPU()) = zerofield(cvol; dev=dev)

"""
    GlaFld(vol::GlaVol; dev::Backend=CPU())

Construct the zero field over a plain volume. Same as `zerofield`.

# Arguments
- `vol::GlaVol`: The volume
- `dev::Backend=CPU()`: The KernelAbstractions backend holding the buffer

# Returns
- `GlaFld`: A field of zeros with one entry per degree of freedom
"""
GlaFld(vol::GlaVol; dev::Backend=CPU()) = zerofield(GlaCmpVol(vol); dev=dev)

Adapt.adapt_structure(to, fld::GlaFld) = GlaFld(adapt(to, fld.dat), fld.cvol)

Base.zero(fld::GlaFld) = GlaFld(zero(fld.dat), fld.cvol)

# Region blocks only line up if the two fields agree on the tiling
function _chkCvl(fldA::GlaFld, fldB::GlaFld)
    if fldA.cvol != fldB.cvol
        throw(ArgumentError("The two fields live on different composite volumes. Operations that mix fields need a common tiling."))
    end
    return nothing
end

Base.size(fld::GlaFld) = size(fld.dat)
Base.IndexStyle(::Type{<:GlaFld}) = IndexLinear()
Base.getindex(fld::GlaFld, idx::Int) = fld.dat[idx]
Base.setindex!(fld::GlaFld, val, idx::Int) = setindex!(fld.dat, val, idx)
Base.parent(fld::GlaFld) = fld.dat

Base.similar(fld::GlaFld) = GlaFld(similar(fld.dat), fld.cvol)
Base.similar(fld::GlaFld, ::Type{T}) where T<:Complex{<:AbstractFloat} =
    GlaFld(similar(fld.dat, T), fld.cvol)
# Only a complex float buffer can be wrapped, so any other eltype comes back raw
Base.similar(fld::GlaFld, ::Type{T}) where T = similar(fld.dat, T)
Base.copy(fld::GlaFld) = GlaFld(copy(fld.dat), fld.cvol)

function Base.show(io::IO, fld::GlaFld)
    dev = nameof(typeof(get_backend(fld)))
    if nregions(fld.cvol) == 1
        reg = regions(fld.cvol)[1]
        print(io, "Field ($(length(fld)) degrees of freedom, $dev) on a (" *
            join(reg.cel, "×") * ") cells, (" * join(reg.scl, "×") * ")λ³ volume")
        return
    end
    print(io, "Composite field ($(length(fld)) degrees of freedom, $dev)\n  ", fld.cvol)
end
Base.show(io::IO, ::MIME"text/plain", fld::GlaFld) = show(io, fld)

LinearAlgebra.dot(fldA::GlaFld, fldB::GlaFld) =
    (_chkCvl(fldA, fldB); dot(fldA.dat, fldB.dat))
LinearAlgebra.norm(fld::GlaFld, p::Real=2) = norm(fld.dat, p)

function LinearAlgebra.axpy!(alp, fldX::GlaFld, fldY::GlaFld)
    _chkCvl(fldX, fldY)
    axpy!(alp, fldX.dat, fldY.dat)
    return fldY
end

function LinearAlgebra.axpby!(alp, fldX::GlaFld, bet, fldY::GlaFld)
    _chkCvl(fldX, fldY)
    axpby!(alp, fldX.dat, bet, fldY.dat)
    return fldY
end

LinearAlgebra.rmul!(fld::GlaFld, alp::Number) = (rmul!(fld.dat, alp); fld)
Base.fill!(fld::GlaFld, val) = (fill!(fld.dat, val); fld)

function Base.copyto!(fldDst::GlaFld, fldSrc::GlaFld)
    _chkCvl(fldDst, fldSrc)
    copyto!(fldDst.dat, fldSrc.dat)
    return fldDst
end

Base.BroadcastStyle(::Type{<:GlaFld}) = Broadcast.ArrayStyle{GlaFld}()

#= Replace every field in a broadcast tree by its buffer. Rebuilding with
broadcasted rather than reusing the wrapper style hands the work to the buffer,
so a GPU field goes through a GPU kernel instead of scalar indexing. =#
_bcUnw(arg) = arg
_bcUnw(fld::GlaFld) = fld.dat
_bcUnw(bc::Broadcast.Broadcasted) =
    Broadcast.broadcasted(bc.f, map(_bcUnw, bc.args)...)

# Walk the same tree for the common composite volume, complaining on a mismatch
_bcCvl(cvol, arg) = cvol
_bcCvl(cvol, bc::Broadcast.Broadcasted) = foldl(_bcCvl, bc.args; init=cvol)
function _bcCvl(cvol, fld::GlaFld)
    if !isnothing(cvol) && cvol != fld.cvol
        throw(ArgumentError("A broadcast mixes fields that live on different composite volumes. Operations that mix fields need a common tiling."))
    end
    return fld.cvol
end

_bcCvl(bc::Broadcast.Broadcasted) = foldl(_bcCvl, bc.args; init=nothing)

function Base.similar(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{GlaFld}},
    ::Type{T}) where T
    cvol = _bcCvl(bc)
    dat = similar(Broadcast.instantiate(_bcUnw(bc)), T)
    T <: Complex{<:AbstractFloat} || return dat
    return GlaFld(dat, cvol)
end

function Base.copy(bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{GlaFld}})
    cvol = _bcCvl(bc)
    dat = Broadcast.materialize(_bcUnw(bc))
    eltype(dat) <: Complex{<:AbstractFloat} || return dat
    return GlaFld(dat, cvol)
end

function Base.copyto!(fld::GlaFld,
    bc::Broadcast.Broadcasted{Broadcast.ArrayStyle{GlaFld}})
    foldl(_bcCvl, bc.args; init=fld.cvol)
    Broadcast.materialize!(fld.dat, _bcUnw(bc))
    return fld
end

"""
    zerofield(::Type{T}, cvol::GlaCmpVol; dev::Backend=CPU())
    zerofield(cvol::GlaCmpVol; dev::Backend=CPU())

Allocate a zero field of storage precision `T` (`dflPrc` when unrequested) over a composite volume.

# Arguments
- `cvol::GlaCmpVol`: The composite volume
- `dev::Backend=CPU()`: The KernelAbstractions backend holding the buffer

# Returns
- `GlaFld{T}`: A field of zeros with one entry per degree of freedom
"""
function zerofield(::Type{T}, cvol::GlaCmpVol; dev::Backend=CPU()) where T<:AbstractFloat
    dat = KernelAbstractions.zeros(dev, Complex{T}, _dofOff(cvol)[end])
    return GlaFld(dat, cvol)
end
zerofield(cvol::GlaCmpVol; dev::Backend=CPU()) = zerofield(dflPrc, cvol; dev=dev)

"""
    zerofield(::Type{T}, vol::GlaVol; dev::Backend=CPU())
    zerofield(vol::GlaVol; dev::Backend=CPU())

Allocate a zero field of storage precision `T` (`dflPrc` when unrequested) over a plain volume.

The volume is taken as a tiling of one region, so the result is an ordinary
`GlaFld` and every operation defined on fields applies to it.

# Arguments
- `vol::GlaVol`: The volume
- `dev::Backend=CPU()`: The KernelAbstractions backend holding the buffer

# Returns
- `GlaFld`: A field of zeros with one entry per degree of freedom
"""
zerofield(::Type{T}, vol::GlaVol; dev::Backend=CPU()) where T<:AbstractFloat = zerofield(T, GlaCmpVol(vol); dev=dev)
zerofield(vol::GlaVol; dev::Backend=CPU()) = zerofield(dflPrc, GlaCmpVol(vol); dev=dev)

"""
    regionview(fld::GlaFld, idx::Integer)

View the block of one region as a 4-tensor.

The result has size `(reg.cel..., 3)` and shares memory with the field, so
writing through it mutates the field. The entries are the stored √ΔV
coefficients, not densities. Dividing the whole view by `sqrt(prod(reg.scl))`
gives the current density, since a region has a single cell size and therefore a
single factor.

The shape and the index order are the ones `GlaOprVac` expects of a source or
target array, so the view can be handed straight to an operator.

# Arguments
- `fld::GlaFld`: The field
- `idx::Integer`: The region index, in the order of `regions(fld.cvol)`

# Returns
- A 4-tensor view of size `(reg.cel..., 3)` into the field buffer
"""
function regionview(fld::GlaFld, idx::Integer)
    reg = regions(fld.cvol)[idx]
    blk = view(fld.dat, (fld.off[idx] + 1):fld.off[idx + 1])
    return reshape(blk, (Int.(reg.cel)..., 3))
end

"""
    eachregion(fld::GlaFld)

Iterate over the region views of a field, in `regions` order.

# Arguments
- `fld::GlaFld`: The field

# Returns
- An iterator of the `regionview` of every region
"""
eachregion(fld::GlaFld) = (regionview(fld, idx) for idx in 1:nregions(fld.cvol))

"""
    discretize!(fld::GlaFld, f)

Fill a field from a function of position.

`f` takes a cell center as an `NTuple{3,Float64}` in wavelengths and returns
something indexable of length three, an `SVector`, a tuple, or a vector, holding
the three components of the current density there. Each value is multiplied by
`sqrt(prod(reg.scl))` on the way into the buffer, which is the conversion from a
density to a stored coefficient. Sampling is setup code, so the values are
always built on the CPU and copied to the buffer at the end.

# Arguments
- `fld::GlaFld`: The field to fill
- `f`: The current density as a function of position

# Returns
- `GlaFld`: The field, filled
"""
function discretize!(fld::GlaFld, f)
    buf = Vector{eltype(fld)}(undef, length(fld))
    for (regIdx, reg) in enumerate(regions(fld.cvol))
        celNum = prod(reg.cel)
        sclFac = sqrt(Float64(prod(reg.scl)))
        for (celIdx, celInd) in enumerate(CartesianIndices(Tuple(reg.cel)))
            pos = ntuple(dir -> Float64(reg.grd[dir][celInd[dir]]), 3)
            val = f(pos)
            for dir in 1:3
                buf[fld.off[regIdx] + (dir - 1) * celNum + celIdx] =
                    val[dir] * sclFac
            end
        end
    end
    copyto!(fld.dat, buf)
    return fld
end

"""
    regrid(fld::GlaFld, scl::NTuple{3,Rational}=finest(fld.cvol))

Read a field off as current densities on a uniform grid.

The grid has cells of size `scl` and covers the bounding box of the composite
volume, which the regions tile exactly. Every uniform cell takes the density of
the source cell holding its center, so the result is piecewise constant and the
regions coarser than `scl` come out as blocks of repeated values. Since `scl`
divides every region scale, cell boundaries never cut across a source cell.

The returned array holds densities, not stored coefficients.

# Arguments
- `fld::GlaFld`: The field to read
- `scl::NTuple{3,Rational}=finest(fld.cvol)`: The uniform cell size

# Returns
- `Array{Complex{T},4}` of size `(cel..., 3)` with `cel` the uniform cell counts, `T` the precision of `fld`

# Throws
- `ArgumentError`: If `scl` is coarser than or incommensurate with the cell size
  of any region, or if it does not evenly divide the bounding box
"""
function regrid(fld::GlaFld, scl::NTuple{3,Rational}=finest(fld.cvol))
    regs = regions(fld.cvol)
    for (idx, reg) in enumerate(regs)
        rat = reg.scl .// scl
        if any(.!isinteger.(rat)) || any(rat .< 1)
            throw(ArgumentError("A resampling scale of $(scl) does not work for region $idx, which has cells of $(reg.scl): the ratio is $(Tuple(rat)). The resampling scale has to divide every region scale a whole number of times."))
        end
    end
    boxLwr = reduce((edgA, edgB) -> min.(edgA, edgB), _lwrEdg.(regs))
    boxUpr = reduce((edgA, edgB) -> max.(edgA, edgB), _uprEdg.(regs))
    celNum = (boxUpr .- boxLwr) .// scl
    if any(.!isinteger.(celNum))
        throw(ArgumentError("A resampling scale of $(scl) does not evenly divide the bounding box, which spans $(Tuple(boxLwr)) to $(Tuple(boxUpr)) for $(Tuple(celNum)) cells."))
    end
    out = zeros(eltype(fld), ntuple(dir -> Int(celNum[dir]), 3)..., 3)
    dat = isa(fld.dat, Array) ? fld.dat : Array(fld.dat)
    for (regIdx, reg) in enumerate(regs)
        rep = ntuple(dir -> Int(reg.scl[dir] // scl[dir]), 3)
        regLwr = _lwrEdg(reg)
        bas = ntuple(dir -> Int((regLwr[dir] - boxLwr[dir]) // scl[dir]), 3)
        celNumReg = prod(reg.cel)
        sclFac = sqrt(Float64(prod(reg.scl)))
        for (celIdx, celInd) in enumerate(CartesianIndices(Tuple(reg.cel)))
            lwr = ntuple(dir -> bas[dir] + (celInd[dir] - 1) * rep[dir], 3)
            rng = ntuple(dir -> (lwr[dir] + 1):(lwr[dir] + rep[dir]), 3)
            for dir in 1:3
                out[rng..., dir] .=
                    dat[fld.off[regIdx] + (dir - 1) * celNumReg + celIdx] / sclFac
            end
        end
    end
    return out
end
