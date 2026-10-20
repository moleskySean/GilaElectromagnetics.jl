module GilaVoxelShapesExt

import GilaElectromagnetics: GlaVol, GlaCmpVol, _regSus
using VoxelShapes: Region, CompositeGrid, Geometry, rasterize, cells, scale, center, regions

GlaVol(reg::Region) = GlaVol(cells(reg), scale(reg), center(reg))
GlaCmpVol(grd::CompositeGrid) = GlaCmpVol(GlaVol.(regions(grd)))

_regSus(::Type{T}, reg::GlaVol, ::Integer, geo::Geometry) where T<:AbstractFloat =
    vec(Array{Complex{T}}(rasterize(geo, Region(reg.cel, reg.scl, reg.org))))

end
