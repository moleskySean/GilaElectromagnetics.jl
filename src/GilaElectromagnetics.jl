"""
Gila implements single (complex) frequency electromagnetic Green functions 
between generalized source and target cuboid ``volumes''. Technical details 
are available in the supporting glaDoc file.

Author: Sean Molesky 
Distribution: The code distributed under GNU LGPL.
"""
module GilaElectroMag
using AbstractFFTs, FFTW, LinearAlgebra, FastGaussQuadrature, Cubature, 
CUDA, Base.Threads, ThreadsX 
#=
glaMem defines the memory structures used in Gila. 

exported definitions
--------------------
GlaVol---basic cuboid memory: relative cell number and size, grid values, etc. 

genGlaVol---generator for GlaVol memory structure. 

GlaKerOpt---hardware options for Green function protocol (egoOpr!), switch 
between CPU and GPU resources, define number of available compute threads, etc.

GlaOprMem---container for the volume interaction elements (as well as internal 
information) used in computing the Green function interaction between a pair of 
volumes. GlaOprMem is effectively a stored Green function.

GlaOprMem---generate operator memory structure, from a volume or pair of 
volumes, for use by egoOpr!

notable internal definitions
----------------------------
GlaExtInf---memory structure for translating between distinct grid layouts.
glaVolEveGen---regenerate and scale a GlaVol to have even number of cells.
=#
include("glaMem.jl")
export GlaVol, GlaKerOpt, GlaOprMem, GlaExtInf
#=
glaInt contains support functions for numerical (real space) integration of 
the cuboid interactions described by the electromagnetic Green function. This 
code is mostly translated from DIRECTFN_E by Athanasios Polimeridis.

Reference: Polimeridis AG, Vipiana F, Mosig JR, Wilton DR. 
DIRECTFN: Fully numerical algorithms for high precision computation of singular 
integrals in Galerkin SIE methods. 
IEEE Transactions on Antennas and Propagation. 2013; 61(6):3112-22.

exported definitions
--------------------

notable internal definitions
----------------------------
=#
include("glaInt.jl")
#=
glaGen calculates the unique elements of the electromagnetic Green functions, 
embedded in circulant form, via calls to the code contained in glaInt. 
Integration tolerances ultimately used by glaInt.jl are introduced in this file.

exported definitions
--------------------

notable internal definitions
----------------------------
genEgoSlf!---computation of interaction elements for sources within one volume.

genEgoExt!---computation of interaction elements between a source and target
volume. The source and target volume are allowed to touch and have cell scales 
that differ by integer factors. 
=#
include("glaGen.jl")
#=
glaAct contains the package protocol for computing matrix vector products---the 
action of the electromagnetic Green function on a specified current 
distribution---for a given volume or pair of volumes. 

exported definitions
--------------------
egoOpr!---given a GlaOprMem structure and source current distribution, 
yield the resulting electromagnetic fields. 

notable internal definitions
----------------------------
=#
include("glaAct.jl")
export egoOpr! 
end