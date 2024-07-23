#=
Conventions for the values returned by the weak functions. Small letters 
correspond to normal face directions; capital letters correspond to grid 
increment directions. 

Self 	xx  yy  zz
     	1   2   3


Edge 	xxY xxZ yyX yyZ zzX zzY xy xz yz
     	1   2   3   4   5   6   7  8  9     


Vertex 	xx  yy  zz  xy  xz  yz 
       	1   2   3   4   5   6

GilaWInt contains all necessary support functions for numerical integration of 
the electromagnetic Green function. This code is translated from DIRECTFN_E by 
Athanasios Polimeridis, and is distributed under the GNU LGPL.

Author: Sean Molesky

Reference: Polimeridis AG, Vipiana F, Mosig JR, Wilton DR. 
DIRECTFN: Fully numerical algorithms for high precision computation of singular 
integrals in Galerkin SIE methods. 
IEEE Transactions on Antennas and Propagation. 2013; 61(6):3112-22.

In what follows the word weak is used in reference to the fact that the scalar 
Green function surface integral is weakly singular: the integrand exhibits a 
singularity proportional to the inverse of the separation distance. The letters 
S, E and V refer, respectively, to integration over self-adjacent triangles, 
edge-adjacent triangles, and vertex-adjacent triangles. 

The article cited above contains useful error comparison plots for the number 
evaluation points considered. 
=#
#=
Returns the scalar (Helmholtz) Green function. The separation dstMag is assumed 
to be scaled by wavelength. 
=#
@inline function sclEgo(dstMag::AbstractFloat, frqPhz::T)::ComplexF64 where
	T<:Union{ComplexF64,ComplexF32}
	return cis(2π * dstMag * frqPhz) / (4 * π * dstMag * frqPhz^2)
end
#=
Returns the scalar (Helmholtz) Green function with the singularity removed. The 
separation distance dstMag is assumed to be scaled by the wavelength. The 
function is used in the included glaIntSup.jl code to improve the convergence of
all weakly singular integrals.
=#
@inline function sclEgoN(dstMag::AbstractFloat, frqPhz::T)::ComplexF64 where 
	T<:Union{ComplexF64,ComplexF32}
    if dstMag > 1e-7
        return (cis(2π * dstMag * frqPhz) - 1) / 
        (4 * π * dstMag * frqPhz^2)
    else
        return ((im / frqPhz) - π * dstMag) / 2
    end
end
#=
Returns the three dimensional Euclidean norm of a vector. 
=#
@inline function dstMag(v1::AbstractFloat, v2::AbstractFloat, 
	v3::AbstractFloat)::Float64
	return sqrt(v1^2 + v2^2 + v3^2)
end
#=
Head function for integration over coincident square panels. The scl vector 
contains the characteristic lengths of a cuboid voxel relative to the 
wavelength. glQud1 is an array of Gauss-Legendre quadrature weights and 
positions. The cmpInf parameter determines the level of precision used for 
integral calculations. Namely, cmpInf.intOrd is used internally in all 
weakly singular integral computations. 
=#
function wekS(scl::NTuple{3,Number}, glQud1::Array{<:AbstractFloat,2}, 
	cmpInf::GlaKerOpt)::Array{ComplexF64,1}

	grdPts = Array{Float64}(undef, 3, 18)
	# weak self integrals for the three characteristic faces of a cuboid 
	# dir = 1 -> xy face (z-nrm)   dir = 2 -> xz face (y-nrm) 
	# dir = 3 -> yz face (x-nrm)
	return [wekSDir(3, scl, grdPts, glQud1, cmpInf) +
	rSrfSlf(Float64(scl[2]), Float64(scl[3]), cmpInf);
	wekSDir(2, scl, grdPts, glQud1, cmpInf) +
	rSrfSlf(Float64(scl[1]), Float64(scl[3]), cmpInf); 
	wekSDir(1, scl, grdPts, glQud1, cmpInf) + 
	rSrfSlf(Float64(scl[1]), Float64(scl[2]), cmpInf)]
end
#=
Weak self-integral of a particular face.
=#
function wekSDir(dir::Integer, scl::NTuple{3,Number}, 
	grdPts::Array{<:AbstractFloat,2}, glQud1::Array{<:AbstractFloat,2}, 
	cmpInf::GlaKerOpt)::ComplexF64

	wekGrdPts!(dir, scl, grdPts)
	return (((
	wekSInt(hcat(grdPts[:,1], grdPts[:,2], grdPts[:,5]), glQud1, 
		cmpInf) +
	wekSInt(hcat(grdPts[:,1], grdPts[:,5], grdPts[:,4]), glQud1, 
		cmpInf) +
	wekEInt(hcat(grdPts[:,1], grdPts[:,2], grdPts[:,5], 
	 	grdPts[:,1], grdPts[:,5], grdPts[:,4]), glQud1, 
	cmpInf) +
	wekEInt(hcat(grdPts[:,1], grdPts[:,5], grdPts[:,4], 
		grdPts[:,1], grdPts[:,2], grdPts[:,5]), glQud1, 
	cmpInf)) + (
	wekSInt(hcat(grdPts[:,4], grdPts[:,1], grdPts[:,2]), glQud1, 
		cmpInf) +
	wekSInt(hcat(grdPts[:,4], grdPts[:,2], grdPts[:,5]), glQud1, 
		cmpInf) +
	wekEInt(hcat(grdPts[:,4], grdPts[:,1], grdPts[:,2], 
	 	grdPts[:,4], grdPts[:,2], grdPts[:,5]), glQud1, 
	cmpInf) +
	wekEInt(hcat(grdPts[:,4], grdPts[:,2], grdPts[:,5], 
		grdPts[:,4], grdPts[:,1], grdPts[:,2]), glQud1, 
	cmpInf))) / 2.0)
end
#=
Head function for integration over edge adjacent square panels. See wekS for 
input parameter descriptions. 
=#
function wekE(scl::NTuple{3,Number}, glQud1::Array{<:AbstractFloat,2}, 
	cmpInf::GlaKerOpt)::Array{ComplexF64,1}
	
	grdPts = Array{Float64,2}(undef, 3, 18)
	# labels are panelDir-panelDir-gridIncrement
	vals = wekEDir(3, scl, grdPts, glQud1, cmpInf)
	# lower case letters reference the normal directions of the rectangles
	# upper case letter reference the increasing axis direction when necessary 
	# first set
	xxY = vals[1] + rSrfEdgFlt(Float64(scl[3]), Float64(scl[2]), cmpInf)
	xxZ = vals[3] + rSrfEdgFlt(Float64(scl[2]), Float64(scl[3]), cmpInf)
	xyA = vals[2] + rSrfEdgCrn(Float64(scl[3]), Float64(scl[2]), 
		Float64(scl[1]), cmpInf)
	xzA = vals[4] + rSrfEdgCrn(Float64(scl[2]), Float64(scl[3]), 
		Float64(scl[1]), cmpInf)
	vals = wekEDir(2, scl, grdPts, glQud1, cmpInf)
	# second set
	yyZ = vals[1] + rSrfEdgFlt(Float64(scl[1]), Float64(scl[3]), cmpInf)
	yyX = vals[3] + rSrfEdgFlt(Float64(scl[3]), Float64(scl[1]), cmpInf)
	yzA = vals[2] + rSrfEdgCrn(Float64(scl[1]), Float64(scl[3]), 
		Float64(scl[2]), cmpInf)
	xyB = vals[4] + rSrfEdgCrn(Float64(scl[3]), Float64(scl[2]), 
		Float64(scl[1]), cmpInf)
	vals = wekEDir(1, scl, grdPts, glQud1, cmpInf)
	# third set
	zzX = vals[1] + rSrfEdgFlt(Float64(scl[2]), Float64(scl[1]), cmpInf)
	zzY = vals[3] + rSrfEdgFlt(Float64(scl[1]), Float64(scl[2]), cmpInf)
	xzB = vals[2] + rSrfEdgCrn(Float64(scl[2]), Float64(scl[3]), 
		Float64(scl[1]), cmpInf)
	yzB = vals[4] + rSrfEdgCrn(Float64(scl[1]), Float64(scl[3]), 
		Float64(scl[2]), cmpInf)
	return [xxY; xxZ; yyX; yyZ; zzX; zzY; (xyA + xyB) / 2.0; (xzA + xzB) / 2.0;
	(yzA + yzB) / 2.0]
end
#= 
Weak edge integrals for a given face as specified by dir.
	dir = 1 -> z face -> [y-edge (++ gridX): zz(x), xz(x);
						  x-edge (++ gridY) zz(y) yz(y)]

	dir = 2 -> y face -> [x-edge (++ gridZ): yy(z), yz(z); 
						  z-edge (++ gridX) yy(x) xy(x)]

	dir = 3 -> x face -> [z-edge (++ gridY): xx(y), xy(y); 
						  y-edge (++ gridZ) xx(z) xz(z)]
=#
function wekEDir(dir::Integer, scl::NTuple{3,Number}, 
	grdPts::Array{<:AbstractFloat,2}, glQud1::Array{<:AbstractFloat,2}, 
	cmpInf::GlaKerOpt)::Array{ComplexF64,1}

	wekGrdPts!(dir, scl, grdPts) 
	return [wekEInt(hcat(grdPts[:,1], grdPts[:,2], grdPts[:,5],
	grdPts[:,2], grdPts[:,3], grdPts[:,5]), glQud1, cmpInf) +
	wekVInt(true, hcat(grdPts[:,1], grdPts[:,2], grdPts[:,5], 
	grdPts[:,3], grdPts[:,6], grdPts[:,5]), glQud1, cmpInf) +
	wekVInt(true, hcat(grdPts[:,1], grdPts[:,5], grdPts[:,4],
	grdPts[:,2], grdPts[:,3], grdPts[:,5]), glQud1, cmpInf) +
	wekVInt(true, hcat(grdPts[:,1], grdPts[:,5], grdPts[:,4], 
	grdPts[:,3], grdPts[:,6], grdPts[:,5]), glQud1, cmpInf);
	wekEInt(hcat(grdPts[:,1], grdPts[:,2], grdPts[:,5],
	grdPts[:,2], grdPts[:,11], grdPts[:,5]), glQud1, cmpInf) +
	wekVInt(true, hcat(grdPts[:,1], grdPts[:,2], grdPts[:,5], 
	grdPts[:,11], grdPts[:,14], grdPts[:,5]), glQud1, cmpInf) +
	wekVInt(true, hcat(grdPts[:,1], grdPts[:,5], grdPts[:,4],
	grdPts[:,2], grdPts[:,11], grdPts[:,5]), glQud1, cmpInf) +
	wekVInt(true, hcat(grdPts[:,1], grdPts[:,5], grdPts[:,4], 
	grdPts[:,11], grdPts[:,14], grdPts[:,5]), glQud1, cmpInf);
	wekVInt(true, hcat(grdPts[:,1], grdPts[:,2], grdPts[:,5], 
	grdPts[:,4], grdPts[:,5], grdPts[:,7]), glQud1, cmpInf) +
	wekVInt(true, hcat(grdPts[:,1], grdPts[:,2], grdPts[:,5], 
	grdPts[:,5], grdPts[:,8], grdPts[:,7]), glQud1, cmpInf) +
	wekEInt(hcat(grdPts[:,1], grdPts[:,5], grdPts[:,4], 
	grdPts[:,4], grdPts[:,5], grdPts[:,7]), glQud1, cmpInf) +
	wekVInt(true, hcat(grdPts[:,1], grdPts[:,5], grdPts[:,4], 
	grdPts[:,5], grdPts[:,8], grdPts[:,7]), glQud1, cmpInf);
	wekVInt(true, hcat(grdPts[:,1], grdPts[:,2], grdPts[:,5], 
	grdPts[:,4], grdPts[:,5], grdPts[:,13]), glQud1, cmpInf) +
	wekVInt(true, hcat(grdPts[:,1], grdPts[:,2], grdPts[:,5], 
	grdPts[:,5], grdPts[:,14], grdPts[:,13]), glQud1, cmpInf) +
	wekEInt(hcat(grdPts[:,1], grdPts[:,5], grdPts[:,4], 
	grdPts[:,4], grdPts[:,5], grdPts[:,13]), glQud1, cmpInf) +
	wekVInt(true, hcat(grdPts[:,1], grdPts[:,5], grdPts[:,4], 
	grdPts[:,5], grdPts[:,14], grdPts[:,13]), glQud1, cmpInf)]
end
#=
Head function returning integral values for the Ego function over vertex 
adjacent square panels. See wekS for input parameter descriptions. 
=#
function wekV(scl::NTuple{3,Number}, glQud1::Array{<:AbstractFloat,2}, 
	cmpInf::GlaKerOpt)::Array{ComplexF64,1}

	grdPts = Array{Float64,2}(undef,3,18)
	# vertex integrals for x-normal face
	vals = wekVDir(3, scl, grdPts, glQud1, cmpInf)
	xxO = vals[1]
	xyA = vals[2]
	xzA = vals[3]
	# vertex integrals for y-normal face
	vals = wekVDir(2, scl, grdPts, glQud1, cmpInf)
	yyO = vals[1]
	yzA = vals[2]
	xyB = vals[3]
	# vertex integrals for z-normal face
	vals = wekVDir(1, scl, grdPts, glQud1, cmpInf)
	zzO = vals[1]
	xzB = vals[2]
	yzB = vals[3]
	return[xxO; yyO; zzO; (xyA + xyB) / 2.0; (xzA + xzB) / 2.0; 
	(yzA + yzB) / 2.0]
end
#= 
Weak edge integrals for a given face as specified by dir.
	dir = 1 -> z face -> [zz zx zy]
	dir = 2 -> y face -> [yy yz yx]
	dir = 3 -> x face -> [xx xy xz]
=#
function wekVDir(dir::Integer, scl::NTuple{3,Number}, 
	grdPts::Array{<:AbstractFloat,2}, glQud1::Array{<:AbstractFloat,2}, 
	cmpInf::GlaKerOpt)::Array{ComplexF64,1}

	wekGrdPts!(dir, scl, grdPts) 
	return [wekVInt(false, hcat(grdPts[:,1], grdPts[:,2], grdPts[:,5], 
	grdPts[:,5], grdPts[:,6], grdPts[:,9]), glQud1, cmpInf) +
	wekVInt(false, hcat(grdPts[:,1], grdPts[:,2], grdPts[:,5], 
	grdPts[:,5], grdPts[:,9], grdPts[:,8]), glQud1, cmpInf) +
	wekVInt(false, hcat(grdPts[:,1], grdPts[:,5], grdPts[:,4], 
	grdPts[:,5], grdPts[:,6], grdPts[:,9]), glQud1, cmpInf) +
	wekVInt(false, hcat(grdPts[:,1], grdPts[:,5], grdPts[:,4], 
	grdPts[:,5], grdPts[:,9], grdPts[:,8]), glQud1, cmpInf);
	wekVInt(false, hcat(grdPts[:,1], grdPts[:,2], grdPts[:,5], 
	grdPts[:,5], grdPts[:,17], grdPts[:,14]), glQud1, cmpInf) +
	wekVInt(false, hcat(grdPts[:,1], grdPts[:,2], grdPts[:,5], 
	grdPts[:,5], grdPts[:,8], grdPts[:,17]), glQud1, cmpInf) +
	wekVInt(false, hcat(grdPts[:,1], grdPts[:,5], grdPts[:,4], 
	grdPts[:,5], grdPts[:,17], grdPts[:,14]), glQud1, cmpInf) +
	wekVInt(false, hcat(grdPts[:,1], grdPts[:,5], grdPts[:,4], 
	grdPts[:,5], grdPts[:,8], grdPts[:,17]), glQud1, cmpInf);
	wekVInt(false, hcat(grdPts[:,1], grdPts[:,2], grdPts[:,5], 
	grdPts[:,5], grdPts[:,15], grdPts[:,14]), glQud1, cmpInf) +
	wekVInt(false, hcat(grdPts[:,1], grdPts[:,2], grdPts[:,5], 
	grdPts[:,5], grdPts[:,6], grdPts[:,15]), glQud1, cmpInf) +
	wekVInt(false, hcat(grdPts[:,1], grdPts[:,5], grdPts[:,4], 
	grdPts[:,5], grdPts[:,15], grdPts[:,14]), glQud1, cmpInf) +
	wekVInt(false, hcat(grdPts[:,1], grdPts[:,5], grdPts[:,4], 
	grdPts[:,5], grdPts[:,6], grdPts[:,15]), glQud1, cmpInf)]
end
#=
Generate all unique pairs of cube faces. 
=#
function facPar()::Array{Integer,2}

	fPairs = Array{Integer,2}(undef, 36, 2)
	for i ∈ 1:6, j ∈ 1:6
		k = (i - 1) * 6 + j
		fPairs[k,1] = i
		fPairs[k,2] = j	
	end
	return fPairs
end
#=
Determine scaling factors for surface integrals.
=#
function srfScl(sclT::NTuple{3,Number}, 
	sclS::NTuple{3,Number})::Array{Float64,1}

	srcScl = 1.0
	trgScl = 1.0
	srfScl = Array{Float64,1}(undef, 36)
	
	for srcFId ∈ 1 : 6
		if srcFId == 1 || srcFId == 2 			srcScl = sclS[2] * sclS[3]
		elseif srcFId == 3 || srcFId == 4 		srcScl = sclS[1] * sclS[3]
		else 									srcScl = sclS[1] * sclS[2]
		end
		# target scaling has been switched to source scaling
		for trgFId ∈ 1 : 6 
			if trgFId == 1 || trgFId == 2 		trgScl = sclT[1]
			elseif trgFId == 3 || trgFId == 4 	trgScl = sclT[2]
			else  								trgScl = sclT[3]
			end			
			srfScl[(srcFId - 1) * 6 + trgFId] = Float64(srcScl / trgScl)
		end
	end
	return srfScl
end
#=
Generate array of cuboid faces based from a characteristic size, l[]. 
L and U reference relative positions on the corresponding normal axis.
Points are number in a counter-clockwise convention when viewing the 
face from the exterior of the cube. 
=#
function cubFac(size::NTuple{3,Number})::Array{Float64,3}
	
	yzL = hcat([-size[1], -size[2], -size[3]], [-size[1], size[2], -size[3]], 
		[-size[1], size[2], size[3]], [-size[1], -size[2], size[3]]) ./ 2
	yzU = hcat([size[1], -size[2], -size[3]], [size[1], -size[2], size[3]], 
		[size[1], size[2], size[3]], [size[1], size[2], -size[3]]) ./ 2
	xzL = hcat([-size[1], -size[2], -size[3]], [-size[1], -size[2], size[3]], 
		[size[1], -size[2], size[3]], [size[1], -size[2], -size[3]]) ./ 2
	xzU = hcat([-size[1], size[2], -size[3]], [size[1], size[2], -size[3]], 
		[size[1], size[2], size[3]], [-size[1], size[2], size[3]]) ./ 2
	xyL = hcat([-size[1], -size[2], -size[3]], [size[1], -size[2], -size[3]], 
		[size[1], size[2], -size[3]], [-size[1], size[2], -size[3]]) ./ 2
	xyU = hcat([-size[1], -size[2], size[3]], [-size[1], size[2], size[3]], 
		[size[1], size[2], size[3]], [size[1], -size[2], size[3]]) ./ 2
	return cat(yzL, yzU, xzL, xzU, xyL, xyU, dims = 3)
end
#=
Determine a directional component, set by dir, of the separation vector for a 
pair points, as determined by ord, which may take on values between zero and 
one. The first pair of entries are coordinates in the source surface, the 
second pair of entries are coordinates in the target surface. 
=#
@inline function cubVecAltAdp(dir::Integer, ordVec::Array{<:AbstractFloat,1}, 
	fp::Integer, trgFaces::Array{<:AbstractFloat,3}, 
	srcFaces::Array{<:AbstractFloat,3}, fPairs::Array{<:Integer,2})::Float64
	
	return (trgFaces[dir,1,fPairs[fp,1]] +
		ordVec[3] * (trgFaces[dir,2,fPairs[fp,1]] - 
			trgFaces[dir,1,fPairs[fp,1]]) +
		ordVec[4] * (trgFaces[dir,4,fPairs[fp,1]] - 
			trgFaces[dir,1,fPairs[fp,1]])) -
	(srcFaces[dir,1,fPairs[fp,2]] +
		ordVec[1] * (srcFaces[dir,2,fPairs[fp,2]] - 
			srcFaces[dir,1,fPairs[fp,2]]) +
		ordVec[2] * (srcFaces[dir,4,fPairs[fp,2]] - 
			srcFaces[dir,1,fPairs[fp,2]]))
end
#=
Create grid point system for calculation for calculation of weakly singular 
integrals. 
=#
function wekGrdPts!(dir::Integer, scl::NTuple{3,Number}, 
	grdPts::Array{<:AbstractFloat,2})::Nothing

	if dir == 1
		# standard orientation
		gridX = Float64(scl[1])
		gridY = Float64(scl[2])
		gridZ = Float64(scl[3])
	elseif dir == 2
		# single coordinate rotation
		gridX = Float64(scl[3])
		gridY = Float64(scl[1])
		gridZ = Float64(scl[2])
	elseif dir == 3
		# double coordinate rotation
		gridX = Float64(scl[2]) 
		gridY = Float64(scl[3])
		gridZ = Float64(scl[1])
	else
		error("Invalid direction selection.")
	end
	grdPts[:,1] = [0.0; 	   	 	0.0; 			0.0]
	grdPts[:,2] = [gridX; 	   	 	0.0; 			0.0]
	grdPts[:,3] = [2.0 * gridX; 	0.0; 			0.0]
	grdPts[:,4] = [0.0;         	gridY; 			0.0]
	grdPts[:,5] = [gridX; 		 	gridY; 			0.0]
	grdPts[:,6] = [2.0 * gridX; 	gridY; 			0.0]
	grdPts[:,7] = [0.0; 		 	2.0 * gridY; 	0.0]
	grdPts[:,8] = [gridX; 		 	2.0 * gridY; 	0.0]
	grdPts[:,9] = [2.0 * gridX; 	2.0 * gridY; 	0.0]
	grdPts[:,10] = [0.0; 	 	  	0.0; 			gridZ]
	grdPts[:,11] = [gridX; 	  		0.0; 			gridZ]
	grdPts[:,12] = [2.0 * gridX; 	0.0; 			gridZ]
	grdPts[:,13] = [0.0; 		  	gridY; 			gridZ]
	grdPts[:,14] = [gridX; 	  		gridY; 			gridZ]
	grdPts[:,15] = [2.0 * gridX; 	gridY; 			gridZ]
	grdPts[:,16] = [0.0; 		  	2.0 * gridY; 	gridZ]
	grdPts[:,17] = [gridX;       	2.0 * gridY; 	gridZ]
	grdPts[:,18] = [2.0 * gridX; 	2.0 * gridY; 	gridZ]
	return nothing
end
#=
The code contained in glaIntSup evaluates the integrands called by the wekS, 
wekE, and wekV head functions using a series of variable transformations 
and analytic integral evaluations---reducing the four dimensional surface 
integrals performed for ``standard'' cells to chains of one dimensional 
integrals. No comments are included in this low level code, which is simply a 
julia translation of DIRECTFN_E by Athanasios Polimeridis with added support for
multi-threading. For a complete description of the steps being performed see 
the article cited above and references included therein. 
=#
include("glaIntSup.jl")
#= 
Direct evaluation of 1 / (4 * π * dstMag) integral for a square panel with 
itself. la and lb are the edge lengths. 
=#
@inline function rSrfSlf(la::AbstractFloat, lb::AbstractFloat, 
	cmpInf::GlaKerOpt)::ComplexF64

    return (1 / (48 * π * cmpInf.frqPhz^2)) * (8 * la^3 + 8 * lb^3 
    - 8 * la^2 * sqrt(la^2 + lb^2) - 8 * lb^2 * sqrt(la^2 + lb^2) - 
    3 * la^2 * lb * (2 * log(la) + 2 * log(la + lb - sqrt(la^2 + lb^2)) + 
    log(sqrt(la^2 + lb^2) - lb) - 5 * log(lb + sqrt(la^2 + lb^2)) - 
    2 * log(lb - la + sqrt(la^2 + lb^2)) - 
    2 * log(la + 2 * lb - sqrt(la^2 + 4 * lb^2)) + 
    log(sqrt(la^2 + 4 * lb^2) - 2 * lb) + 
    2 * log(la - 2 * lb + sqrt(la^2 + 4 * lb^2)) + 
    log(2 * lb + sqrt(la^2 + 4 * lb^2)) + 
    2 * log(2 * lb - la + sqrt(la^2 + 4 * lb^2)) - 
    2 * log(la + 2 * lb + sqrt(la^2 + 4 * lb^2))) + 6 * la * lb^2 * 
    (log(64) + 4 * log(lb) + 2 * log(sqrt(la^2 + lb^2) - la) + 
    3 * log(la + sqrt(la^2 + lb^2)) - 3 * log(sqrt(la^2 + 4 * lb^2) - la) - 
    3 * log(sqrt(la^4 + 5 * la^2 * lb^2 + 4 * lb^4) + 
    la * (sqrt(la^2 + lb^2) - la - sqrt(la^2 + 4 * lb^2)))))
end
#= 
Direct evaluation of 1 / (4 * π * dstMag) integral for a pair of cornered edge 
panels. la, lb, and lc are the edge lengths, and la is assumed to be common to 
both panels. 
=#
@inline function rSrfEdgCrn(la::AbstractFloat, lb::AbstractFloat, 
	lc::AbstractFloat, cmpInf::GlaKerOpt)::ComplexF64

   	return (1 / (48 * π * cmpInf.frqPhz^2)) * (8 * lb * lc * 
   	sqrt(lb^2 + lc^2) - 8 * lb * lc * sqrt(la^2 + lb^2 + lc^2) - 12 * la^3 * 
    acot(la * lc / (la^2 + lb^2 - lb * sqrt(la^2 + lb^2 + lc^2))) + 
    12 * la^3 * atan(la / lc) - 
    12 * la * lc^2 * atan(la * lb / (lc * sqrt(la^2 + lb^2 + lc^2))) - 
    12 * la * lb^2 * atan(la * lc / (lb * sqrt(la^2 + lb^2 + lc^2))) - 
    16 * la^3 * atan(lb * lc / (la * sqrt(la^2 + lb^2 + lc^2))) + 
    6 * lc^3 * atanh(lb / sqrt(lb^2 + lc^2)) - 
    6 * lc * (la^2 + lc^2) * atanh(lb / sqrt(la^2 + lb^2 + lc^2)) - 
    15 * la^2 * lc * log(la^2 + lc^2) - lc^3 * log(la^2 + lc^2) + 
    2 * lc^3 * log(lc / (lb + sqrt(lb^2 + lc^2))) + 
    6 * la^2 * lc * log(sqrt(la^2 + lb^2 + lc^2) - lb) + 
    24 * la^2 * lc * log(sqrt(la^2 + lb^2 + lc^2) + lb) + 
    2 * lc^3 * log(sqrt(la^2 + lb^2 + lc^2) + lb) + 
    6 * la * lb * (-2 * la * log(la^2 + lb^2) - 
    lc * log((lb^2 + lc^2) * (sqrt(la^2 + lb^2 + lc^2) - la)) + 
    3 * lc * log(la + sqrt(la^2 + lb^2 + lc^2)) + 
    la * log(sqrt(la^2 + lb^2 + lc^2) - lc) + 
    3 * la * log(sqrt(la^2 + lb^2 + lc^2) + lc)) + 
    2 * lb^3 * (
    log((sqrt(la^2 + lb^2 + lc^2) - lc) / (lc + sqrt(la^2 + lb^2 + lc^2))) + 
    log(1 + (2 * lc * (lc + sqrt(lb^2 + lc^2))) / lb^2)))
end
#= 
Direct evaluation of 1 / (4 * π * dstMag) integral for a pair of flat edge 
panels. la and lb are the edge lengths, and lb is assumed to be ``doubled''. 
=#
@inline function rSrfEdgFlt(la::AbstractFloat, lb::AbstractFloat, 
	cmpInf::GlaKerOpt)::ComplexF64
       
    return (1 / (12 * π * cmpInf.frqPhz^2)) * (-la^3 + 2 * lb^2 * 
    (3 * lb + sqrt(la^2 + lb^2) - 2 * sqrt(la^2 + 4 * lb^2)) + la^2 * 
    (2 * sqrt(la^2 + lb^2) - sqrt(la^2 + 4 * lb^2))) + 
    (1 / (64 * π)) * la * lb * (lb * (-62 * log(2) - 
    5 * log(-la + sqrt(la^2 + lb^2)) + 
    4 * log(8 * lb^2 * (-la + sqrt(la^2 + lb^2))) - 
    33 * log(la + sqrt(la^2 + lb^2)) + 17 * log(-la + sqrt(la^2 + 4 * lb^2)) - 
    24 * log(lb * (-la + sqrt(la^2 + 4 * lb^2))) + 
    57 * log(la + sqrt(la^2 + 4 * lb^2))) + 
    4 * la * (-8 * asinh(lb / la) + 6 * asinh(2 * lb / la) + 
    6 * atanh(lb / sqrt(la^2 + lb^2)) + 12 * log(la) - 
    13 * log(-lb + sqrt(la^2 + lb^2)) + log((-lb + sqrt(la^2 + lb^2)) / la) + 
    log(la / (lb + sqrt(la^2 + lb^2))) - 7 * log(lb + sqrt(la^2 + lb^2)) - 
    2 * log((lb + sqrt(la^2 + lb^2))/la) - 
    3 * log(-(((lb + sqrt(la^2 + lb^2)) * 
    (2 * lb - sqrt(la^2 + 4 * lb^2))) / (la^2))) - 
    3 * log((-lb + sqrt(la^2 + lb^2)) / (-2 * lb + sqrt(la^2 + 4 * lb^2))) + 
    11 * log(-2 * lb + sqrt(la^2 + 4 * lb^2)) - 
    3 * log((lb + sqrt(la^2 + lb^2)) / (2 * lb + sqrt(la^2 + 4 * lb^2))) + 
    log(2 * lb + sqrt(la^2 + 4 * lb^2)) + 
    9 * log((2 * lb + sqrt(la^2 + 4 * lb^2)) / (lb + sqrt(la^2 + lb^2))) - 
    2 * log(la^2 + 2 * lb * (lb - sqrt(la^2 + lb^2)))))
end