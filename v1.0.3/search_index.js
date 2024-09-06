var documenterSearchIndex = {"docs":
[{"location":"docIndex.html#The-GilaElectromagnetics-Module","page":"An other page","title":"The GilaElectromagnetics Module","text":"","category":"section"},{"location":"docIndex.html","page":"An other page","title":"An other page","text":"GilaElectromagnetics","category":"page"},{"location":"docIndex.html#GilaElectromagnetics","page":"An other page","title":"GilaElectromagnetics","text":"GilaElectromagnetics implements single (complex) frequency electromagnetic Green functions between generalized source and target cuboid ``volumes''. Technical  details are available in the supporting document files.\n\nAuthor: Sean Molesky  Distribution: The code distributed under GNU LGPL.\n\n\n\n\n\n","category":"module"},{"location":"docIndex.html#Module-Index","page":"An other page","title":"Module Index","text":"","category":"section"},{"location":"docIndex.html","page":"An other page","title":"An other page","text":"Modules = [GilaElectromagnetics]\nOrder   = [:constant, :type, :function, :macro]","category":"page"},{"location":"docIndex.html#Detailed-API","page":"An other page","title":"Detailed API","text":"","category":"section"},{"location":"docIndex.html","page":"An other page","title":"An other page","text":"Modules = [GilaElectromagnetics]\nOrder   = [:constant, :type, :function, :macro]","category":"page"},{"location":"docIndex.html#GilaElectromagnetics.GlaKerOpt-Tuple{Bool}","page":"An other page","title":"GilaElectromagnetics.GlaKerOpt","text":"GlaKerOpt(devStt::Bool)\n\nSimplified GlaKerOpt constructor.\n\n\n\n\n\n","category":"method"},{"location":"docIndex.html#GilaElectromagnetics.GlaOpr","page":"An other page","title":"GilaElectromagnetics.GlaOpr","text":"GlaOpr(cel::NTuple{3, Int}, scl::NTuple{3, Rational}, \norg::NTuple{3, Rational}=(0//1, 0//1, 0//1); \nuseGpu::Bool=false, setTyp::DataType=ComplexF64)\n\nConstruct a self Green operator.\n\nArguments\n\ncel::NTuple{3, Int}: The number of cells in each dimension.\nscl::NTuple{3, Rational}: The size of each cell in each dimension \n\n(in units of wavelength).\n\norg::NTuple{3, Rational}=(0//1, 0//1, 0//1): The origin of the volume in \n\neach dimension (in units of wavelength).\n\nuseGpu::Bool=false: Whether to use the GPU (true) or CPU (false).\nsetTyp::DataType=ComplexF64: The element type of the operator. Must be a\n\nsubtype of Complex.\n\n\n\n\n\n","category":"type"},{"location":"docIndex.html#GilaElectromagnetics.GlaOpr-2","page":"An other page","title":"GilaElectromagnetics.GlaOpr","text":"GlaOpr\n\nAbstraction wrapper for GlaOprMem. \n\nFields\n\nmem::GlaOprMem: Data to process the Green function.\n\n\n\n\n\n","category":"type"},{"location":"docIndex.html#GilaElectromagnetics.GlaOpr-Tuple{Tuple{Int64, Int64, Int64}, Tuple{Rational, Rational, Rational}, Tuple{Rational, Rational, Rational}, Tuple{Int64, Int64, Int64}, Tuple{Rational, Rational, Rational}, Tuple{Rational, Rational, Rational}}","page":"An other page","title":"GilaElectromagnetics.GlaOpr","text":"GlaOpr(celSrc::NTuple{3, Int}, sclSrc::NTuple{3, Rational}, \norgSrc::NTuple{3, Rational}, celTrg::NTuple{3, Int}, \nsclTrg::NTuple{3, Rational}, orgTrg::NTuple{3, Rational}; \nuseGpu::Bool=false, setTyp::DataType=ComplexF64)\n\nConstruct an external Green's operator.\n\nArguments\n\ncelSrc::NTuple{3, Int}: The number of cells in each dimension of the source\n\nvolume.\n\nsclSrc::NTuple{3, Rational}: The size of each cell in each dimension of the\n\nsource volume (in units of wavelength).\n\norgSrc::NTuple{3, Rational}: The origin of the source volume in each\n\ndimension (in units of wavelength).\n\ncelTrg::NTuple{3, Int}: The number of cells in each dimension of the target\n\nvolume.\n\nsclTrg::NTuple{3, Rational}: The size of each cell in each dimension of the\n\ntarget volume (in units of wavelength).\n\norgTrg::NTuple{3, Rational}: The origin of the target volume in each\n\ndimension (in units of wavelength).\n\nuseGpu::Bool=false: Whether to use the GPU (true) or CPU (false).\nsetTyp::DataType=ComplexF64: The element type of the operator. Must be a\n\nsubtype of Complex.\n\n\n\n\n\n","category":"method"},{"location":"docIndex.html#GilaElectromagnetics.GlaOprMem","page":"An other page","title":"GilaElectromagnetics.GlaOprMem","text":"GlaOprMem\n\nStorage structure for a Green function operator. .cmpInf–-computation information see GlaKerOpt .trgVol–-target volume of Green function .srcVol–-source volume of Green function .mixInf–-information for matching source and target grids, see GlaExtInf .dimInfC–-dimension information for Green function volumes, host side .dimInfD–-dimension information for Green function volumes, device side .egoFur–-unique Fourier transform data for circulant Green function .fftPlnFwd–-forward Fourier transform plans .fftPlnRev–-reverse Fourier transform plans .phzInf–-phase vector for splitting Fourier transforms\n\n\n\n\n\n","category":"type"},{"location":"docIndex.html#GilaElectromagnetics.GlaOprMem-Union{Tuple{T}, Tuple{GlaKerOpt, GlaVol}, Tuple{GlaKerOpt, GlaVol, Union{Nothing, GlaVol}}} where T<:Union{ComplexF64, ComplexF32}","page":"An other page","title":"GilaElectromagnetics.GlaOprMem","text":"function GlaOprMem(cmpInf::GlaKerOpt, trgVol::GlaVol,\nsrcVol::Union{GlaVol,Nothing}=nothing, \negoFur::Union{AbstractArray{<:AbstractArray{T}},\nNothing}=nothing)::GlaOprMem where T<:Union{ComplexF64,ComplexF32}\n\nPrepare memory for green function operator–-when called with a single GlaVol,  or identical source and target volumes, yields the self construction. \n\n\n\n\n\n","category":"method"},{"location":"docIndex.html#GilaElectromagnetics.GlaVol","page":"An other page","title":"GilaElectromagnetics.GlaVol","text":"GlaVol(cel::Array{<:Integer,1}, celScl::NTuple{3,Rational}, \norg::NTuple{3,Rational}, grdScl::NTuple{3,Rational}=celScl)::GlaVol\n\nConstructor for Gila Volumes.\n\n\n\n\n\n","category":"type"},{"location":"docIndex.html#GilaElectromagnetics.egoBrnDev!-Union{Tuple{T}, Tuple{GlaOprMem, Integer, Integer, AbstractArray{T}}} where T<:Union{ComplexF64, ComplexF32}","page":"An other page","title":"GilaElectromagnetics.egoBrnDev!","text":"egoBrnDev!(egoMem::GlaOprMem, lvl::Integer, bId::Integer, \nactVec::AbstractArray{T})::AbstractArray{T} where \nT<:Union{ComplexF64,ComplexF32}\n\nHead branching function implementing Green function action on device.\n\n\n\n\n\n","category":"method"},{"location":"docIndex.html#GilaElectromagnetics.egoBrnHst!-Union{Tuple{T}, Tuple{GlaOprMem, Integer, Integer, AbstractArray{T}}} where T<:Union{ComplexF64, ComplexF32}","page":"An other page","title":"GilaElectromagnetics.egoBrnHst!","text":"egoBrnHst!(egoMem::GlaOprMem, lvl::Integer, bId::Integer, \nactVec::AbstractArray{T})::AbstractArray{T} where \nT<:Union{ComplexF64,ComplexF32}\n\nHead branching function implementing Green function action on host.\n\n\n\n\n\n","category":"method"},{"location":"docIndex.html#GilaElectromagnetics.genEgoExt!-Union{Tuple{T}, Tuple{AbstractArray{T, 5}, GlaVol, GlaVol, GlaKerOpt}} where T<:Union{ComplexF64, ComplexF32}","page":"An other page","title":"GilaElectromagnetics.genEgoExt!","text":"genEgoExt!(egoCrcExt::AbstractArray{T,5}, trgVol::GlaVol, \nsrcVol::GlaVol, cmpInf::GlaKerOpt)::Nothing where \nT<:Union{ComplexF64,ComplexF32}\n\nCalculate circulant vector for the Green function between a target volume,  trgVol, and source volume, srcVol.\n\n\n\n\n\n","category":"method"},{"location":"docIndex.html#GilaElectromagnetics.genEgoMat-Tuple{Tuple{var\"#s16\", var\"#s16\", var\"#s16\"} where var\"#s16\"<:Rational, Tuple{var\"#s15\", var\"#s15\", var\"#s15\"} where var\"#s15\"<:Integer}","page":"An other page","title":"GilaElectromagnetics.genEgoMat","text":"function genEgoMat(celScl::{3,<:Rational}, \ncelNum::Ntuple{3,<:Integer})::Array{ComplexF64,2}\n\nGenerate dense matrix of a Green function. \n\n\n\n\n\n","category":"method"},{"location":"docIndex.html#GilaElectromagnetics.genEgoSlf!-Union{Tuple{T}, Tuple{AbstractArray{T, 5}, GlaVol, GlaKerOpt}} where T<:Union{ComplexF64, ComplexF32}","page":"An other page","title":"GilaElectromagnetics.genEgoSlf!","text":"genEgoSlf!(egoCrc::Array{ComplexF64}, slfVol::GlaVol, \ncmpInf::GlaKerOpt)::Nothing\n\nCalculate circulant vector of the Green function on a single domain.\n\n\n\n\n\n","category":"method"},{"location":"docIndex.html#GilaElectromagnetics.glaSze-Tuple{GlaOpr, Int64}","page":"An other page","title":"GilaElectromagnetics.glaSze","text":"glaSze(op::GlaOpr, dim::Int)\n\nReturns the size of the input/output arrays for a GlaOpr in tensor form.\n\nArguments\n\nop::GlaOpr: The operator to check.\ndim::Int: The length of the dimension to check.\n\nReturns\n\nThe size of the input/output arrays for a GlaOpr in tensor form.\n\n\n\n\n\n","category":"method"},{"location":"docIndex.html#GilaElectromagnetics.glaSze-Tuple{GlaOpr}","page":"An other page","title":"GilaElectromagnetics.glaSze","text":"glaSze(opr::GlaOpr)\n\nReturns the size of the input/output arrays for a GlaOpr in tensor form.\n\nArguments\n\nop::GlaOpr: The operator to check.\n\nReturns\n\nA tuple of the sizes of the input and output arrays in tensor form.\n\n\n\n\n\n","category":"method"},{"location":"docIndex.html#GilaElectromagnetics.isadjoint-Tuple{GlaOpr}","page":"An other page","title":"GilaElectromagnetics.isadjoint","text":"isadjoint(opr::GlaOpr)\n\nReturns true if the operator is the adjoint of the Greens operator.\n\nArguments\n\nopr::GlaOpr: The operator to check.\n\nReturns\n\ntrue if the operator is the adjoint, false otherwise.\n\n\n\n\n\n","category":"method"},{"location":"docIndex.html#GilaElectromagnetics.isexternaloperator-Tuple{GlaOpr}","page":"An other page","title":"GilaElectromagnetics.isexternaloperator","text":"isexternaloperator(opr::GlaOpr)\n\nReturns true if the operator is an external Greens operator.\n\nArguments\n\nopr::GlaOpr: The operator to check.\n\nReturns\n\ntrue if the operator is an external Greens operator, false otherwise.\n\n\n\n\n\n","category":"method"},{"location":"docIndex.html#GilaElectromagnetics.isselfoperator-Tuple{GlaOpr}","page":"An other page","title":"GilaElectromagnetics.isselfoperator","text":"isselfoperator(opr::GlaOpr)\n\nReturns true if the operator is a self Greens operator.\n\nArguments\n\nopr::GlaOpr: The operator to check.\n\nReturns\n\ntrue if the operator is a self Greens operator, false otherwise.\n\n\n\n\n\n","category":"method"},{"location":"index.html#GilaElectromagnetics.jl","page":"Index","title":"GilaElectromagnetics.jl","text":"","category":"section"},{"location":"index.html","page":"Index","title":"Index","text":"Documentation for GilaElectromagnetics.jl https://github.com/moleskySean/GilaElectromagnetics.jl","category":"page"}]
}