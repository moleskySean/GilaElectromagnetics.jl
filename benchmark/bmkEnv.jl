# Shared environment bootstrap for the benchmark scripts. Activates
# benchmark/Project.toml and dev-links the parent package on first use, so
# `julia --project=. benchmark/<script>.jl` works from any starting
# environment (the root [extras]/bench target is not loadable via --project=.).
import Pkg
Pkg.activate(@__DIR__; io = devnull)
if !isfile(joinpath(@__DIR__, "Manifest.toml"))
    Pkg.develop(Pkg.PackageSpec(path = dirname(@__DIR__)))
end
Pkg.instantiate()
