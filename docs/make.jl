using Documenter
using GilaElectromagnetics

push!(LOAD_PATH,"../src/")
makedocs(sitename="GilaElectromagnetics.jl Documentation",
         pages = [
            "GilaElectromagnetics" => "index.md",
            "Concepts" => "concepts.md",
            "Usage" => "usage.md",
            "Examples" => "examples.md",
            "API Reference" => "library.md",
         ],
         format = Documenter.HTML(prettyurls = true),
           repo = Remotes.GitHub("moleskySean", "GilaElectromagnetics.jl")
         )

deploydocs(
    repo = "github.com/moleskySean/GilaElectromagnetics.jl.git",
    devbranch = "main"
)
