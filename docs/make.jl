using Documenter
using GilaElectromagnetics

push!(LOAD_PATH,"../src/")
makedocs(sitename="GilaElectromagnetics.jl Documentation",
         pages = [
            "GilaElectromagnetics" => "index.md",
            "Concepts" => "concepts.md",
            "Usage" => "usage.md",
            "Examples" => "examples.md",
            "Library" => "library.md",
         ],
         format = Documenter.HTML(prettyurls = true, repolink = "https://github.com/emilegp/GilaElectromagnetics.jl"),
         repo = "github.com/emilegp/GilaElectromagnetics.jl"
)
# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
#deploydocs(
#    repo = "github.com/emilegp/GilaElectromagnetics.jl.git",
#    devbranch = "main"
#)
