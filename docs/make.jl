using Documenter
using GilaElectromagnetics

push!(LOAD_PATH,"../src/")
makedocs(sitename="GilaElectromagnetics.jl Documentation",
         pages = [
            "Index" => "index.md",
            "An other page" => "docIndex.md",
         ],
         format = Documenter.HTML(prettyurls = false)
)
# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/moleskySean/GilaElectromagnetics.jl.git",
    devbranch = "main"
)