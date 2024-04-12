using Documenter
using Gila

push!(LOAD_PATH,"../src/")
makedocs(sitename="Gila.jl Documentation",
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
    repo = "github.com/moleskySean/Gila.jl.git",
    devbranch = "main"
)