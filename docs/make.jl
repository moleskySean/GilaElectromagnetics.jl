using Documenter
using DocumenterCitations
using GilaElectromagnetics

mth = MathJax3(Dict(:tex => Dict(
    "inlineMath" => [["\$", "\$"], ["\\(", "\\)"]],
    "tags" => "none",
    "packages" => ["base", "ams", "autoload", "newcommand", "configmacros", "boldsymbol", "color", "physics", "braket"],
    "macros" => Dict(
        "bm" => ["\\boldsymbol{#1}", 1],
        "vb" => ["\\boldsymbol{#1}", 1, ["*"]],
        "ab" => "\\qty",
        "bra" => ["\\left\\langle #1\\right\\vert", 1, ["<", "|"]],
        "ket" => ["\\left\\vert #1\\right\\rangle", 1, ["|", ">"]],
        "braket" => ["\\Braket{#1}", 1, ["<", ">"]],
        "ketbra" => ["\\left\\vert #1\\vphantom{#2}\\right\\rangle\\left\\langle #2\\vphantom{#1}\\right\\vert", 2, ["|", "><", "|"]],
        "dd" => ["\\mathinner{\\mathrm{d}^{#1}}\\!#2", 2, ""],
        "pdv" => ["\\frac{\\partial #1}{\\partial #2}", 2],
        "abs" => ["\\left\\vert #1\\right\\vert", 1],
        "norm" => ["\\left\\Vert #1\\right\\Vert", 1],
        "order" => ["\\mathcal{O}\\left(#1\\right)", 1],
        "curl" => "\\mathinner{\\nabla}\\!\\times",
        "div" => "\\mathinner{\\nabla}\\!\\cdot",
        "bbone" => "\\mathbb{1}",
        "dprime" => "\\prime\\prime",
        "coloneq" => "\\mathrel{\\vcenter{:}}=",
        "eqcolon" => "=\\mathrel{\\vcenter{:}}",
        "Im" => "\\operatorname{Im}",
        "exp" => "\\operatorname{exp}",
        "log" => "\\operatorname{log}",
        "red" => ["\\textcolor[RGB]{190,52,48}{#1}", 1],
        "blue" => ["\\textcolor[RGB]{33,92,168}{#1}", 1],
    ),
)))

push!(LOAD_PATH,"../src/")
makedocs(sitename="GilaElectromagnetics.jl Documentation",
         pages = [
            "Home" => "index.md",
            "Theory" => [
               "theory/maxwell.md",
               "theory/jvie.md",
               "theory/discretization.md",
               "theory/matrix_elements.md",
               "theory/toeplitz.md",
               "theory/solving.md",
               "theory/fourier.md",
               "theory/references.md",
            ],
            "The Integrals of Gila" => "integrals/index.md",
            "Tutorials" => "tutorials/index.md",
            "How-to Guide" => [
               "howto/volumes.md",
               "howto/fields.md",
               "howto/vacuum.md",
               "howto/materials.md",
               "howto/scattering.md",
               "howto/boundaries.md",
               "howto/frequency.md",
               "howto/precision.md",
               "howto/performance.md",
            ],
            "Extensions" => "extensions/index.md",
            "API Reference" => "library.md",
         ],
         format = Documenter.HTML(prettyurls = true, mathengine = mth, assets = ["assets/theory.css"]),
         plugins = [CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style = :numeric)],
           repo = Remotes.GitHub("moleskySean", "GilaElectromagnetics.jl")
         )

deploydocs(
    repo = "github.com/moleskySean/GilaElectromagnetics.jl.git",
    devbranch = "main"
)
