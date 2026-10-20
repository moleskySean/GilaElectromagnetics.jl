using Documenter
using DocumenterCitations
using GilaElectromagnetics
using Literate

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

# Each examples/NN_name.jl becomes a tutorial page whose code Documenter runs,
# and each figure on it is shown in a light and a dark version
include("tutorial_figures.jl")
themeFigures(src) = replace(src, r"^fig$"m => "fig\nMain.TutorialFigures.themed(fig) #hide")
examples = joinpath(@__DIR__, "..", "examples")
tutorials = filter(contains(r"^\d\d_.*\.jl$"), readdir(examples))
notExecuted = ["08_band_structure.jl"] # too slow for a docs build
for file in tutorials
    kw = file in notExecuted ? (; codefence = "```julia" => "```") : (; preprocess = themeFigures)
    Literate.markdown(joinpath(examples, file), joinpath(@__DIR__, "src", "tutorials"); kw...)
end

makedocs(sitename="GilaElectromagnetics.jl",
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
            "Tutorials" => ["tutorials/index.md"; ["tutorials/" * replace(file, ".jl" => ".md") for file in tutorials]],
            "How-to Guide" => [
               "howto/volumes.md",
               "howto/fields.md",
               "howto/units.md",
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
