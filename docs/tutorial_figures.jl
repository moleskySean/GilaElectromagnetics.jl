# Every tutorial figure is saved twice, as drawn and recolored for the dark themes,
# and the page shows the one matching the theme (see assets/theory.css).
module TutorialFigures

using CairoMakie

const out = joinpath(@__DIR__, "build", "assets", "tutorials", "generated")
# Named colors first, since some attributes (a Legend's) only take a Symbol
const ink = (:gray85, RGBf(0.85, 0.85, 0.85))
const faint = ((:white, 0.15), RGBAf(1, 1, 1, 0.15))
const clear = (:transparent, RGBAf(0, 0, 0, 0))

function recolor!(block, name, colors)
    name in propertynames(block) || return
    for color in colors
        try
            return setproperty!(block, name, color)
        catch err
            err isa MethodError || rethrow()
        end
    end
end

const inkAttrs = (:titlecolor, :xlabelcolor, :ylabelcolor, :zlabelcolor, :xticklabelcolor, :yticklabelcolor,
                  :zticklabelcolor, :labelcolor, :ticklabelcolor, :xtickcolor, :ytickcolor, :ztickcolor, :tickcolor,
                  :leftspinecolor, :rightspinecolor, :topspinecolor, :bottomspinecolor, :spinecolor, :framecolor,
                  :xspinecolor_1, :xspinecolor_2, :xspinecolor_3, :yspinecolor_1, :yspinecolor_2, :yspinecolor_3,
                  :zspinecolor_1, :zspinecolor_2, :zspinecolor_3, :color)
const faintAttrs = (:xgridcolor, :ygridcolor, :zgridcolor)

function darken!(fig)
    fig.scene.backgroundcolor[] = last(clear)
    for block in fig.content
        block isa Union{Axis, Axis3, Colorbar, Legend, Label} || continue
        for name in inkAttrs
            name == :color && !(block isa Label) && continue
            recolor!(block, name, ink)
        end
        foreach(name -> recolor!(block, name, faint), faintAttrs)
        recolor!(block, :backgroundcolor, clear)
    end
    return fig
end

const count = Ref(0)

function themed(fig)
    mkpath(out)
    name = "fig$(count[] += 1)"
    save(joinpath(out, "$name-light.png"), fig)
    save(joinpath(out, "$name-dark.png"), darken!(fig))
    src = "../../assets/tutorials/generated/$name"
    return HTML("""<img class="fig-light" src="$src-light.png"><img class="fig-dark" src="$src-dark.png">""")
end

end
