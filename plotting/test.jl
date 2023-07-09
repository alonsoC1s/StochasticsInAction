using GLMakie

function plot_board()
    # By default the scene is [-1, 1]^2
    scene = Scene(backgroundcolor=:white)
    const width = 2 / 4
    const center = width / 2


    # Filling the rect list
    square_coords = reduce(vcat, [
        [(-1, -1 + width * i) for i = 1:2], # Left
        [(-1 + width * i, -1 + 3 * width) for i = 1:2], # Top
        [(1 - width, -1 + width * i) for i = 1:2], # Right
        [(-1 + width * i, -1) for i = 1:2] # Bottom
    ])

    colors = [:red, :red, :green, :green, :purple, :purple, :blue, :blue]

    for (coord, color, label) in zip(square_coords, colors, 1:8)
        poly!(scene,
            Rect2f(coord..., width, width),
            color=color, strokewidth=2, strokecolor=:black
        )
        text!(scene,
            first(coord) + center, last(coord) + center,
            text=string(label), fontsize=30, justification=:center
        )
    end
    scene
end