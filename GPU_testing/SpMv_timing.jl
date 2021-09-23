include("../numerical")
include("../physical_params.jl")

using CUDA
using Printf

let 
    
    Lw = 1
    D = .25


    # Basin Params
    B_p = (μ_out = 36.0,
           ρ_out = 2.8,
           μ_in = 8.0,
           ρ_in = 2.0,
           c = (Lw/2)/D,
           r̄ = (Lw/2)^2,
           r_w = 1 + (Lw/2)/D)

    (x1, x2, x3, x4) = (0, Lw, 0, Lw)
    (y1, y2, y3, y4) = (0, 0, Lw, Lw)
    xt, yt = transfinite(x1, x2, x3, x4, y1, y2, y3, y4)

    nes = 2 * 8 .^ (1:10)

    for ne in nns
        
        nn = ne + 1
        metrics = create_metrics(ne,ne, B_p, μ, xt, yt)
        
        @printf "nn: %d" nn

    end



end
