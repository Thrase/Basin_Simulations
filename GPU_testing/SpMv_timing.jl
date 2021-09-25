include("../numerical.jl")
include("../physical_params.jl")
include("../domain.jl")

using CUDA.CUSPARSE
using Printf
using LinearAlgebra
using SparseArrays

let 
    
    p = 2 
    
    Lw = 1
    D = .25

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

    nes = 8 * 2 .^ (4:8)

    for ne in nes
        
        nn = ne + 1
        
        @printf "nn: %d\n" nn
        
        metrics = create_metrics(ne,ne, B_p, μ, xt, yt)
        Ã = sbpA(p, ne, ne, metrics)
        
        block_ops = (Nn = nn^2,
                     nn = nn,
                     Ã = Ã)

        Λ = dynamicblock(block_ops)
        u = rand(size(Λ)[2])

        @time for _ in 1:100
            mul!(u, Λ, u, 1, 0)
        end
        
        dΛ = CuSparseMatrixCSR(Λ)
        du = CuArray(u)

        @time for _ in 1:100
            CUSPARSE.mv!('N', 1, dΛ, du, 0, du, 'a')
        end

    end
end
