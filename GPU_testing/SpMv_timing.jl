include("../numerical.jl")
include("../physical_params.jl")
include("../domain.jl")

#using CUDA
#using CUDA.CUSPARSE
using Printf
using LinearAlgebra
using SparseArrays
#CUDA.versioninfo()

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

    R = (-1, 0, 0, 1)
    ne = 8 * 2 .^ (7:7)
    ne = ne[1]
    #for ne in nes
        
    nn = ne + 1
        
    @printf "nn: %d\n" nn
        
    metrics = create_metrics(ne, ne, B_p, μ, xt, yt)
        
    LFToB = [BC_DIRICHLET, BC_DIRICHLET, BC_NEUMANN, BC_NEUMANN]
    
    @time loc = locoperator(p, ne, ne, B_p, μ, ρ, metrics, LFToB)
    
    #=
    ops = (Nn = nn^2,
           nn = nn,
           Ã = loc.Ã,
           L = loc.L,
           H = loc.H,
           R = R,
           Z̃f = loc.Z̃f,
           nBBCΓL = loc.nBBCΓL,
           BCTHL = loc.BCTHL,
           nCnΓ = loc.nCnΓ,
           BCTH = loc.BCTH,
           JIHP = loc.JIHP)
    
    @time Λ = dynamicblock(ops)
    =#
    @show (length(loc.Λ.nzval) * 8) / (1024)^3

        #=
        u1 = rand((-1.0, 1.0), size(Λ)[2])

        @show maximum(u1), minimum(u1), length(u1)
        
        u2 = zeros(size(Λ)[2])

        dΛ = CuSparseMatrixCSR(Λ)
        du1 = CuArray(u1)
        du2 = CuArray(u2)
        
        @show (length(Λ.nzval) * 8) / (1024)^3
        
        mul!(u2, Λ, u1)
        
        @time begin
            mul!(u1, Λ, u2)
        end

        du2 .= dΛ * du1

        @time begin
            du1 .= dΛ * du2
        end

        du_res = Array(du2)
        error = norm(u2 .- du_res)
        @show norm(u2)
        @show norm(du_res)
        @show error
       =# 
    #end
end
