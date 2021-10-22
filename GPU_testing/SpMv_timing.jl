include("../numerical.jl")
include("../physical_params.jl")
include("../domain.jl")
include("../MMS/mms_funcs.jl")

using CUDA
using CUDA.CUSPARSE
using Printf
using LinearAlgebra
using SparseArrays
using MatrixMarket
using UnicodePlots


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
    
    MMS = (wl = Lw/2,
           amp = .5,
           ϵ = .01)

    RS = (Hvw = 12,
          Ht = 6,
          σn = 50.0,
          a = .015,
          b0 = .02,
          bmin = 0.0,
          Dc = 10e6,
          f0 = .6,
          V0 = 1e-6)

    (x1, x2, x3, x4) = (0, Lw, 0, Lw)
    (y1, y2, y3, y4) = (0, 0, Lw, Lw)
    xt, yt = transfinite(x1, x2, x3, x4, y1, y2, y3, y4)

    R = (-1, 0, 0, 1)
    nes = 8 * 2 .^ (1:5)
    
    for ne in nes
        
        nn = ne + 1
        
        @printf "nn: %d\n" nn
        
        metrics = create_metrics(ne, ne, B_p, μ, xt, yt)
        
        x = metrics.coord[1]
        y = metrics.coord[2]

        LFToB = [BC_DIRICHLET, BC_DIRICHLET, BC_NEUMANN, BC_NEUMANN]
    
        faces = [1 2 3 4]
        d_ops_waveprop = operators_dynamic(p, ne, ne, B_p, μ, ρ, R, faces, metrics, LFToB)
        Λ = d_ops_waveprop.Λ
        

        u0 = ue(x[:], y[:], 0.0, MMS)
        v0 = ue_t(x[:], y[:], 0.0, MMS)
        q1 = [u0;v0]
        for i in 1:4
            q1 = vcat(q1, d_ops_waveprop.L[i]*u0)
        end
        q1 = vcat(q1, ψe(metrics.facecoord[1][1],
                         metrics.facecoord[2][1],
                         0, B_p, RS, MMS))     

        dΛ = CuSparseMatrixCSR(Λ)
        dq = CuArray(q1)
        dΔq = CuArray(zeros(length(dq)))
        
        q = deepcopy(q1)
        Δq = zeros(length(q1))
        
        
        @time Δq .= Λ * q
        @time dΔq .= dΛ * dq

        dΔq_res = Array(dΔq)
        error = norm(dΔq_res .- Δq)
        @show error
        
    end
end
