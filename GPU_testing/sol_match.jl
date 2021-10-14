using CUDA
using CUDA.CUSPARSE
include("../numerical.jl")
include("../physical_params.jl")
include("../domain.jl")
include("../MMS/mms_funcs.jl")
include("../numerical.jl")
include("../solvers.jl")

CUDA.allowscalar(false)
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

    R = [-1, 0, 0, 1]

    ne = 8 #* 2^2
    nn = ne + 1

    #
    # Get operators and domain
    #
    metrics = create_metrics(ne, ne, B_p, μ, xt, yt)
    LFToB = [BC_DIRICHLET, BC_DIRICHLET, BC_NEUMANN, BC_NEUMANN]
    faces = [0 2 3 4]
    ops = operators_dynamic(p, ne, ne, B_p, μ, ρ, R, faces, metrics, LFToB)
    b = b_fun(metrics.facecoord[2][1], RS)
    x = metrics.coord[1]
    y = metrics.coord[2]
    
    #
    # get initial condtions
    #
    u0 = ue(x[:], y[:], 0.0, MMS)
    v0 = ue_t(x[:], y[:], 0.0, MMS)
    q = [u0;v0]
    for i in 1:4
        q = vcat(q, ops.L[i]*u0)
    end
    q = vcat(q, ψe(metrics.facecoord[1][1],
                   metrics.facecoord[2][1],
                   0.0, B_p, RS, MMS))
   
    q1 = deepcopy(q)
    #
    # set time and cfl condition
    #
    tspan = (0, .0005)
    dt_scale = .0001
    dt = dt_scale * 2 * ops.hmin / (sqrt(B_p.μ_out/B_p.ρ_out))

    vf = Array{Float64, 1}(undef, nn)
    τ̃f = Array{Float64, 1}(undef, nn)
    v̂_fric = Array{Float64, 1}(undef, nn)
    
    operators = (nn = nn,
                 d_ops = ops,
                 coord = metrics.coord,
                 fc = metrics.facecoord,
                 sJ = metrics.sJ,
                 R = R,
                 B_p = B_p,
                 RS = RS,
                 b = b,
                 MMS = MMS,
                 vf = vf,
                 τ̃f = τ̃f,
                 v̂_fric = v̂_fric,
                 CHAR_SOURCE = S_c,
                 STATE_SOURCE = S_rs,
                 FORCE = Forcing)
    

    GPU_operators = (nn = nn,
                 Λ = ops.Λ,
                 Z̃f = ops.Z̃f[1],
                 L = ops.L[1],
                 H = ops.H[1],
                 P̃I = ops.P̃I,
                 JIHP = ops.JIHP,
                 nCnΓ1 = ops.nCnΓ1,
                 nBBCΓL1 = ops.nBBCΓL1,
                 sJ = metrics.sJ[1],
                 RS = RS,
                 b = b,
                 vf = vf,
                 τ̃f = τ̃f,
                 v̂_fric = v̂_fric)
    
    #@time rk4!(q, ODE_RHS_BLOCK_CPU_FAULT!, operators, dt, tspan)
    @time ODE_RHS_GPU_FAULT!(q1, GPU_operators, dt, tspan)
    #@time rk4!(dq, dΛ, dt, tspan)
    #display(q)
    #q_end = Array(dq)
    #@show norm(q - q_end)


end
