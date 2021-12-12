include("numerical.jl")
include("domain.jl")
include("physical_params.jl")
include("read_in.jl")
using IterativeSolvers
using LinearAlgebra
let

    p,
    T,
    N,
    Lw,
    r̂,
    l,
    D,
    dynamic_flag,
    d_to_s,
    dt_scale,
    ic_file,
    ic_t_file,
    Dc,
    force = read_params(ARGS[1])
   
    nn = N + 1
    
    # Basin Params
    B_p = (μ_out = 36.0,
           ρ_out = 2.8,
           μ_in = 8.0,
           ρ_in = 2.0,
           c = (Lw/2)/D,
           r̄ = (Lw/2)^2,
           r_w = 1 + (Lw/2)/D)
    
    
    # Get grid
    grid_t = @elapsed begin
        xt, yt = transforms_e(Lw, r̂, l)
        metrics = create_metrics(N, N, B_p, μ, ρ, xt, yt)
    end

    fc = metrics.facecoord[2][1]

    δNp, 
    gNp, 
    VWp, 
    RS = fault_params(fc, Dc)

    # getting discrete operators
    faces = [0 2 3 4]
    R = [-1 0 1 0]

    ops = operators(p, N, N, μ, ρ, R, B_p, faces, metrics)

    M̃1 = ops.M̃
    M̃2 = copy(M̃1)
    M̃2 = cholesky(M̃2)
    u1 = zeros(size(M̃)[1])
    u2 = copy(u1)
    ge1 = ones(size(u1))
    ge2 = copy(ge1)

    @time 

end
