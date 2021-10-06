using CUDA
using CUDA.CUSPARSE
include("../numerical.jl")
include("../physical_params.jl")
include("../domain.jl")
include("../MMS/mms_funcs.jl")

CUDA.allowscalar(false)

function rk4!(q, Λ, dt, tspan)

    nstep = ceil(Int, (tspan[2] - tspan[1]) / dt)
    dt = (tspan[2] - tspan[1]) / nstep
    Δq = similar(q)
    Δq2 = similar(q)
    
    for step in 1:nstep
        
        Δq .= 0

        t = tspan[1] + (step - 1) * dt

        Δq2 .= Λ * q
        Δq .+= 1/6 * dt * Δq2

        Δq2 .= Λ * (q + (1/2) * dt * Δq2)
        Δq .+= 1/6 * dt * 2Δq2

        Δq2 .= Λ * (q + (1/2) * dt * Δq2)
        Δq .+= 1/6 * dt * 2Δq2

        Δq2 .= Λ * (q + dt * Δq2)

        Δq .+= 1/6 * dt * Δq2

        q .+= Δq
        
    end
    nothing
end


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

    # MMS params
    MMS = (wl = Lw/2,
           amp = .5,
           ϵ = 0.0)


    (x1, x2, x3, x4) = (0, Lw, 0, Lw)
    (y1, y2, y3, y4) = (0, 0, Lw, Lw)
    xt, yt = transfinite(x1, x2, x3, x4, y1, y2, y3, y4)

    R = (-1, 0, 0, 1)

    ne = 8 * 2^6
    nn = ne + 1

    #
    # Get operators and domain
    #
    metrics = create_metrics(ne, ne, B_p, μ, xt, yt)
    LFToB = [BC_DIRICHLET, BC_DIRICHLET, BC_NEUMANN, BC_NEUMANN]
    loc = locoperator(p, ne, ne, B_p, μ, ρ, R, metrics, LFToB)
    
    x = metrics.coord[1]
    y = metrics.coord[2]
    #
    # get initial condtions
    #
    u0 = ue(x[:], y[:], 0.0, MMS)
    v0 = ue_t(x[:], y[:], 0.0, MMS)
    q = [u0;v0]
    for i in 1:4
        q = vcat(q, loc.L[i]*u0)
    end
    q = vcat(q, zeros(nn))
    @show Base.summarysize(q)/(1024)^3
    @show 8 * length(q)/(1024)^3
    #
    # set time and cfl condition
    #
    tspan = (0, .01)
    dt_scale = .1
    dt = dt_scale * 2 * loc.hmin / (sqrt(B_p.μ_out/B_p.ρ_out))

    dΛ = CuSparseMatrixCSR(loc.Λ)
    dq = CuArray(q)
    @show Base.summarysize(loc.Λ) / (1024)^3
    @show Base.summarysize(dq) / (1024)^3
    
    @time rk4!(q, loc.Λ, dt, tspan)
    @time rk4!(dq, dΛ, dt, tspan)

    q_end = Array(dq)
    @show norm(q - q_end)


end
