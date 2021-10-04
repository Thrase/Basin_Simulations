using CUDA
using CUDA.CUSPARSE
include("../numerical.jl")
include("../physical_params.jl")
include("../domain.jl")
include("../MMS/mms_funcs.jl")


function rk4!(Λ, q, dt)
    
    k1 = Λ * q

    dq2 = q + (dt/2 * k1)
    k2 = Λ * dq1

    dq3 = q + (dt/2 * k2)
    k3 = Λ * dq2

    dq4 = q + (dt * k3)
    k4 = Λ * dq3
    
    q = q + 1/6 * (k1 + 2k2 + 2k3 + k4)

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

    (x1, x2, x3, x4) = (0, Lw, 0, Lw)
    (y1, y2, y3, y4) = (0, 0, Lw, Lw)
    xt, yt = transfinite(x1, x2, x3, x4, y1, y2, y3, y4)

    R = (-1, 0, 0, 1)

    ne = 8 * 2^5
    nn = ne + 1

    metrics = create_metrics(ne, ne, B_p, μ, xt, yt)

    LFToB = [BC_DIRICHLET, BC_DIRICHLET, BC_NEUMANN, BC_NEUMANN]
    
    @time loc = locoperator(p, ne, ne, B_p, μ, ρ, R, metrics, LFToB)
    
    q = ue(x,y, 0)
    
    tspan = (0, 1)
    dt_scale = .1
    dt = dt_scale * 2 * lop.hmin / (sqrt(B_p.μ_out/B_p.ρ_out))



    #=
    RKA = (
        T(0),
        T(-567301805773 // 1357537059087),
        T(-2404267990393 // 2016746695238),
        T(-3550918686646 // 2091501179385),
        T(-1275806237668 // 842570457699),
    )

    RKB = (
        T(1432997174477 // 9575080441755),
        T(5161836677717 // 13612068292357),
        T(1720146321549 // 2090206949498),
        T(3134564353537 // 4481467310338),
        T(2277821191437 // 14882151754819),
    )

    RKC = (
        T(0),
        T(1432997174477 // 9575080441755),
        T(2526269341429 // 6820363962896),
        T(2006345519317 // 3224310063776),
        T(2802321613138 // 2924317926251),
    )

    nstep = ceil(Int, (tspan[1] - tspan[2]) / dt)
    dt = (tspan[1] - tspan[2]) / nstep

    fill!(Δq, 0)
    fill!(Δq2, 0)
    for step in 1:nstep
        t = t0 + (step - 1) * dt
        for s in 1:length(RKA)
            f!(Δq2, q, p, t + RKC[s] * dt)
            Δq .+= Δq2
            q .+= RKB[s] * dt * Δq
            Δq .*= RKA[s % length(RKA) + 1]
        end
    end
    =#

end
