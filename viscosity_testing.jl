include("numerical.jl")
include("solvers.jl")
include("physical_params.jl")
include("read_in.jl")
include("domain.jl")
include("write_out.jl")

using Printf
using OrdinaryDiffEq

let

    ### input parameters
    p = 2
    T = 10
    N = 15
    Lw = 24
    r̂ = .75
    l = .05
    D = 4
    dt_scale = .5
    Dc = .008
    B_on = 1
    μ_in = 8.0
    dynamic_flag = 0
    d_to_s = .005
    dir_out = "./testing/"
    
    year_seconds = 31556952
    sim_seconds = T * year_seconds
    t_now = 0.0

    nn = N + 1
    
    ### basin params
    B_p = (μ_out = 36.0,
           ρ_out = 2.8,
           μ_in = μ_in,
           ρ_in = 2.0,
           c = (Lw/2)/D,
           r̄ = (Lw/2)^2,
           r_w = 1 + (Lw/2)/D,
           on = B_on)

    ### get grid
    grid_t = @elapsed begin
        xt, yt = transforms_e(Lw, r̂, l)
        metrics = create_metrics(N, N, B_p, μ, ρ, xt, yt)
    end

    ### get fault params
    fc = Array(metrics.facecoord[2][1])
    (x, y) = metrics.coord

    η = metrics.η

    δNp, 
    gNp, 
    VWp,
    RS = fault_params(fc, Dc, Lw)

    ### get discrete operators
    R = [-1 0 1 0]
    opt_t = @elapsed begin
        ops = operators(p, N, N, μ, ρ, R, B_p, metrics)
    end
    @printf "Got operators in %f seconds\n" opt_t
    flush(stdout)

    ### initial conditions
    ψδγ = zeros(2nn + nn^2)
    for n in 1:nn
        ψδγ[n] = RS.a * log(2*(RS.V0/RS.Vp) * sinh((RS.τ_inf - η[n]*RS.Vp)/(RS.σn*RS.a)))
    end
    ψδγ[nn + 1: 2nn] .= 0
    ψδγ[2nn + 1: 2nn + nn^2] .= 0

    vars = (u_prev = zeros(nn^2),
            t_prev = [0.0, 0.0],
            Δτ = zeros(nn),
            vf = zeros(nn),
            u = zeros(nn^2),
            uf2 = zeros(nn),
            ge = zeros(nn^2),
            δ_end = zeros(nn),
            ψ_end = zeros(nn),
            t_end = [0.0])

    vars.uf2 .= (RS.τ_inf * Lw) ./ metrics.μf2

    fault_name,
    station_name,
    remote_name,
    volume_name = new_dir(dir_out, ARGS[1], [0.0], fc, x[1:2:nn, 1], y[1, 1:2:nn])
    
    slip_plot=nothing
    
    ### parameter orginzation
    io = (dir_name = dir_out,
          fault_name = fault_name,
          station_name = station_name,
          remote_name = remote_name,
          volume_name = volume_name,
          pf = [0.0, 0.0, 0.0],
          vp = nothing,
          slip_plot = [slip_plot])
    
    static_params = (year_seconds,
                     reject_step = [false],
                     dynamic_flag = dynamic_flag,
                     Lw = Lw,
                     nn = nn,
                     δNp = δNp,
                     d_to_s = d_to_s,
                     vars = vars,
                     ops = ops,
                     metrics = metrics,
                     fc = Array(metrics.facecoord[2][1]),
                     io = io,
                     RS = RS,
                     vf = zeros(nn),
                     cycles = [0])


    ### run inter-seismic period
    static_params.reject_step[1] = false
    stopper = DiscreteCallback(STOPFUN_Q, terminate!)
    
    inter_time = @elapsed begin

        t_span = (0.0, sim_seconds)
        dts = (year_seconds, dt_scale * 2 * ops.hmin / (sqrt(B_p.μ_out/B_p.ρ_out)))
        
        prob = ODEProblem(Q_DYNAMIC!, ψδγ, t_span, static_params)
        
        sol = solve(prob, Tsit5(); isoutofdomain=stepcheck,
                    dt=dts[2],
                    abstol = 1e-12,
                    reltol = 1e-12,
                    gamma = .3,
                    save_everystep=false,
                    internalnorm=(x, _)->norm(x, Inf),
                    #saveat = year_seconds,
                    callback=stopper)
    end

    nothing

end
