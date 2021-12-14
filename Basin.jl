include("numerical.jl")
include("solvers.jl")
include("physical_params.jl")
include("read_in.jl")
include("domain.jl")
include("write_out.jl")

using DelimitedFiles
using Printf
using OrdinaryDiffEq
using CUDA
using CUDA.CUSPARSE
using Plots

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
    Dc = read_params(ARGS[1])
    
    @show d_to_s
    nn = N + 1

    # get simulation time
    year_seconds = 31556952
    sim_seconds = T * year_seconds
    t_now = 0.0
    if ic_t_file != "None"
        t_now = readdlm(ic_t_file)
        t_span = (t_now, sim_seconds)

    else
        t_span = (0.0, sim_seconds)
    end

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

    @printf "got grid in %f seconds\n" grid_t
    flush(stdout)

    # fault Params
    fc = metrics.facecoord[2][1]
    (x, y) = metrics.coord
    for i in 2:length(fc)
        @show fc[i], fc[i] - fc[i-1]
    end
    #quit()

    η = metrics.η

    δNp, 
    gNp, 
    VWp, 
    RS = fault_params(fc, Dc)

    # io stuffs
    dir_name,
    slip_file,
    slip_rate_file,
    stress_file,
    state_file = make_ss(fc, δNp, ARGS[1])
    station_names = make_stations(dir_name)
    u_file, v_file = make_uv_files(dir_name, x[1:2:δNp, 1], y[1, 1:2:δNp])
    @printf "set-up io\n"
    flush(stdout)

    # getting discrete operators
    faces = [0 2 3 4]
    R = [-1 0 1 0]
    opt_t = @elapsed begin
        ops = operators(p, N, N, μ, ρ, R, B_p, faces, metrics)
    end
    @printf "Got operators\n"
    flush(stdout)
    # getting initial condtions
    if ic_file != "None"
        ψδ = readdlm(ic_file)
    else
        ψδ = zeros(2nn)
        for n in 1:nn
            ψδ[n] = RS.a * log(2*(RS.V0/RS.Vp) * sinh((RS.τ_inf - η[n]*RS.Vp)/(RS.σn*RS.a)))
        end
        ψδ[δNp .+ (1:nn)] .= 0
    end

    q = Array{Float64, 1}(undef, 2nn^2 + 5*nn)

    @printf "Got initial conditions\n"
    flush(stdout)
    # getting everything together for solvers
    io = (dir_name = dir_name,
          slip_file = slip_file,
          slip_rate_file = slip_rate_file,
          stress_file = stress_file,
          state_file = state_file,
          station_names = station_names,
          pf = [0, 0.0, 0.0])
    
    vars = (u_prev = zeros(nn^2),
            t_prev = [0.0, 0.0],
            year_seconds = year_seconds,
            Δτ = zeros(nn),
            τ = zeros(nn),
            u = zeros(nn^2),
            ge = zeros(nn^2))

    
    static_params = (reject_step = [false],
                     Lw = Lw,
                     nn = nn,
                     d_to_s = d_to_s,
                     vars = vars,
                     ops = ops,
                     metrics = metrics,
                     io = io,
                     RS = RS,
                     vf = zeros(nn),
                     cycles = [0])

    threads = 512
    dynamic_params = (nn = nn,
                      threads = threads,
                      blocks = cld(nn, threads),
                      Λ = CuSparseMatrixCSC(ops.Λ),
                      sJ = CuArray(metrics.sJ[1]),
                      Z̃f = CuArray(ops.Z̃f[1]),
                      L = CuSparseMatrixCSC(ops.L[1]),
                      H = CuArray(diag(ops.H[1])),
                      JIHP = CuSparseMatrixCSC(ops.JIHP),
                      nCnΓ1 = CuSparseMatrixCSC(ops.nCnΓ1),
                      nBBCΓL1 = CuSparseMatrixCSC(ops.nBBCΓL1),
                      RS = CuArray([RS.a, RS.σn, RS.V0, RS.Dc, RS.f0, nn]),
                      b = CuArray(RS.b),
                      τ̃f = CuArray(zeros(nn)),
                      v̂= CuArray(zeros(nn)),
                      fc = metrics.facecoord[2][1],
                      Lw = Lw,
                      io = io,
                      d_to_s = d_to_s)

        
    @printf "Approximately %f Gib to GPU\n" Base.summarysize(dynamic_params)/1e9
    flush(stdout)
    dts = (year_seconds, dt_scale * 2 * ops.hmin / (sqrt(B_p.μ_out/B_p.ρ_out)))
    
    cycles = 1
    plot()
    while t_now < T * year_seconds
        static_params.cycles[1] = cycles
        @printf "On cycle %d\n" cycles
        
        @printf "Begining Inter-seismic period...\n"
        flush(stdout)
        static_params.reject_step[1] = false
        stopper = DiscreteCallback(STOPFUN_Q, terminate!)
        prob = ODEProblem(Q_DYNAMIC!, ψδ, t_span, static_params)
        inter_time = @elapsed begin
            sol = solve(prob, Tsit5(); isoutofdomain=stepcheck, dt=dts[2],
                        atol = 1e-12, rtol = 1e-12, save_everystep=true,
                        internalnorm=(x, _)->norm(x, Inf), callback=stopper)
        end
        
        @printf "Interseismic period took %s seconds. \n" inter_time
        flush(stdout)
        # dynamic inital conditions
        t_now = sol.t[end]
        t_span = (t_now,  sim_seconds)

        @printf "Simulation time is now %s years. \n\n" t_span[1]/year_seconds

        q = Array(q)
        q[1:nn^2] .= static_params.vars.u[:]
        q[nn^2 + 1 : 2nn^2] .= (static_params.vars.u - static_params.vars.u_prev)/(sol.t[end] - static_params.vars.t_prev[1])
        
        for i in 1:4
            q[2nn^2 + (i-1)*nn + 1 : 2nn^2 + i*nn] .= ops.L[i]*static_params.vars.u
        end
        q[2nn^2 + 4nn + 1 : 2nn^2 + 5nn] .= sol.u[end][1:nn]
        
        temp_io = open(io.slip_file, "a")
        writedlm(temp_io, ["BREAK"])
        close(temp_io)

        @printf "Begining Co-sesimic period...\n"
        flush(stdout)
        co_time = @elapsed begin
            q = CuArray(q)
            t_now = timestep_write!(q, FAULT_GPU!, dynamic_params, dts[2], t_span)
        end
        @printf "Coseismic period took %s seconds. \n" co_time
        flush(stdout)
        ψδ[1:nn] .= Array(q[2nn^2 + 4*nn + 1 : 2nn^2 + 5*nn])
        ψδ[nn + 1: 2nn] .= Array(2 * q[1 : nn : nn^2])
        static_params.vars.t_prev[2] = t_now/year_seconds
        t_span = (t_now, sim_seconds)
        
        temp_io = open(io.slip_file, "a")
        writedlm(temp_io, ["BREAK"])
        close(temp_io)

        @printf "Simulation time is now %s years. \n\n" t_span[1]/year_seconds
        static_params.io.pf[1] = 0
        static_params.io.pf[2] = 0
        cycles += 1
    end

end


