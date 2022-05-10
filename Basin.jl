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

    ### input parameters
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
    B_on,
    dir_out,
    volume_plots,
    cycle_flag,
    num_cycles,
    intime_plotting,
    μ_in= read_params(ARGS[1])
    dir_out = string("../../../erickson/output_files/", dir_out)

    nn = N + 1

    ### get simulation time
    year_seconds = 31556952
    sim_seconds = T * year_seconds
    t_now = 0.0
    if ic_t_file != "None"
        t_now = readdlm(ic_t_file)[1]
        @show typeof(t_now)
        t_span = (t_now, sim_seconds)
    else
        t_span = (0.0, sim_seconds)
    end

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
        xt, yt = transforms_n(Lw)
        metrics = create_metrics(N, N, B_p, μ, ρ, xt, yt)
    end

    @printf "got grid in %f seconds\n" grid_t
    flush(stdout)

    ### get fault params
    fc = Array(metrics.facecoord[2][1])
    (x, y) = metrics.coord
    #for i in 2:length(fc)
    #    @show fc[i], fc[i] - fc[i-1]
    #end
    
    η = metrics.η

    δNp, 
    gNp, 
    VWp, 
    RS = fault_params(fc, Dc)

    ### setup io
    stations = collect(0.0:2.0:22.0)
    fault_name,
    station_name,
    station_r_name,
    volume_name = new_dir(dir_out, ARGS[1], stations, fc, x[1:2:nn, 1], y[1, 1:2:nn])

    @printf "set-up io\n"
    flush(stdout)

    ### get discrete operators
    faces = [0 2 3 4]
    R = [-1 0 1 0]
    opt_t = @elapsed begin
        ops = operators(p, N, N, μ, ρ, R, B_p, metrics)
    end
    @printf "Got operators in %f seconds\n" opt_t
    flush(stdout)
    
    # make plot opject for slip plot if plotting while running
    if intime_plotting == 1
        slip_plot = plot(legend=false, yflip = true, ylabel="Depth(Km)", xlabel="Slip(m)")
    else
        slip_plot = nothing
    end

    ### get initial condtions
    if ic_file != "None"
        ψδ = readdlm(ic_file)
    else
        ψδ = zeros(2nn)
        for n in 1:nn
            ψδ[n] = RS.a * log(2*(RS.V0/RS.Vp) * sinh((RS.τ_inf - η[n]*RS.Vp)/(RS.σn*RS.a)))
        end
        ψδ[nn + 1: 2nn] .= 0
    end

    q = Array{Float64, 1}(undef, 2nn^2 + 5*nn)

    @printf "Got initial conditions\n"
    flush(stdout)

    ### parameter orginzation
    io = (dir_name = dir_out,
          fault_name = fault_name,
          station_name = station_name,
          station_r_name = station_r_name,
          volume_name = volume_name,
          stations = stations,
          pf = [0, 0.0, 0.0],
          vp = volume_plots,
          slip_plot = [slip_plot])
    
    vars = (u_prev = zeros(nn^2),
            t_prev = [0.0, 0.0],
            Δτ = zeros(nn),
            vf = zeros(nn),
            u = zeros(nn^2),
            uf2 = zeros(nn),
            ge = zeros(nn^2))

    vars.uf2 .= (RS.τ_inf * Lw) ./ metrics.μf2
    
    static_params = (year_seconds,
                     reject_step = [false],
                     Lw = Lw,
                     nn = nn,
                     d_to_s = d_to_s,
                     vars = vars,
                     ops = ops,
                     metrics = metrics,
                     fc = Array(metrics.facecoord[2][1]),
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
                      Z̃f1 = CuArray(ops.Z̃f[1]),
                      Z̃f2 = CuArray(ops.Z̃f[2]),
                      Z̃f3 = CuArray(ops.Z̃f[3]),
                      L2 = CuSparseMatrixCSC(ops.L[2]),
                      L3 = CuSparseMatrixCSC(ops.L[3]),
                      H = CuArray(diag(ops.H[1])),
                      JIHP = CuSparseMatrixCSC(ops.JIHP),
                      nCnΓ1 = CuSparseMatrixCSC(ops.nCnΓ1),
                      HIGΓL1 = CuSparseMatrixCSC(ops.HIGΓL1),
                      HIG = CuSparseMatrixCSC(ops.HI[1] * ops.G[1]),
                      RS = CuArray([RS.a, RS.σn, RS.V0, RS.Dc, RS.f0, nn]),
                      b = CuArray(RS.b),
                      τ̃f = CuArray(zeros(nn)),
                      v̂ = CuArray(zeros(nn)),
                      source2 = CuArray(zeros(nn)),
                      source3 = CuArray(zeros(nn)),
                      fc = Array(metrics.facecoord[2][1]),
                      Lw = Lw,
                      io = io,
                      d_to_s = d_to_s,
                      RS_cpu = RS,
                      δ = zeros(nn),
                      v̂_cpu = zeros(nn),
                      τ̂_cpu = zeros(nn),
                      τ̃_cpu = zeros(nn),
                      ψ_cpu = zeros(nn))
    
    @printf "Approximately %f Gib to GPU\n\n" Base.summarysize(dynamic_params)/1e9
    flush(stdout)

    ### set dynamic timestep
    dts = (year_seconds, dt_scale * 2 * ops.hmin / (sqrt(B_p.μ_out/B_p.ρ_out)))
    
    ### begin cycles
    cycles = 1
    done = false
    while break_con(t_now, sim_seconds, cycle_flag, cycles, num_cycles)

        static_params.cycles[1] = cycles
        @printf "On cycle %d\n" cycles
        @printf "Begining Inter-seismic period...\n"
        flush(stdout)

        ### run inter-seismic period
        static_params.reject_step[1] = false
        stopper = DiscreteCallback(STOPFUN_Q, terminate!)
        prob = ODEProblem(Q_DYNAMIC!, ψδ, t_span, static_params)
        
        inter_time = @elapsed begin
            # integrate
            prob = ODEProblem(Q_DYNAMIC!, ψδ, t_span, static_params)
            sol = solve(prob, Tsit5(); isoutofdomain=stepcheck,
                        dt=dts[2],
                        atol = 1e-12,
                        rtol = 1e-12,
                        gamma = .3,
                        save_everystep=true,
                        internalnorm=(x, _)->norm(x, Inf),
                        #saveat = year_seconds,
                        callback=stopper)
            end
            
        @printf "Finished Interseismic\n"
        @printf "Interseismic period took %s seconds. \n" inter_time
        flush(stdout)
        
        ### get dynamic inital conditions
        t_now = sol.t[end]
        t_span = (t_now,  sim_seconds)
        @printf "Simulation time is now %s years. \n\n" t_span[1]/year_seconds
        flush(stdout)
        q = Array(q)
        q[1:nn^2] .= static_params.vars.u[:]
        q[nn^2 + 1 : 2nn^2] .=
            (static_params.vars.u - static_params.vars.u_prev) / 
            (sol.t[end] - static_params.vars.t_prev[1])
        
        q[2nn^2 + 1 : 2nn^2 + nn] .= sol.u[end][nn+1:2nn]./2
        
        for i in 2:4
            q[2nn^2 + (i-1)*nn + 1 : 2nn^2 + i*nn] .= ops.L[i]*static_params.vars.u
        end
        
        q[2nn^2 + 4nn + 1 : 2nn^2 + 5nn] .= sol.u[end][1:nn]
        
        @printf "Begining Co-seismic period...\n"
        flush(stdout)

        co_time = @elapsed begin
            
            ### getting source terms for non-reflecting boundaries
            dynamic_params.source2 .= CuArray(metrics.sJ[2] .* (ops.Z̃f[2] .*
                ops.L[2] * q[nn^2 + 1 : 2nn^2] +
                traction(ops, metrics, 2, q[1:nn^2],
                         ops.L[2] * q[1:nn^2])))

            dynamic_params.source3 .= CuArray(metrics.sJ[3] .* (ops.Z̃f[3] .*
            ops.L[3] * q[nn^2 + 1 : 2nn^2] +
            traction(static_params.ops, metrics, 3, q[1:nn^2],
                     ops.L[3] * q[1:nn^2])))
            
            q = CuArray(q)

            ### run inter-seismic solver
            t_now = timestep_write!(q, FAULT_GPU!, dynamic_params, dts[2], t_span)
        end
        @printf "Finised Co-seismic period\n"
        @printf "Coseismic period took %s seconds. \n" co_time
        flush(stdout)

        ### get inter-seismic initial conditions
        if t_now != nothing

            ψδ[1:nn] .= Array(q[2nn^2 + 4*nn + 1 : 2nn^2 + 5*nn])
            ψδ[nn + 1: 2nn] .= Array(2 * q[2nn^2 + 1 : 2nn^2 + nn])
            static_params.vars.t_prev[2] = t_now
            static_params.vars.uf2 .= Array(dynamic_params.L2 * q[1:nn^2])
            t_span = (t_now, sim_seconds)
            
            
            @printf "Simulation time is now %s years. \n\n" t_span[1]/year_seconds
            static_params.io.pf[1] = 0
            static_params.io.pf[2] = 0
            cycles += 1

        end
    end
    
end


