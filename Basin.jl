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
    dir_out=read_params(ARGS[1])
    dir_out = string("../../erickson/output_files/", dir_out)

    nn = N + 1

    ### get simulation time
    year_seconds = 31556952
    sim_seconds = T * year_seconds
    t_now = 0.0
    if ic_t_file != "None"
        t_now = readdlm(ic_t_file)[1]
        t_span = (t_now, sim_seconds)
    else
        t_span = (0.0, sim_seconds)
    end

    ### basin Params
    B_p = (μ_out = 36.0,
           ρ_out = 2.8,
           μ_in = 8.0,
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

    @printf "got grid in %f seconds\n" grid_t
    flush(stdout)

    ### get fault params
    fc = metrics.facecoord[2][1]
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
    slip_file,
    slip_rate_file,
    stress_file,
    state_file = make_ss(dir_out, fc, δNp, ARGS[1])
    station_names = make_stations(dir_out)
    u_file, v_file = make_uv_files(dir_out, x[1:2:nn, 1], y[1, 1:2:nn])
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


    ### get initial condtions
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

    ### parameter orginzation
    io = (dir_name = dir_out,
          slip_file = slip_file,
          slip_rate_file = slip_rate_file,
          stress_file = stress_file,
          state_file = state_file,
          station_names = station_names,
          pf = [0, 0.0, 0.0],
          u_file = u_file,
          v_file = v_file)
    
    vars = (u_prev = zeros(nn^2),
            t_prev = [0.0, 0.0],
            Δτ = zeros(nn),
            vf = zeros(nn),
            u = zeros(nn^2),
            ge = zeros(nn^2))

    static_params = (year_seconds,
                     reject_step = [false],
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
                      Z̃f1 = CuArray(ops.Z̃f[1]),
                      Z̃f2 = CuArray(ops.Z̃f[2]),
                      Z̃f3 = CuArray(ops.Z̃f[3]),
                      L2 = CuSparseMatrixCSC(ops.L[2]),
                      L3 = CuSparseMatrixCSC(ops.L[3]),
                      H = CuArray(diag(ops.H[1])),
                      JIHP = CuSparseMatrixCSC(ops.JIHP),
                      nCnΓ1 = CuSparseMatrixCSC(ops.nCnΓ1),
                      HIGΓL1 = CuSparseMatrixCSC(ops.HIGΓL1),
                      RS = CuArray([RS.a, RS.σn, RS.V0, RS.Dc, RS.f0, nn]),
                      b = CuArray(RS.b),
                      τ̃f = CuArray(zeros(nn)),
                      v̂ = CuArray(zeros(nn)),
                      source2 = CuArray(zeros(nn)),
                      source3 = CuArray(zeros(nn)),
                      fc = metrics.facecoord[2][1],
                      Lw = Lw,
                      io = io,
                      d_to_s = d_to_s,
                      RS_cpu = RS)
    
    @printf "Approximately %f Gib to GPU\n\n" Base.summarysize(dynamic_params)/1e9
    flush(stdout)

    ### set dynamic timestep
    dts = (year_seconds, dt_scale * 2 * ops.hmin / (sqrt(B_p.μ_out/B_p.ρ_out)))
    
    ### begin cycles
    cycles = 1
    done = false
    while t_now < T * year_seconds

        static_params.cycles[1] = cycles
        @printf "On cycle %d\n" cycles
        @printf "Begining Inter-seismic period...\n"
        flush(stdout)

        ### run inter-seismic period
        static_params.reject_step[1] = false
        stopper = DiscreteCallback(STOPFUN_Q, terminate!)
        prob = ODEProblem(Q_DYNAMIC!, ψδ, t_span, static_params)
        
        inter_flag = false
        # while intergration is not finished
        while !inter_flag
            inter_time = @elapsed begin
                # integrate
                prob = ODEProblem(Q_DYNAMIC!, ψδ, t_span, static_params)
                sol = solve(prob, Tsit5(); isoutofdomain=stepcheck, dt=dts[2],
                            atol = 1e-12, rtol = 1e-12, save_everystep=true,
                            internalnorm=(x, _)->norm(x, Inf), callback=stopper)
            end
            
            # if the timestepper fails
            if sol.retcode == :Unstable
                # try again
                @printf "Broke out of Integrator early\n"
                @printf "Restarting...\n"
                nt_begin = reset_quasidynamic!(ψδ, static_params)
                t_span = (nt_begin, sim_seconds)
                #otherwise break
            elseif sol.retcode == :Terminated
                @printf "Finished Interseismic\n"
                @printf "Interseismic period took %s seconds. \n" inter_time
                flush(stdout)
                
                ### get dynamic inital conditions
                t_now = sol.t[end]
                t_span = (t_now,  sim_seconds)
                @printf "Simulation time is now %s years. \n\n" t_span[1]/year_seconds

                q = Array(q)
                q[1:nn^2] .= static_params.vars.u[:]
                q[nn^2 + 1 : 2nn^2] .=
                    (static_params.vars.u - static_params.vars.u_prev) / 
                    (sol.t[end] - static_params.vars.t_prev[1])
                
                for i in 1:4
                    q[2nn^2 + (i-1)*nn + 1 : 2nn^2 + i*nn] .= ops.L[i]*static_params.vars.u
                end
                q[2nn^2 + 4nn + 1 : 2nn^2 + 5nn] .= sol.u[end][1:nn]

                inter_flag = true

            elseif sol.retcode == :Success
                inter_flag = true
                @printf "Finished Interseismic\n"
                @printf "Simulation time is now %s year. \n\n" sol.t[end]/year_seconds
                done = true
            end
        end 

        # to breakout on last cycle
        if done == true
            break
        end

        ### write break to output file
        temp_io = open(io.slip_file, "a")
        writedlm(temp_io, ["BREAK"])
        close(temp_io)

        @printf "Begining Co-seismic period...\n"
        flush(stdout)

        co_time = @elapsed begin
            
            ### getting source terms for non-reflecting boundaries
            dynamic_params.source2 .= CuArray(metrics.sJ[2] .* (ops.Z̃f[2] .*
                ops.L[2] * q[nn^2 + 1 : 2nn^2] +
                traction(ops, metrics, 2, q[1:nn^2],
                         f2_data(RS, metrics.μf2, Lw, t_now))))

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

        ### get inter-seismic so
        if t_now != nothing

            ψδ[1:nn] .= Array(q[2nn^2 + 4*nn + 1 : 2nn^2 + 5*nn])
            ψδ[nn + 1: 2nn] .= Array(2 * q[1 : nn : nn^2])
            static_params.vars.t_prev[2] = t_now/year_seconds
            t_span = (t_now, sim_seconds)
            
            temp_io = open(io.slip_file, "a")
            temp_io2 = open(u_file, "a")
            temp_io3 = open(v_file, "a")
            
            writedlm(temp_io, ["BREAK"])
            writedlm(temp_io2, ["BREAK"])
            writedlm(temp_io3, ["BREAK"])
            
            close(temp_io)
            close(temp_io2)
            close(temp_io3)
            
            @printf "Simulation time is now %s years. \n\n" t_span[1]/year_seconds
            static_params.io.pf[1] = 0
            static_params.io.pf[2] = 0
            cycles += 1

        end
    end
    
end


