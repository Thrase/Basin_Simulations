using Plots
using Printf
using CUDA
using CUDA.CUSPARSE

include("../physical_params.jl")
include("mms_funcs.jl")
include("../domain.jl")
include("../numerical.jl")
include("../solvers.jl")

function refine(ps, ns, t_span, Lw, D, B_p, RS, R, MMS)
    

    #xt, yt = transforms_e(Lw, .1, .05)
    # expand to (0,Lw) × (0, Lw)
    (x1, x2, x3, x4) = (0, 1, 0, 1)
    (y1, y2, y3, y4) = (0, 0, 1, 1)
    xt, yt = transfinite(x1, x2, x3, x4, y1, y2, y3, y4)
    
    for p in ps
        err1 = Vector{Float64}(undef, length(ns))
        err2 = Vector{Float64}(undef, length(ns))
        err3 = Vector{Float64}(undef, length(ns))
        err4 = Vector{Float64}(undef, length(ns))
        err5 = Vector{Float64}(undef, length(ns))
        for (iter, N) in enumerate(ns)
            
            nn = N + 1
            Nn = nn^2
            mt = @elapsed begin
                metrics = create_metrics(N, N, B_p, μ, xt, yt)
            end

            @printf "Got metrics: %s s\n" mt
       
            x = metrics.coord[1]
            y = metrics.coord[2]

            ot = @elapsed begin
                faces_fault = [0 2 3 4]
                d_ops = operators_dynamic(p, N, N, μ, ρ, R, B_p, faces_fault, metrics)
                
                #faces = [1 2 3 4]
                #d_ops_waveprop = operators_dynamic(p, N, N, B_p, μ, ρ, R, faces, metrics, LFtoB)
                b = b_fun(metrics.facecoord[2][1], RS)
                τ̃f = Array{Float64, 1}(undef, nn)
                vf = Array{Float64, 1}(undef, nn)
                v̂_fric = Array{Float64, 1}(undef, nn)
                
                cpu_operators = (d_ops = d_ops,
                                 Λ_waveprop = d_ops.Λ,
                                 JIHP = d_ops.JIHP,
                                 nn = nn,
                                 fc = metrics.facecoord,
                                 coord = metrics.coord,
                                 R = R,
                                 B_p = B_p,
                                 MMS = MMS,
                                 RS = RS,
                                 b = b,
                                 sJ = metrics.sJ,
                                 τ̃f = τ̃f,
                                 vf = vf,
                                 v̂_fric = v̂_fric,
                                 CHAR_SOURCE = S_c,
                                 STATE_SOURCE = S_rs,
                                 FORCE = Forcing)
                

                dt_scale = .0001
                dt = dt_scale * 2 * d_ops.hmin / (sqrt(B_p.μ_out/B_p.ρ_out))
                nstep = ceil(Int, (t_span[2] - t_span[1]) / dt)
                dt = (t_span[2] - t_span[1]) / nstep

                x = metrics.coord[1]
                y = metrics.coord[2]
                fc = metrics.facecoord

                GPU_operators = (nn = nn,
                                 #Λ_fault = d_ops_fault.Λ,
                                 Λ_waveprop = CuArray(d_ops.Λ),
                                 Z̃f = d_ops.Z̃f,
                                 L = d_ops.L,
                                 H = d_ops.H,
                                 P̃I = d_ops.P̃I,
                                 JIHP = CuArray(d_ops.JIHP),
                                 nCnΓ1 = d_ops.nCnΓ1,
                                 nBBCΓL1 = d_ops.nBBCΓL1,
                                 sJ = metrics.sJ,
                                 RS = RS,
                                 b = b,
                                 vf = vf,
                                 τ̃f = τ̃f,
                                 v̂_fric = v̂_fric)
                
            end

            @printf "Got Operators: %s s\n" ot

            it = @elapsed begin
                u0 = ue(x[:], y[:], 0.0, MMS)
                v0 = ue_t(x[:], y[:], 0.0, MMS)
                q1 = [u0;v0]
                for i in 1:4
                    q1 = vcat(q1, d_ops.L[i]*u0)
                end
                q1 = vcat(q1, ψe(metrics.facecoord[1][1],
                                metrics.facecoord[2][1],
                                 0, B_p, RS, MMS))

                q3 = CuArray(deepcopy(q1))
                q4 = deepcopy(q1)
                q5 = deepcopy(q1)
                @assert length(q1) == 2nn^2 + 5nn

            end

            @printf "Got initial conditions: %s s\n" it
            @printf "Running simulations with %s nodes...\n" nn
            @printf "\n___________________________________\n"
            
            #=
            st3 = @elapsed begin
                timestep!(q3, WAVEPROP!, GPU_operators, dt, t_span)
                #Euler_GPU_WAVEPROP!(q3, GPU_operators, dt, t_span)
            end
            =#

            #@printf "Ran GPU to time %s in: %s s \n\n" t_span[2] st3

            st4 = @elapsed begin
               timestep!(q4, MMS_FAULT_CPU!, cpu_operators, dt, t_span)
            end
            
            @printf "Ran CPU MMS to time %s in: %s s \n\n" t_span[2] st4
            
            #=
            st5 = @elapsed begin
                timestep!(q5, WAVEPROP!, cpu_operators, dt, t_span)
            end

            @printf "Ran CPU to time %s in: %s s \n\n" t_span[2] st5
            =#
            
            #u_end3 = @view Array(q3)[1:Nn]
            #u_end5 = @view q5[1:Nn]

            u_end4 = @view q4[1:Nn]
            diff_u4 = u_end4 - ue(x[:], y[:], t_span[2], MMS)
            err4[iter] = sqrt(diff_u4' * d_ops.JH * diff_u4)
            
            #=
            contour(x[:,1], x[1,:],
                    (reshape(u_end3, (nn, nn)) .- ue(x, y, t_span[2], MMS))',
                    xlabel="off fault", ylabel="depth", fill=true, yflip=true)
            gui()
            =#

            #@printf "L2 error displacements between CPU and GPU waveprop: %e\n\n" norm(u_end5 - u_end3)
            #@printf "GPU fault error: %e\n\n" err1[iter]
            #@printf "CPU fault  error: %e\n\n" err2[iter]
            #@printf "GPU waveprop error: %e\n\n" err3[iter]
            @printf "CPU error: %e\n\n" err4[iter]
            if iter > 1
                #@printf "GPU fault rate: %f\n" log(2, err1[iter - 1]/err1[iter])
                #@printf "CPU fault rate: %f\n" log(2, err2[iter - 1]/err2[iter])
                #@printf "GPU waveprop rate: %f\n" log(2, err3[iter - 1]/err3[iter])
                @printf "CPU rate: %f\n" log(2, err4[iter - 1]/err4[iter])
            end
            
            @printf "___________________________________\n\n"
        end
    end

end
